"""OpenCode v2 client tests against a mocked ``/api/*`` server.

Payload shapes are taken from a live ``opencode serve`` 2.1.0."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from a2a.types import TaskState

from opencode_a2a.execution.stream_events import (
    _extract_interrupt_asked_event,
    _extract_interrupt_resolved_event,
    _extract_stream_terminal_signal,
    _extract_token_usage,
    _extract_upstream_error_from_response,
)
from opencode_a2a.opencode_upstream_client import UpstreamContractError
from opencode_a2a.opencode_upstream_client_v2 import (
    SYSTEM_INSTRUCTION_KEY,
    OpencodeV2UpstreamClient,
    _just_before,
)
from tests.support.settings import make_settings

SID = "ses_ea62df78dffeychz5Lb3WupGBQ"
USER_MSG = "msg_159d208c4001tLNDeTH6y952cl"
ASSISTANT_MSG = "msg_159d20906001LpnF9zP1nIeOEw"
IDLE_MSG = "msg_159d216680018cycWhYvX67tMY"


def _assistant(text: str = "PONG", *, msg_id: str = ASSISTANT_MSG, **extra: Any) -> dict:
    message = {
        "id": msg_id,
        "type": "assistant",
        "agent": "build",
        "model": {"id": "big-pickle", "providerID": "opencode"},
        "content": [
            {"type": "reasoning", "text": "thinking"},
            {"type": "text", "text": text},
        ],
        "finish": "stop",
        "cost": 0,
        "tokens": {"input": 19, "output": 4, "reasoning": 12, "cache": {"read": 1, "write": 0}},
    }
    message.update(extra)
    return message


def _idle(outcome: str = "succeeded", *, msg_id: str = IDLE_MSG, error: Any = None) -> dict:
    message: dict[str, Any] = {"id": msg_id, "type": "idle", "outcome": outcome}
    if error is not None:
        message["error"] = error
    return message


def _user(msg_id: str = USER_MSG, text: str = "hi") -> dict:
    return {"id": msg_id, "type": "user", "text": text}


class FakeV2Server:
    """Records requests and serves canned v2 responses."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.messages: list[dict] = [_idle(), _assistant(), _user()]  # newest first
        self.wait_calls = 0
        self.idle_after_waits = 1
        self.sse_lines: list[str] = []
        self.routes: dict[tuple[str, str], Any] = {}

    def body(self, request: httpx.Request) -> Any:
        return json.loads(request.content) if request.content else None

    def calls(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        if key in self.routes:
            route = self.routes[key]
            return route(request) if callable(route) else route
        path = request.url.path
        if request.method == "GET" and path == "/api/event":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="\n".join(self.sse_lines).encode(),
            )
        if request.method == "POST" and path == "/api/session":
            return httpx.Response(200, json={"data": {"id": SID, "title": "t"}})
        if request.method == "POST" and path.endswith("/prompt"):
            return httpx.Response(
                200, json={"data": {"id": USER_MSG, "sessionID": SID, "type": "user"}}
            )
        if request.method == "POST" and path.endswith("/wait"):
            self.wait_calls += 1
            return httpx.Response(204)
        if request.method == "GET" and path.endswith("/message"):
            if self.wait_calls < self.idle_after_waits:
                data = [m for m in self.messages if m["type"] != "idle"]
            else:
                data = list(self.messages)
            limit = int(request.url.params.get("limit", "100"))
            if request.url.params.get("order") == "asc":
                data = list(reversed(data))
            return httpx.Response(
                200, json={"data": data[:limit], "cursor": {"next": "c-next", "previous": "c-p"}}
            )
        if request.method == "POST" and path.split("/")[-1] in {"agent", "model"}:
            return httpx.Response(204)
        if request.method == "PUT" and "/instructions/entries/" in path:
            return httpx.Response(204)
        return httpx.Response(404, json={"_tag": "NotFound", "message": path})


def _client(server: FakeV2Server, **settings: Any) -> OpencodeV2UpstreamClient:
    defaults: dict[str, Any] = {
        "opencode_workspace_root": "/workspace",
        "opencode_auth_password": "pw",  # pragma: allowlist secret
    }
    defaults.update(settings)
    client = OpencodeV2UpstreamClient(make_settings(**defaults))
    headers = dict(client._client.headers)
    client._client = httpx.AsyncClient(
        base_url="http://opencode", transport=httpx.MockTransport(server.handler), headers=headers
    )
    return client


@pytest.mark.asyncio
async def test_create_session_sends_location_title_and_agent() -> None:
    server = FakeV2Server()
    client = _client(server, opencode_agent="plan")
    assert await client.create_session("hello") == SID
    body = server.body(server.calls("POST", "/api/session")[0])
    assert body == {"title": "hello", "location": {"directory": "/workspace"}, "agent": "plan"}
    assert server.requests[0].headers["Authorization"].startswith("Basic ")
    await client.close()


@pytest.mark.asyncio
async def test_create_session_without_id_raises() -> None:
    server = FakeV2Server()
    server.routes[("POST", "/api/session")] = httpx.Response(200, json={"data": {}})
    client = _client(server)
    with pytest.raises(RuntimeError):
        await client.create_session()
    await client.close()


@pytest.mark.asyncio
async def test_send_message_waits_for_idle_and_maps_to_v1_shape() -> None:
    server = FakeV2Server()
    server.idle_after_waits = 2
    client = _client(server)
    message = await client.send_message(SID, "Reply with PONG")
    assert message.text == "PONG"
    assert message.message_id == ASSISTANT_MSG
    assert message.raw["info"]["role"] == "assistant"
    assert "error" not in message.raw["info"]
    assert [p["type"] for p in message.raw["parts"]] == ["reasoning", "text"]
    assert server.body(server.calls("POST", f"/api/session/{SID}/prompt")[0]) == {
        "text": "Reply with PONG"
    }
    # First wait returned before the turn started; the client waited again.
    assert server.wait_calls == 2
    assert _extract_upstream_error_from_response(message.raw) is None
    await client.close()


@pytest.mark.asyncio
async def test_send_message_failed_turn_surfaces_v1_error() -> None:
    server = FakeV2Server()
    server.messages = [
        _idle("failed", error={"type": "provider.no-route", "message": "Model unavailable"}),
        _user(),
    ]
    client = _client(server)
    message = await client.send_message(SID, "hi")
    error = _extract_upstream_error_from_response(message.raw)
    assert error is not None
    assert "Model unavailable" in error.message
    assert message.message_id == IDLE_MSG
    await client.close()


@pytest.mark.asyncio
async def test_send_message_auth_failure_maps_to_provider_auth_error() -> None:
    server = FakeV2Server()
    server.messages = [
        _idle("failed", error={"type": "provider.auth", "message": "bad key", "status": 401}),
        _user(),
    ]
    client = _client(server)
    message = await client.send_message(SID, "hi")
    assert message.raw["info"]["error"]["name"] == "ProviderAuthError"
    assert message.raw["info"]["error"]["data"]["statusCode"] == 401
    await client.close()


@pytest.mark.asyncio
async def test_send_message_interrupted_turn_is_aborted_error() -> None:
    server = FakeV2Server()
    server.messages = [_idle("interrupted"), _assistant(), _user()]
    client = _client(server)
    message = await client.send_message(SID, "hi")
    assert message.raw["info"]["error"]["name"] == "MessageAbortedError"
    await client.close()


@pytest.mark.asyncio
async def test_send_message_parts_files_model_variant_and_system() -> None:
    server = FakeV2Server()
    client = _client(server, opencode_system="Be brief.", opencode_variant="high")
    await client.send_message(
        SID,
        None,
        parts=[
            {"type": "text", "text": "Summarize"},
            {"type": "file", "url": "file:///w/a.pdf", "filename": "a.pdf", "mime": "x"},
            {"type": "tool", "tool": "a2a_call", "call_id": "c1", "output": "remote says hi"},
        ],
        model_override={"providerID": "opencode", "modelID": "big-pickle"},
    )
    prompt = server.body(server.calls("POST", f"/api/session/{SID}/prompt")[0])
    assert prompt["files"] == [{"uri": "file:///w/a.pdf", "name": "a.pdf"}]
    assert prompt["text"].startswith("Summarize")
    assert "[a2a_call result c1] remote says hi" in prompt["text"]
    model = server.body(server.calls("POST", f"/api/session/{SID}/model")[0])
    assert model == {"model": {"id": "big-pickle", "providerID": "opencode", "variant": "high"}}
    entry = server.calls(
        "PUT", f"/api/experimental/session/{SID}/instructions/entries/{SYSTEM_INSTRUCTION_KEY}"
    )
    assert server.body(entry[0]) == {"value": "Be brief."}

    # Same settings again: nothing is re-applied.
    await client.send_message(
        SID, "again", model_override={"providerID": "opencode", "modelID": "big-pickle"}
    )
    assert len(server.calls("POST", f"/api/session/{SID}/model")) == 1
    assert len(entry) == 1
    await client.close()


@pytest.mark.asyncio
async def test_model_override_is_reset_to_default_on_next_plain_turn() -> None:
    server = FakeV2Server()
    server.routes[("GET", "/api/model/default")] = httpx.Response(
        200, json={"data": {"id": "big-pickle", "providerID": "opencode"}}
    )
    client = _client(server)
    await client.send_message(SID, "a", model_override={"providerID": "x", "modelID": "y"})
    await client.send_message(SID, "b")
    models = [server.body(r) for r in server.calls("POST", f"/api/session/{SID}/model")]
    assert models == [
        {"model": {"id": "y", "providerID": "x"}},
        {"model": {"id": "big-pickle", "providerID": "opencode"}},
    ]
    await client.close()


@pytest.mark.asyncio
async def test_send_message_requires_text_or_parts() -> None:
    client = _client(FakeV2Server())
    with pytest.raises(ValueError):
        await client.send_message(SID)
    with pytest.raises(ValueError):
        await client.send_message(SID, parts=[])
    await client.close()


@pytest.mark.asyncio
async def test_send_message_times_out() -> None:
    server = FakeV2Server()
    server.idle_after_waits = 10**6
    client = _client(server)
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0])
    client._monotonic = lambda: next(ticks)
    with pytest.raises(httpx.TimeoutException):
        await client.send_message(SID, "hi", timeout_override=5.0)
    await client.close()


@pytest.mark.asyncio
async def test_prompt_without_id_is_contract_error() -> None:
    server = FakeV2Server()
    server.routes[("POST", f"/api/session/{SID}/prompt")] = httpx.Response(200, json={"data": {}})
    client = _client(server)
    with pytest.raises(UpstreamContractError):
        await client.send_message(SID, "hi")
    await client.close()


@pytest.mark.asyncio
async def test_http_errors_propagate() -> None:
    server = FakeV2Server()
    server.routes[("POST", f"/api/session/{SID}/prompt")] = httpx.Response(
        404, json={"_tag": "SessionNotFoundError", "message": "Session not found"}
    )
    client = _client(server)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.send_message(SID, "hi")
    assert exc_info.value.response.status_code == 404
    await client.close()


@pytest.mark.asyncio
async def test_list_messages_reverses_to_oldest_first_and_maps_cursor() -> None:
    server = FakeV2Server()
    client = _client(server)
    page = await client.list_messages(SID, params={"limit": 10, "before": "c-0"})
    request = server.calls("GET", f"/api/session/{SID}/message")[0]
    assert request.url.params["order"] == "desc"
    assert request.url.params["cursor"] == "c-0"
    assert [m["info"]["role"] for m in page.payload] == ["user", "assistant"]
    assert page.next_cursor == "c-next"
    await client.close()


@pytest.mark.asyncio
async def test_get_message_maps_and_rejects_non_chat_types() -> None:
    server = FakeV2Server()
    server.routes[("GET", f"/api/session/{SID}/message/{ASSISTANT_MSG}")] = httpx.Response(
        200, json={"data": _assistant()}
    )
    server.routes[("GET", f"/api/session/{SID}/message/{IDLE_MSG}")] = httpx.Response(
        200, json={"data": _idle()}
    )
    client = _client(server)
    message = await client.get_message(SID, ASSISTANT_MSG)
    assert message["info"]["modelID"] == "big-pickle"
    with pytest.raises(UpstreamContractError):
        await client.get_message(SID, IDLE_MSG)
    await client.close()


@pytest.mark.asyncio
async def test_session_queries_map_to_v1_shapes() -> None:
    server = FakeV2Server()
    session = {
        "id": SID,
        "title": "t",
        "location": {"directory": "/workspace"},
        "time": {"created": 1, "updated": 50},
    }
    child = {"id": "ses_child", "parentID": SID, "time": {"updated": 100}}
    server.routes[("GET", f"/api/session/{SID}")] = httpx.Response(200, json={"data": session})
    server.routes[("GET", "/api/session")] = httpx.Response(200, json={"data": [session, child]})
    server.routes[("GET", "/api/session/active")] = httpx.Response(
        200, json={"data": {SID: {"since": 1}}}
    )
    client = _client(server)
    assert (await client.get_session(SID))["directory"] == "/workspace"
    roots = await client.list_sessions(params={"roots": True, "limit": 5, "search": "t"})
    assert [s["id"] for s in roots] == [SID]
    recent = await client.list_sessions(params={"start": 60})
    assert [s["id"] for s in recent] == ["ses_child"]
    list_request = server.calls("GET", "/api/session")[0]
    assert list_request.url.params["directory"] == "/workspace"
    assert list_request.url.params["limit"] == "5"
    children = await client.list_child_sessions(SID)
    assert server.calls("GET", "/api/session")[-1].url.params["parentID"] == SID
    assert len(children) == 2
    assert await client.session_status() == {SID: {"type": "busy"}}
    assert await client.get_session_todo(SID) == []
    await client.close()


@pytest.mark.asyncio
async def test_session_controls_use_v2_routes() -> None:
    server = FakeV2Server()
    session = {"id": SID, "location": {"directory": "/workspace"}}
    server.routes[("GET", f"/api/session/{SID}")] = httpx.Response(200, json={"data": session})
    server.routes[("POST", f"/api/session/{SID}/interrupt")] = httpx.Response(
        200, json={"interrupted": False}
    )
    server.routes[("POST", f"/api/session/{SID}/compact")] = httpx.Response(
        200, json={"data": {"id": "msg_x"}}
    )
    server.routes[("POST", f"/api/session/{SID}/fork")] = httpx.Response(
        200, json={"data": {"id": "ses_fork", "location": {"directory": "/w"}}}
    )
    server.routes[("POST", f"/api/session/{SID}/revert/stage")] = httpx.Response(
        200, json={"data": {"messageID": USER_MSG}}
    )
    server.routes[("DELETE", f"/api/session/{SID}/revert")] = httpx.Response(204)
    server.routes[("GET", f"/api/session/{SID}/diff")] = httpx.Response(
        200, json={"data": [{"file": "a", "additions": 1, "deletions": 0}]}
    )
    client = _client(server)
    assert await client.abort_session(SID) is True
    assert await client.summarize_session(SID, {"providerID": "p", "modelID": "m"}) is True
    forked = await client.fork_session(SID, {"messageID": USER_MSG})
    assert forked["id"] == "ses_fork" and forked["directory"] == "/w"
    assert server.body(server.calls("POST", f"/api/session/{SID}/fork")[0]) == {"before": USER_MSG}
    assert (await client.revert_session(SID, {"messageID": USER_MSG}))["id"] == SID
    with pytest.raises(ValueError):
        await client.revert_session(SID, {})
    assert (await client.unrevert_session(SID))["id"] == SID
    diff = await client.get_session_diff(SID, params={"messageID": "m", "context": 3})
    assert diff[0]["file"] == "a"
    assert server.calls("GET", f"/api/session/{SID}/diff")[0].url.params.get("context") == "3"
    await client.close()


@pytest.mark.asyncio
async def test_prompt_async_command_and_shell() -> None:
    server = FakeV2Server()
    server.routes[("POST", f"/api/session/{SID}/command")] = httpx.Response(204)
    server.routes[("POST", f"/api/session/{SID}/shell")] = httpx.Response(204)
    server.routes[("GET", "/api/model/default")] = httpx.Response(
        200, json={"data": {"id": "big-pickle", "providerID": "opencode"}}
    )
    baseline_then_turn = iter(
        [
            [_user("msg_000")],
            [_idle(), _assistant("command output"), _user("msg_000")],
            [_user("msg_000")],
            [
                _idle(),
                {"id": "msg_shell", "type": "shell", "command": "ls", "output": {"text": "a\nb"}},
                _user("msg_000"),
            ],
        ]
    )

    def messages(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": next(baseline_then_turn), "cursor": {}})

    server.routes[("GET", f"/api/session/{SID}/message")] = messages
    client = _client(server)
    await client.session_prompt_async(
        SID,
        {
            "parts": [{"type": "text", "text": "later"}],
            "agent": "plan",
            "model": {"providerID": "p", "modelID": "m"},
        },
    )
    assert server.body(server.calls("POST", f"/api/session/{SID}/prompt")[0]) == {"text": "later"}
    assert server.body(server.calls("POST", f"/api/session/{SID}/agent")[0]) == {"agent": "plan"}

    command = await client.session_command(
        SID,
        {
            "command": "/review",
            "arguments": "src",
            "parts": [{"type": "file", "url": "file:///x", "mime": "text/plain"}],
        },
    )
    assert server.body(server.calls("POST", f"/api/session/{SID}/command")[0]) == {
        "name": "review",
        "text": "src",
        "files": [{"uri": "file:///x"}],
    }
    assert command["parts"][-1]["text"] == "command output"

    shell = await client.session_shell(SID, {"agent": "build", "command": "ls"})
    assert shell["parts"][0]["text"] == "a\nb"
    await client.close()


@pytest.mark.asyncio
async def test_unsupported_operations_raise_501() -> None:
    client = _client(FakeV2Server())
    for call in (
        client.share_session(SID),
        client.unshare_session(SID),
        client.list_workspaces(),
        client.create_workspace({"type": "x"}),
        client.remove_workspace("w"),
        client.reset_worktree({"directory": "/w"}),
    ):
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await call
        assert exc_info.value.response.status_code == 501
    await client.close()


@pytest.mark.asyncio
async def test_provider_catalog_is_assembled_from_v2_endpoints() -> None:
    server = FakeV2Server()
    server.routes[("GET", "/api/provider")] = httpx.Response(
        200, json={"data": [{"id": "opencode", "name": "OpenCode", "activation": "enabled"}]}
    )
    server.routes[("GET", "/api/model")] = httpx.Response(
        200,
        json={
            "data": [
                {
                    "id": "big-pickle",
                    "providerID": "opencode",
                    "name": "Big Pickle",
                    "status": "active",
                    "limit": {"context": 200000, "output": 32000},
                    "capabilities": {"tools": True, "input": ["text"], "reasoning": True},
                },
                {"id": "orphan", "providerID": "other"},
            ]
        },
    )
    server.routes[("GET", "/api/model/default")] = httpx.Response(
        200, json={"data": {"id": "big-pickle", "providerID": "opencode"}}
    )
    client = _client(server)
    catalog = await client.list_provider_catalog()
    assert catalog["default"] == {"opencode": "big-pickle"}
    assert catalog["connected"] == ["opencode"]
    model = catalog["all"][0]["models"]["big-pickle"]
    assert model["capabilities"] == {"toolcall": True, "reasoning": True, "attachment": False}
    assert {p["id"] for p in catalog["all"]} == {"opencode", "other"}
    assert server.calls("GET", "/api/provider")[0].url.params["location[directory]"] == "/workspace"
    await client.close()


@pytest.mark.asyncio
async def test_projects_and_worktrees() -> None:
    server = FakeV2Server()
    location = {
        "directory": "/workspace",
        "project": {"id": "prj1", "directory": "/workspace", "canonical": "/workspace"},
    }
    server.routes[("GET", "/api/location")] = httpx.Response(200, json=location)
    server.routes[("GET", "/api/project")] = httpx.Response(
        200, json=[{"id": "prj1", "canonical": "/workspace"}]
    )
    server.routes[("GET", "/api/worktree")] = httpx.Response(
        200, json={"data": [{"directory": "/wt/feature-a"}]}
    )
    server.routes[("POST", "/api/worktree")] = httpx.Response(
        200, json={"data": {"directory": "/wt/feature-b"}}
    )
    server.routes[("DELETE", "/api/worktree")] = httpx.Response(204)
    client = _client(server)
    assert (await client.list_projects())[0]["name"] == "workspace"
    assert (await client.get_current_project())["id"] == "prj1"
    assert await client.list_worktrees() == [{"directory": "/wt/feature-a", "name": "feature-a"}]
    created = await client.create_worktree({"name": "feature-b"})
    assert created == {"directory": "/wt/feature-b", "name": "feature-b"}
    assert server.body(server.calls("POST", "/api/worktree")[0]) == {
        "projectID": "prj1",
        "name": "feature-b",
    }
    assert await client.remove_worktree({"directory": "/wt/feature-b"}) is True
    assert server.body(server.calls("DELETE", "/api/worktree")[0]) == {
        "projectID": "prj1",
        "directory": "/wt/feature-b",
        "force": False,
    }
    await client.close()


@pytest.mark.asyncio
async def test_project_id_missing_is_contract_error() -> None:
    server = FakeV2Server()
    server.routes[("GET", "/api/location")] = httpx.Response(200, json={"directory": "/w"})
    client = _client(server)
    assert await client.get_current_project() is None
    with pytest.raises(UpstreamContractError):
        await client.list_worktrees()
    await client.close()


@pytest.mark.asyncio
async def test_permission_and_question_replies_route_through_session() -> None:
    server = FakeV2Server()
    form = {
        "id": "frm_1",
        "sessionID": SID,
        "fields": [
            {"key": "q0", "type": "string", "title": "Color"},
            {"key": "q1", "type": "multiselect", "title": "Pets"},
            {"key": "q2", "type": "boolean", "title": "Sure"},
            {"key": "q3", "type": "integer", "title": "Count"},
            {"key": "hidden", "type": "string", "hidden": True},
        ],
    }
    server.routes[("GET", f"/api/session/{SID}/form/frm_1")] = httpx.Response(
        200, json={"data": form}
    )
    server.routes[("POST", f"/api/session/{SID}/form/frm_1/reply")] = httpx.Response(204)
    server.routes[("DELETE", f"/api/session/{SID}/form/frm_1")] = httpx.Response(204)
    server.routes[("POST", f"/api/session/{SID}/permission/per_1/reply")] = httpx.Response(204)
    server.routes[("GET", "/api/permission/request")] = httpx.Response(
        200, json={"data": [{"id": "per_1", "sessionID": SID}]}
    )
    client = _client(server)
    # Not remembered locally: falls back to the server's pending list.
    assert await client.permission_reply("per_1", reply="always", message="ok") is True
    assert server.body(server.calls("POST", f"/api/session/{SID}/permission/per_1/reply")[0]) == {
        "decision": "always",
        "message": "ok",
    }
    await client.remember_interrupt_request(
        request_id="frm_1", session_id=SID, interrupt_type="question"
    )
    assert await client.question_reply(
        "frm_1", answers=[["Blue"], ["cats", "dogs"], ["yes"], ["3"]]
    )
    assert server.body(server.calls("POST", f"/api/session/{SID}/form/frm_1/reply")[0]) == {
        "answer": {"q0": "Blue", "q1": ["cats", "dogs"], "q2": True, "q3": 3}
    }
    assert await client.question_reject("frm_1") is True
    await client.close()


@pytest.mark.asyncio
async def test_unknown_interrupt_is_404() -> None:
    server = FakeV2Server()
    server.routes[("GET", "/api/form")] = httpx.Response(200, json={"data": []})
    client = _client(server)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.question_reject("frm_missing")
    assert exc_info.value.response.status_code == 404
    await client.close()


V2_STREAM = [
    {"type": "server.connected", "data": {}},
    {"type": "session.execution.started", "data": {"sessionID": SID}},
    {
        "type": "session.step.started",
        "data": {"sessionID": SID, "assistantMessageID": ASSISTANT_MSG},
    },
    {
        "type": "session.reasoning.started",
        "data": {"sessionID": SID, "assistantMessageID": ASSISTANT_MSG, "ordinal": 0},
    },
    {
        "type": "session.reasoning.delta",
        "data": {
            "sessionID": SID,
            "assistantMessageID": ASSISTANT_MSG,
            "ordinal": 0,
            "delta": "The",
        },
    },
    {
        "type": "session.tool.input.started",
        "data": {
            "sessionID": SID,
            "assistantMessageID": ASSISTANT_MSG,
            "id": "call_1",
            "name": "shell",
        },
    },
    {
        "type": "session.tool.called",
        "data": {
            "sessionID": SID,
            "assistantMessageID": ASSISTANT_MSG,
            "id": "call_1",
            "input": {"command": "echo hi"},
        },
    },
    {
        "type": "permission.asked",
        "data": {"id": "per_1", "sessionID": SID, "action": "shell", "resources": ["echo hi"]},
    },
    {
        "type": "permission.replied",
        "data": {"sessionID": SID, "requestID": "per_1", "reply": "once"},
    },
    {
        "type": "session.tool.success",
        "data": {
            "sessionID": SID,
            "assistantMessageID": ASSISTANT_MSG,
            "id": "call_1",
            "content": [{"type": "text", "text": "hi\n"}],
        },
    },
    {
        "type": "session.text.started",
        "data": {"sessionID": SID, "assistantMessageID": ASSISTANT_MSG, "ordinal": 0},
    },
    {
        "type": "session.text.delta",
        "data": {
            "sessionID": SID,
            "assistantMessageID": ASSISTANT_MSG,
            "ordinal": 0,
            "delta": "PONG",
        },
    },
    {
        "type": "session.text.ended",
        "data": {
            "sessionID": SID,
            "assistantMessageID": ASSISTANT_MSG,
            "ordinal": 0,
            "text": "PONG",
        },
    },
    {
        "type": "session.step.ended",
        "data": {
            "sessionID": SID,
            "assistantMessageID": ASSISTANT_MSG,
            "cost": 0,
            "tokens": {"input": 19, "output": 4, "reasoning": 12, "cache": {"read": 1}},
        },
    },
    {"type": "session.execution.succeeded", "data": {"sessionID": SID}},
]


@pytest.mark.asyncio
async def test_stream_events_translates_v2_events_to_v1() -> None:
    server = FakeV2Server()
    server.sse_lines = [": heartbeat", ""]
    for event in V2_STREAM:
        server.sse_lines += [f"data: {json.dumps(event)}", ""]
    server.sse_lines += ["data: not-json", "", "data: [1]", ""]
    client = _client(server)
    events = [event async for event in client.stream_events()]
    types = [e["type"] for e in events]
    assert types == [
        "message.part.updated",  # step-start
        "message.part.updated",  # reasoning started
        "message.part.delta",
        "message.part.updated",  # tool pending
        "message.part.updated",  # tool running
        "permission.asked",
        "permission.replied",
        "message.part.updated",  # tool completed
        "message.part.updated",  # text started
        "message.part.delta",
        "message.part.updated",  # text ended
        "message.part.updated",  # step-finish
        "session.idle",
    ]
    tool_done = events[7]["properties"]["part"]
    assert tool_done["tool"] == "shell"
    assert tool_done["state"] == {
        "status": "completed",
        "input": {"command": "echo hi"},
        "output": "hi\n",
    }
    delta = events[9]["properties"]
    assert delta["partID"] == f"{ASSISTANT_MSG}:text:0" and delta["delta"] == "PONG"
    assert _extract_token_usage(events[11])["input_tokens"] == 19
    asked = _extract_interrupt_asked_event(events[5])
    assert asked == {
        "request_id": "per_1",
        "interrupt_type": "permission",
        "details": {"permission": "shell", "patterns": ["echo hi"]},
    }
    assert _extract_interrupt_resolved_event(events[6])["resolution"] == "replied"
    signal = _extract_stream_terminal_signal(events[-1])
    assert signal is not None and signal.state == TaskState.TASK_STATE_COMPLETED
    assert server.calls("GET", "/api/event")
    await client.close()


def test_translator_failure_interrupt_and_forms() -> None:
    from opencode_a2a.opencode_v2 import V2EventTranslator

    translator = V2EventTranslator()
    failed = translator.translate(
        {
            "type": "session.execution.failed",
            "data": {"sessionID": SID, "error": {"type": "provider.no-route", "message": "nope"}},
        }
    )[0]
    signal = _extract_stream_terminal_signal(failed)
    assert signal is not None and signal.state == TaskState.TASK_STATE_FAILED
    assert "nope" in (signal.message or "")
    interrupted = translator.translate(
        {"type": "session.execution.interrupted", "data": {"sessionID": SID, "reason": "x"}}
    )[0]
    assert interrupted["properties"]["error"]["name"] == "MessageAbortedError"
    form = {
        "id": "frm_1",
        "sessionID": SID,
        "fields": [
            {
                "key": "q0",
                "title": "Color",
                "description": "Red or blue?",
                "type": "string",
                "options": [{"value": "Red", "label": "Red"}],
            }
        ],
    }
    asked = translator.translate({"type": "form.created", "data": {"form": form}})[0]
    extracted = _extract_interrupt_asked_event(asked)
    assert extracted is not None
    assert extracted["interrupt_type"] == "question"
    assert extracted["details"]["questions"] == [
        {
            "header": "Color",
            "question": "Red or blue?",
            "options": [{"label": "Red", "value": "Red"}],
        }
    ]
    replied = translator.translate(
        {"type": "form.replied", "data": {"id": "frm_1", "sessionID": SID}}
    )[0]
    cancelled = translator.translate(
        {"type": "form.cancelled", "data": {"id": "frm_1", "sessionID": SID}}
    )[0]
    assert _extract_interrupt_resolved_event(replied)["resolution"] == "replied"
    assert _extract_interrupt_resolved_event(cancelled)["resolution"] == "rejected"
    failed_tool = translator.translate(
        {
            "type": "session.tool.failed",
            "data": {"sessionID": SID, "id": "c9", "error": {"type": "aborted", "message": "no"}},
        }
    )[0]
    assert failed_tool["properties"]["part"]["state"] == {"status": "error", "error": "no"}
    assert translator.translate({"type": "session.tool.progress", "data": {"id": "c9"}}) == []
    assert translator.translate({"type": "session.tool.called", "data": {}}) == []
    assert translator.translate({"type": "form.created", "data": {"form": None}}) == []
    assert translator.translate({"type": "session.text.delta", "data": {"delta": 1}}) == []
    assert translator.translate({"type": 1, "data": {}}) == []


def test_normalizer_edge_cases() -> None:
    from opencode_a2a import opencode_v2 as v2

    assert v2.unwrap_data([1]) == [1]
    assert v2.v1_error_from_v2("x") is None
    assert v2.v1_error_from_v2({"type": "aborted"}) == {"name": "MessageAbortedError", "data": {}}
    assert v2.v1_session_from_v2("x") == "x"
    assert v2.v1_sessions_from_v2("x") == "x"
    assert v2.v1_sessions_from_v2(["bad", {"id": "s"}]) == [{"id": "s"}]
    assert v2.v1_session_status_from_v2([]) == {}
    assert v2.v1_session_status_from_v2({"s": {"type": "retry"}, 1: {}}) == {"s": {"type": "retry"}}
    assert v2.v1_message_from_v2("x") is None
    assert v2.v1_message_from_v2({"type": "user"}) is None
    system = v2.v1_message_from_v2({"id": "m", "type": "system", "text": "s"}, session_id=SID)
    assert system is not None and system["info"]["role"] == "system"
    assert v2.v1_messages_from_v2("x", session_id=SID) == []
    shell = v2.v1_message_from_v2({"id": "m", "type": "shell", "output": {"exit": 0}})
    assert shell is not None and json.loads(shell["parts"][0]["text"]) == {"exit": 0}
    assert (
        v2.v1_message_from_v2({"id": "m", "type": "shell", "output": "plain"})["parts"][0]["text"]
        == "plain"
    )
    assert v2.v1_message_from_v2({"id": "m", "type": "shell"})["parts"][0]["text"] == ""
    tool_message = v2.v1_message_from_v2(
        {
            "id": "m",
            "type": "assistant",
            "content": [
                "bad",
                {"type": "tool", "id": "c", "name": "shell", "state": {"status": "streaming"}},
                {"type": "tool", "state": {"status": "error", "error": "boom"}},
            ],
            "error": {"type": "x", "message": "y"},
        }
    )
    assert tool_message is not None
    assert tool_message["parts"][0]["state"]["status"] == "pending"
    assert tool_message["parts"][1]["state"]["error"] == "boom"
    assert tool_message["info"]["error"]["name"] == "x"
    turn = v2.combine_turn_messages([_idle("failed")], session_id=SID)
    assert turn["info"]["error"]["name"] == "UnknownError"
    assert v2.v2_model_ref(None) is None
    assert v2.v2_model_ref({"providerID": "p"}) is None
    assert v2.v2_prompt_body_from_parts("fallback", [{"type": "file", "url": ""}]) == {
        "text": "fallback"
    }
    assert v2.v2_prompt_body_from_parts(None, [{"type": "tool", "tool": "t", "error": "bad"}]) == {
        "text": "[t result] error: bad"
    }
    catalog = v2.v1_provider_catalog_from_v2("x", "y", None)
    assert catalog == {"all": [], "default": {}, "connected": []}
    catalog = v2.v1_provider_catalog_from_v2(["bad", {"name": "no id"}], ["bad", {"id": "m"}], {})
    assert catalog["all"] == []
    assert v2.v1_questions_from_form({"fields": "x"}) == []
    assert v2.v1_questions_from_form({"fields": ["bad", {"hidden": True}, {"title": "T"}]}) == [
        {"header": "T", "question": "T"}
    ]
    answer = v2.v2_form_answer(
        {"fields": [{"key": "n", "type": "number"}, {"key": "i", "type": "integer"}]},
        [["x"], ["2.0"]],
    )
    assert answer == {"i": 2}
    assert _just_before("") == ""
    assert _just_before("msg_b") == "msg_a"
