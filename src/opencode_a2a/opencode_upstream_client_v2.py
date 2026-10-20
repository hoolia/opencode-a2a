"""OpenCode v2 (``/api/*``) upstream client.

Presents the same Python interface as :class:`OpencodeUpstreamClient` (the OpenCode 1.x
client) and returns payloads in the 1.x shape, so the execution, streaming and
JSON-RPC layers work unchanged against OpenCode 2.x. See ``docs/opencode-v2.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from .config import Settings
from .opencode_upstream_client import (
    _UNSET,
    OpencodeMessage,
    OpencodeMessagePage,
    OpencodeUpstreamClient,
    UpstreamContractError,
)
from .opencode_v2 import (
    V2EventTranslator,
    combine_turn_messages,
    unwrap_data,
    v1_message_from_v2,
    v1_messages_from_v2,
    v1_provider_catalog_from_v2,
    v1_session_from_v2,
    v1_session_status_from_v2,
    v1_sessions_from_v2,
    v2_form_answer,
    v2_model_ref,
    v2_prompt_body_from_parts,
)
from .parts.text import extract_text_from_parts
from .server.state_store import InterruptRequestRepository

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION_KEY = "opencode-a2a-system"
_TURN_POLL_INTERVAL_SECONDS = 0.25
_TURN_MESSAGE_PAGE_LIMIT = 200
_TURN_MESSAGE_MAX_PAGES = 25


def _unsupported(endpoint: str, *, reason: str) -> httpx.HTTPStatusError:
    """Build the error raised for v1 operations without a v2 counterpart.

    A synthetic ``501 Not Implemented`` flows through the existing upstream HTTP
    error mapping, so JSON-RPC callers receive a regular upstream error."""
    request = httpx.Request("POST", f"opencode-v2://unsupported{endpoint}")
    body = json.dumps({"_tag": "NotSupportedByOpenCodeV2", "message": reason})
    response = httpx.Response(
        501,
        request=request,
        content=body.encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    return httpx.HTTPStatusError(
        f"{endpoint} is not supported by OpenCode v2: {reason}",
        request=request,
        response=response,
    )


class OpencodeV2UpstreamClient(OpencodeUpstreamClient):
    def __init__(
        self,
        settings: Settings,
        *,
        interrupt_request_repository: InterruptRequestRepository | None = None,
    ) -> None:
        super().__init__(settings, interrupt_request_repository=interrupt_request_repository)
        # session id -> tuple of settings already applied (agent, model, system).
        self._applied_session_settings: dict[str, tuple[Any, ...]] = {}
        self._session_settings_lock = asyncio.Lock()
        self._monotonic = time.monotonic

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _location_params(self, directory: str | None = None) -> dict[str, str]:
        resolved = directory or self._directory
        if not resolved:
            return {}
        return {"location[directory]": resolved}

    def _location_body(self, directory: str | None = None) -> dict[str, str] | None:
        resolved = directory or self._directory
        if not resolved:
            return None
        return {"directory": resolved}

    async def _v2(
        self,
        method: str,
        path: str,
        *,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        json_body: Any = _UNSET,
        timeout: float | None | object = _UNSET,
        raw: bool = False,
    ) -> Any:
        request_kwargs: dict[str, Any] = {}
        if json_body is not _UNSET:
            request_kwargs["json"] = json_body
        if timeout is not _UNSET:
            request_kwargs["timeout"] = timeout
        async with self._request_budget.reserve(operation=endpoint):
            response = await self._client.request(
                method,
                path,
                params=params,
                headers=self._request_headers(),
                **request_kwargs,
            )
            response.raise_for_status()
            if response.status_code == 204 or not response.content:
                return None
            payload = self._decode_json_response(response, endpoint=endpoint)
        if raw:
            return payload
        return unwrap_data(payload)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    async def stream_events(
        self,
        stop_event: asyncio.Event | None = None,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        del directory, workspace_id
        translator = V2EventTranslator()
        async with self._stream_budget.reserve(operation="/api/event"):
            async with self._client.stream(
                "GET",
                "/api/event",
                timeout=None,
                headers=self._request_headers({"Accept": "text/event-stream"}),
            ) as response:
                response.raise_for_status()
                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if stop_event and stop_event.is_set():
                        break
                    if line.startswith(":"):
                        continue
                    if line == "":
                        if not data_lines:
                            continue
                        payload = "\n".join(data_lines).strip()
                        data_lines.clear()
                        if not payload:
                            continue
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        for translated in translator.translate(event):
                            yield translated
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def create_session(
        self,
        title: str | None = None,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> str:
        del workspace_id
        payload: dict[str, Any] = {}
        if title:
            payload["title"] = title
        location = self._location_body(directory)
        if location is not None:
            payload["location"] = location
        if self._agent:
            payload["agent"] = self._agent
        data = await self._v2("POST", "/api/session", endpoint="/api/session", json_body=payload)
        session_id = data.get("id") if isinstance(data, Mapping) else None
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("OpenCode session response missing id")
        return session_id

    async def session_status(
        self,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del directory, workspace_id
        data = await self._v2("GET", "/api/session/active", endpoint="/api/session/active")
        return v1_session_status_from_v2(data)

    async def get_session(
        self,
        session_id: str,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del directory, workspace_id
        data = await self._v2(
            "GET", f"/api/session/{session_id}", endpoint="/api/session/{sessionID}"
        )
        return v1_session_from_v2(data)

    async def list_child_sessions(
        self,
        session_id: str,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del directory, workspace_id
        data = await self._v2(
            "GET",
            "/api/session",
            endpoint="/api/session",
            params={"parentID": session_id},
        )
        return v1_sessions_from_v2(data)

    async def get_session_todo(
        self,
        session_id: str,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        # OpenCode v2 has no session todo list; report an empty one.
        del session_id, directory, workspace_id
        return []

    async def get_session_diff(
        self,
        session_id: str,
        *,
        params: dict[str, Any] | None = None,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del directory, workspace_id
        query: dict[str, Any] = {}
        for key, value in (params or {}).items():
            if value is None or key in {"directory", "workspace", "messageID"}:
                continue
            query[key] = value if isinstance(value, str) else str(value)
        return await self._v2(
            "GET",
            f"/api/session/{session_id}/diff",
            endpoint="/api/session/{sessionID}/diff",
            params=query,
        )

    async def abort_session(
        self,
        session_id: str,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> bool:
        del directory, workspace_id
        await self._v2(
            "POST",
            f"/api/session/{session_id}/interrupt",
            endpoint="/api/session/{sessionID}/interrupt",
        )
        # v1 returned true once the abort was accepted, whether or not a turn ran.
        return True

    async def list_sessions(
        self,
        *,
        params: dict[str, Any] | None = None,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del workspace_id
        query: dict[str, Any] = {}
        roots: bool | None = None
        start: int | None = None
        for key, value in (params or {}).items():
            if value is None:
                continue
            if key == "roots":
                roots = value in (True, "true", "True", "1")
            elif key == "start":
                try:
                    start = int(value)
                except (TypeError, ValueError):
                    start = None
            elif key in {"limit", "search"}:
                query[key] = value if isinstance(value, str) else str(value)
        resolved_directory = directory or self._directory
        if resolved_directory:
            query["directory"] = resolved_directory
        data = await self._v2("GET", "/api/session", endpoint="/api/session", params=query)
        return v1_sessions_from_v2(data, roots=roots, start=start)

    async def list_messages(
        self,
        session_id: str,
        *,
        params: dict[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> OpencodeMessagePage:
        del workspace_id
        query: dict[str, Any] = {"order": "desc"}
        for key, value in (params or {}).items():
            if value is None:
                continue
            if key == "limit":
                query["limit"] = str(value)
            elif key == "before":
                query["cursor"] = str(value)
        payload = await self._v2(
            "GET",
            f"/api/session/{session_id}/message",
            endpoint="/api/session/{sessionID}/message",
            params=query,
            raw=True,
        )
        messages = unwrap_data(payload)
        if not isinstance(messages, list):
            raise UpstreamContractError("OpenCode v2 message list payload must be an array")
        # v2 pages newest-first; v1 returned the page oldest-first.
        mapped = v1_messages_from_v2(list(reversed(messages)), session_id=session_id)
        next_cursor = None
        cursor = payload.get("cursor") if isinstance(payload, Mapping) else None
        if isinstance(cursor, Mapping) and messages:
            raw_next = cursor.get("next")
            if isinstance(raw_next, str) and raw_next.strip():
                next_cursor = raw_next.strip()
        return OpencodeMessagePage(payload=mapped, next_cursor=next_cursor)

    async def get_message(
        self,
        session_id: str,
        message_id: str,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del directory, workspace_id
        data = await self._v2(
            "GET",
            f"/api/session/{session_id}/message/{message_id}",
            endpoint="/api/session/{sessionID}/message/{messageID}",
        )
        mapped = v1_message_from_v2(data, session_id=session_id)
        if mapped is None:
            raise UpstreamContractError(
                "OpenCode v2 message has no A2A-visible content (non-chat message type)"
            )
        return mapped

    # ------------------------------------------------------------------
    # Per-session settings (agent, model, system instruction)
    # ------------------------------------------------------------------

    async def _resolve_variant_model(self, directory: str | None) -> dict[str, str] | None:
        data = await self._v2(
            "GET",
            "/api/model/default",
            endpoint="/api/model/default",
            params=self._location_params(directory),
        )
        if not isinstance(data, Mapping):
            return None
        return v2_model_ref(
            {"providerID": data.get("providerID"), "id": data.get("id")},
            variant=self._variant,
        )

    async def _ensure_session_settings(
        self,
        session_id: str,
        *,
        model_override: Mapping[str, Any] | None,
        agent_override: str | None = None,
        directory: str | None = None,
    ) -> None:
        agent = agent_override or self._agent
        model = v2_model_ref(model_override, variant=self._variant)
        wanted = (agent, json.dumps(model, sort_keys=True), self._system)
        async with self._session_settings_lock:
            previous = self._applied_session_settings.get(session_id)
            if previous == wanted:
                return
        if agent and (previous is None or previous[0] != agent):
            await self._v2(
                "POST",
                f"/api/session/{session_id}/agent",
                endpoint="/api/session/{sessionID}/agent",
                json_body={"agent": agent},
            )
        # v1 chose the model per message; v2 keeps it on the session, so a turn
        # without an override after one with an override returns to the default.
        previous_had_model = previous is not None and previous[1] != "null"
        if model is None and ((self._variant and previous is None) or previous_had_model):
            model = await self._resolve_variant_model(directory)
        if model is not None:
            await self._v2(
                "POST",
                f"/api/session/{session_id}/model",
                endpoint="/api/session/{sessionID}/model",
                json_body={"model": model},
            )
        if self._system and (previous is None or previous[2] != self._system):
            await self._v2(
                "PUT",
                f"/api/experimental/session/{session_id}/instructions/entries/"
                f"{SYSTEM_INSTRUCTION_KEY}",
                endpoint="/api/experimental/session/{sessionID}/instructions/entries/{key}",
                json_body={"value": self._system},
            )
        async with self._session_settings_lock:
            self._applied_session_settings[session_id] = wanted

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    async def _turn_messages(self, session_id: str, after_message_id: str) -> list[dict[str, Any]]:
        """Return the messages newer than ``after_message_id``, oldest first."""
        collected: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(_TURN_MESSAGE_MAX_PAGES):
            query: dict[str, Any] = {"order": "desc", "limit": str(_TURN_MESSAGE_PAGE_LIMIT)}
            if cursor:
                query["cursor"] = cursor
            payload = await self._v2(
                "GET",
                f"/api/session/{session_id}/message",
                endpoint="/api/session/{sessionID}/message",
                params=query,
                raw=True,
            )
            page = unwrap_data(payload)
            if not isinstance(page, list) or not page:
                break
            reached = False
            for message in page:
                if not isinstance(message, dict):
                    continue
                message_id = message.get("id")
                if not isinstance(message_id, str):
                    continue
                # v2 message ids are time-ordered, so "newer" is a string comparison.
                if message_id <= after_message_id:
                    reached = True
                    break
                collected.append(message)
            if reached:
                break
            next_cursor = payload.get("cursor") if isinstance(payload, Mapping) else None
            cursor = next_cursor.get("next") if isinstance(next_cursor, Mapping) else None
            if not isinstance(cursor, str) or not cursor:
                break
        collected.reverse()
        return collected

    async def _await_turn(
        self,
        session_id: str,
        after_message_id: str,
        *,
        timeout: float | None,
    ) -> list[dict[str, Any]]:
        deadline = None if timeout is None else self._monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - self._monotonic())
            if remaining is not None and remaining <= 0:
                raise httpx.ReadTimeout(
                    f"OpenCode session {session_id} did not finish the turn in time"
                )
            await self._v2(
                "POST",
                f"/api/experimental/session/{session_id}/wait",
                endpoint="/api/experimental/session/{sessionID}/wait",
                timeout=remaining,
            )
            messages = await self._turn_messages(session_id, after_message_id)
            if any(message.get("type") == "idle" for message in messages):
                return messages
            # The wait endpoint returns at once while the loop has not started yet.
            await asyncio.sleep(_TURN_POLL_INTERVAL_SECONDS)

    def _turn_timeout(self, timeout_override: float | None | object) -> float | None:
        if timeout_override is _UNSET:
            return self._settings.opencode_timeout
        if timeout_override is None:
            return None
        return float(timeout_override)  # type: ignore[arg-type]

    async def _submit_and_wait(
        self,
        session_id: str,
        path: str,
        endpoint: str,
        body: dict[str, Any],
        *,
        timeout_override: float | None | object,
    ) -> dict[str, Any]:
        # Snapshot the newest message id first: command and shell return no id.
        baseline = await self._turn_messages_baseline(session_id)
        result = await self._v2("POST", path, endpoint=endpoint, json_body=body)
        after = baseline
        if isinstance(result, Mapping) and isinstance(result.get("id"), str):
            after = _just_before(result["id"])
        messages = await self._await_turn(
            session_id, after, timeout=self._turn_timeout(timeout_override)
        )
        return combine_turn_messages(messages, session_id=session_id)

    async def _turn_messages_baseline(self, session_id: str) -> str:
        payload = await self._v2(
            "GET",
            f"/api/session/{session_id}/message",
            endpoint="/api/session/{sessionID}/message",
            params={"order": "desc", "limit": "1"},
        )
        if isinstance(payload, list) and payload and isinstance(payload[0], Mapping):
            message_id = payload[0].get("id")
            if isinstance(message_id, str):
                return message_id
        return ""

    async def send_message(
        self,
        session_id: str,
        text: str | None = None,
        *,
        parts: Sequence[Mapping[str, Any]] | None = None,
        directory: str | None = None,
        workspace_id: str | None = None,
        model_override: Mapping[str, Any] | None = None,
        timeout_override: float | None | object = _UNSET,
    ) -> OpencodeMessage:
        del workspace_id
        body = v2_prompt_body_from_parts(text, parts)
        await self._ensure_session_settings(
            session_id, model_override=model_override, directory=directory
        )
        if self._log_payloads:
            logger.debug("OpenCode request payload=%s", body)
        prompt = await self._v2(
            "POST",
            f"/api/session/{session_id}/prompt",
            endpoint="/api/session/{sessionID}/prompt",
            json_body=body,
        )
        user_message_id = prompt.get("id") if isinstance(prompt, Mapping) else None
        if not isinstance(user_message_id, str) or not user_message_id:
            raise UpstreamContractError("OpenCode v2 prompt response missing message id")
        messages = await self._await_turn(
            session_id,
            user_message_id,
            timeout=self._turn_timeout(timeout_override),
        )
        data = combine_turn_messages(messages, session_id=session_id)
        if self._log_payloads:
            logger.debug("OpenCode response payload=%s", data)
        return OpencodeMessage(
            text=extract_text_from_parts(data.get("parts", [])),
            session_id=session_id,
            message_id=data["info"].get("id"),
            raw=data,
        )

    async def session_prompt_async(
        self,
        session_id: str,
        request: dict[str, Any],
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        del workspace_id
        body = v2_prompt_body_from_parts(None, request.get("parts") or [])
        model = request.get("model") if isinstance(request.get("model"), Mapping) else None
        agent = request.get("agent") if isinstance(request.get("agent"), str) else None
        await self._ensure_session_settings(
            session_id, model_override=model, agent_override=agent, directory=directory
        )
        await self._v2(
            "POST",
            f"/api/session/{session_id}/prompt",
            endpoint="/api/session/{sessionID}/prompt",
            json_body=body,
        )

    async def session_command(
        self,
        session_id: str,
        request: dict[str, Any],
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del workspace_id
        model = request.get("model") if isinstance(request.get("model"), Mapping) else None
        agent = request.get("agent") if isinstance(request.get("agent"), str) else None
        await self._ensure_session_settings(
            session_id, model_override=model, agent_override=agent, directory=directory
        )
        body: dict[str, Any] = {
            "name": str(request.get("command", "")).lstrip("/"),
            "text": str(request.get("arguments", "")),
        }
        files = (
            v2_prompt_body_from_parts(None, request["parts"]).get("files")
            if request.get("parts")
            else None
        )
        if files:
            body["files"] = files
        return await self._submit_and_wait(
            session_id,
            f"/api/session/{session_id}/command",
            "/api/session/{sessionID}/command",
            body,
            timeout_override=_UNSET,
        )

    async def session_shell(
        self,
        session_id: str,
        request: dict[str, Any],
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del workspace_id, directory
        # v2 shell commands run outside an agent; request.agent/model do not apply.
        return await self._submit_and_wait(
            session_id,
            f"/api/session/{session_id}/shell",
            "/api/session/{sessionID}/shell",
            {"command": str(request.get("command", ""))},
            timeout_override=_UNSET,
        )

    async def fork_session(
        self,
        session_id: str,
        request: dict[str, Any] | None = None,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del directory, workspace_id
        body: dict[str, Any] = {}
        message_id = (request or {}).get("messageID")
        if isinstance(message_id, str) and message_id.strip():
            body["before"] = message_id.strip()
        data = await self._v2(
            "POST",
            f"/api/session/{session_id}/fork",
            endpoint="/api/session/{sessionID}/fork",
            json_body=body,
        )
        return v1_session_from_v2(data)

    async def share_session(
        self,
        session_id: str,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del session_id, directory, workspace_id
        raise _unsupported("/session/{sessionID}/share", reason="session sharing was removed")

    async def unshare_session(
        self,
        session_id: str,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del session_id, directory, workspace_id
        raise _unsupported("/session/{sessionID}/share", reason="session sharing was removed")

    async def summarize_session(
        self,
        session_id: str,
        request: dict[str, Any] | None = None,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> bool:
        # v2 compacts with the session model; v1 request.providerID/modelID are ignored.
        del request, directory, workspace_id
        await self._v2(
            "POST",
            f"/api/session/{session_id}/compact",
            endpoint="/api/session/{sessionID}/compact",
            json_body={},
        )
        return True

    async def revert_session(
        self,
        session_id: str,
        request: dict[str, Any],
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        # v1 revert hid messages until the next prompt; v2 "stage" is the same operation.
        message_id = request.get("messageID")
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("revert requires request.messageID")
        await self._v2(
            "POST",
            f"/api/session/{session_id}/revert/stage",
            endpoint="/api/session/{sessionID}/revert/stage",
            json_body={"messageID": message_id.strip()},
        )
        return await self.get_session(session_id, directory=directory, workspace_id=workspace_id)

    async def unrevert_session(
        self,
        session_id: str,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        await self._v2(
            "DELETE",
            f"/api/session/{session_id}/revert",
            endpoint="/api/session/{sessionID}/revert",
        )
        return await self.get_session(session_id, directory=directory, workspace_id=workspace_id)

    # ------------------------------------------------------------------
    # Discovery, projects, workspaces, worktrees
    # ------------------------------------------------------------------

    async def list_provider_catalog(
        self,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> Any:
        del workspace_id
        params = self._location_params(directory)
        providers, models, default_model = await asyncio.gather(
            self._v2("GET", "/api/provider", endpoint="/api/provider", params=params),
            self._v2("GET", "/api/model", endpoint="/api/model", params=params),
            self._v2("GET", "/api/model/default", endpoint="/api/model/default", params=params),
        )
        return v1_provider_catalog_from_v2(providers, models, default_model)

    async def _current_location(self) -> Mapping[str, Any]:
        data = await self._v2(
            "GET",
            "/api/location",
            endpoint="/api/location",
            params=self._location_params(),
            raw=True,
        )
        return data if isinstance(data, Mapping) else {}

    async def list_projects(self) -> Any:
        data = await self._v2("GET", "/api/project", endpoint="/api/project")
        if not isinstance(data, list):
            return data
        projects: list[Any] = []
        for item in data:
            if isinstance(item, Mapping):
                project = dict(item)
                canonical = project.get("canonical")
                if isinstance(canonical, str):
                    project.setdefault("worktree", canonical)
                    project.setdefault("name", os.path.basename(canonical.rstrip("/")))
                projects.append(project)
        return projects

    async def get_current_project(self) -> Any:
        location = await self._current_location()
        project = location.get("project")
        if not isinstance(project, Mapping):
            return None
        result = dict(project)
        directory = result.get("canonical") or result.get("directory")
        if isinstance(directory, str):
            result.setdefault("worktree", directory)
            result.setdefault("name", os.path.basename(directory.rstrip("/")))
        return result

    async def list_workspaces(self) -> Any:
        raise _unsupported("/experimental/workspace", reason="workspaces were removed")

    async def create_workspace(self, request: dict[str, Any]) -> Any:
        del request
        raise _unsupported("/experimental/workspace", reason="workspaces were removed")

    async def remove_workspace(self, workspace_id: str) -> Any:
        del workspace_id
        raise _unsupported("/experimental/workspace/{id}", reason="workspaces were removed")

    async def _project_id(self) -> str:
        location = await self._current_location()
        project = location.get("project")
        project_id = project.get("id") if isinstance(project, Mapping) else None
        if not isinstance(project_id, str) or not project_id:
            raise UpstreamContractError("OpenCode v2 location has no project id")
        return project_id

    @staticmethod
    def _worktree_item(item: Any) -> Any:
        if not isinstance(item, Mapping):
            return item
        result = dict(item)
        directory = result.get("directory")
        if isinstance(directory, str):
            result.setdefault("name", os.path.basename(directory.rstrip("/")))
        return result

    async def list_worktrees(self) -> Any:
        project_id = await self._project_id()
        data = await self._v2(
            "GET", "/api/worktree", endpoint="/api/worktree", params={"projectID": project_id}
        )
        if not isinstance(data, list):
            return data
        return [self._worktree_item(item) for item in data]

    async def create_worktree(self, request: dict[str, Any]) -> Any:
        body: dict[str, Any] = {"projectID": await self._project_id()}
        name = request.get("name")
        if isinstance(name, str) and name.strip():
            body["name"] = name.strip()
        data = await self._v2("POST", "/api/worktree", endpoint="/api/worktree", json_body=body)
        item = self._worktree_item(data)
        if isinstance(item, dict) and isinstance(name, str) and name.strip():
            item["name"] = name.strip()
        return item

    async def remove_worktree(self, request: dict[str, Any]) -> bool:
        body = {
            "projectID": await self._project_id(),
            "directory": request.get("directory"),
            "force": False,
        }
        await self._v2("DELETE", "/api/worktree", endpoint="/api/worktree", json_body=body)
        return True

    async def reset_worktree(self, request: dict[str, Any]) -> bool:
        del request
        raise _unsupported("/experimental/worktree/reset", reason="worktree reset was removed")

    # ------------------------------------------------------------------
    # Interrupts (permissions and questions)
    # ------------------------------------------------------------------

    async def _interrupt_session_id(self, request_id: str, *, kind: str) -> str:
        session_id = await self.resolve_interrupt_session(request_id)
        if session_id:
            return session_id
        # Fall back to the server's pending lists (for example after a restart).
        path = "/api/permission/request" if kind == "permission" else "/api/form"
        data = await self._v2("GET", path, endpoint=path, params=self._location_params())
        if isinstance(data, list):
            for item in data:
                if isinstance(item, Mapping) and item.get("id") == request_id:
                    candidate = item.get("sessionID")
                    if isinstance(candidate, str) and candidate:
                        return candidate
        request = httpx.Request("POST", f"opencode-v2://{kind}/{request_id}")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError(
            f"OpenCode {kind} request {request_id} not found",
            request=request,
            response=response,
        )

    async def permission_reply(
        self,
        request_id: str,
        *,
        reply: str,
        message: str | None = None,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> bool:
        del directory, workspace_id
        session_id = await self._interrupt_session_id(request_id, kind="permission")
        payload: dict[str, Any] = {"decision": reply}
        if message:
            payload["message"] = message
        await self._v2(
            "POST",
            f"/api/session/{session_id}/permission/{request_id}/reply",
            endpoint="/api/session/{sessionID}/permission/{requestID}/reply",
            json_body=payload,
        )
        return True

    async def question_reply(
        self,
        request_id: str,
        *,
        answers: list[list[str]],
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> bool:
        del directory, workspace_id
        session_id = await self._interrupt_session_id(request_id, kind="question")
        form = await self._v2(
            "GET",
            f"/api/session/{session_id}/form/{request_id}",
            endpoint="/api/session/{sessionID}/form/{formID}",
        )
        form_info = form.get("form", form) if isinstance(form, Mapping) else {}
        if not isinstance(form_info, Mapping):
            form_info = {}
        await self._v2(
            "POST",
            f"/api/session/{session_id}/form/{request_id}/reply",
            endpoint="/api/session/{sessionID}/form/{formID}/reply",
            json_body={"answer": v2_form_answer(form_info, answers)},
        )
        return True

    async def question_reject(
        self,
        request_id: str,
        *,
        directory: str | None = None,
        workspace_id: str | None = None,
    ) -> bool:
        del directory, workspace_id
        session_id = await self._interrupt_session_id(request_id, kind="question")
        await self._v2(
            "DELETE",
            f"/api/session/{session_id}/form/{request_id}",
            endpoint="/api/session/{sessionID}/form/{formID}",
        )
        return True


def _just_before(message_id: str) -> str:
    """Return a string that sorts immediately before ``message_id`` so the message
    itself is included in a "newer than" scan."""
    if not message_id:
        return message_id
    return message_id[:-1] + chr(ord(message_id[-1]) - 1)
