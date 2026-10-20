"""Translation between the OpenCode v2 HTTP API and the legacy (v1) payload shapes.

The adapter's execution, streaming, and JSON-RPC layers were written against the
OpenCode 1.x wire format. OpenCode 2.x moved everything under ``/api/*``, wraps
responses in ``{"data": ...}``, and renamed its events. Rather than rewriting the
consumers, :class:`~opencode_a2a.opencode_upstream_client_v2.OpencodeV2UpstreamClient`
translates every v2 payload into the v1 shape the rest of the package expects, using
the pure functions in this module.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"

# v2 tool states -> v1 tool states.
_TOOL_STATUS_MAP = {
    "streaming": "pending",
    "running": "running",
    "completed": "completed",
    "error": "error",
}


def unwrap_data(payload: Any) -> Any:
    """Return ``payload["data"]`` for v2 envelopes, else the payload unchanged."""
    if isinstance(payload, Mapping) and "data" in payload:
        return payload["data"]
    return payload


def _str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def v1_error_from_v2(error: Any) -> dict[str, Any] | None:
    """Map a v2 ``Session.StructuredError`` (``{type, message, status?}``) to the v1
    ``{name, data: {message, statusCode?}}`` shape used by the error translator."""
    if not isinstance(error, Mapping):
        return None
    error_type = _str(error.get("type")) or "UnknownError"
    message = _str(error.get("message"))
    status = error.get("status")
    name = error_type
    lowered = error_type.lower()
    if lowered.startswith("provider.auth") or status == 401:
        name = "ProviderAuthError"
    elif lowered == "aborted":
        name = "MessageAbortedError"
    data: dict[str, Any] = {}
    if message:
        data["message"] = message
    if isinstance(status, int) and not isinstance(status, bool):
        data["statusCode"] = status
    return {"name": name, "data": data}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def v1_session_from_v2(session: Any) -> Any:
    """Map a v2 ``Session.Info`` to the v1 ``Session`` shape."""
    if not isinstance(session, Mapping):
        return session
    result: dict[str, Any] = dict(session)
    location = session.get("location")
    if isinstance(location, Mapping):
        directory = _str(location.get("directory"))
        if directory:
            result["directory"] = directory
    return result


def v1_sessions_from_v2(
    sessions: Any,
    *,
    roots: bool | None = None,
    start: int | None = None,
) -> Any:
    if not isinstance(sessions, list):
        return sessions
    items: list[Any] = []
    for session in sessions:
        if not isinstance(session, Mapping):
            continue
        if roots and _str(session.get("parentID")):
            continue
        if start is not None:
            time_info = session.get("time")
            updated = time_info.get("updated") if isinstance(time_info, Mapping) else None
            if isinstance(updated, (int, float)) and updated < start:
                continue
        items.append(v1_session_from_v2(session))
    return items


def v1_session_status_from_v2(active: Any) -> dict[str, Any]:
    """v2 ``GET /api/session/active`` lists busy sessions; v1 returned a
    ``{sessionID: {type}}`` map."""
    result: dict[str, Any] = {}
    if not isinstance(active, Mapping):
        return result
    for session_id, info in active.items():
        if not isinstance(session_id, str):
            continue
        entry: dict[str, Any] = {"type": "busy"}
        if isinstance(info, Mapping):
            status_type = _str(info.get("type")) or _str(info.get("status"))
            if status_type:
                entry["type"] = status_type
        result[session_id] = entry
    return result


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _tool_content_text(content: Any) -> str | None:
    if not isinstance(content, list):
        return None
    texts: list[str] = []
    for item in content:
        if isinstance(item, Mapping) and isinstance(item.get("text"), str):
            texts.append(item["text"])
    if not texts:
        return None
    return "".join(texts)


def v1_tool_part(
    *,
    call_id: str,
    tool: str | None,
    status: str,
    session_id: str | None,
    message_id: str | None,
    tool_input: Any = None,
    output: str | None = None,
    error: Any = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {"status": _TOOL_STATUS_MAP.get(status, status)}
    if tool_input is not None:
        state["input"] = tool_input
    if output is not None:
        state["output"] = output
    if error is not None:
        if isinstance(error, Mapping):
            state["error"] = _str(error.get("message")) or json.dumps(dict(error))
        else:
            state["error"] = str(error)
    part: dict[str, Any] = {
        "id": call_id,
        "type": "tool",
        "callID": call_id,
        "state": state,
    }
    if tool:
        part["tool"] = tool
    if session_id:
        part["sessionID"] = session_id
    if message_id:
        part["messageID"] = message_id
    return part


def _v1_parts_from_assistant_content(
    content: Any,
    *,
    session_id: str | None,
    message_id: str | None,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if not isinstance(content, list):
        return parts
    ordinals: dict[str, int] = {"text": 0, "reasoning": 0}
    for item in content:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type in ("text", "reasoning"):
            ordinal = ordinals[item_type]
            ordinals[item_type] += 1
            part: dict[str, Any] = {
                "id": f"{message_id}:{item_type}:{ordinal}",
                "type": item_type,
                "text": item.get("text") if isinstance(item.get("text"), str) else "",
            }
            if session_id:
                part["sessionID"] = session_id
            if message_id:
                part["messageID"] = message_id
            parts.append(part)
        elif item_type == "tool":
            raw_state = item.get("state")
            state: Mapping[str, Any] = raw_state if isinstance(raw_state, Mapping) else {}
            parts.append(
                v1_tool_part(
                    call_id=_str(item.get("id")) or f"{message_id}:tool:{len(parts)}",
                    tool=_str(item.get("name")),
                    status=_str(state.get("status")) or "pending",
                    session_id=session_id,
                    message_id=message_id,
                    tool_input=state.get("input"),
                    output=_tool_content_text(state.get("content")),
                    error=state.get("error"),
                )
            )
    return parts


def _shell_output_text(message: Mapping[str, Any]) -> str:
    output = message.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, Mapping):
        for key in ("text", "stdout", "content", "preview"):
            value = output.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(dict(output), ensure_ascii=False)
    return ""


def _message_text(message: Mapping[str, Any]) -> str:
    text = message.get("text")
    return text if isinstance(text, str) else ""


def v1_message_from_v2(message: Any, *, session_id: str | None = None) -> dict[str, Any] | None:
    """Map one v2 ``Session.Message.Info`` to the v1 ``{info, parts}`` shape.

    Returns ``None`` for v2 message types with no v1 counterpart (idle markers,
    agent/model/location selection, compaction)."""
    if not isinstance(message, Mapping):
        return None
    message_id = _str(message.get("id"))
    if message_id is None:
        return None
    message_type = message.get("type")
    info: dict[str, Any] = {"id": message_id}
    if session_id:
        info["sessionID"] = session_id
    if isinstance(message.get("time"), Mapping):
        info["time"] = dict(message["time"])

    def _text_part(text: str) -> dict[str, Any]:
        part: dict[str, Any] = {"id": f"{message_id}:text:0", "type": "text", "text": text}
        if session_id:
            part["sessionID"] = session_id
        part["messageID"] = message_id
        return part

    if message_type in ("user", "synthetic"):
        info["role"] = USER_ROLE
        return {"info": info, "parts": [_text_part(_message_text(message))]}
    if message_type == "system":
        info["role"] = "system"
        return {"info": info, "parts": [_text_part(_message_text(message))]}
    if message_type == "shell":
        info["role"] = ASSISTANT_ROLE
        return {"info": info, "parts": [_text_part(_shell_output_text(message))]}
    if message_type == "assistant":
        info["role"] = ASSISTANT_ROLE
        model = message.get("model")
        if isinstance(model, Mapping):
            if _str(model.get("id")):
                info["modelID"] = model["id"]
            if _str(model.get("providerID")):
                info["providerID"] = model["providerID"]
        for key in ("agent", "cost", "tokens", "finish"):
            if key in message:
                info[key] = message[key]
        error = v1_error_from_v2(message.get("error"))
        if error is not None:
            info["error"] = error
        return {
            "info": info,
            "parts": _v1_parts_from_assistant_content(
                message.get("content"), session_id=session_id, message_id=message_id
            ),
        }
    return None


def v1_messages_from_v2(messages: Any, *, session_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return items
    for message in messages:
        mapped = v1_message_from_v2(message, session_id=session_id)
        if mapped is not None:
            items.append(mapped)
    return items


def combine_turn_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
) -> dict[str, Any]:
    """Fold the v2 messages produced by one prompt turn (oldest first) into one v1
    assistant message, matching what v1 ``POST /session/{id}/message`` returned."""
    parts: list[dict[str, Any]] = []
    last_assistant: Mapping[str, Any] | None = None
    idle: Mapping[str, Any] | None = None
    for message in messages:
        message_type = message.get("type")
        if message_type == "idle":
            idle = message
            continue
        if message_type not in ("assistant", "shell"):
            continue
        mapped = v1_message_from_v2(message, session_id=session_id)
        if mapped is None:
            continue
        parts.extend(mapped["parts"])
        last_assistant = message
    info: dict[str, Any] = {"role": ASSISTANT_ROLE, "sessionID": session_id}
    if last_assistant is not None:
        mapped_last = v1_message_from_v2(last_assistant, session_id=session_id)
        if mapped_last is not None:
            info.update(mapped_last["info"])
    elif idle is not None and _str(idle.get("id")):
        info["id"] = idle["id"]
    error = None
    if idle is not None:
        outcome = idle.get("outcome")
        if outcome == "failed":
            error = v1_error_from_v2(idle.get("error")) or {
                "name": "UnknownError",
                "data": {"message": "OpenCode execution failed"},
            }
        elif outcome == "interrupted":
            error = v1_error_from_v2(idle.get("error")) or {
                "name": "MessageAbortedError",
                "data": {"message": "OpenCode execution interrupted"},
            }
    # The idle marker carries the turn outcome; per-step errors that the loop
    # recovered from (retries) must not fail the turn.
    info.pop("error", None)
    if error is not None:
        info["error"] = error
    return {"info": info, "parts": parts}


# ---------------------------------------------------------------------------
# Prompt input
# ---------------------------------------------------------------------------


def _tool_result_text(part: Mapping[str, Any]) -> str:
    tool = _str(part.get("tool")) or "tool"
    call_id = _str(part.get("call_id")) or _str(part.get("callID")) or ""
    error = part.get("error")
    output = part.get("output")
    header = f"[{tool} result{f' {call_id}' if call_id else ''}]"
    if error:
        return f"{header} error: {error}"
    return f"{header} {output if output is not None else ''}".rstrip()


def v2_prompt_body_from_parts(
    text: str | None,
    parts: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Build a v2 ``POST /api/session/{id}/prompt`` body from v1-style input parts.

    v2 takes one ``text`` string plus ``files``; text parts are concatenated, file
    parts become ``files`` entries, and adapter-internal tool-result parts are
    rendered as text."""
    if parts is None:
        if not isinstance(text, str):
            raise ValueError("send_message requires either text or parts")
        return {"text": text}
    normalized = [dict(part) for part in parts]
    if not normalized:
        raise ValueError("send_message parts must not be empty")
    texts: list[str] = []
    files: list[dict[str, Any]] = []
    for part in normalized:
        part_type = part.get("type")
        if part_type == "text":
            value = part.get("text")
            if isinstance(value, str):
                texts.append(value)
        elif part_type == "file":
            uri = _str(part.get("url")) or _str(part.get("uri"))
            if uri is None:
                continue
            attachment: dict[str, Any] = {"uri": uri}
            name = _str(part.get("filename")) or _str(part.get("name"))
            if name:
                attachment["name"] = name
            files.append(attachment)
        elif part_type == "tool":
            texts.append(_tool_result_text(part))
    if not texts and isinstance(text, str) and text:
        texts.append(text)
    body: dict[str, Any] = {"text": "\n\n".join(texts)}
    if files:
        body["files"] = files
    return body


def v2_model_ref(
    value: Mapping[str, Any] | None,
    *,
    variant: str | None = None,
) -> dict[str, str] | None:
    """Map a v1 ``{providerID, modelID}`` ref to a v2 ``Model.Ref``."""
    if value is None:
        return None
    provider = _str(value.get("providerID"))
    model = _str(value.get("modelID")) or _str(value.get("id"))
    if provider is None or model is None:
        return None
    ref = {"id": model, "providerID": provider}
    if variant:
        ref["variant"] = variant
    return ref


# ---------------------------------------------------------------------------
# Provider catalog
# ---------------------------------------------------------------------------


def v1_provider_catalog_from_v2(
    providers: Any,
    models: Any,
    default_model: Any,
) -> dict[str, Any]:
    """Assemble the v1 ``GET /provider`` shape (``{all, default, connected}``) from
    v2 ``/api/provider``, ``/api/model`` and ``/api/model/default``."""
    by_provider: dict[str, dict[str, Any]] = {}
    connected: list[str] = []
    if isinstance(providers, list):
        for provider in providers:
            if not isinstance(provider, Mapping):
                continue
            provider_id = _str(provider.get("id"))
            if provider_id is None:
                continue
            entry: dict[str, Any] = {
                "id": provider_id,
                "name": provider.get("name")
                if isinstance(provider.get("name"), str)
                else provider_id,
                "models": {},
            }
            by_provider[provider_id] = entry
            if provider.get("activation", "enabled") == "enabled":
                connected.append(provider_id)
    if isinstance(models, list):
        for model in models:
            if not isinstance(model, Mapping):
                continue
            provider_id = _str(model.get("providerID"))
            model_id = _str(model.get("id")) or _str(model.get("modelID"))
            if provider_id is None or model_id is None:
                continue
            entry = by_provider.setdefault(
                provider_id, {"id": provider_id, "name": provider_id, "models": {}}
            )
            capabilities = model.get("capabilities")
            v1_caps: dict[str, Any] = {}
            if isinstance(capabilities, Mapping):
                if isinstance(capabilities.get("tools"), bool):
                    v1_caps["toolcall"] = capabilities["tools"]
                if isinstance(capabilities.get("reasoning"), bool):
                    v1_caps["reasoning"] = capabilities["reasoning"]
                inputs = capabilities.get("input")
                if isinstance(inputs, list):
                    v1_caps["attachment"] = any(item != "text" for item in inputs)
            v1_model: dict[str, Any] = {
                "id": model_id,
                "name": model.get("name") if isinstance(model.get("name"), str) else model_id,
            }
            if isinstance(model.get("status"), str):
                v1_model["status"] = model["status"]
            if isinstance(model.get("limit"), Mapping):
                v1_model["limit"] = dict(model["limit"])
            if v1_caps:
                v1_model["capabilities"] = v1_caps
            entry["models"][model_id] = v1_model
    default: dict[str, str] = {}
    if isinstance(default_model, Mapping):
        provider_id = _str(default_model.get("providerID"))
        model_id = _str(default_model.get("id")) or _str(default_model.get("modelID"))
        if provider_id and model_id:
            default[provider_id] = model_id
    return {"all": list(by_provider.values()), "default": default, "connected": connected}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _event(event_type: str, properties: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": event_type, "properties": dict(properties)}


def _block_part(
    data: Mapping[str, Any],
    block: str,
    *,
    text: str | None = None,
) -> dict[str, Any]:
    message_id = _str(data.get("assistantMessageID"))
    ordinal = data.get("ordinal", 0)
    part: dict[str, Any] = {
        "id": f"{message_id}:{block}:{ordinal}",
        "type": block,
        "sessionID": data.get("sessionID"),
        "messageID": message_id,
        "role": ASSISTANT_ROLE,
    }
    if text is not None:
        part["text"] = text
    return part


class V2EventTranslator:
    """Stateful translator from v2 ``/api/event`` payloads to v1 event dicts.

    Tool events in v2 carry only the call id after ``session.tool.input.started``;
    the translator remembers the tool name and input per call id so that every
    emitted v1 tool part is complete."""

    def __init__(self) -> None:
        self._tool_names: dict[str, str] = {}
        self._tool_inputs: dict[str, Any] = {}

    def translate(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        event_type = event.get("type")
        data = event.get("data")
        if not isinstance(event_type, str) or not isinstance(data, Mapping):
            return []
        session_id = data.get("sessionID")
        message_id = _str(data.get("assistantMessageID"))

        if event_type in ("session.text.started", "session.reasoning.started"):
            block = event_type.split(".")[1]
            part = _block_part(data, block, text="")
            return [_event("message.part.updated", {"sessionID": session_id, "part": part})]
        if event_type in ("session.text.delta", "session.reasoning.delta"):
            block = event_type.split(".")[1]
            delta = data.get("delta")
            if not isinstance(delta, str):
                return []
            part = _block_part(data, block)
            return [
                _event(
                    "message.part.delta",
                    {
                        "sessionID": session_id,
                        "messageID": message_id,
                        "partID": part["id"],
                        "field": "text",
                        "delta": delta,
                    },
                )
            ]
        if event_type in ("session.text.ended", "session.reasoning.ended"):
            block = event_type.split(".")[1]
            text = data.get("text") if isinstance(data.get("text"), str) else ""
            part = _block_part(data, block, text=text)
            return [_event("message.part.updated", {"sessionID": session_id, "part": part})]

        if event_type.startswith("session.tool."):
            return self._translate_tool(event_type, data)

        if event_type == "session.step.started":
            part = {
                "id": f"{message_id}:step-start",
                "type": "step-start",
                "sessionID": session_id,
                "messageID": message_id,
            }
            return [_event("message.part.updated", {"sessionID": session_id, "part": part})]
        if event_type in ("session.step.ended", "session.step.failed"):
            part = {
                "id": f"{message_id}:step-finish",
                "type": "step-finish",
                "sessionID": session_id,
                "messageID": message_id,
            }
            for key in ("tokens", "cost"):
                if key in data:
                    part[key] = data[key]
            return [_event("message.part.updated", {"sessionID": session_id, "part": part})]

        if event_type == "session.execution.succeeded":
            return [_event("session.idle", {"sessionID": session_id})]
        if event_type == "session.execution.failed":
            error = v1_error_from_v2(data.get("error")) or {
                "name": "UnknownError",
                "data": {"message": "OpenCode execution failed"},
            }
            return [_event("session.error", {"sessionID": session_id, "error": error})]
        if event_type == "session.execution.interrupted":
            reason = _str(data.get("reason")) or "interrupted"
            error = {
                "name": "MessageAbortedError",
                "data": {"message": f"OpenCode execution interrupted ({reason})"},
            }
            return [_event("session.error", {"sessionID": session_id, "error": error})]

        if event_type == "permission.asked":
            return [
                _event(
                    "permission.asked",
                    {
                        "id": data.get("id"),
                        "sessionID": session_id,
                        "permission": data.get("action"),
                        "patterns": data.get("resources") or [],
                        "metadata": data.get("metadata") or {},
                    },
                )
            ]
        if event_type == "permission.replied":
            return [
                _event(
                    "permission.replied",
                    {
                        "sessionID": session_id,
                        "requestID": data.get("requestID"),
                        "reply": data.get("reply"),
                    },
                )
            ]

        if event_type == "form.created":
            form = data.get("form")
            if not isinstance(form, Mapping):
                return []
            return [
                _event(
                    "question.asked",
                    {
                        "id": form.get("id"),
                        "sessionID": form.get("sessionID"),
                        "questions": v1_questions_from_form(form),
                    },
                )
            ]
        if event_type == "form.replied":
            return [
                _event(
                    "question.replied",
                    {"sessionID": session_id, "requestID": data.get("id")},
                )
            ]
        if event_type == "form.cancelled":
            return [
                _event(
                    "question.rejected",
                    {"sessionID": session_id, "requestID": data.get("id")},
                )
            ]
        return []

    def _translate_tool(self, event_type: str, data: Mapping[str, Any]) -> list[dict[str, Any]]:
        call_id = _str(data.get("id"))
        if call_id is None:
            return []
        session_id = data.get("sessionID")
        message_id = _str(data.get("assistantMessageID"))
        extra: dict[str, Any] = {}
        if event_type == "session.tool.input.started":
            name = _str(data.get("name"))
            if name:
                self._tool_names[call_id] = name
            status = "pending"
        elif event_type == "session.tool.called":
            self._tool_inputs[call_id] = data.get("input")
            status = "running"
        elif event_type == "session.tool.success":
            status = "completed"
            extra = {"output": _tool_content_text(data.get("content")) or ""}
        elif event_type == "session.tool.failed":
            status = "error"
            extra["error"] = data.get("error")
        else:
            return []
        part = v1_tool_part(
            call_id=call_id,
            tool=self._tool_names.get(call_id),
            status=status,
            session_id=session_id if isinstance(session_id, str) else None,
            message_id=message_id,
            tool_input=self._tool_inputs.get(call_id),
            **extra,
        )
        part["role"] = ASSISTANT_ROLE
        if status in ("completed", "error"):
            self._tool_names.pop(call_id, None)
            self._tool_inputs.pop(call_id, None)
        return [_event("message.part.updated", {"sessionID": session_id, "part": part})]


def v1_questions_from_form(form: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = form.get("fields")
    questions: list[dict[str, Any]] = []
    if not isinstance(fields, list):
        return questions
    for field in fields:
        if not isinstance(field, Mapping) or field.get("hidden"):
            continue
        question: dict[str, Any] = {}
        header = _str(field.get("title"))
        text = _str(field.get("description")) or header
        if header:
            question["header"] = header
        if text:
            question["question"] = text
        options = field.get("options")
        if isinstance(options, list):
            question["options"] = [
                dict(option) for option in options if isinstance(option, Mapping)
            ]
        questions.append(question)
    return questions


def v2_form_answer(form: Mapping[str, Any], answers: Iterable[Sequence[str]]) -> dict[str, Any]:
    """Map v1 positional ``answers`` (one list of strings per question) onto the
    form's visible field keys."""
    fields = [
        field
        for field in (form.get("fields") or [])
        if isinstance(field, Mapping) and not field.get("hidden") and _str(field.get("key"))
    ]
    answer: dict[str, Any] = {}
    for field, values in zip(fields, answers, strict=False):
        selected = [value for value in values if isinstance(value, str)]
        field_type = field.get("type")
        key = field["key"]
        if field_type == "multiselect":
            answer[key] = selected
        elif field_type == "boolean":
            answer[key] = bool(selected) and selected[0].strip().lower() in (
                "true",
                "yes",
                "y",
                "1",
            )
        elif field_type in ("number", "integer"):
            if selected:
                try:
                    number = float(selected[0])
                except ValueError:
                    continue
                answer[key] = int(number) if field_type == "integer" else number
        else:
            if selected:
                answer[key] = ", ".join(selected)
    return answer
