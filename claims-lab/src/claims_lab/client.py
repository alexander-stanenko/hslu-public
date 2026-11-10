"""OpenAI's own Python client for notebook 01, which works without a framework.

    client = cl.get_client()
    response = client.chat.completions.create(model="gpt-5.4-mini", messages=[...], tools=[...])
    response = client.chat.completions.parse(model="gpt-5.4-mini", messages=[...], response_format=cl.ClaimDecision)

Requests and answers are the openai package's own (`AzureOpenAI`). Around each call the course adds what
`get_model()` adds to a LangChain model from notebook 02 on:

- the session cost meter and spending cap (`cl.cost_report()`), and the meter of a scoreboard run;
- inside `with cl.replay():`, the recorded answer for the same request, if there is one (instant and free);
- with CLAIMS_LAB_PROVIDER=offline, a scripted stand-in that needs no key (tests).
"""
from __future__ import annotations

import json
import os

from .costs import RUN_METERS, SESSION_METER

_INFO: dict[int, tuple] = {}  # id(object) -> (object, (model, input tokens, output tokens, replayed)), for show_trace


def _remember(obj, info: tuple) -> None:
    _INFO[id(obj)] = (obj, info)


def info_of(obj) -> tuple | None:
    """(model, input tokens, output tokens, replayed) of an answer or its message from this client, else None."""
    entry = _INFO.get(id(obj))
    return entry[1] if entry and entry[0] is obj else None


def usage_of(response) -> tuple:
    """(model, input tokens, output tokens, replayed) of a response from client.chat.completions.create or .parse."""
    info = info_of(response)
    if info:
        return info
    u = getattr(response, "usage", None)
    return getattr(response, "model", None), getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0), False


def plain(x):
    """Messages, tool calls and answers as plain JSON data (dicts and lists)."""
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json", exclude_none=True, warnings=False)
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def _request(kind: str, kwargs: dict) -> dict:
    body = {k: plain(v) for k, v in kwargs.items() if k != "response_format"}
    rf = kwargs.get("response_format")
    if isinstance(rf, type):
        body["response_format"] = {"name": rf.__name__, "schema": rf.model_json_schema()}
    elif rf is not None:
        body["response_format"] = plain(rf)
    return {"kind": kind, **body}


def _rebuild(kind: str, kwargs: dict, data: dict):
    from openai.types.chat import ChatCompletion, ParsedChatCompletion
    rf = kwargs.get("response_format")
    if kind == "parse" and isinstance(rf, type):
        return ParsedChatCompletion[rf].model_validate(data)
    return ChatCompletion.model_validate(data)


class _Completions:
    def __init__(self, owner: "CourseClient"):
        self._owner = owner

    def create(self, **kwargs):
        """client.chat.completions.create(model=..., messages=[...], tools=[...]): one model call."""
        return self._call("create", kwargs)

    def parse(self, **kwargs):
        """client.chat.completions.parse(model=..., messages=[...], response_format=SomeClass): one model call whose
        answer fills in the class; the filled-in object is response.choices[0].message.parsed."""
        return self._call("parse", kwargs)

    def _call(self, kind: str, kwargs: dict):
        from langchain_core.globals import get_llm_cache

        from .cache import SQLiteReplayCache
        kwargs.setdefault("max_completion_tokens", int(os.environ.get("CLAIMS_LAB_MAX_OUTPUT_TOKENS", 4000)))
        request = _request(kind, kwargs)
        cache = get_llm_cache()
        cache = cache if isinstance(cache, SQLiteReplayCache) else None
        key, llm = json.dumps(request, sort_keys=True), f"openai-client:{kwargs.get('model')}"
        data = cache.lookup_raw(key, llm) if cache else None
        if data is not None:
            response, replayed = _rebuild(kind, kwargs, data), True
        else:
            SESSION_METER._before(len(json.dumps(request.get("messages", []))) // 4)
            for meter in RUN_METERS.get():
                meter._before()
            response, replayed = self._owner._send(kind, kwargs), False
            if cache:
                cache.update_raw(key, llm, response.model_dump(mode="json", warnings=False))
        u = response.usage
        info = (response.model, u.prompt_tokens if u else 0, u.completion_tokens if u else 0, replayed)
        SESSION_METER._record(*info)
        for meter in RUN_METERS.get():
            meter._record(*info)
        _remember(response, info)
        _remember(response.choices[0].message, info)
        return response


class _Chat:
    def __init__(self, owner: "CourseClient"):
        self.completions = _Completions(owner)


class CourseClient:
    """OpenAI's client with the course key, the cost meter and the replay (see the module docstring)."""

    def __init__(self):
        from .models import COURSE_AZURE, provider
        self.provider = provider()
        self.chat = _Chat(self)
        self._client = None
        if self.provider == "azure_openai":
            from openai import AzureOpenAI
            self._client = AzureOpenAI(api_version=os.environ.get("OPENAI_API_VERSION", COURSE_AZURE["OPENAI_API_VERSION"]),
                                       max_retries=6)   # wait and retry after a rate limit
        elif self.provider != "offline":
            raise RuntimeError("Notebook 01 uses OpenAI's client, which needs the course's Azure OpenAI key. "
                               "Run the first cell of the notebook, or ask the instructor.")

    def _send(self, kind: str, kwargs: dict):
        if self._client is None:
            return _offline(kind, kwargs)
        api = self._client.chat.completions
        return api.parse(**kwargs) if kind == "parse" else api.create(**kwargs)

    def __repr__(self) -> str:
        return f"<OpenAI client for the course ({self.provider}), with the cost meter and the replay>"


def get_client() -> CourseClient:
    """OpenAI's Python client with the course key, the cost meter and the replay of recorded answers (notebook 01)."""
    return CourseClient()


# ------------------------------------------------------------------ messages in LangChain's form (for helpers)
def to_langchain(messages: list) -> list:
    """Notebook 01's messages (dicts and the client's reply objects) as LangChain messages, with the tokens of each
    answer, so that the course's helpers (show_trace, the offline stand-in) read both kinds the same way."""
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

    from .costs import CACHED_FLAG
    out, names = [], {}
    for m in messages:
        if isinstance(m, BaseMessage) or not (isinstance(m, dict) or hasattr(m, "model_dump")):
            out.append(m)  # already a LangChain message
            continue
        info = info_of(m)
        d = plain(m)
        role, content = d.get("role"), d.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content))
        elif role == "user":
            out.append(HumanMessage(content))
        elif role == "assistant":
            calls = []
            for tc in d.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {"arguments": fn.get("arguments")}
                names[tc.get("id")] = fn.get("name")
                calls.append({"name": fn.get("name"), "args": args, "id": tc.get("id"), "type": "tool_call"})
            ai = AIMessage(content=content, tool_calls=calls)
            if info:
                model, i, o, replayed = info
                ai.usage_metadata = {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}
                ai.response_metadata = {"model_name": model, **({CACHED_FLAG: True} if replayed else {})}
            out.append(ai)
        elif role == "tool":
            out.append(ToolMessage(content=str(content), tool_call_id=d.get("tool_call_id", ""),
                                   name=names.get(d.get("tool_call_id"))))
    return out


def _alone(invoke, messages):
    """Run the stand-in outside any surrounding LangChain run (a scoreboard run): the client has already reported
    this call to the meters, which must not count it a second time through the run's callbacks."""
    import contextvars

    from langchain_core.runnables.config import var_child_runnable_config

    def run():
        var_child_runnable_config.set(None)
        return invoke(messages)
    return contextvars.copy_context().run(run)


def _offline(kind: str, kwargs: dict):
    """The scripted stand-in (claims_lab.offline) answering in the shape of OpenAI's client."""
    from openai.types.chat import ChatCompletion, ParsedChatCompletion

    from .offline import OfflineChatModel
    lc = to_langchain(kwargs.get("messages", []))
    model = OfflineChatModel()
    rf = kwargs.get("response_format")
    n_in = len(json.dumps(plain(kwargs.get("messages", [])))) // 4
    base = {"id": "offline", "object": "chat.completion", "created": 0, "model": "offline"}
    if kind == "parse" and isinstance(rf, type):
        obj = _alone(model.with_structured_output(rf).invoke, lc)
        content = obj.model_dump_json()
        data = {**base, "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": content, "parsed": obj.model_dump()}}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": len(content) // 4, "total_tokens": n_in + len(content) // 4}}
        return ParsedChatCompletion[rf].model_validate(data)
    ai = _alone((model.bind_tools(kwargs["tools"]) if kwargs.get("tools") else model).invoke, lc)
    calls = [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
             for tc in ai.tool_calls]
    message = {"role": "assistant", "content": ai.text or None, **({"tool_calls": calls} if calls else {})}
    n_out = max(1, len(ai.text) // 4 + 20 * len(calls))
    data = {**base, "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": message}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}}
    return ChatCompletion.model_validate(data)
