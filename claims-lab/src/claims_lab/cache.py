"""Replay mode: a small SQLite cache for LLM responses.

Reference and demo cells run inside `with replay():`. If the exact same request was recorded before
(during the instructor's dry run), the stored answer is returned instantly and for free; otherwise
the live model is called and the answer is recorded.
"""
from __future__ import annotations

import json
import re
import os
import sqlite3
import threading
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

from langchain_core.caches import BaseCache
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.outputs import ChatGeneration, Generation

from .costs import CACHED_FLAG


def _key(prompt: str) -> str:
    """The request without what the model never sees and what differs between a recording and its replay: message ids,
    the random ids of content blocks, and each answer's metadata and usage (lookup() marks replayed answers there, and LangChain adds a cost of 0).
    Without this, an agent's second call, whose history holds the replayed first answer, would never match the
    recording, and every step after the first would run live."""
    try:
        data = json.loads(prompt)
    except ValueError:
        return prompt

    def clean(x):
        if isinstance(x, dict):
            if isinstance(x.get("id"), str) and x["id"].startswith("lc_") and "type" in x:
                del x["id"]                        # a content block's random id (MCP tool results carry one)
            kwargs = x.get("kwargs")
            if isinstance(kwargs, dict):
                for name in ("id", "response_metadata", "usage_metadata"):
                    kwargs.pop(name, None)
            for v in x.values():
                clean(v)
        elif isinstance(x, list):
            for v in x:
                clean(v)

    clean(data)
    return json.dumps(data, sort_keys=True)


_ENDPOINT = re.compile(r'"(azure_endpoint|base_url|openai_api_base)": "[^"]*"(, )?')


_INDENT = re.compile(r"\\n +")


def _llm_key(llm_string: str) -> str:
    """The model settings without the address of the service: the same deployment answers the same way whether its
    endpoint is written as the regional or the resource address, so a recording made here replays in class.

    Indentation after a line break is dropped too. The tool descriptions in these settings come from docstrings, and
    Python 3.13 and later strip a docstring's indentation while 3.12 keeps it, so without this a recording made on one
    Python version would never replay on the other."""
    return _INDENT.sub(r"\\n", _ENDPOINT.sub("", llm_string))


class SQLiteReplayCache(BaseCache):
    """LLM cache stored in one SQLite file. Cached messages are marked so the cost meter skips them."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache(prompt TEXT, llm TEXT, value TEXT, PRIMARY KEY(prompt, llm))")
        self._conn.commit()

    def lookup(self, prompt: str, llm_string: str):
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE prompt=? AND llm=?", (_key(prompt), _llm_key(llm_string))).fetchone()
        if row is None:
            if os.environ.get("CLAIMS_LAB_REPLAY_ONLY"):  # the pre-lecture check: a demo cell must not call the model
                raise RuntimeError("replay miss: this request is not in the recording")
            return None
        out = []
        for g in json.loads(row[0]):
            if "message" in g:
                msg = messages_from_dict([g["message"]])[0]
                msg.response_metadata = {**(msg.response_metadata or {}), CACHED_FLAG: True}
                out.append(ChatGeneration(message=msg))
            else:
                out.append(Generation(text=g["text"]))
        return out

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        data = [{"message": message_to_dict(g.message)} if isinstance(g, ChatGeneration) else {"text": g.text} for g in return_val]
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (_key(prompt), _llm_key(llm_string), json.dumps(data)))
            self._conn.commit()

    def lookup_raw(self, key: str, llm: str):
        """A recorded answer of OpenAI's plain client (notebook 01), as JSON data, or None."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE prompt=? AND llm=?", (key, llm)).fetchone()
        if row is None and os.environ.get("CLAIMS_LAB_REPLAY_ONLY"):
            raise RuntimeError("replay miss: this request is not in the recording")
        return json.loads(row[0]) if row else None

    def update_raw(self, key: str, llm: str, value) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (key, llm, json.dumps(value)))
            self._conn.commit()

    def clear(self, **kwargs) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]


def _default_path() -> Path:
    if os.environ.get("CLAIMS_LAB_REPLAY_DB"):  # e.g. a scratch file while testing
        return Path(os.environ["CLAIMS_LAB_REPLAY_DB"])
    from .models import provider
    if provider() == "offline":  # never mix scripted answers into the recorded cache
        from .data import workdir
        return workdir() / "replay_offline.db"
    return Path(str(files("claims_lab"))) / "replay" / "replay.db"  # outside data/, which the generator rebuilds


@contextmanager
def replay(path: str | Path | None = None):
    """Run the enclosed cells with the replay cache switched on.

    >>> with replay():
    ...     result = agent.invoke(...)
    """
    path = Path(path) if path else _default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = get_llm_cache()
    set_llm_cache(SQLiteReplayCache(path))
    try:
        yield
    finally:
        set_llm_cache(previous)
