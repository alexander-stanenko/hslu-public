"""Token and cost metering, and hard spending guards, for LangChain model calls.

A `CostMeter` is a callback handler: attach it to a model (`callbacks=[meter]`) or pass it in a run
config. It counts live calls, and separately the calls replayed from the cache with the tokens and cost they had
when they were recorded (so a replayed baseline is compared at its real cost). It can STOP a run before a call is
sent when a limit is reached:

- max_calls:        model calls per run (stops runaway agent loops)
- max_input_tokens: estimated size of a single request (stops ever-growing conversations early)
- max_usd:          total spend of this meter

Why this exists: LangChain's `create_agent` allows up to 9,999 steps. An agent stuck in a tool loop
re-sends its whole, growing history on every step; one such run cost ~$100 during course preparation.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

# USD per 1M tokens (input, output). Azure prices can be set with CLAIMS_LAB_PRICE_IN / _OUT.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5-mini": (0.25, 2.00),
    "offline": (0.0, 0.0),  # the scripted stand-in model (claims_lab.offline)
}
EMBEDDING_PRICE = 0.02  # USD per 1M tokens (text-embedding-3-small)
LONG_CONTEXT = {"claude-haiku-5-5": (100_000, 0.50, 2.50)}  # above this many input tokens a higher rate applies
CACHED_FLAG = "claims_lab_cached"


class BudgetExceeded(RuntimeError):
    """Raised before a model call that would break a spending guard."""


# ------------------------------------------------------------------ shared spending ledger (across processes)
# When CLAIMS_LAB_LEDGER points to a file, every live model and embedding call is appended to it, and NO call is
# sent once the total reaches CLAIMS_LAB_LEDGER_CAP_USD. This caps a whole test session (several notebooks/kernels).
def _ledger_path():
    p = os.environ.get("CLAIMS_LAB_LEDGER")
    return Path(p) if p else None


def ledger_total() -> float:
    p = _ledger_path()
    if not p or not p.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in p.read_text().splitlines() if line.strip())


def ledger_check() -> None:
    cap = float(os.environ.get("CLAIMS_LAB_LEDGER_CAP_USD", "inf"))
    if _ledger_path() and (spent := ledger_total()) >= cap:
        raise BudgetExceeded(f"Stopped: the shared spending cap of ${cap:.2f} is reached (spent ${spent:.4f}).")


def ledger_add(kind: str, model: str | None, input_tokens: int, output_tokens: int, usd: float) -> None:
    p = _ledger_path()
    if not p:
        return
    rec = dict(t=round(time.time(), 1), pid=os.getpid(), kind=kind, model=model, i=input_tokens, o=output_tokens, usd=usd)
    with open(p, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def price_for(model_name: str | None, input_tokens: int = 0) -> tuple[float, float]:
    if model_name:
        for key, p in PRICES.items():
            if model_name.startswith(key):
                if key in LONG_CONTEXT and input_tokens > LONG_CONTEXT[key][0]:
                    return LONG_CONTEXT[key][1:]
                return p
    return (float(os.environ.get("CLAIMS_LAB_PRICE_IN", 0.15)), float(os.environ.get("CLAIMS_LAB_PRICE_OUT", 0.60)))


def _estimate_tokens(messages) -> int:
    chars = 0
    for batch in messages:
        for m in batch:
            chars += len(m.content) if isinstance(m.content, str) else len(str(m.content))
            chars += len(str(getattr(m, "tool_calls", "") or ""))
    return chars // 4


@dataclass(eq=False)  # identity hash: LangChain keeps handlers in sets
class CostMeter(BaseCallbackHandler):
    """Counts LLM calls, tokens and USD cost, and enforces optional limits. Thread-safe.

    Live calls go into calls / tokens / cost_usd (what this session spends). Calls replayed from the cache go into
    replayed_calls / replayed_cost_usd, with the usage recorded at the time (what the design costs when it runs live).
    """
    max_calls: int | None = None
    max_input_tokens: int | None = None
    max_usd: float | None = None
    calls: int = 0
    started: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0
    replayed_calls: int = 0
    replayed_cost_usd: float = 0.0
    raise_error: bool = True  # let BudgetExceeded propagate to the caller
    ledger: bool = False      # record live calls in the shared ledger (only the session meter does, to avoid doubles)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        if self.ledger:
            ledger_check()
        with self._lock:
            if self.max_calls is not None and self.started >= self.max_calls:
                raise BudgetExceeded(f"Stopped: this run already made {self.started} model calls (limit {self.max_calls}). "
                                     f"The agent is probably stuck in a loop.")
            if self.max_usd is not None and self.cost_usd >= self.max_usd:
                raise BudgetExceeded(f"Stopped: spending limit of ${self.max_usd:.2f} reached (spent ${self.cost_usd:.4f}). "
                                     f"Tell the instructor if you need more.")
            est = _estimate_tokens(messages)
            if self.max_input_tokens is not None and est > self.max_input_tokens:
                raise BudgetExceeded(f"Stopped: the next request would send ~{est:,} tokens (limit {self.max_input_tokens:,}). "
                                     f"The conversation has grown too long.")
            self.started += 1

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:  # non-chat models
        with self._lock:
            if self.max_calls is not None and self.started >= self.max_calls:
                raise BudgetExceeded(f"Stopped: {self.started} model calls (limit {self.max_calls}).")
            self.started += 1

    def on_llm_end(self, response, **kwargs) -> None:
        for gens in response.generations:
            for gen in gens:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                meta = getattr(msg, "response_metadata", {}) or {}
                usage = getattr(msg, "usage_metadata", None) or {}
                i, o = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                pin, pout = price_for(meta.get("model_name") or meta.get("model"), i)
                usd = i * pin / 1e6 + o * pout / 1e6
                with self._lock:
                    if meta.get(CACHED_FLAG):  # replayed: free now, but report what it cost when recorded
                        self.cache_hits += 1
                        self.replayed_calls += 1
                        self.replayed_cost_usd += usd
                        continue
                    self.calls += 1
                    self.input_tokens += i
                    self.output_tokens += o
                    self.cost_usd += usd
                if self.ledger:
                    ledger_add("chat", meta.get("model_name") or meta.get("model"), i, o, usd)

    def reset(self) -> None:
        with self._lock:
            self.calls = self.started = self.input_tokens = self.output_tokens = self.cache_hits = self.replayed_calls = 0
            self.cost_usd = self.replayed_cost_usd = 0.0

    def summary(self) -> dict:
        return dict(llm_calls=self.calls, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                    cost_usd=round(self.cost_usd, 5), cache_hits=self.cache_hits, replayed_calls=self.replayed_calls,
                    replayed_cost_usd=round(self.replayed_cost_usd, 5))

    def __str__(self) -> str:
        s = f"{self.calls} LLM calls, {self.input_tokens:,} in / {self.output_tokens:,} out tokens, ${self.cost_usd:.4f}"
        return s + (f", {self.cache_hits} replayed from cache (recorded cost ${self.replayed_cost_usd:.4f})" if self.cache_hits else "")


# Session-wide guard on every model from get_model(): spending cap per notebook session and a cap on the size
# of any single request. Defaults are far above normal course use (~$0.10 per notebook, ~10k tokens per request).
SESSION_METER = CostMeter(max_usd=float(os.environ.get("CLAIMS_LAB_SESSION_BUDGET_USD", 2.0)),
                          max_input_tokens=int(os.environ.get("CLAIMS_LAB_MAX_INPUT_TOKENS", 60_000)), ledger=True)

# Per-run limits used by the scoreboard (one claim package = one run).
RUN_MAX_CALLS = int(os.environ.get("CLAIMS_LAB_RUN_MAX_CALLS", 25))


def cost_report() -> str:
    """Total LLM usage of this notebook session (all models created with get_model())."""
    print(f"Session so far: {SESSION_METER}  (session limit ${SESSION_METER.max_usd:.2f})")
    return str(SESSION_METER)
