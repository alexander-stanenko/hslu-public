"""The scoreboard: run a claims system on labelled packages and compare designs on accuracy, safety and cost.

    from claims_lab import scoreboard
    def my_system(package) -> ClaimDecision: ...
    scoreboard(my_system, label="better tools")                    # sync systems
    await ascoreboard(my_async_system, label="via MCP")            # async systems (e.g. MCP tools)
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from langchain_core.runnables import RunnableLambda

from .costs import RUN_MAX_CALLS, RUN_METERS, CostMeter
from .data import Package, labels, load_package
from .schemas import ClaimDecision
from .trace import indented

HISTORY: list[dict] = []
RUNS: dict[str, pd.DataFrame] = {}  # the latest per-claim table of each design, for the claims grid

# phrases that count as naming an expected issue or missing item. They must start a word ("late" does not match
# "calculated"); the issues and missing_items lists are checked first, then the reasoning.
SYNONYMS = {
    "duplicate_invoice": ["duplicate", "already used", "already paid", "same invoice", "reused", "used before", "earlier claim"],
    "possible_exclusion": ["exclusion", "excluded", "wear", "maintenance", "tread", "tyre", "tire", "rust", "key", "unlocked", "§3"],
    "above_fast_track_limit": ["5,000", "5000", "5'000", "fast-track limit", "fast_track_limit", "above the limit", "exceeds the limit", "above_fast_track"],
    "not_covered": ["not covered", "not_covered", "does not cover", "doesn't cover", "only covers", "not included", "not insured"],
    "policy_inactive": ["inactive", "not active", "not valid", "no longer valid", "expired", "ended", "valid_to", "policy_inactive"],
    "police_report": ["police"],
    "invoice": ["invoice"],
    "policy_id": ["policy number", "policy_id", "policy id", "not identified", "cannot identify", "could not identify", "which policy"],
}


def as_system(x, max_steps: int = 12):
    """A claims system (package in, decision out) from what the course builds: a function is used as it is; an agent
    from create_agent gets the package as its user message; the pipeline of notebook 04 gets it in its state."""
    if not hasattr(x, "invoke") or not hasattr(x, "get_graph"):
        return x
    from .models import run_config
    keys = set(x.get_input_jsonschema().get("properties", {}))
    if "package" in keys:
        return lambda pkg: x.invoke({"package": pkg}, config=run_config(25))["decision"]
    return lambda pkg: x.invoke({"messages": [("user", pkg.text())]}, config=run_config(max_steps))


def as_async_system(x, max_steps: int = 12):
    """Async version of as_system, for agents whose tools are MCP tools (they can only be called asynchronously)."""
    if not hasattr(x, "ainvoke") or not hasattr(x, "get_graph"):
        return x
    from .models import run_config

    async def system(pkg):
        return await x.ainvoke({"messages": [("user", pkg.text())]}, config=run_config(max_steps))
    return system


def _sample(split: str, n: int | None, ids: list[str] | None) -> list[dict]:
    pool = [l for l in (labels("dev") if split == "quick" else labels(split)) if not ids or l["package_id"] in ids]
    if split == "quick" and not ids:
        pool = [l for l in pool if l.get("quick")]
    if ids or n is None or n >= len(pool):
        return pool
    by_cat: dict[str, list] = {}
    for l in pool:
        by_cat.setdefault(l["category"], []).append(l)
    out, i = [], 0
    while len(out) < n:  # deterministic round-robin over categories -> stratified sample
        for cat in sorted(by_cat):
            if i < len(by_cat[cat]) and len(out) < n:
                out.append(by_cat[cat][i])
        i += 1
    return out


def _to_decision(x) -> ClaimDecision:
    if isinstance(x, ClaimDecision):
        return x
    if isinstance(x, dict) and "structured_response" in x:
        return _to_decision(x["structured_response"])
    if isinstance(x, dict):
        return ClaimDecision(**x)
    if isinstance(x, str):
        return ClaimDecision(**json.loads(x))
    return ClaimDecision.model_validate(x, from_attributes=True)


def _evidence(tag: str, d: ClaimDecision) -> bool:
    """Does the decision name this issue? Its issues and missing items first, then its reasoning."""
    listed = [x.lower().strip() for x in d.issues + d.missing_items]
    if tag in listed:
        return True
    phrases = [tag.replace("_", " ")] + SYNONYMS.get(tag, [])
    hay = " ".join(listed + [d.reasoning.lower()])
    return any(re.search(r"\b" + re.escape(p), hay) for p in phrases)


def _score(lab: dict, d: ClaimDecision | None, err: str | None) -> dict:
    exp = lab["expected"]
    row = dict(package=lab["package_id"], claim=lab.get("nickname", ""), category=lab["category"], expected=exp["decision"],
               predicted=d.decision if d else "ERROR", correct=bool(d and d.decision == exp["decision"]))
    row["payout_ok"] = None
    if exp["decision"] == "fast_track":
        row["payout_ok"] = bool(d and d.decision == "fast_track" and d.payout_chf is not None
                                and abs(d.payout_chf - exp["payout_chf"]) <= 1)
    row["unsafe_approval"] = bool(d and d.decision == "fast_track" and exp["decision"] != "fast_track")
    tags = exp["issues"] + exp["missing_items"]
    row["issues_found"] = (sum(_evidence(t, d) for t in tags) / len(tags)) if (tags and d) else (None if not tags else 0.0)
    row["expected_payout"] = exp["payout_chf"]
    row["predicted_payout"] = d.payout_chf if d else None
    row["error"] = err
    return row


def _row(lab: dict, d, err, meter: CostMeter) -> dict:
    """One scored claim. Calls, cost and model time include replayed calls as recorded, so replayed designs compare
    fairly. seconds is the time spent waiting for the model (None if a replayed call was recorded without it)."""
    row = _score(lab, d, err)
    row.update(llm_calls=meter.calls + meter.replayed_calls, tool_calls=meter.tool_calls,
               cost_usd=round(meter.cost_usd + meter.replayed_cost_usd, 5),
               seconds=None if meter.seconds_unknown else round(meter.model_seconds, 1), replayed_calls=meter.replayed_calls)
    return row


def _summarise(rows: list[dict], label: str) -> dict:
    df = pd.DataFrame(rows)
    ran = df[df["error"].isna()]  # claims that failed (e.g. a rate limit) are not counted as wrong decisions
    pay = ran["payout_ok"].dropna()
    iss = ran["issues_found"].dropna()
    return dict(label=label, n=len(df), accuracy=round(ran["correct"].mean(), 3) if len(ran) else 0.0,
                payout_accuracy=round(pay.mean(), 3) if len(pay) else None, payouts_checked=len(pay),
                unsafe_approvals=int(df["unsafe_approval"].sum()), issue_recall=round(iss.mean(), 3) if len(iss) else None,
                errors=int(df["error"].notna().sum()), llm_calls_per_pkg=round(df["llm_calls"].mean(), 1),
                tool_calls_per_pkg=round(df["tool_calls"].mean(), 1),
                cost_usd_total=round(df["cost_usd"].sum(), 4), cost_usd_per_pkg=round(df["cost_usd"].mean(), 5),
                seconds_per_pkg=None if df["seconds"].isna().any() else round(df["seconds"].mean(), 1),
                replayed=bool(df["replayed_calls"].sum()),
                to_person=int((ran["predicted"] == "refer_adjuster").sum()),     # work for an adjuster
                to_person_expected=int((ran["expected"] == "refer_adjuster").sum()))


LEGEND = ("correct: the expected decision (the clerk's, from the cheat sheet); payout_ok: right amount (claims to pay "
          "only); unsafe_approval: paid although it should not have been; llm_calls, tool_calls: model calls and the tools "
          "the model asked for, for this claim")
RETRYABLE = ("429", "rate limit", "ratelimit", "timeout", "timed out", "temporarily", "overloaded", "connection")


def _retryable(err: str | None) -> bool:
    return bool(err) and "BudgetExceeded" not in err and any(k in err.lower() for k in RETRYABLE)


def _report(rows, summary, show: bool):
    df = pd.DataFrame(rows)
    HISTORY.append(summary)
    RUNS[summary["label"]] = df
    if show:
        s = summary
        print(f"SCOREBOARD: {s['label']}  ({s['n']} claims)" + (". Replayed: calls, cost and time as recorded in the live run" if s["replayed"] else ""))
        if s["errors"]:
            print(f"  Warning: {s['errors']} of {s['n']} claims failed (see the error column), probably the rate limit. They are left out of "
                  f"the accuracy. Wait 30 seconds and run the cell again.")
        pay_n = s.get("payouts_checked")
        print(f"  decision accuracy {s['accuracy']:.0%}, payout accuracy {s['payout_accuracy'] if s['payout_accuracy'] is None else format(s['payout_accuracy'], '.0%')}"
              + (f" (of {pay_n} claim{'s' if pay_n != 1 else ''} to pay)" if pay_n else "")
              + f", unsafe approvals {s['unsafe_approvals']}, issues found {s['issue_recall'] if s['issue_recall'] is None else format(s['issue_recall'], '.0%')}")
        secs = f", {s['seconds_per_pkg']} s of model time" if s["seconds_per_pkg"] is not None else ""
        print(f"  per claim: {s['llm_calls_per_pkg']} model calls, {s['tool_calls_per_pkg']} tool calls{secs}, "
              f"${s['cost_usd_per_pkg']:.4f}" + (f"; {s['errors']} errors" if s["errors"] else ""))
        print(f"  sent to a person (refer_adjuster): {s['to_person']} of {s['n'] - s['errors']} claims, "
              f"{s['to_person_expected']} expected")
        try:
            from IPython.display import display
            cols = ["package", "claim", "expected", "predicted", "correct", "payout_ok", "unsafe_approval", "llm_calls",
                    "tool_calls"]
            cols += ["error"] if df["error"].notna().any() else []
            from .hslu_palette import HSLU
            warn = HSLU["accents"]["magenta"]["hell1"]
            try:
                display(df[cols].style.apply(lambda r: [f"background-color: {warn}" if r["unsafe_approval"] else "" for _ in r], axis=1))
            except (ImportError, AttributeError):  # pandas styling needs jinja2; show the plain table without it
                display(df[cols])
            print(LEGEND)
        except Exception:
            pass
    return df


def scoreboard(system, split: str = "quick", n: int | None = None, ids: list[str] | None = None, concurrency: int = 2,
               label: str = "run", show: bool = True) -> pd.DataFrame:
    """Run a claims system on labelled packages and score it.

    Args:
        system: an agent from create_agent, the pipeline of notebook 04, or a function taking a Package and returning a
            ClaimDecision (or a dict / agent result containing one).
        split: 'quick' (default: PKG-001 to PKG-008), 'dev' (PKG-001 to PKG-016) or 'heldout' (PKG-101 to PKG-108).
        n: number of packages (stratified sample; None = all of the split). ids: explicit package IDs instead.
        concurrency: packages processed in parallel (keep it small to respect rate limits).
        label: name of this design, shown in scoreboard_history().
    """
    labs = _sample(split, n, ids)
    system = as_system(system)
    runnable = RunnableLambda(lambda pkg: system(pkg))

    def run_one(lab):
        pkg = load_package(lab["package_id"])
        for attempt in (1, 2):  # one retry after a rate limit or a timeout
            meter = CostMeter(max_calls=RUN_MAX_CALLS, max_input_tokens=60_000)
            token = RUN_METERS.set((meter,))  # OpenAI's plain client (notebook 01) reports here; LangChain via the config
            try:
                d, err = _to_decision(runnable.invoke(pkg, config={"callbacks": [meter]})), None
            except Exception as e:  # keep going; the error is shown in the table
                d, err = None, f"{type(e).__name__}: {str(e)[:120]}"
            finally:
                RUN_METERS.reset(token)
            if not (attempt == 1 and _retryable(err)):
                break
            time.sleep(5)
        return _row(lab, d, err, meter)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        rows = list(ex.map(run_one, labs))
    return _report(rows, _summarise(rows, label), show)


async def ascoreboard(system, split: str = "quick", n: int | None = None, ids: list[str] | None = None, concurrency: int = 2,
                      label: str = "run", show: bool = True) -> pd.DataFrame:
    """Async version of `scoreboard` for agents with MCP tools (or functions defined with `async def`)."""
    labs = _sample(split, n, ids)
    system = as_async_system(system)
    sem = asyncio.Semaphore(concurrency)

    async def call(pkg: Package):
        return await system(pkg)

    runnable = RunnableLambda(lambda pkg: None, afunc=call)

    async def run_one(lab):
        async with sem:
            pkg = load_package(lab["package_id"])
            for attempt in (1, 2):  # one retry after a rate limit or a timeout
                meter = CostMeter(max_calls=RUN_MAX_CALLS, max_input_tokens=60_000)
                try:
                    d, err = _to_decision(await runnable.ainvoke(pkg, config={"callbacks": [meter]})), None
                except Exception as e:
                    d, err = None, f"{type(e).__name__}: {str(e)[:120]}"
                if not (attempt == 1 and _retryable(err)):
                    break
                await asyncio.sleep(5)
            return _row(lab, d, err, meter)

    rows = await asyncio.gather(*(run_one(l) for l in labs))
    return _report(list(rows), _summarise(list(rows), label), show)


def spot_check(system, *package_ids: str) -> None:
    """Run a system on a few packages and print each decision next to the expected one (no scoreboard entry)."""
    system = as_system(system)
    for pid in package_ids:
        try:
            d = _to_decision(system(load_package(pid)))
            got, why = d.decision, d.reasoning
        except Exception as e:  # show the problem, keep going
            got, why = "ERROR", f"{type(e).__name__}: {e}"
        lab = next(l for l in labels() if l["package_id"] == pid)
        expected = lab["expected"]["decision"]
        print(f"{'yes' if got == expected else 'no '} {pid} ({lab['nickname']}): {got} (expected {expected})\n{indented(why)}\n")


async def aspot_check(system, *package_ids: str) -> None:
    """Async version of `spot_check` for agents with MCP tools (or functions defined with `async def`)."""
    system = as_async_system(system)
    for pid in package_ids:
        try:
            d = _to_decision(await system(load_package(pid)))
            got, why = d.decision, d.reasoning
        except Exception as e:  # show the problem, keep going
            got, why = "ERROR", f"{type(e).__name__}: {e}"
        lab = next(l for l in labels() if l["package_id"] == pid)
        expected = lab["expected"]["decision"]
        print(f"{'yes' if got == expected else 'no '} {pid} ({lab['nickname']}): {got} (expected {expected})\n{indented(why)}\n")


def compare(a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, str] = ("A", "B")) -> pd.DataFrame:
    """Two scoreboard runs side by side, one row per claim: which claims one design decided correctly and the other did not.

    a, b: the tables returned by scoreboard() or ascoreboard() on the same claims.
    """
    cols = ["package", "claim", "expected", "predicted", "correct", "unsafe_approval", "llm_calls"]
    cols += ["tool_calls"] if "tool_calls" in a and "tool_calls" in b else []
    m = a[cols].merge(b[cols], on=["package", "claim", "expected"], suffixes=(f" {names[0]}", f" {names[1]}"))
    ca, cb = m[f"correct {names[0]}"], m[f"correct {names[1]}"]
    m["change"] = ["fixed" if (not x and y) else "broken" if (x and not y) else "" for x, y in zip(ca, cb)]
    return m


def scoreboard_history(plot: bool = True, grid: bool = True) -> pd.DataFrame:
    """All scoreboard runs of this session: a chart of accuracy against cost, and a grid of which design decided which
    claim correctly."""
    df = pd.DataFrame(HISTORY)
    if plot and len(df):
        from .style import plot_claims_grid, plot_history
        plot_history(df)
        if grid:
            plot_claims_grid(RUNS)
    cols = ["label", "accuracy", "unsafe_approvals", "payout_accuracy", "issue_recall", "to_person", "llm_calls_per_pkg",
            "tool_calls_per_pkg", "cost_usd_per_pkg", "seconds_per_pkg"]
    if len(df) and df["errors"].any():
        cols.append("errors")
    return df[[c for c in cols if c in df]] if len(df) else df
