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

from .costs import RUN_MAX_CALLS, CostMeter
from .data import Package, labels, load_package
from .schemas import ClaimDecision
from .trace import indented

HISTORY: list[dict] = []
RUNS: dict[str, pd.DataFrame] = {}  # the latest per-claim table of each design, for the claims grid

# phrases that count as naming an expected issue or missing item. They must start a word ("late" does not match
# "calculated"); the issues and missing_items lists are checked first, then the reasoning.
SYNONYMS = {
    "invoice_before_incident": ["before the incident", "before incident", "predates", "invoice_before", "dated before", "earlier than the incident", "prior to the incident"],
    "duplicate_invoice": ["duplicate", "already used", "already paid", "same invoice", "reused", "used before"],
    "frequency": ["frequency", "third claim", "3 claims", "three claims", "3rd", "multiple claims", "claims within 12 months"],
    "amount_above_typical": ["above typical", "amount_above", "exceeds typical", "inflated", "unusually high", "far above", "price range"],
    "no_weather_event": ["no weather", "no hail", "no storm", "no event", "weather_event", "not recorded"],
    "late_report": ["late", "30 days", "reporting deadline", "delay"],
    "possible_exclusion": ["exclusion", "wear", "maintenance", "gradual", "negligence", "key in the ignition", "unlocked"],
    "inconsistent_documents": ["inconsistent", "contradict", "mismatch", "rear", "does not match", "discrepanc"],
    "above_fast_track_limit": ["5,000", "5000", "fast-track limit", "fast_track_limit", "above the limit", "exceeds the limit", "above_fast_track"],
    "not_covered": ["not covered", "not_covered", "no partial", "liability only", "liability-only", "add-on", "collision module", "not included"],
    "policy_inactive": ["inactive", "not active", "cancelled", "expired", "ended", "policy_inactive"],
    "police_report": ["police"],
    "proof_of_ownership": ["ownership", "receipt", "proof"],
    "invoice_or_estimate": ["invoice or estimate", "invoice or cost estimate", "repair invoice", "cost estimate", "no invoice", "missing invoice", "estimate", "quote"],
    "policy_id": ["policy number", "policy_id", "policy id", "not identified", "cannot identify", "could not identify", "which policy"],
}


def _sample(split: str, n: int | None, ids: list[str] | None) -> list[dict]:
    pool = [l for l in labels(split) if not ids or l["package_id"] in ids]
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
    row = dict(package=lab["package_id"], category=lab["category"], expected=exp["decision"],
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


def _row(lab: dict, d, err, meter: CostMeter, t0: float) -> dict:
    """One scored claim. Calls and cost include replayed calls at their recorded cost, so replayed designs compare fairly."""
    row = _score(lab, d, err)
    row.update(llm_calls=meter.calls + meter.replayed_calls, cost_usd=round(meter.cost_usd + meter.replayed_cost_usd, 5),
               seconds=round(time.time() - t0, 1), replayed_calls=meter.replayed_calls)
    return row


def _summarise(rows: list[dict], label: str, seconds: float) -> dict:
    df = pd.DataFrame(rows)
    ran = df[df["error"].isna()]  # claims that failed (e.g. a rate limit) are not counted as wrong decisions
    pay = ran["payout_ok"].dropna()
    iss = ran["issues_found"].dropna()
    return dict(label=label, n=len(df), accuracy=round(ran["correct"].mean(), 3) if len(ran) else 0.0,
                payout_accuracy=round(pay.mean(), 3) if len(pay) else None, payouts_checked=len(pay),
                unsafe_approvals=int(df["unsafe_approval"].sum()), issue_recall=round(iss.mean(), 3) if len(iss) else None,
                errors=int(df["error"].notna().sum()), llm_calls_per_pkg=round(df["llm_calls"].mean(), 1),
                cost_usd_total=round(df["cost_usd"].sum(), 4), cost_usd_per_pkg=round(df["cost_usd"].mean(), 5),
                seconds_per_pkg=round(df["seconds"].mean(), 1), wall_seconds=round(seconds, 1),
                replayed=bool(df["replayed_calls"].sum()),
                to_person=int((ran["predicted"] == "refer_adjuster").sum()),     # work for an adjuster
                to_person_expected=int((ran["expected"] == "refer_adjuster").sum()))


LEGEND = ("correct: same decision as the experienced clerk; payout_ok: right amount (claims to pay only); "
          "unsafe_approval: paid although it should not have been; issues_found: share of the expected issues the decision names")
RETRYABLE = ("429", "rate limit", "ratelimit", "timeout", "timed out", "temporarily", "overloaded", "connection")


def _retryable(err: str | None) -> bool:
    return bool(err) and "BudgetExceeded" not in err and any(k in err.lower() for k in RETRYABLE)


def _report(rows, summary, show: bool):
    df = pd.DataFrame(rows)
    HISTORY.append(summary)
    RUNS[summary["label"]] = df
    if show:
        s = summary
        print(f"SCOREBOARD: {s['label']}  ({s['n']} claims)" + (". Replayed: calls and cost as recorded in the live run" if s["replayed"] else ""))
        if s["errors"]:
            print(f"  Warning: {s['errors']} of {s['n']} claims failed (see the error column), probably the rate limit. They are left out of "
                  f"the accuracy. Wait 30 seconds and run the cell again.")
        pay_n = s.get("payouts_checked")
        print(f"  decision accuracy {s['accuracy']:.0%}, payout accuracy {s['payout_accuracy'] if s['payout_accuracy'] is None else format(s['payout_accuracy'], '.0%')}"
              + (f" (of {pay_n} claim{'s' if pay_n != 1 else ''} to pay)" if pay_n else "")
              + f", unsafe approvals {s['unsafe_approvals']}, issues found {s['issue_recall'] if s['issue_recall'] is None else format(s['issue_recall'], '.0%')}")
        print(f"  {s['llm_calls_per_pkg']} model calls per claim, ${s['cost_usd_per_pkg']:.4f} per claim "
              f"(about ${s['cost_usd_per_pkg'] * 10000:,.0f} per 10,000 claims, model calls only), {s['seconds_per_pkg']} s per claim"
              + (f", {s['errors']} errors" if s["errors"] else ""))
        print(f"  sent to a person (refer_adjuster): {s['to_person']} of {s['n'] - s['errors']} claims, "
              f"{s['to_person_expected']} expected")
        try:
            from IPython.display import display
            cols = ["package", "category", "expected", "predicted", "correct", "payout_ok", "unsafe_approval", "issues_found", "llm_calls", "cost_usd", "seconds", "error"]
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


def scoreboard(system, split: str = "dev", n: int | None = 8, ids: list[str] | None = None, concurrency: int = 2,
               label: str = "run", show: bool = True) -> pd.DataFrame:
    """Run `system(package) -> ClaimDecision` on labelled packages and score it.

    Args:
        system: a function taking a Package and returning a ClaimDecision (or a dict / agent result containing one).
        split: 'dev' (default) or 'heldout'.
        n: number of packages (stratified sample; None = all of the split). ids: explicit package IDs instead.
        concurrency: packages processed in parallel (keep it small to respect rate limits).
        label: name of this design, shown in scoreboard_history().
    """
    labs = _sample(split, n, ids)
    runnable = RunnableLambda(lambda pkg: system(pkg))

    def run_one(lab):
        pkg = load_package(lab["package_id"])
        for attempt in (1, 2):  # one retry after a rate limit or a timeout
            meter = CostMeter(max_calls=RUN_MAX_CALLS, max_input_tokens=60_000)
            t0 = time.time()
            try:
                d, err = _to_decision(runnable.invoke(pkg, config={"callbacks": [meter]})), None
            except Exception as e:  # keep going; the error is shown in the table
                d, err = None, f"{type(e).__name__}: {str(e)[:120]}"
            if not (attempt == 1 and _retryable(err)):
                break
            time.sleep(5)
        return _row(lab, d, err, meter, t0)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        rows = list(ex.map(run_one, labs))
    return _report(rows, _summarise(rows, label, time.time() - t0), show)


async def ascoreboard(system, split: str = "dev", n: int | None = 8, ids: list[str] | None = None, concurrency: int = 2,
                      label: str = "run", show: bool = True) -> pd.DataFrame:
    """Async version of `scoreboard` for systems defined with `async def` (e.g. agents using MCP tools)."""
    labs = _sample(split, n, ids)
    sem = asyncio.Semaphore(concurrency)

    async def call(pkg: Package):
        return await system(pkg)

    runnable = RunnableLambda(lambda pkg: None, afunc=call)

    async def run_one(lab):
        async with sem:
            pkg = load_package(lab["package_id"])
            for attempt in (1, 2):  # one retry after a rate limit or a timeout
                meter = CostMeter(max_calls=RUN_MAX_CALLS, max_input_tokens=60_000)
                t0 = time.time()
                try:
                    d, err = _to_decision(await runnable.ainvoke(pkg, config={"callbacks": [meter]})), None
                except Exception as e:
                    d, err = None, f"{type(e).__name__}: {str(e)[:120]}"
                if not (attempt == 1 and _retryable(err)):
                    break
                await asyncio.sleep(5)
            return _row(lab, d, err, meter, t0)

    t0 = time.time()
    rows = await asyncio.gather(*(run_one(l) for l in labs))
    return _report(list(rows), _summarise(list(rows), label, time.time() - t0), show)


def spot_check(system, *package_ids: str) -> None:
    """Run a system on a few packages and print each decision next to the expected one (no scoreboard entry)."""
    for pid in package_ids:
        try:
            d = _to_decision(system(load_package(pid)))
            got, why = d.decision, d.reasoning
        except Exception as e:  # show the problem, keep going
            got, why = "ERROR", f"{type(e).__name__}: {e}"
        expected = next(l for l in labels() if l["package_id"] == pid)["expected"]["decision"]
        print(f"{'yes' if got == expected else 'no '} {pid}: {got} (expected {expected})\n{indented(why)}\n")


async def aspot_check(system, *package_ids: str) -> None:
    """Async version of `spot_check` for systems defined with `async def` (e.g. agents using MCP tools)."""
    for pid in package_ids:
        try:
            d = _to_decision(await system(load_package(pid)))
            got, why = d.decision, d.reasoning
        except Exception as e:  # show the problem, keep going
            got, why = "ERROR", f"{type(e).__name__}: {e}"
        expected = next(l for l in labels() if l["package_id"] == pid)["expected"]["decision"]
        print(f"{'yes' if got == expected else 'no '} {pid}: {got} (expected {expected})\n{indented(why)}\n")


def compare(a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, str] = ("A", "B")) -> pd.DataFrame:
    """Two scoreboard runs side by side, one row per claim: which claims one design decided correctly and the other did not.

    a, b: the tables returned by scoreboard() or ascoreboard() on the same claims.
    """
    cols = ["package", "category", "expected", "predicted", "correct", "unsafe_approval", "llm_calls"]
    m = a[cols].merge(b[cols], on=["package", "category", "expected"], suffixes=(f" {names[0]}", f" {names[1]}"))
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
    cols = ["label", "n", "accuracy", "payout_accuracy", "unsafe_approvals", "to_person", "issue_recall", "llm_calls_per_pkg", "cost_usd_per_pkg",
            "cost_usd_total", "seconds_per_pkg", "wall_seconds", "replayed", "errors"]
    return df[[c for c in cols if c in df]] if len(df) else df
