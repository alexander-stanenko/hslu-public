"""Offline stand-in for the course model: no API key, no network, deterministic, free.

Switch it on with `CLAIMS_LAB_PROVIDER=offline` (or `cl.setup(provider="offline")`). It is used to execute
the notebooks end to end in tests and to develop without spending money. It is NOT a language model:

- With tools bound it calls the course tools it recognises (policy lookup, invoice check, ...) once each,
  with arguments read from the claim text by regular expressions, then gives a short final answer.
- For structured output (`with_structured_output`, or an output tool like `ClaimDecision`) it fills the
  schema from the same regular expressions and simple defaults.

Its answers are often wrong, and that is fine: it checks that the code runs, not that the design is good.
"""
from __future__ import annotations

import itertools
import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

_ids = itertools.count(1)

CLAIM_TYPE_WORDS = [  # first match wins
    ("theft_outside", ["stolen at", "theft outside", "bike stolen"]), ("burglary", ["burglary", "break-in", "broke in"]),
    ("theft", ["stolen", "theft"]), ("vandalism", ["vandal", "scratched"]), ("animal_collision", ["deer", "animal", "wildlife"]),
    ("hail", ["hail"]), ("storm", ["storm", "gust"]), ("water_damage", ["water", "pipe", "leak", "damp"]),
    ("fire", ["fire", "smoke"]), ("glass", ["windscreen", "windshield", "glass"]), ("collision", ["collision", "pillar", "bumper"]),
]

DOC_TYPE_WORDS = [  # first match wins, on the file name
    ("email", ["email"]), ("police_report", ["polizei", "police"]), ("purchase_receipt", ["quittung", "receipt"]),
    ("estimate", ["offerte", "estimate", "kv-"]), ("invoice", ["rechnung", "invoice", "garage", "scan_002", "scan_003"]),
    ("claim_form", ["schadenmeldung", "claim_form", "scan_001"]), ("photo_note", ["img", "photo"]),
]


def _text(messages: list[BaseMessage]) -> str:
    return "\n".join(m.content if isinstance(m.content, str) else json.dumps(m.content, default=str) for m in messages)


def _facts(text: str) -> dict:
    """What the regular expressions can read from a claim package (or a conversation about one)."""
    f: dict[str, Any] = {}
    if m := re.search(r"\b(PKG-\d{3})\b", text):
        f["package_id"] = m.group(1)
    if m := re.search(r"\b(P-\d{5})\b", text):
        f["policy_id"] = m.group(1)
    if m := re.search(r"\b(?!PKG-|CHE-)([A-Z]{2,4}-\d{3,6})\b", text):  # invoice numbers look like GH-4471
        f["invoice_no"] = m.group(1)
    amounts = [float(a.replace("'", "").replace(",", "")) for a in re.findall(r"CHF\s*(\d[\d',]*(?:\.\d{2})?)", text)]
    amounts += [float(a.replace("'", "")) for a in re.findall(r"TOTAL\D{0,12}([\d']+\.\d{2})", text, re.I)]
    if amounts:
        f["amount_chf"] = max(amounts)
    if m := re.search(r"(?:incident|loss|damage)[^\n:]{0,20}:\s*(\d{1,2})\.(\d{1,2})\.(\d{4})", text, re.I):
        f["incident_date"] = f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    elif m := re.search(r"(?:incident|loss|damage)[^\n:]{0,20}:\s*(\d{4}-\d{2}-\d{2})", text, re.I):
        f["incident_date"] = m.group(1)
    if m := re.search(r"\b(\d{4})\s+[A-ZÄÖÜ][a-zäöüé]+", text):
        f["postcode"] = m.group(1)
    if m := re.search(r"(?:Policyholder|Name)\s*:\s*([A-ZÄÖÜ][^\n,]+)", text):
        f["name"] = m.group(1).strip()
    elif m := re.search(r"From:\s*([^<\n]+?)\s*<", text):
        f["name"] = m.group(1).strip()
    low = text.lower()
    f["claim_type"] = next((t for t, words in CLAIM_TYPE_WORDS if any(w in low for w in words)), "other")
    docs = []  # document types, from the '=== filename ===' headers of a package
    for name in re.findall(r"^=== (.+?) ===$", text, re.M):
        n = name.lower()
        docs.append(next((t for t, words in DOC_TYPE_WORDS if any(w in n for w in words)), "other"))
    if docs:
        f["documents_present"] = sorted(set(docs))
    return f


def _decision(text: str, messages: list[BaseMessage] = ()) -> dict:
    """A naive clerk: pays unless a tool result showed an earlier claim with the same invoice."""
    if any(isinstance(m, ToolMessage) and (m.name or "").startswith("ask_") and "already used" in m.text for m in messages):
        return dict(decision="refer_adjuster", payout_chf=None, missing_items=[], issues=["duplicate_invoice"],
                    reasoning="[offline model] A specialist found that the invoice number was already used in an earlier claim.")
    f = _facts(text)
    if m := re.search(r"Decision: (fast_track|request_info|refer_adjuster|decline)\b([^\n]*)", text):  # its own earlier answer
        payout = re.search(r"payout CHF ([\d,]+\.\d{2})", m.group(2))
        return dict(decision=m.group(1), payout_chf=float(payout.group(1).replace(",", "")) if payout else None, missing_items=[],
                    issues=["duplicate_invoice"] if "already used" in m.group(2) else [],
                    reasoning="[offline model] Copied from its own answer: " + m.group(0))
    if any(isinstance(m, ToolMessage) and m.name == "search_claims_by_invoice" and m.text.strip() not in ("", "[]")
           for m in messages):
        return dict(decision="refer_adjuster", payout_chf=None, missing_items=[], issues=["duplicate_invoice"],
                    reasoning="[offline model] The invoice number was already used in an earlier claim.")
    amount = f.get("amount_chf")
    return dict(decision="fast_track", payout_chf=max(0.0, amount - 200) if amount else None, missing_items=[], issues=[],
                reasoning="[offline model] Looks like a routine claim. This is a scripted answer, not a real model.")


def _fill(schema: dict, text: str, defs: dict | None = None, messages: list[BaseMessage] = ()) -> Any:
    """Build arguments that validate against a JSON schema, preferring facts read from the text."""
    defs = defs or schema.get("$defs", {})
    if "$ref" in schema:
        return _fill(defs[schema["$ref"].split("/")[-1]], text, defs, messages)
    if schema.get("title") == "PolicyDecisions" or set(schema.get("properties", {})) == {"decisions"}:  # one decision per policy number in the text (Block 5, Part E)
        ids = list(dict.fromkeys(re.findall(r"\b(P-\d{5})\b", text)))
        return {"decisions": [dict(_decision(text, messages), policy_id=pid) for pid in ids]}
    if set(schema.get("properties", {})) == {"parts"}:  # the parts of a package, one per policy (Block 5, Part E)
        docs = [n for n in re.findall(r"^=== (.+?) ===", text, re.M)]
        return {"parts": [{"policy_id": pid, "what": "offline", "documents": docs}
                          for pid in dict.fromkeys(re.findall(r"\b(P-\d{5})\b", text))]}
    if schema.get("title") == "ClaimDecision" or set(schema.get("properties", {})) >= {"decision", "reasoning"}:
        d = _decision(text, messages)
        return dict(d, policy_id=_facts(text).get("policy_id", "unknown")) if "policy_id" in schema.get("properties", {}) else d
    if "anyOf" in schema:
        options = [s for s in schema["anyOf"] if s.get("type") != "null"]
        return _fill(options[0], text, defs) if options else None
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object" or "properties" in schema:
        facts, out = _facts(text), {}
        for name, prop in schema.get("properties", {}).items():
            enum = prop.get("enum") or next((s.get("enum") for s in prop.get("anyOf", []) if s.get("enum")), None)
            if name in facts and (not enum or facts[name] in enum):
                out[name] = facts[name]
            elif name in schema.get("required", []):  # optional fields stay empty unless the text gives them
                out[name] = _fill(prop, text, defs)
        return out
    return {"string": "offline", "number": 0.0, "integer": 0, "boolean": False, "array": []}.get(t, None)


def _course_tool_args(name: str, f: dict) -> dict | None:
    """Arguments for the course tools the offline model knows, or None if it can't call that tool."""
    pid, inv = f.get("policy_id"), f.get("invoice_no")
    return {
        "read_package": {"package_id": f["package_id"]} if "package_id" in f else None,
        "open_claim": {"package_id": f["package_id"], **({"policy_id": pid} if pid else {})} if "package_id" in f else None,
        "lookup_policy": {"policy_id": pid} if pid else None,
        "find_policies": {"name": f["name"]} if not pid and f.get("name") else None,
        "get_claim_history": {"policy_id": pid} if pid else None,
        "search_claims_by_invoice": {"invoice_no": inv} if inv else None,
        "search_policy_terms": {"query": f"{f['claim_type']} cover"},
        "check_weather": ({"postcode": f["postcode"], "date": f["incident_date"]}
                          if f["claim_type"] in ("hail", "storm") and f.get("postcode") and f.get("incident_date") else None),
    }.get(name)


def _question(spec: dict, f: dict) -> dict | None:
    """Arguments for a specialist wrapped as a tool (ask_..., one text argument, Block 5): the claim's facts written
    out, so that the specialist, which does not see the package, can find them again."""
    props, required = spec["parameters"].get("properties", {}), spec["parameters"].get("required", [])
    if not spec["name"].startswith("ask_") or len(required) != 1 or props.get(required[0], {}).get("type") != "string":
        return None
    facts = [f"Claim {f.get('package_id', '')}", f"policy {f['policy_id']}" if f.get("policy_id") else
             f"customer {f.get('name', 'unknown')}", f"claim type {f['claim_type']}",
             f"incident date: {f['incident_date']}" if f.get("incident_date") else "",
             f"amount CHF {f['amount_chf']:.2f}" if f.get("amount_chf") else "",
             f"invoice {f['invoice_no']}" if f.get("invoice_no") else "",
             f"postcode {f['postcode']} Area" if f.get("postcode") else ""]
    return {required[0]: "; ".join(x for x in facts if x) + ". What do you find?"}


class OfflineChatModel(BaseChatModel):
    """Scripted, free stand-in for the course model (see the module docstring)."""
    model_name: str = "offline"
    max_tool_calls: int = 4

    @property
    def _llm_type(self) -> str:
        return "claims-lab-offline"

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], tool_choice=tool_choice, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, tools=None, tool_choice=None, **kwargs) -> ChatResult:
        text = _text(messages)
        specs = [t["function"] for t in (tools or [])]
        by_name = {s["name"]: s for s in specs}
        msg = self._respond(messages, text, specs, by_name, tool_choice)
        n_in = (len(text) + len(json.dumps(specs))) // 4
        n_out = max(1, len(msg.content) // 4 + 20 * len(msg.tool_calls))
        msg.usage_metadata = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
        msg.response_metadata = {"model_name": self.model_name}
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _respond(self, messages, text, specs, by_name, tool_choice) -> AIMessage:
        def call(name: str, args: dict) -> AIMessage:
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"offline_{next(_ids)}", "type": "tool_call"}])

        # forced structured output (with_structured_output / tool_choice): fill the one schema
        forced = tool_choice if isinstance(tool_choice, str) and tool_choice in by_name else None
        if forced or (tool_choice in ("any", "required") and len(specs) == 1):
            spec = by_name[forced] if forced else specs[0]
            return call(spec["name"], _fill(spec["parameters"], text, messages=messages))
        if not specs:
            d = _decision(text, messages)
            return AIMessage(content=f"Decision: {d['decision']}. {d['reasoning']}")

        done = [tc["name"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
        if len(done) < self.max_tool_calls:
            facts = _facts(text)
            for spec in specs:
                if spec["name"] not in done and ((args := _course_tool_args(spec["name"], facts)) is not None
                                                 or (args := _question(spec, facts)) is not None):
                    return call(spec["name"], args)
        # an output tool (create_agent's structured response) is named after its schema, e.g. ClaimDecision
        output_tool = next((s for s in specs if s["name"][:1].isupper()), None)
        if output_tool:
            return call(output_tool["name"], _fill(output_tool["parameters"], text, messages=messages))
        d = _decision(text, messages)
        return AIMessage(content=f"Decision: {d['decision']}"
                                 + (f", payout CHF {d['payout_chf']:,.2f}" if d["payout_chf"] else "") + f". {d['reasoning']}")
