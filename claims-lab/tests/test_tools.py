from claims_lab import data
from claims_lab import tools as T


def test_lookup_policy():
    p = T.lookup_policy("P-1001")
    assert p["holder"] == "Anna Brunner" and p["plate"] == "LU 248 113" and p["car"] == "VW Golf (2021)"
    assert p["covers"] == ["glass", "hail", "theft", "collision"] and p["deductible_chf"] == 500
    assert (p["valid_from"], p["valid_to"]) == ("2025-01-01", "2027-12-31")
    assert T.lookup_policy(" p-1003 ")["covers"] == ["collision"]   # the number is cleaned up
    assert "error" in T.lookup_policy("P-9999")


def test_find_policies_ambiguous_and_unique():
    assert len(T.find_policies(name="Thomas Meier")) == 2                # PKG-004: the name alone is not enough
    assert [p["policy_id"] for p in T.find_policies(name="Thomas Meier", plate="LU 310 877")] == ["P-1004"]
    assert [p["policy_id"] for p in T.find_policies(name="Claudia Roth", plate="ZG 61 340")] == ["P-1016"]
    assert len(T.find_policies(plate="lu248113")) == 1                   # spaces and case do not matter
    assert T.find_policies(name="Nobody Here") == []


def test_red_flag_tools():
    assert [r["policy_id"] for r in T.search_claims_by_invoice("GH-3318")] == ["P-1002"]   # PKG-002's invoice, paid before
    assert [r["policy_id"] for r in T.search_claims_by_invoice("AZ-41207")] == ["P-1001"]  # PKG-101's, on another policy
    assert T.search_claims_by_invoice("GH-4471") == []
    history = T.get_claim_history("P-1001")
    assert len(history) == 1 and history[0]["invoice_no"] == "AZ-41207" and history[0]["damage"] == "glass"
    assert T.get_claim_history("P-1003") == []


def test_calc_payout():
    assert T.calc_payout(1150, 500) == 650
    assert T.calc_payout(100, 200) == 0


def test_search_terms_finds_the_right_section():
    assert "§2" in [h["source"] for h in T.search_terms("deductible and payout")]
    assert "§3" in [h["source"] for h in T.search_terms("tyres worn below the legal minimum")]
    assert "§4" in [h["source"] for h in T.search_terms("which documents does a theft need")]
    assert all(set(h) == {"source", "title", "text"} for h in T.search_terms("glass"))


def test_write_tools_use_the_workdir():
    msg = T.request_documents("PKG-005", ["police_report"], "Please send the police report.")
    assert "police_report" in msg and "laura.frei@mail.example" in msg
    assert len(data.query("SELECT * FROM outbox")) == 1
    T.record_decision("PKG-001", "fast_track", 650.0)
    assert data.query("SELECT decision FROM decisions")[0]["decision"] == "fast_track"
    data.reset_workdir()
    assert data.query("SELECT * FROM outbox") == []


def test_langchain_tool_schemas():
    from claims_lab.reference import block2
    names = {t.name for t in block2.FINAL_TOOLS}
    assert names == {"lookup_policy", "find_policies", "get_claim_history", "search_claims_by_invoice", "search_terms",
                     "calc_payout"}
    for t in block2.FINAL_TOOLS + block2.BAD_TOOLS:
        assert t.args is not None
    assert len(block2.BAD_TOOLS) == 5


def test_every_tool_argument_has_a_type():
    """OpenAI / Azure reject a tool whose argument schema has no 'type' (Anthropic accepts it)."""
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from claims_lab.reference import block1, block2
    for t in block2.BAD_TOOLS + block1.AGENT_TOOLS + block2.FINAL_TOOLS:   # block1's are plain functions
        spec = convert_to_openai_tool(t)["function"]
        for name, prop in spec["parameters"].get("properties", {}).items():
            assert "type" in prop or "anyOf" in prop, f"{spec['name']}.{name} has no type"


def test_mcp_servers_offer_the_block2_tools_word_for_word():
    """Over MCP the agent sees exactly the Block 2 tool descriptions: only the wiring changes (notebook 03)."""
    import asyncio
    from claims_lab.mcp_servers import claims_systems, policy_docs
    from claims_lab.reference import block2
    ours = {t.name: " ".join(t.description.split()) for t in block2.FINAL_TOOLS}
    seen = {}
    for server in (claims_systems.mcp, policy_docs.mcp):
        for tool in asyncio.run(server.list_tools()):
            seen[tool.name] = " ".join(tool.description.split())
    assert seen == ours
