from claims_lab import data
from claims_lab import tools as T


def test_lookup_policy():
    p = T.lookup_policy("P-10001")
    assert p["holder"] == "Anna Brunner" and p["product"] == "motor"
    assert p["deductibles_chf"]["glass"] == 200 and p["modules"]["partial"] is True
    assert "error" in T.lookup_policy("P-99999")


def test_find_policies_ambiguous_and_unique():
    assert len(T.find_policies(name="Thomas Meier")) == 2
    assert len(T.find_policies(name="Jonas Widmer", postcode="8404")) == 1
    assert len(T.find_policies(plate="lu248113")) == 1


def test_red_flag_tools():
    assert len(T.search_claims_by_invoice("GH-3318")) == 1
    assert T.search_claims_by_invoice("GH-4471") == []
    assert len(T.get_claim_history("P-10026")) == 2
    assert T.check_weather("6010", "2026-09-14")[0]["event"] == "hail"
    assert T.check_weather("3110", "2026-09-21") == []
    assert T.get_price_range("windscreen_replacement")["typical_max_chf"] == 1400
    assert "known_repair_types" in T.get_price_range("spaceship")


def test_calc_payout():
    assert T.calc_payout(1150, 200) == 950
    assert T.calc_payout(100, 200) == 0
    assert T.calc_payout(90000, 200, 80000) == 79800


def test_search_policy_terms_finds_the_right_section():
    top = [h["source"] for h in T.search_policy_terms("deductible for windscreen glass breakage")]
    assert "motor §6" in top
    top = [h["source"] for h in T.search_policy_terms("bicycle stolen at the train station", product="household")]
    assert "household §3" in top
    top = [h["source"] for h in T.search_policy_terms("which documents are required for vandalism")]
    assert any(s in top for s in ("guideline §3", "motor §8"))


def test_write_tools_use_the_workdir():
    msg = T.request_documents("PKG-001", ["police_report"], "Please send the police report.")
    assert "police_report" in msg
    assert len(data.query("SELECT * FROM outbox")) == 1
    T.record_decision("PKG-001", "fast_track", 950.0)
    assert data.query("SELECT decision FROM decisions")[0]["decision"] == "fast_track"
    data.reset_workdir()
    assert data.query("SELECT * FROM outbox") == []


def test_langchain_tool_schemas():
    from claims_lab.reference import block2
    names = {t.name for t in block2.FINAL_TOOLS}
    assert {"lookup_policy", "find_policies", "check_weather", "search_policy_terms"} <= names
    for t in block2.FINAL_TOOLS + block2.BAD_TOOLS:
        assert t.args is not None
    assert len(block2.BAD_TOOLS) == 6


def test_every_tool_argument_has_a_type():
    """OpenAI / Azure reject a tool whose argument schema has no 'type' (Anthropic accepts it)."""
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from claims_lab.reference import block1, block2
    for t in block2.BAD_TOOLS + block1.AGENT_TOOLS + block2.FINAL_TOOLS:
        for name, prop in convert_to_openai_tool(t)["function"]["parameters"].get("properties", {}).items():
            assert "type" in prop or "anyOf" in prop, f"{t.name}.{name} has no type"


def test_mcp_servers_offer_the_block2_tools_word_for_word():
    """Over MCP the agent sees exactly the Block 2 tool descriptions: only the wiring changes (notebook 03)."""
    import asyncio
    from claims_lab.mcp_servers import claims_systems, policy_docs, reference_data
    from claims_lab.reference import block2
    ours = {t.name: " ".join(t.description.split()) for t in block2.FINAL_TOOLS}
    seen = {}
    for server in (claims_systems.mcp, policy_docs.mcp, reference_data.mcp):
        for tool in asyncio.run(server.list_tools()):
            seen[tool.name] = " ".join(tool.description.split())
    assert seen == ours
