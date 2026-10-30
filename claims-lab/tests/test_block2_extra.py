"""Notebook 02 helpers: the fallback for broken tool cells, the spot check with steps, the limit middleware, and
that the notebook continues with the reference when an exercise cell fails (all offline)."""
import sys
from pathlib import Path

import pytest

import claims_lab as cl
from claims_lab import tools
from claims_lab.reference import block2
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CLAIMS_LAB_PROVIDER", "offline")
    monkeypatch.setenv("CLAIMS_LAB_WORKDIR", str(tmp_path / "work"))
    monkeypatch.chdir(tmp_path)


def test_check_tools_keeps_good_tools_and_falls_back_otherwise(capsys):
    mine = [block2.lookup_policy, block2.get_claim_history]
    assert block2.check_tools(mine) == mine
    assert capsys.readouterr().out == ""
    assert block2.check_tools([tools.lookup_policy]) == block2.FIXED_TOOLS  # a plain function, @tool missing
    assert "is @tool missing" in capsys.readouterr().out
    assert block2.check_tools(mine + mine, block2.FINAL_TOOLS, "final_tools") == block2.FINAL_TOOLS
    assert "same name" in capsys.readouterr().out


def test_check_tools_catches_an_argument_renamed_only_in_the_signature(capsys):
    """Notebook 02 B2: renaming `x` or `id` in the def line but not in the return line breaks the tool at call time."""
    @tool
    def get_claim_history(policy_id: str) -> str:
        """All earlier claims on a policy ('P-10001')."""
        return str(tools.get_claim_history(str(x)))  # noqa: F821 - the participant's mistake

    @tool
    def lookup_policy(policy_id: str) -> str:
        """The policy record for a policy number like 'P-10001'."""
        return str(tools.lookup_policy(id))  # the built-in id: no NameError, but the wrong argument

    for broken in (get_claim_history, lookup_policy):
        assert block2.check_tools([broken, block2.search_policy_terms]) == block2.FIXED_TOOLS
        out = capsys.readouterr().out
        assert "calling a tool once failed" in out and "rename it in the tool's return line too" in out


def test_spot_check_steps_reports_calls_tools_and_cost(capsys):
    runs = block2.spot_check_steps(block2.final_agent(cl.get_model(effort="low")), "PKG-015")
    out = capsys.readouterr().out
    assert "PKG-015" in out and "model calls" in out and "output tokens per call" in out and "search_claims_by_invoice" in out
    assert runs["PKG-015"]["structured_response"].decision == "refer_adjuster"


def test_model_call_limit_ends_the_run_without_an_exception():
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
    agent = block2.final_agent(middleware=[ModelCallLimitMiddleware(run_limit=3, exit_behavior="end"),
                                           ToolCallLimitMiddleware(run_limit=10, exit_behavior="end")])
    result = block2.run(agent)(cl.load_package("PKG-025"))
    assert result.get("structured_response") is None
    assert "limit" in result["messages"][-1].text.lower()


def test_exercise_notebook_continues_with_the_reference_when_an_exercise_cell_fails(tmp_path):
    pytest.importorskip("nbclient")
    if not (ROOT / "notebooks" / "build.py").exists():
        pytest.skip("needs the notebook sources (course repo only)")
    from nbclient import NotebookClient
    sys.path.insert(0, str(ROOT / "notebooks"))
    import build
    nb = build.build(ROOT / "notebooks" / "src" / "02_tools_and_context.py", solution=False)
    nb.cells = [c for c in nb.cells if "install" not in c.metadata.get("tags", [])]
    b2 = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "my_tools = [lookup_policy, history" in c.source)
    c1 = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.startswith("MY_PROMPT ="))
    nb.cells[b2].source = "raise RuntimeError('a broken exercise cell')"
    nb.cells[c1].source = "raise RuntimeError('a broken exercise cell')"
    end = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "show_trace(result, chart=True)" in c.source)
    nb.cells = nb.cells[: end + 1]
    NotebookClient(nb, timeout=120, kernel_name="python3", allow_errors=True,
                   resources={"metadata": {"path": str(tmp_path)}}).execute()
    for i, cell in enumerate(nb.cells):
        errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        assert not errors or i in (b2, c1), f"cell {i} failed: {errors[0]['evalue']}\n{cell.source[:300]}"
    text = lambda i: "".join(o.get("text", "") for o in nb.cells[i].outputs)  # noqa: E731
    assert "Continuing with our reference tools" in text(b2 + 2)
    assert "continuing with our checklist, block2.PLAN_PROMPT" in text(c1 + 2)
    assert "continuing with our eight tools, block2.FINAL_TOOLS" in text(c1 + 2)


def test_check_tools_warns_when_a_tool_keeps_its_bad_description(capsys):
    kept = block2.check_tools(block2.BAD_TOOLS[:2] + list(block2.FIXED_TOOLS[2:]))
    assert len(kept) == len(block2.FIXED_TOOLS)  # the tools are used: the warning only says what to fix
    assert "one-word description" in capsys.readouterr().out
    block2.check_tools(block2.FIXED_TOOLS)
    assert "one-word description" not in capsys.readouterr().out
