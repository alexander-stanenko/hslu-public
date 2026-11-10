"""The thread helpers: where we are, the result card, the picks table, figures, and the new charts."""
import matplotlib

matplotlib.use("Agg")

import pytest

import claims_lab as cl
from claims_lab import course
from claims_lab.reference import block1, block4


@pytest.fixture(autouse=True)
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIMS_LAB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("CLAIMS_LAB_PROVIDER", "offline")


def test_where_we_are_names_the_block_and_what_is_open(capsys):
    cl.where_we_are(3)
    out = capsys.readouterr().out
    assert course.DAY_QUESTION in out and "other applications" in out and "inside one notebook" in out


def test_every_block_with_a_result_has_three_sentences():
    for b in range(5):
        assert len(course.BLOCKS[b]["tell"]) == 3


def test_result_card_shows_recorded_runs_and_mine(capsys):
    cl.scoreboard(block1.chatbot, ids=["PKG-001"], label="mine", show=False)
    cl.result_card(4)
    out = capsys.readouterr().out
    assert "hybrid pipeline" in out and "mine" in out and "for your team" in out


def test_picks_are_saved_merged_and_shown_with_evidence(capsys):
    with pytest.raises(ValueError):
        cl.save_picks({"Weird": ("agent", "x")})
    cl.save_picks({"Suspicious": ("agent", "it can look things up")})
    assert "Saved your picks for 1 kind of claim." in capsys.readouterr().out
    cl.save_picks({"Not covered": ("workflow", "a fixed rule")})
    df = course.picks_table(4).set_index("kind of claim")
    assert df.loc["Suspicious", "your pick"] == "agent" and df.loc["Not covered", "your pick"] == "workflow"
    assert "notebook 01:" in df.loc["Suspicious", "what the measurements say so far"]
    assert "notebook 04:" in df.loc["Suspicious", "what the measurements say so far"]
    assert "notebook 04:" not in course.picks_table(2).set_index("kind of claim").loc["Suspicious", "what the measurements say so far"]


def test_show_picks_without_saved_picks_explains_why(capsys):
    cl.show_picks(1)
    assert "fill in the picks cell above" in capsys.readouterr().out
    cl.show_picks(3)
    assert "Your picks are not in this session" in capsys.readouterr().out


def test_figures_are_shipped():
    from importlib.resources import files
    for name in course.FIGURES:
        assert (files("claims_lab") / "figures" / f"{name}.png").is_file()
    with pytest.raises(ValueError):
        cl.figure("nope")


def test_charts_draw_without_errors(monkeypatch):
    from claims_lab.style import plot_context, plot_tokens
    plot_tokens([(900, 40), (1400, 30), (2100, 60)])
    plot_context({"system prompt": 300, "tool definitions": 800, "claim package (user)": 1200, "tool results": 400,
                  "model's own steps": 100})
    from claims_lab.trace import _draw_locally
    _draw_locally(block4.build_pipeline(), block4.KINDS, {"extract_facts", "rule_checks"})


def test_empty_picks_are_not_saved_and_saved_picks_print_a_paste_line(capsys):
    cl.save_picks({"Suspicious": ("", ""), "Not covered": ("", "")})
    assert "No pick filled in yet" in capsys.readouterr().out
    assert course.picks_table(1)["your pick"].eq("").all()
    cl.save_picks({"Suspicious": ("agent", "it can look things up"), "Not covered": ("", "")})
    out = capsys.readouterr().out
    assert "Saved your picks for 1 kind of claim." in out
    assert 'cl.save_picks({"Suspicious": ["agent", "it can look things up"]})' in out   # to paste into the next notebook
    table = course.picks_table(2)
    assert table.set_index("kind of claim").loc["Suspicious", "your pick"] == "agent"
    assert table.set_index("kind of claim").loc["Not covered", "your pick"] == ""


def test_block_6_has_one_row_per_design_with_ranges():
    rows = [r for r in course.REFERENCE_RUNS if r["block"] == 6 and "held-out" not in r["label"]]
    assert [r["label"] for r in rows] == ["colleague's tools", "described tools", "checklist agent",
                                          "checklist agent over MCP", "hybrid pipeline", "claims team"]
    assert all(r["runs"] >= 3 and isinstance(r["correct"], str) for r in rows)
    assert len([r for r in course.REFERENCE_RUNS if "held-out" in r["label"]]) == 3


def test_result_card_marks_a_replayed_run_as_a_demo(capsys, monkeypatch):
    import sys
    sb = sys.modules["claims_lab.scoreboard"]  # the module; cl.scoreboard is the function
    monkeypatch.setattr(sb, "HISTORY", [{"label": "bad tools", "n": 8, "accuracy": 0.25, "unsafe_approvals": 4,
                                         "llm_calls_per_pkg": 2.6, "cost_usd_per_pkg": 0.0028, "replayed": True}])
    cl.result_card(2)
    assert "demo (replayed)" in capsys.readouterr().out


def test_a_pick_can_be_one_word_without_a_reason(capsys):
    """Notebook 01, Exercise D: a plain string is a pick without a reason; an empty string is no pick."""
    cl.save_picks({"Routine and complete": "workflow", "Incomplete": ""})
    assert "Saved your picks for 1 kind of claim." in capsys.readouterr().out
    df = course.picks_table(1).set_index("kind of claim")
    assert df.loc["Routine and complete", "your pick"] == "workflow" and df.loc["Incomplete", "your pick"] == ""
