"""claims_lab: helper package for the HSLU module 'AI Agents and Controlled Workflows'.

All data is synthetic and describes the fictional Lakeside Insurance AG.

    import claims_lab as cl
    cl.setup()
    pkg = cl.load_package("PKG-001"); print(pkg.text())
"""
from .cache import replay
from .client import get_client
from .costs import CostMeter, cost_report
from .course import figure, reference_runs, result_card, save_picks, show_picks, where_we_are
from .data import COMPANY, Package, cheat_sheet, label, labels, list_packages, load_package, query, reset_workdir, show_claims, terms
from .env import setup
from .costs import BudgetExceeded
from .models import get_embeddings, get_model, provider, run_config
from .schemas import ClaimDecision, ClaimFacts, DocInfo
from .scoreboard import ascoreboard, aspot_check, compare, scoreboard, scoreboard_history, spot_check
from .trace import context_breakdown, show_graph, show_trace

__version__ = "0.1.0"
__all__ = ["setup", "get_model", "get_client", "run_config", "BudgetExceeded", "get_embeddings", "provider", "load_package", "list_packages", "cheat_sheet", "labels", "label", "show_claims",
           "terms", "query", "reset_workdir", "Package", "ClaimDecision", "ClaimFacts", "DocInfo", "scoreboard",
           "ascoreboard", "scoreboard_history", "compare", "show_graph", "spot_check", "aspot_check", "show_trace", "context_breakdown", "replay", "CostMeter", "cost_report", "COMPANY",
           "where_we_are", "result_card", "save_picks", "show_picks", "figure", "reference_runs"]
