# claims-lab

Helper package and synthetic data for the HSLU continuing-education module **"AI Agents and Controlled Workflows"** (Module 3 of the CAS LLMs and AI Agents).

> All data describes the **fictional** insurer *Lakeside Insurance AG*. People, companies, policies, invoices and
> police reports are invented for teaching purposes. Any resemblance to real persons or businesses is coincidental.

## In a notebook (Colab)
```python
!pip install -q "claims-lab @ git+https://github.com/alexander-stanenko/hslu-public.git#subdirectory=claims-lab"
import claims_lab as cl
cl.setup()                      # key from Colab Secrets, version check, model ping
pkg = cl.load_package("PKG-001")
print(pkg.text())
```

## What's inside
| | |
|---|---|
| `cl.load_package`, `cl.list_packages`, `cl.labels` | 28 dev + 12 held-out claim packages (+1 demo), ground truth |
| `cl.get_model()`, `cl.structured()` | the course model (Anthropic or Azure OpenAI via `CLAIMS_LAB_PROVIDER`; `offline` = scripted stand-in, no key, no cost) |
| `cl.cheat_sheet()` | the clerk's cheat sheet: 4 decisions, 5 questions |
| `claims_lab.reference.block1`, `.block2` | reference solutions per block (catch-up cells and solutions) |
| `cl.spot_check`, `cl.context_breakdown` | quick decision check on a few packages · what fills the context window |
| `claims_lab.tools` | reference tools (policy admin, claim history, prices, weather, terms search) + `BAD_TOOLS` |
| `claims_lab.mcp_servers` | FastMCP servers for the same systems (`python -m claims_lab.mcp_servers.claims_systems`) |
| `cl.scoreboard`, `cl.ascoreboard`, `cl.scoreboard_history` | accuracy / unsafe approvals / issues found / cost per design |
| `cl.show_trace` | step-by-step view of an agent run |
| `cl.replay()` | replay cache for demo cells |
| `cl.CostMeter`, `cl.run_config()`, `cl.cost_report()` | spending guards: per-run call limit, per-request size limit, session budget |

## Development
```bash
python generator/build.py        # regenerate data (deterministic, checks labels against the rule oracle)
pip install -e ".[dev]" && pytest
```
