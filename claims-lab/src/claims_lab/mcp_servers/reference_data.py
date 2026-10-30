"""MCP server 'lakeside-reference-data': price list, weather service and the payout calculation.

Run as a separate process (stdio):  python -m claims_lab.mcp_servers.reference_data
"""
from __future__ import annotations

import json

from fastmcp import FastMCP

from claims_lab import tools as T
from claims_lab.mcp_servers import READ_ONLY

mcp = FastMCP("lakeside-reference-data", instructions="Typical repair prices, recorded weather events and the payout calculation.")


@mcp.tool(annotations=READ_ONLY)
def get_price_range(repair_type: str) -> str:
    """Typical price range in CHF for a repair type, to spot inflated invoices. Repair types:
    windscreen_chip_repair, windscreen_replacement, hail_dent_repair, front_end_repair, bumper_repair, paint_repair,
    mirror_replacement, body_repair, water_damage_drying, pipe_repair, roof_repair, glass_household."""
    return json.dumps(T.get_price_range(repair_type), ensure_ascii=False)


@mcp.tool(annotations=READ_ONLY)
def check_weather(postcode: str, date: str) -> str:
    """Recorded hail, storm or heavy-rain events for a Swiss postcode (e.g. '6003') on a date (YYYY-MM-DD).
    Use it for every hail or storm claim. An empty list means no event was recorded there that day."""
    return json.dumps(T.check_weather(postcode, date), ensure_ascii=False)


@mcp.tool(annotations=READ_ONLY)
def calc_payout(amount_chf: float, deductible_chf: float, sum_insured_chf: float | None = None) -> str:
    """Payout in CHF for an approved claim: the claimed amount (capped at the sum insured) minus the deductible,
    never below 0. Always use it instead of calculating yourself."""
    return json.dumps(T.calc_payout(amount_chf, deductible_chf, sum_insured_chf), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(show_banner=False)
