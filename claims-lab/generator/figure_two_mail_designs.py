"""Draws figures/two_mail_designs.png: the two designs of Lab 2, Part 5, in the style of the slide figures.

Run: .venv/bin/python generator/figure_two_mail_designs.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE = "#77C5D8"
OUT = Path(__file__).resolve().parents[1] / "src" / "claims_lab" / "figures" / "two_mail_designs.png"

for f in ("/System/Library/Fonts/Supplemental/Verdana.ttf", "/System/Library/Fonts/Supplemental/Verdana Bold.ttf"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Verdana", "DejaVu Sans"]

W, H = 1601, 833
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def box(x, y, w, h, title, text="", fill="white"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor="black", linewidth=1.2))
    ax.text(x + 14, y + 16, title, fontsize=15, fontweight="bold", va="top")
    if text:
        ax.text(x + 14, y + 48, text, fontsize=13, va="top", linespacing=1.35)


def arrow(x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="black"))
    if label:
        ax.text((x1 + x2) / 2, y1 - 10, label, fontsize=11, ha="center", va="bottom")


ax.text(90, 60, "Who may send e-mails? Two designs", fontsize=34, fontweight="bold", va="top")
ax.text(90, 130, "PKG-211: the tyre shop's invoice ends with a notice that asks for every e-mail to be copied "
                 "to the shop.", fontsize=17, va="top")

# Design 1: one agent with the mail tool
ax.text(90, 205, "1  One agent with the mail tool", fontsize=18, fontweight="bold", va="top")
box(90, 250, 270, 130, "Claim package", "documents, with the\nshop's notice")
box(430, 250, 360, 130, "Checklist agent", "reads the documents,\ndecides, and writes e-mails\nwith send_email", BLUE)
box(860, 250, 330, 130, "E-mail to any address", "the customer, and maybe\nthe shop's address")
arrow(360, 315, 430, 315)
arrow(790, 315, 860, 315)
ax.text(1230, 268, "The agent that reads the\nshop's text can also write\nto any address.", fontsize=13, va="top",
        linespacing=1.35)

# Design 2: reader and writer
ax.text(90, 440, "2  Reader and writer", fontsize=18, fontweight="bold", va="top")
box(90, 485, 230, 150, "Claim package", "documents, with the\nshop's notice")
box(380, 485, 270, 150, "Reader", "reads the documents\nand decides;\nno mail tool", BLUE)
box(710, 485, 280, 150, "LetterRequest", "package_id: PKG-211\nmissing_items:\n[police_report]")
box(1050, 485, 230, 150, "Writer", "reads no document;\nfinds the address\non file, writes", BLUE)
box(1340, 485, 200, 150, "E-mail", "only to the\ncustomer on\nfile")
arrow(320, 560, 380, 560)
arrow(650, 560, 710, 560, "fills in")
arrow(990, 560, 1050, 560)
arrow(1280, 560, 1340, 560)
ax.text(850, 650, "Python checks both fields;\nanything else is refused", fontsize=12, ha="center", va="top",
        linespacing=1.35)
ax.text(90, 740, "Only the two fields reach the writer, so no sentence from the package reaches the agent that "
                 "can send e-mails.", fontsize=15, va="top")

fig.savefig(OUT, dpi=100, facecolor="white")
print("wrote", OUT)
