"""Access to the Lakeside Insurance AG data: claim packages (inbox), labels, database, policy terms."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

DATA_DIR = Path(str(files("claims_lab") / "data"))
COMPANY = "Lakeside Insurance AG"


@dataclass
class Package:
    """One incoming claim package: an email plus attachments (all plain text)."""
    package_id: str
    received_at: str
    channel: str
    sender: str
    documents: dict[str, str] = field(default_factory=dict)

    def text(self, max_chars_per_doc: int | None = None) -> str:
        """All documents as one string, each under a '=== filename ===' header."""
        parts = []
        for name, body in self.documents.items():
            body = body if max_chars_per_doc is None else body[:max_chars_per_doc]
            parts.append(f"=== {name} ===\n{body.strip()}")
        return f"Package {self.package_id} (received {self.received_at} via {self.channel})\n\n" + "\n\n".join(parts)

    def __repr__(self) -> str:
        return f"Package({self.package_id}, {len(self.documents)} documents: {', '.join(self.documents)})"


def list_packages(split: str = "dev") -> list[str]:
    """Package IDs of a split: 'quick' (PKG-001 to PKG-008), 'dev' (PKG-001 to PKG-016, for the exercises), 'heldout'
    (PKG-101 to PKG-108, the final check in notebook 05) or 'all' (dev and held-out)."""
    if split == "quick":
        return [l["package_id"] for l in labels() if l.get("quick")]
    return [l["package_id"] for l in labels() if split == "all" or l["split"] == split]


def load_package(package_id: str) -> Package:
    """Load a claim package from the inbox."""
    pdir = DATA_DIR / "inbox" / package_id
    if not pdir.exists():
        raise KeyError(f"Unknown package {package_id!r}. Try list_packages('dev').")
    meta = json.loads((pdir / "meta.json").read_text())
    docs = {"email.txt": (pdir / "email.txt").read_text()}
    for name in meta["attachments"]:
        docs[name] = (pdir / name).read_text()
    return Package(meta["package_id"], meta["received_at"], meta["channel"], meta["sender"], docs)


@lru_cache(maxsize=1)
def _labels() -> tuple:
    with open(DATA_DIR / "labels" / "labels.jsonl") as f:
        return tuple(json.loads(line) for line in f)


def labels(split: str | None = None) -> list[dict]:
    """Ground-truth labels (expected decision, payout, missing items, issues) per package."""
    return [dict(l) for l in _labels() if split in (None, "all") or l["split"] == split]


def label(package_id: str) -> dict:
    return next(l for l in labels() if l["package_id"] == package_id)


def show_claims(split: str = "quick") -> None:
    """Print the claims of a split, one line each: the claim, its nickname and the expected decision."""
    for l in labels():
        if (split == "quick" and l.get("quick")) or l["split"] == split or split == "all":
            e = l["expected"]
            out = e["decision"] + (f" CHF {e['payout_chf']:,.0f}" if e.get("payout_chf") else "") + \
                (f" ({', '.join(e['missing_items'])})" if e.get("missing_items") else "")
            print(f"{l['package_id']}  {l['nickname']:<40} expected: {out}")


def terms(name: str = "car_terms") -> str:
    """Full text of Lakeside's general terms for car insurance (§1 to §4)."""
    return (DATA_DIR / "terms" / f"{name}.md").read_text()


def cheat_sheet(show: bool = True) -> str | None:
    """The clerk's cheat sheet: the 4 possible decisions and the 4 questions a clerk asks, in order.
    show=True displays it (formatted in a notebook); show=False returns the text, e.g. for a system prompt."""
    text = (files("claims_lab") / "cheat_sheet.md").read_text(encoding="utf-8")
    if show:
        try:
            from IPython import get_ipython
            from IPython.display import Markdown, display
            if get_ipython() is not None:
                display(Markdown(text))
                return None
        except ImportError:
            pass
        print(text)
        return None
    return text


def workdir() -> Path:
    """Writable working directory (a copy of the database lives here, so the package data stays pristine)."""
    w = Path(os.environ.get("CLAIMS_LAB_WORKDIR", "claims_lab_work"))
    w.mkdir(exist_ok=True)
    return w


@lru_cache(maxsize=1)
def _db_version() -> str:
    import hashlib
    return hashlib.sha256((DATA_DIR / "lakeside.db").read_bytes()).hexdigest()[:16]


_DB_LOCK = threading.Lock()


def db_path() -> Path:
    """The working copy of the database. A copy made from an older version of the package data is replaced."""
    p, stamp = workdir() / "lakeside.db", workdir() / "lakeside.db.version"
    with _DB_LOCK:  # the scoreboard runs claims in threads; only one of them may make the copy
        if p.exists() and (not stamp.exists() or stamp.read_text() != _db_version()):
            p.unlink()  # made from other package data: its policies and claims may be out of date
        if not p.exists():
            tmp = p.with_suffix(".db.tmp")
            shutil.copy(DATA_DIR / "lakeside.db", tmp)
            con = sqlite3.connect(tmp)
            con.executescript("""
                CREATE TABLE IF NOT EXISTS decisions(package_id TEXT, decision TEXT, payout_chf REAL, note TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS outbox(package_id TEXT, recipient TEXT, subject TEXT, body TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP);
            """)
            con.commit()
            con.close()
            stamp.write_text(_db_version())
            os.replace(tmp, p)
    return p


def reset_workdir() -> None:
    """Throw away all changes (decisions, sent messages) and start from the original database."""
    p = workdir() / "lakeside.db"
    if p.exists():
        p.unlink()
    db_path()


def query(sql: str, params: tuple | list = ()) -> list[dict]:
    """Run a SQL query against the working copy of the Lakeside database and return rows as dicts."""
    con = sqlite3.connect(db_path())
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def execute(sql: str, params: tuple | list = ()) -> None:
    con = sqlite3.connect(db_path())
    try:
        con.execute(sql, params)
        con.commit()
    finally:
        con.close()
