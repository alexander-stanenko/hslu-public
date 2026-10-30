"""Search over the policy terms and the internal claims guideline (RAG as a tool).

Chunks are the numbered sections (§) of each document. With provider embeddings configured
(Azure) we use them; otherwise a local TF-IDF stand-in is used, so it also works offline.
"""
from __future__ import annotations

import re

import numpy as np

from .data import DATA_DIR

DOCS = {"motor": "motor", "household": "household", "building": "building", "claims_guideline": "guideline"}
_index = None


def chunks() -> list[dict]:
    out = []
    for fname, short in DOCS.items():
        text = (DATA_DIR / "terms" / f"{fname}.md").read_text()
        for sec in re.split(r"\n(?=## )", text):
            m = re.match(r"## (§\d+) (.*)", sec)
            if m:
                out.append(dict(source=f"{short} {m.group(1)}", product=short, title=m.group(2).strip(),
                                text=sec.split("\n", 1)[1].strip() if "\n" in sec else ""))
    return out


def _embedding_call(texts: list[str]) -> None:
    """Embedding calls go through the shared spending ledger too (estimated tokens, tiny cost)."""
    from .costs import EMBEDDING_PRICE, ledger_add, ledger_check
    ledger_check()
    tokens = sum(len(t) for t in texts) // 4
    ledger_add("embedding", "text-embedding", tokens, 0, tokens * EMBEDDING_PRICE / 1e6)


def _cached_vectors(docs: list[str], emb) -> np.ndarray:
    """Embed the sections once per working directory. MCP servers run as short-lived processes (one per tool call
    with the LangChain adapters), so without this cache every search would embed all sections again."""
    import hashlib

    from .data import workdir
    key = hashlib.sha256(("\n\0".join(docs) + str(getattr(emb, "deployment", "")) + str(getattr(emb, "model", ""))).encode()).hexdigest()[:16]
    path = workdir() / f"terms_vectors_{key}.npy"
    if path.exists():
        return np.load(path)
    _embedding_call(docs)
    vecs = np.array(emb.embed_documents(docs))
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, vecs)
    tmp.replace(path)  # atomic, in case several server processes build it at once
    return vecs


def _build():
    global _index
    from .models import get_embeddings
    cs = chunks()
    docs = [f"{c['title']}\n{c['text']}" for c in cs]
    emb = get_embeddings()
    if emb is not None:
        vecs = _cached_vectors(docs, emb)
        _index = ("provider", cs, vecs / np.linalg.norm(vecs, axis=1, keepdims=True), emb)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        m = vec.fit_transform(docs)
        _index = ("tfidf", cs, m, vec)
    return _index


def search_policy_terms(query: str, product: str | None = None, k: int = 3) -> list[dict]:
    """Return the k most relevant sections of the terms / guideline for a question.

    Args:
        query: natural-language question, e.g. "deductible for windscreen replacement".
        product: optional filter: 'motor', 'household', 'building' or 'guideline'.
        k: number of sections to return.
    """
    kind, cs, mat, model = _index or _build()
    if kind == "provider":
        _embedding_call([query])
        q = np.array(model.embed_query(query))
        scores = mat @ (q / np.linalg.norm(q))
    else:
        scores = (mat @ model.transform([query]).T).toarray().ravel()
    order = np.argsort(-scores)
    hits = [dict(cs[i], score=round(float(scores[i]), 3)) for i in order if product in (None, cs[i]["product"])]
    return hits[:k]
