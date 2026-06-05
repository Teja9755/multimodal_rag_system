import re
import os
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from src.core.db import get_vector_store

load_dotenv()

_PG_CONN = os.getenv("PG_CONNECTION_STRING", "").replace(
    "postgresql+psycopg://", "postgresql://"
)

# ---------------- MODE DETECTION ----------------

_KEYWORD_PATTERNS = [
    r"[A-Z]{2,}-\d{4}-\w+",
    r"\b[A-Z]{2,5}\b",
    r"\d{6,}",
]

_KEYWORD_RE = re.compile("|".join(_KEYWORD_PATTERNS))


def _detect_mode(query: str) -> str:
    q = query.strip()

    if _KEYWORD_RE.search(q):
        return "keyword"

    if len(q.split()) <= 3:
        return "hybrid"

    return "vector"


# ---------------- MAIN API ----------------

def query_documents(query: str, k: int = 5) -> list[dict]:
    mode = _detect_mode(query)

    if mode == "keyword":
        return fts_search(query, k)

    if mode == "hybrid":
        return hybrid_search(query, k)

    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)

    return [
        {
            "content": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ]


# ---------------- FULL TEXT SEARCH ----------------

def fts_search(query: str, k: int = 5, collection: str = None) -> list[dict]:
    sql = """
        SELECT
            e.document AS content,
            e.cmetadata AS metadata,
            ts_rank(
                to_tsvector('english', e.document),
                plainto_tsquery('english', %(query)s)
            ) AS rank
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c
            ON c.uuid = e.collection_id
        WHERE to_tsvector('english', e.document)
              @@ plainto_tsquery('english', %(query)s)
        ORDER BY rank DESC
        LIMIT %(k)s;
    """

    with psycopg.connect(_PG_CONN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"query": query, "k": k})
            rows = cur.fetchall()

    return [
        {
            "content": r["content"],
            "metadata": r["metadata"],
            "score": float(r["rank"]),
        }
        for r in rows
    ]


# ---------------- HYBRID (RRF) ----------------

def hybrid_search(query: str, k: int = 5) -> list[dict]:
    vector_store = get_vector_store()

    vector_docs = vector_store.similarity_search(query, k=k)
    fts_docs = fts_search(query, k=k)

    scores: dict[str, float] = {}
    store: dict[str, dict] = {}

    # ---------------- VECTOR RESULTS ----------------
    for rank, doc in enumerate(vector_docs):
        key = doc.page_content[:120]

        scores[key] = scores.get(key, 0) + 1 / (60 + rank + 1)

        store[key] = {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }

    # ---------------- FTS RESULTS ----------------
    for rank, item in enumerate(fts_docs):
        key = item["content"][:120]

        scores[key] = scores.get(key, 0) + 1 / (60 + rank + 1)

        store[key] = {
            "content": item["content"],
            "metadata": item["metadata"],
        }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [store[k] for k, _ in ranked[:k]]


# ---------------- DEBUG ----------------

if __name__ == "__main__":
    q = "what is the leave policy for employees?"
    results = query_documents(q, k=5)

    print(f"\nTop {len(results)} results:\n{'='*60}")

    for i, r in enumerate(results, 1):
        meta = r["metadata"]

        print(f"\n[{i}] file: {meta.get('source_file')} | page: {meta.get('page_number')}")
        print(r["content"][:400])