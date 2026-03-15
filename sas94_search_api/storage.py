from __future__ import annotations

import json
import sqlite3
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient, models

from sas94_search_api.retrieval_models import RetrievedChunk, RetrievalConfig
from sas94_search_api.scoring import lexical_post_score
from sas94_search_api.text_utils import normalize_query_text, split_search_query, tokenize


_QDRANT_CLIENT_CACHE: OrderedDict[tuple[str | None, str, str | None, int | None], QdrantClient] = OrderedDict()
QDRANT_CLIENT_CACHE_SIZE = 4


def match_payload_filters(
    payload: dict[str, object],
    *,
    docsets: tuple[str, ...],
    section_kinds: tuple[str, ...],
) -> bool:
    if docsets and str(payload.get("docset")) not in set(docsets):
        return False
    if section_kinds and str(payload.get("section_kind")) not in set(section_kinds):
        return False
    return True


def build_qdrant_filter(config: RetrievalConfig) -> models.Filter | None:
    clauses: list[models.FieldCondition] = []
    if config.docsets:
        clauses.append(models.FieldCondition(key="docset", match=models.MatchAny(any=list(config.docsets))))
    if config.section_kinds:
        clauses.append(
            models.FieldCondition(
                key="section_kind",
                match=models.MatchAny(any=list(config.section_kinds)),
            )
        )
    if not clauses:
        return None
    return models.Filter(must=clauses)


def build_qdrant_client(config: RetrievalConfig) -> QdrantClient:
    cache_key = (config.qdrant_url, config.qdrant_path, config.qdrant_api_key, config.qdrant_timeout)
    cached = _QDRANT_CLIENT_CACHE.get(cache_key)
    if cached is not None:
        _QDRANT_CLIENT_CACHE.move_to_end(cache_key)
        return cached

    if config.qdrant_url:
        client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            timeout=config.qdrant_timeout,
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            client = QdrantClient(path=config.qdrant_path, timeout=config.qdrant_timeout)

    _QDRANT_CLIENT_CACHE[cache_key] = client
    if len(_QDRANT_CLIENT_CACHE) > QDRANT_CLIENT_CACHE_SIZE:
        _QDRANT_CLIENT_CACHE.popitem(last=False)
    return client


def retrieve_dense(query_text: str, config: RetrievalConfig) -> list[RetrievedChunk]:
    client = build_qdrant_client(config)
    client.set_model(config.embedding_model)
    response = client.query_points(
        collection_name=config.collection,
        query=models.Document(text=query_text, model=config.embedding_model),
        using=client.get_vector_field_name(),
        query_filter=build_qdrant_filter(config),
        limit=config.dense_limit,
        with_payload=True,
    )

    hits: list[RetrievedChunk] = []
    for rank, point in enumerate(response.points, start=1):
        payload = dict(point.payload or {})
        if not payload:
            continue
        hits.append(
            RetrievedChunk(
                score=float(point.score),
                payload=payload,
                source="dense",
                dense_rank=rank,
                stage_scores={"dense": float(point.score)},
            )
        )
    return hits


def corpus_rows(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def score_corpus_row(query_tokens: list[str], row: dict[str, object]) -> float:
    haystack = " ".join(
        [
            str(row.get("title", "")),
            str(row.get("section_path_text", "")),
            str(row.get("retrieval_text", "")),
        ]
    ).lower()
    if not haystack:
        return 0.0

    score = 0.0
    for token in query_tokens:
        count = haystack.count(token)
        if count:
            score += 1.0 + min(count, 5) * 0.2
            if token in str(row.get("section_path_text", "")).lower():
                score += 0.15
            if token in str(row.get("title", "")).lower():
                score += 0.1
    return score


def retrieve_corpus_scan(query_text: str, config: RetrievalConfig) -> list[RetrievedChunk]:
    path = Path(config.corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus fallback not found: {path}")

    query_tokens = tokenize(query_text)
    hits: list[RetrievedChunk] = []
    for row in corpus_rows(path):
        if not match_payload_filters(row, docsets=config.docsets, section_kinds=config.section_kinds):
            continue
        score = score_corpus_row(query_tokens, row)
        if score <= 0:
            continue
        hits.append(
            RetrievedChunk(
                score=score,
                payload=row,
                source="corpus",
                stage_scores={"lexical": score},
            )
        )

    hits.sort(key=lambda item: item.score, reverse=True)
    limited = hits[: config.lexical_limit]
    for rank, hit in enumerate(limited, start=1):
        hit.lexical_rank = rank
    return limited


def build_fts_match_query(query_text: str) -> str:
    query_text = normalize_query_text(query_text)
    base_query, expanded_terms = split_search_query(query_text)
    tokens = tokenize(base_query)
    if not tokens:
        return ""

    parts: list[str] = []
    phrase = base_query.strip().replace('"', " ").strip()
    if " " in phrase and len(phrase) <= 120:
        parts.append(f'"{phrase}"')

    for term in expanded_terms[:8]:
        normalized = normalize_query_text(term).replace('"', " ").strip()
        if not normalized:
            continue
        if " " in normalized:
            parts.append(f'"{normalized}"')
        else:
            parts.append(f'"{normalized.lower()}"')

    seen: set[str] = set()
    for token in tokens[:12]:
        if token in seen:
            continue
        seen.add(token)
        parts.append(f'"{token}"')
    return " OR ".join(parts)


def retrieve_lexical(query_text: str, config: RetrievalConfig) -> list[RetrievedChunk]:
    db_path = Path(config.fts_db_path)
    if not db_path.exists():
        return retrieve_corpus_scan(query_text, config)

    match_query = build_fts_match_query(query_text)
    if not match_query:
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        sql = [
            "SELECT m.payload_json, bm25(chunks_fts, 10.0, 8.0, 4.0, 1.0) AS bm25_score",
            "FROM chunks_fts",
            "JOIN chunks_meta m ON m.source_id = chunks_fts.source_id",
            "WHERE chunks_fts MATCH ?",
        ]
        params: list[object] = [match_query]

        if config.docsets:
            sql.append(f"AND m.docset IN ({','.join('?' for _ in config.docsets)})")
            params.extend(config.docsets)
        if config.section_kinds:
            sql.append(f"AND m.section_kind IN ({','.join('?' for _ in config.section_kinds)})")
            params.extend(config.section_kinds)

        sql.append("ORDER BY bm25_score")
        sql.append("LIMIT ?")
        params.append(config.lexical_limit)

        rows = conn.execute("\n".join(sql), params).fetchall()
    finally:
        conn.close()

    hits: list[RetrievedChunk] = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        raw_score = float(row["bm25_score"])
        score = lexical_post_score(query_text, -raw_score, payload)
        hits.append(
            RetrievedChunk(
                score=score,
                payload=payload,
                source="lexical",
                stage_scores={"lexical": -raw_score},
            )
        )
    hits.sort(key=lambda item: item.score, reverse=True)
    for rank, hit in enumerate(hits, start=1):
        hit.lexical_rank = rank
    return hits
