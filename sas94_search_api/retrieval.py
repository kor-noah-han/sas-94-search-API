from __future__ import annotations

import time

from sas94_search_api.rerank import rerank_hits
from sas94_search_api.retrieval_models import (
    DEFAULT_COLLECTION,
    DEFAULT_CORPUS_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FTS_DB_PATH,
    DEFAULT_QDRANT_PATH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_ROUTE_INDEX_PATH,
    DEFAULT_TERM_DICTIONARY,
    RetrievedChunk,
    RetrievalConfig,
    RetrievalResult,
    SectionRoute,
)
from sas94_search_api.route_index import load_section_routes, rank_section_routes
from sas94_search_api.scoring import fuse_hits, route_bonus, should_skip_dense
from sas94_search_api.storage import retrieve_dense, retrieve_lexical
from sas94_search_api.text_utils import env_default, expand_query, load_dotenv, normalize_query_text


def retrieve_hybrid(query: str, config: RetrievalConfig) -> RetrievalResult:
    timings_ms: dict[str, float] = {}
    dense_hits: list[RetrievedChunk] = []
    lexical_hits: list[RetrievedChunk] = []
    dense_error: str | None = None
    lexical_error: str | None = None
    query = normalize_query_text(query)
    query_text, expanded_terms = expand_query(query, config)

    started = time.perf_counter()
    routes = rank_section_routes(query_text, config)
    timings_ms["route"] = (time.perf_counter() - started) * 1000

    if config.enable_lexical:
        started = time.perf_counter()
        try:
            lexical_hits = retrieve_lexical(query_text, config)
            for hit in lexical_hits:
                hit.score += route_bonus(hit.payload, routes)
            lexical_hits.sort(key=lambda item: item.score, reverse=True)
            for rank, hit in enumerate(lexical_hits, start=1):
                hit.lexical_rank = rank
        except Exception as exc:
            lexical_error = str(exc)
        timings_ms["lexical"] = (time.perf_counter() - started) * 1000

    skip_dense = (
        config.enable_dense
        and config.enable_lexical
        and should_skip_dense(query_text, lexical_hits, routes)
    )

    if config.enable_dense and not skip_dense:
        started = time.perf_counter()
        try:
            dense_hits = retrieve_dense(query_text, config)
            for hit in dense_hits:
                hit.score += route_bonus(hit.payload, routes)
            dense_hits.sort(key=lambda item: item.score, reverse=True)
        except Exception as exc:
            dense_error = str(exc)
        timings_ms["dense"] = (time.perf_counter() - started) * 1000
    elif config.enable_dense:
        timings_ms["dense"] = 0.0

    if dense_hits and lexical_hits:
        hits = fuse_hits(query, dense_hits, lexical_hits, config, routes)
        mode = "hybrid"
    elif dense_hits:
        hits = dense_hits
        mode = "dense"
    else:
        hits = lexical_hits
        mode = "lexical"

    started = time.perf_counter()
    hits, reranked = rerank_hits(query, hits, config)
    timings_ms["rerank"] = (time.perf_counter() - started) * 1000

    return RetrievalResult(
        hits=hits[: config.top_k],
        mode=mode,
        timings_ms=timings_ms,
        query_text=query_text,
        expanded_terms=expanded_terms,
        dense_error=dense_error,
        lexical_error=lexical_error,
        reranked=reranked,
    )


def public_hit_dict(rank: int, hit: RetrievedChunk) -> dict[str, object]:
    payload = hit.payload
    return {
        "rank": rank,
        "score": hit.rerank_score if hit.rerank_score is not None else hit.fused_score or hit.score,
        "dense_rank": hit.dense_rank,
        "lexical_rank": hit.lexical_rank,
        "docset": payload.get("docset"),
        "title": payload.get("title"),
        "section_path_text": payload.get("section_path_text"),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "source_html": payload.get("source_html"),
        "text_preview": str(payload.get("text", ""))[:500],
    }


def format_page_range(payload: dict[str, object]) -> str:
    page_start = payload.get("page_start")
    page_end = payload.get("page_end")
    if page_start and page_end and page_start != page_end:
        return f"p.{page_start}-{page_end}"
    if page_start:
        return f"p.{page_start}"
    return "page n/a"


def format_source_label(payload: dict[str, object]) -> str:
    return f"{payload.get('docset', 'unknown')} {format_page_range(payload)}"


__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_CORPUS_PATH",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_FTS_DB_PATH",
    "DEFAULT_QDRANT_PATH",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_ROUTE_INDEX_PATH",
    "DEFAULT_TERM_DICTIONARY",
    "RetrievedChunk",
    "RetrievalConfig",
    "RetrievalResult",
    "SectionRoute",
    "env_default",
    "format_page_range",
    "format_source_label",
    "load_dotenv",
    "load_section_routes",
    "public_hit_dict",
    "retrieve_hybrid",
]
