from __future__ import annotations

import argparse
import json

from sas94_search_api.generation_stub import GenerationConfig  # type: ignore
from sas94_search_api.retrieval import (
    DEFAULT_COLLECTION,
    DEFAULT_CORPUS_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FTS_DB_PATH,
    DEFAULT_QDRANT_PATH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_ROUTE_INDEX_PATH,
    RetrievalConfig,
    RetrievalResult,
    env_default,
    format_source_label,
    public_hit_dict,
)


def add_retrieval_args(
    parser: argparse.ArgumentParser,
    *,
    top_k_default: int,
    top_k_help: str,
    include_query: bool = False,
) -> None:
    if include_query:
        parser.add_argument("query", nargs="?", help="Question to ask. If omitted, start interactive mode.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name.")
    parser.add_argument(
        "--qdrant-path",
        default=env_default("QDRANT_PATH", DEFAULT_QDRANT_PATH),
        help="Optional local Qdrant storage path for dense/local mode.",
    )
    parser.add_argument("--url", default=env_default("QDRANT_URL"), help="Remote Qdrant URL for server mode.")
    parser.add_argument("--api-key", default=env_default("QDRANT_API_KEY"), help="Qdrant API key for remote mode.")
    parser.add_argument(
        "--embedding-model",
        default=env_default("QDRANT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="Dense embedding model.",
    )
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_PATH, help="Fallback corpus JSONL path.")
    parser.add_argument("--fts-db", default=DEFAULT_FTS_DB_PATH, help="SQLite FTS index path.")
    parser.add_argument("--top-k", type=int, default=top_k_default, help=top_k_help)
    parser.add_argument("--dense-limit", type=int, default=24, help="Candidate pool size for dense retrieval.")
    parser.add_argument("--lexical-limit", type=int, default=24, help="Candidate pool size for lexical retrieval.")
    parser.add_argument("--docset", action="append", help="Optional docset filter.")
    parser.add_argument("--section-kind", action="append", help="Optional section kind filter.")
    parser.add_argument(
        "--mode",
        choices=["dense", "lexical", "hybrid"],
        default="lexical",
        help="Retrieval mode. The default release bundle is lexical-first.",
    )
    parser.add_argument(
        "--rerank-model",
        default=env_default("RAG_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        help="FastEmbed reranker model.",
    )
    parser.add_argument("--rerank-limit", type=int, default=12, help="How many fused candidates to rerank.")
    parser.add_argument("--rerank", action="store_true", help="Enable FastEmbed reranking on top fused candidates.")
    parser.add_argument(
        "--no-term-expansion",
        action="store_true",
        help="Disable Korean SAS term expansion before retrieval.",
    )


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float, default=0.1, help="Gemini generation temperature.")
    parser.add_argument("--model", help="Gemini model name. Overrides GEMINI_MODEL when provided.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification for Gemini API.")


def build_retrieval_config(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        collection=args.collection,
        qdrant_path=args.qdrant_path,
        qdrant_url=args.url,
        qdrant_api_key=args.api_key,
        embedding_model=args.embedding_model,
        corpus_path=args.corpus,
        fts_db_path=args.fts_db,
        route_index_path=env_default("RAG_ROUTE_INDEX_PATH", DEFAULT_ROUTE_INDEX_PATH),
        top_k=args.top_k,
        dense_limit=args.dense_limit,
        lexical_limit=args.lexical_limit,
        docsets=tuple(args.docset or ()),
        section_kinds=tuple(args.section_kind or ()),
        enable_dense=args.mode in {"dense", "hybrid"},
        enable_lexical=args.mode in {"lexical", "hybrid"},
        enable_rerank=args.rerank,
        enable_term_expansion=not args.no_term_expansion,
        rerank_model=args.rerank_model,
        rerank_limit=args.rerank_limit,
    )


def retrieval_response_dict(result: RetrievalResult) -> dict[str, object]:
    return {
        "mode": result.mode,
        "query_text": result.query_text,
        "expanded_terms": result.expanded_terms,
        "timings_ms": {key: round(value, 2) for key, value in result.timings_ms.items()},
        "dense_error": result.dense_error,
        "lexical_error": result.lexical_error,
        "reranked": result.reranked,
        "hits": [public_hit_dict(index, hit) for index, hit in enumerate(result.hits, start=1)],
    }


def retrieval_debug_dict(result: RetrievalResult) -> dict[str, object]:
    return {
        "mode": result.mode,
        "query_text": result.query_text,
        "expanded_terms": result.expanded_terms,
        "timings_ms": {key: round(value, 2) for key, value in result.timings_ms.items()},
        "dense_error": result.dense_error,
        "lexical_error": result.lexical_error,
        "reranked": result.reranked,
    }


def print_hits(result: RetrievalResult) -> None:
    for index, hit in enumerate(result.hits, start=1):
        print(json.dumps(public_hit_dict(index, hit), ensure_ascii=False))
