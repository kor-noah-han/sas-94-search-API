#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import _bootstrap  # noqa: F401

from sas_rag.app import add_retrieval_args, build_retrieval_config
from sas_rag.retrieval import load_dotenv, retrieve_hybrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SAS retrieval quality and latency with a JSONL query set."
    )
    parser.add_argument(
        "--queries",
        default="data/eval/sas-rag-smoke.jsonl",
        help="JSONL file containing query and expected metadata.",
    )
    add_retrieval_args(parser, top_k_default=5, top_k_help="Top-k cutoff for evaluation.")
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise SystemExit(f"Query file not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_hit_rank(result_hits, key: str, value: str) -> int | None:
    for index, hit in enumerate(result_hits, start=1):
        if str(hit.payload.get(key)) == value:
            return index
    return None


def section_contains_rank(result_hits, fragment: str) -> int | None:
    fragment = fragment.lower()
    for index, hit in enumerate(result_hits, start=1):
        section_path = str(hit.payload.get("section_path_text", "")).lower()
        if fragment in section_path:
            return index
    return None


def main() -> int:
    load_dotenv()
    args = parse_args()
    queries = load_queries(Path(args.queries))
    config = build_retrieval_config(args)
    config.dense_limit = max(args.top_k * 4, 24)
    config.lexical_limit = max(args.top_k * 4, 24)

    top1_docset_hits = 0
    topk_docset_hits = 0
    topk_section_hits = 0
    latency_totals = {"dense": 0.0, "lexical": 0.0, "rerank": 0.0}
    total = len(queries)

    for item in queries:
        query = str(item["query"])
        result = retrieve_hybrid(query, config)
        expected_docset = item.get("expected_docset")
        expected_section = item.get("expected_section_contains")

        row = {
            "query": query,
            "mode": result.mode,
            "query_text": result.query_text,
            "expanded_terms": result.expanded_terms,
            "timings_ms": {key: round(value, 2) for key, value in result.timings_ms.items()},
            "dense_error": result.dense_error,
            "lexical_error": result.lexical_error,
            "top_hit_docset": result.hits[0].payload.get("docset") if result.hits else None,
        }

        if expected_docset:
            rank = expected_hit_rank(result.hits, "docset", str(expected_docset))
            row["expected_docset"] = expected_docset
            row["docset_hit_rank"] = rank
            if rank == 1:
                top1_docset_hits += 1
            if rank is not None and rank <= args.top_k:
                topk_docset_hits += 1

        if expected_section:
            rank = section_contains_rank(result.hits, str(expected_section))
            row["expected_section_contains"] = expected_section
            row["section_hit_rank"] = rank
            if rank is not None and rank <= args.top_k:
                topk_section_hits += 1

        for key in latency_totals:
            latency_totals[key] += result.timings_ms.get(key, 0.0)

        print(json.dumps(row, ensure_ascii=False))

    summary = {
        "queries": total,
        "top1_docset_accuracy": round(top1_docset_hits / total, 3) if total else 0.0,
        "topk_docset_accuracy": round(topk_docset_hits / total, 3) if total else 0.0,
        "topk_section_accuracy": round(topk_section_hits / total, 3) if total else 0.0,
        "avg_timings_ms": {
            key: round(value / total, 2) if total else 0.0 for key, value in latency_totals.items()
        },
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
