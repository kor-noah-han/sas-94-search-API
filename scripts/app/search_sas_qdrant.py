#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _bootstrap  # noqa: F401

from sas94_search_api.app import add_retrieval_args, build_retrieval_config, print_hits, retrieval_debug_dict
from sas94_search_api.retrieval import load_dotenv, retrieve_hybrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the SAS corpus with dense, lexical, or hybrid retrieval."
    )
    parser.add_argument("query", help="Search query text.")
    add_retrieval_args(parser, top_k_default=5, top_k_help="Maximum hits to return.")
    parser.add_argument("--limit", dest="top_k", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="Print retrieval timings and fallback information.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    result = retrieve_hybrid(args.query, build_retrieval_config(args))
    if args.show_debug:
        print(json.dumps(retrieval_debug_dict(result), ensure_ascii=False))
    print_hits(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
