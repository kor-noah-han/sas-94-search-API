from __future__ import annotations

import json
import sqlite3
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from sas94_search_api.retrieval_models import LEXICAL_FAMILY_MARKERS, RetrievalConfig, SectionRoute
from sas94_search_api.text_utils import extract_proc_names, split_search_query, tokenize


def corpus_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


@lru_cache(maxsize=2)
def load_section_routes(route_index_path_str: str, fts_db_path_str: str, corpus_path_str: str) -> list[dict[str, object]]:
    route_index_path = Path(route_index_path_str)
    if route_index_path.exists():
        return json.loads(route_index_path.read_text(encoding="utf-8"))

    db_path = Path(fts_db_path_str)
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT docset, title, section_path_text
                FROM chunks_meta
                GROUP BY docset, title, section_path_text
                """
            ).fetchall()
        finally:
            conn.close()

        routes: list[dict[str, object]] = []
        for row in rows:
            docset = str(row["docset"] or "")
            section_path_text = str(row["section_path_text"] or "").strip()
            title = str(row["title"] or "").strip()
            if not docset or not section_path_text:
                continue
            routes.append(
                {
                    "docset": docset,
                    "section_path_text": section_path_text,
                    "chapter_title": None,
                    "section_title": None,
                    "search_text": "\n".join(part for part in [title, section_path_text] if part).lower(),
                }
            )
        return routes

    path = Path(corpus_path_str)
    if not path.exists():
        return []

    grouped: OrderedDict[tuple[str, str], dict[str, object]] = OrderedDict()
    for row in corpus_rows(path):
        docset = str(row.get("docset") or row.get("doc_id") or "")
        section_path_text = str(row.get("section_path_text") or "").strip()
        if not docset or not section_path_text:
            continue
        key = (docset, section_path_text)
        if key in grouped:
            continue
        grouped[key] = {
            "docset": docset,
            "section_path_text": section_path_text,
            "chapter_title": row.get("chapter_title"),
            "section_title": row.get("section_title"),
            "search_text": "\n".join(
                part
                for part in [
                    str(row.get("title") or ""),
                    section_path_text,
                    str(row.get("chapter_title") or ""),
                    str(row.get("section_title") or ""),
                ]
                if part
            ).lower(),
        }
    return list(grouped.values())


def score_section_route(query: str, route: dict[str, object]) -> float:
    base_query, expanded_terms = split_search_query(query)
    query_tokens = tokenize(base_query)
    searchable = str(route["search_text"])
    section_path_text = str(route["section_path_text"]).lower()
    chapter_title = str(route.get("chapter_title") or "").lower()
    score = 0.0

    phrase = base_query.lower().strip().replace("?", "")
    if phrase and phrase in section_path_text:
        score += 3.0

    for token in query_tokens:
        if token in section_path_text:
            score += 0.7
        elif token in chapter_title:
            score += 0.25
        elif token in searchable:
            score += 0.12

    for term in expanded_terms[:8]:
        lowered = term.lower()
        if lowered in section_path_text:
            score += 1.0 if "procedure" in lowered else 0.5
        elif lowered in searchable:
            score += 0.25

    proc_names = extract_proc_names(f"{base_query} {' '.join(expanded_terms)}")
    for proc_name in proc_names:
        if f"{proc_name} procedure" in searchable:
            score += 2.2
        if f"proc {proc_name}" in searchable:
            score += 2.2

    query_scope = f"{base_query.lower()} {' '.join(term.lower() for term in expanded_terms)}"
    for family_terms in LEXICAL_FAMILY_MARKERS.values():
        if any(term in query_scope for term in family_terms):
            score += sum(0.25 for term in family_terms if term in searchable)

    return score


def rank_section_routes(query: str, config: RetrievalConfig, limit: int = 8) -> list[SectionRoute]:
    routes = load_section_routes(config.route_index_path, config.fts_db_path, config.corpus_path)
    if not routes:
        return []

    allowed_docsets = set(config.docsets)
    scored: list[SectionRoute] = []
    for route in routes:
        if allowed_docsets and route["docset"] not in allowed_docsets:
            continue
        score = score_section_route(query, route)
        if score <= 0:
            continue
        scored.append(
            SectionRoute(
                docset=str(route["docset"]),
                section_path_text=str(route["section_path_text"]),
                chapter_title=str(route.get("chapter_title") or "") or None,
                section_title=str(route.get("section_title") or "") or None,
                score=score,
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]
