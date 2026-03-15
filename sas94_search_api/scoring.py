from __future__ import annotations

from sas94_search_api.retrieval_models import (
    DEFINITION_SECTION_MARKERS,
    GRAPHICS_SECTION_MARKERS,
    HOWTO_QUERY_MARKERS,
    HOWTO_SECTION_MARKERS,
    LEXICAL_FAMILY_MARKERS,
    LIBRARY_ASSIGNMENT_MARKERS,
    RetrievedChunk,
    RetrievalConfig,
    SectionRoute,
)
from sas94_search_api.text_utils import extract_proc_names, split_search_query, tokenize


def metadata_bonus(query_tokens: list[str], payload: dict[str, object]) -> float:
    searchable = " ".join(
        [
            str(payload.get("title", "")),
            str(payload.get("section_path_text", "")),
            str(payload.get("chapter_title", "")),
        ]
    ).lower()
    bonus = 0.0
    for token in query_tokens:
        if token in searchable:
            bonus += 0.005
    return bonus


def procedure_bonus(query: str, payload: dict[str, object]) -> float:
    proc_names = extract_proc_names(query)
    if not proc_names:
        return 0.0

    title = str(payload.get("title", "")).lower()
    section_path = str(payload.get("section_path_text", "")).lower()
    chapter_title = str(payload.get("chapter_title", "")).lower()
    text_preview = str(payload.get("text", ""))[:1800].lower()
    searchable = "\n".join([title, chapter_title, section_path, text_preview])

    bonus = 0.0
    for proc_name in proc_names:
        exact_phrase = f"proc {proc_name}"
        procedure_phrase = f"{proc_name} procedure"
        if exact_phrase in searchable:
            bonus += 1.8
        if procedure_phrase in searchable:
            bonus += 1.2
        if proc_name in section_path:
            bonus += 0.8
        if proc_name in title or proc_name in chapter_title:
            bonus += 0.5
    return bonus


def reference_penalty(payload: dict[str, object]) -> float:
    section_path = str(payload.get("section_path_text", "")).lower()
    title = str(payload.get("title", "")).lower()
    text_preview = str(payload.get("text", ""))[:1200].lower()
    combined = f"{title}\n{section_path}\n{text_preview}"

    penalty = 0.0
    low_signal_markers = [
        "appendix",
        "special sas data sets",
        "ods tables",
        "ods table",
        "brief descriptions",
        "procedure concepts",
        "contents",
        "index",
    ]
    for marker in low_signal_markers:
        if marker in combined:
            penalty += 0.45

    if "table " in text_preview or "table." in text_preview:
        penalty += 0.25
    if "ods table" in text_preview or "ods tables produced" in text_preview:
        penalty += 0.8
    proc_mentions = len(set(extract_proc_names(text_preview)))
    if proc_mentions >= 4:
        penalty += 0.7
    elif proc_mentions >= 2:
        penalty += 0.35

    if "examples:" in combined:
        penalty += 0.15
    return penalty


def lexical_post_score(query: str, base_score: float, payload: dict[str, object]) -> float:
    base_query, expanded_terms = split_search_query(query)
    query_tokens = tokenize(base_query)
    section_path = str(payload.get("section_path_text", "")).lower()
    title = str(payload.get("title", "")).lower()
    combined = f"{title} {section_path}"
    score = base_score

    phrase = base_query.lower().strip().replace("?", "")
    if phrase and phrase in combined:
        score += 2.0

    for token in query_tokens:
        if token in section_path:
            score += 0.7
        elif token in title:
            score += 0.35

    if "procedure" in section_path and "proc" in phrase:
        score += 0.5
    if "library" in phrase and "library" in section_path:
        score += 0.75
    if "macro" in phrase and "macro" in section_path:
        score += 0.75
    query_scope = f"{phrase} {' '.join(term.lower() for term in expanded_terms)}"
    if any(marker in query_scope for marker in HOWTO_QUERY_MARKERS):
        if any(marker in section_path for marker in HOWTO_SECTION_MARKERS):
            score += 0.9
        if any(marker in section_path for marker in DEFINITION_SECTION_MARKERS):
            score -= 0.6
    if ("library" in query_scope or "libname" in query_scope) and ("할당" in query_scope or "assign" in query_scope):
        if any(marker in section_path for marker in LIBRARY_ASSIGNMENT_MARKERS):
            score += 1.1
    if any(marker in query_scope for marker in ("graphics", "graph", "plot", "sgplot", "ods graphics", "statistical graphics")):
        if any(marker in section_path for marker in GRAPHICS_SECTION_MARKERS):
            score += 1.2
        if any(marker in title for marker in GRAPHICS_SECTION_MARKERS):
            score += 0.8
    for term in expanded_terms[:8]:
        lowered = term.lower()
        if lowered in combined:
            score += 0.8 if "procedure" in lowered else 0.4
    score += procedure_bonus(query, payload)
    score -= reference_penalty(payload)
    return score


def route_bonus(payload: dict[str, object], routes: list[SectionRoute]) -> float:
    if not routes:
        return 0.0
    section_path = str(payload.get("section_path_text", "")).lower()
    chapter_title = str(payload.get("chapter_title", "")).lower()
    bonus = 0.0
    for index, route in enumerate(routes[:3], start=1):
        weight = max(0.6 - (index - 1) * 0.18, 0.2)
        route_path = route.section_path_text.lower()
        route_chapter = (route.chapter_title or "").lower()
        if section_path == route_path:
            bonus += weight
        elif route_path and section_path.startswith(route_path):
            bonus += weight * 0.7
        elif route_chapter and chapter_title and route_chapter == chapter_title:
            bonus += weight * 0.35
    return bonus


def should_skip_dense(query: str, lexical_hits: list[RetrievedChunk], routes: list[SectionRoute]) -> bool:
    if not lexical_hits:
        return False
    top_hit = lexical_hits[0]
    base_query, expanded_terms = split_search_query(query)
    section_path = str(top_hit.payload.get("section_path_text", "")).lower()
    searchable = " ".join(
        [
            str(top_hit.payload.get("title", "")).lower(),
            section_path,
            str(top_hit.payload.get("chapter_title", "")).lower(),
            str(top_hit.payload.get("text", "")).lower()[:1500],
        ]
    )

    proc_names = extract_proc_names(f"{base_query} {' '.join(expanded_terms)}")
    if proc_names and any(f"{proc} procedure" in searchable for proc in proc_names):
        return True

    if routes and routes[0].score >= 2.5 and route_bonus(top_hit.payload, routes[:1]) >= 0.5:
        return True

    phrase = base_query.lower().strip().replace("?", "")
    if phrase and len(phrase.split()) >= 2 and phrase in searchable:
        return True

    matched_expansions = 0
    for term in expanded_terms[:8]:
        lowered = term.lower()
        if len(lowered) >= 6 and lowered in searchable:
            matched_expansions += 1
    if matched_expansions >= 2:
        return True

    query_scope = f"{base_query.lower()} {' '.join(term.lower() for term in expanded_terms)}"
    for family_terms in LEXICAL_FAMILY_MARKERS.values():
        if not any(term in query_scope for term in family_terms):
            continue
        matches = sum(1 for term in family_terms if term in searchable)
        if matches >= 2:
            return True
        if matches >= 1 and any(term in section_path for term in family_terms):
            return True
    return False


def fuse_hits(
    query: str,
    dense_hits: list[RetrievedChunk],
    lexical_hits: list[RetrievedChunk],
    config: RetrievalConfig,
    routes: list[SectionRoute],
) -> list[RetrievedChunk]:
    query_tokens = tokenize(query)
    merged: dict[str, RetrievedChunk] = {}

    def key_for(hit: RetrievedChunk) -> str:
        return str(hit.payload.get("source_id") or hit.payload.get("id") or id(hit))

    for hit in dense_hits:
        merged[key_for(hit)] = hit

    for hit in lexical_hits:
        key = key_for(hit)
        current = merged.get(key)
        if current is None:
            merged[key] = hit
            continue
        if current.dense_rank is None and hit.dense_rank is not None:
            current.dense_rank = hit.dense_rank
        if current.lexical_rank is None and hit.lexical_rank is not None:
            current.lexical_rank = hit.lexical_rank
        current.stage_scores.update(hit.stage_scores)

    fused: list[RetrievedChunk] = []
    for hit in merged.values():
        score = 0.0
        if hit.dense_rank is not None:
            score += config.dense_weight / (config.rrf_k + hit.dense_rank)
        if hit.lexical_rank is not None:
            score += config.lexical_weight / (config.rrf_k + hit.lexical_rank)
        score += metadata_bonus(query_tokens, hit.payload)
        score += procedure_bonus(query, hit.payload) * 0.08
        score -= reference_penalty(hit.payload) * 0.05
        score += route_bonus(hit.payload, routes)
        hit.fused_score = score
        fused.append(hit)

    fused.sort(key=lambda item: item.fused_score or 0.0, reverse=True)
    return fused
