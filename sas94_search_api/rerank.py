from __future__ import annotations

from sas94_search_api.retrieval_models import RetrievedChunk, RetrievalConfig
from sas94_search_api.text_utils import has_hangul

try:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
except Exception:
    TextCrossEncoder = None


_RERANKER_CACHE: dict[str, TextCrossEncoder] = {}


def rerank_document_text(payload: dict[str, object], limit: int = 1800) -> str:
    parts = [
        f"Document: {payload.get('title', '')}",
        f"Section: {payload.get('section_path_text', '')}",
        str(payload.get("text", ""))[:limit],
    ]
    return "\n".join(part for part in parts if part)


def get_reranker(model_name: str) -> TextCrossEncoder:
    if TextCrossEncoder is None:
        raise RuntimeError("fastembed reranker is not available.")
    reranker = _RERANKER_CACHE.get(model_name)
    if reranker is None:
        reranker = TextCrossEncoder(model_name=model_name, lazy_load=False)
        _RERANKER_CACHE[model_name] = reranker
    return reranker


def rerank_hits(query: str, hits: list[RetrievedChunk], config: RetrievalConfig) -> tuple[list[RetrievedChunk], bool]:
    if not config.enable_rerank or len(hits) <= 1:
        return hits, False
    if has_hangul(query):
        return hits, False

    rerank_pool = hits[: config.rerank_limit]
    documents = [rerank_document_text(hit.payload) for hit in rerank_pool]
    reranker = get_reranker(config.rerank_model)
    scores = list(reranker.rerank(query=query, documents=documents, batch_size=32))

    reranked: list[RetrievedChunk] = []
    original_order: dict[str, int] = {}
    for hit, score in zip(rerank_pool, scores):
        hit.rerank_score = float(score)
        key = str(hit.payload.get("source_id") or hit.payload.get("id") or id(hit))
        original_order[key] = len(original_order) + 1
        reranked.append(hit)

    rerank_order = sorted(
        reranked,
        key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
        reverse=True,
    )
    rerank_rank_by_key: dict[str, int] = {}
    for index, hit in enumerate(rerank_order, start=1):
        key = str(hit.payload.get("source_id") or hit.payload.get("id") or id(hit))
        rerank_rank_by_key[key] = index

    def combined_score(item: RetrievedChunk) -> float:
        key = str(item.payload.get("source_id") or item.payload.get("id") or id(item))
        return (1.0 / (30 + rerank_rank_by_key[key])) + (0.35 / (30 + original_order[key]))

    reranked.sort(key=combined_score, reverse=True)
    reranked.extend(hits[config.rerank_limit :])
    return reranked, True
