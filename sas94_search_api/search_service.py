from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import lru_cache

from sas94_search_api.app import retrieval_response_dict
from sas94_search_api.retrieval import RetrievalConfig, RetrievalResult, retrieve_hybrid


@dataclass
class SearchServiceResponse:
    query: str
    retrieval: dict[str, object]
    result: RetrievalResult


def config_cache_dict(config: RetrievalConfig) -> dict[str, object]:
    return {
        "collection": config.collection,
        "qdrant_path": config.qdrant_path,
        "qdrant_url": config.qdrant_url,
        "qdrant_api_key": config.qdrant_api_key,
        "qdrant_timeout": config.qdrant_timeout,
        "embedding_model": config.embedding_model,
        "corpus_path": config.corpus_path,
        "fts_db_path": config.fts_db_path,
        "route_index_path": config.route_index_path,
        "term_dictionary_path": config.term_dictionary_path,
        "top_k": config.top_k,
        "dense_limit": config.dense_limit,
        "lexical_limit": config.lexical_limit,
        "docsets": list(config.docsets),
        "section_kinds": list(config.section_kinds),
        "enable_dense": config.enable_dense,
        "enable_lexical": config.enable_lexical,
        "enable_rerank": config.enable_rerank,
        "enable_term_expansion": config.enable_term_expansion,
        "rerank_model": config.rerank_model,
        "rerank_limit": config.rerank_limit,
        "dense_weight": config.dense_weight,
        "lexical_weight": config.lexical_weight,
        "rrf_k": config.rrf_k,
    }


def _config_cache_key(config: RetrievalConfig) -> str:
    return json.dumps(
        config_cache_dict(config),
        ensure_ascii=False,
        sort_keys=True,
    )


@lru_cache(maxsize=128)
def _cached_result(query: str, config_key: str) -> RetrievalResult:
    config = RetrievalConfig(**json.loads(config_key))
    return retrieve_hybrid(query, config)


def run_search(query: str, config: RetrievalConfig) -> SearchServiceResponse:
    result = copy.deepcopy(_cached_result(query, _config_cache_key(config)))
    return SearchServiceResponse(
        query=query,
        retrieval=retrieval_response_dict(result),
        result=result,
    )
