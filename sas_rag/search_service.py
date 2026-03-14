from __future__ import annotations

from dataclasses import dataclass

from sas_rag.app import retrieval_response_dict
from sas_rag.retrieval import RetrievalConfig, RetrievalResult, retrieve_hybrid


@dataclass
class SearchServiceResponse:
    query: str
    retrieval: dict[str, object]
    result: RetrievalResult


def run_search(query: str, config: RetrievalConfig) -> SearchServiceResponse:
    result = retrieve_hybrid(query, config)
    return SearchServiceResponse(
        query=query,
        retrieval=retrieval_response_dict(result),
        result=result,
    )

