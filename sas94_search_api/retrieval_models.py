from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ENV_PATH = Path(".env")
DEFAULT_COLLECTION = "sas9_pdf_chunks"
DEFAULT_CORPUS_PATH = "data/processed/sas-rag/corpus/sas9-pdf-corpus.jsonl"
DEFAULT_FTS_DB_PATH = "data/processed/sas-rag/search/sas9-pdf-fts.db"
DEFAULT_ROUTE_INDEX_PATH = "data/processed/sas-rag/search/sas9-pdf-route-index.json"
DEFAULT_QDRANT_PATH = "data/qdrant/sas9_pdf"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DEFAULT_TERM_DICTIONARY = "data/config/sas-ko-en-terms.json"

LEXICAL_FAMILY_MARKERS: dict[str, tuple[str, ...]] = {
    "library": ("library", "libname", "libref", "assignment"),
    "data step": ("data step", "data-step", "set statement", "merge statement", "data statement"),
}
HOWTO_QUERY_MARKERS = (
    "how",
    "how to",
    "어떻게",
    "방법",
    "사용",
    "쓰",
    "구문",
    "syntax",
    "example",
    "예제",
    "statement",
    "할당",
)
HOWTO_SECTION_MARKERS = (
    "syntax",
    "example",
    "examples",
    "usage",
    "getting started",
    "overview",
    "statement",
    "elements of",
)
DEFINITION_SECTION_MARKERS = (
    "definitions",
    "definition",
    "terms to be familiar with",
)
LIBRARY_ASSIGNMENT_MARKERS = (
    "assignment",
    "elements of a library assignment",
    "libname",
)


@dataclass
class RetrievedChunk:
    score: float
    payload: dict[str, object]
    source: str
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None
    fused_score: float | None = None
    stage_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    hits: list[RetrievedChunk]
    mode: str
    timings_ms: dict[str, float]
    query_text: str
    expanded_terms: list[str]
    dense_error: str | None = None
    lexical_error: str | None = None
    reranked: bool = False


@dataclass
class SectionRoute:
    docset: str
    section_path_text: str
    chapter_title: str | None
    section_title: str | None
    score: float


@dataclass
class RetrievalConfig:
    collection: str = DEFAULT_COLLECTION
    qdrant_path: str = DEFAULT_QDRANT_PATH
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_timeout: int | None = 30
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    corpus_path: str = DEFAULT_CORPUS_PATH
    fts_db_path: str = DEFAULT_FTS_DB_PATH
    route_index_path: str = DEFAULT_ROUTE_INDEX_PATH
    term_dictionary_path: str = DEFAULT_TERM_DICTIONARY
    top_k: int = 5
    dense_limit: int = 12
    lexical_limit: int = 16
    docsets: tuple[str, ...] = ()
    section_kinds: tuple[str, ...] = ()
    enable_dense: bool = False
    enable_lexical: bool = True
    enable_rerank: bool = False
    enable_term_expansion: bool = True
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_limit: int = 8
    dense_weight: float = 0.8
    lexical_weight: float = 1.2
    rrf_k: int = 60
