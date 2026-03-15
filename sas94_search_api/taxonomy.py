from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sas94_search_api.text_utils import split_search_query


INTENT_MARKERS: dict[str, tuple[str, ...]] = {
    "howto": ("how", "how to", "어떻게", "방법", "하는 법", "만드는 법", "만들기", "작성 방법", "알려줘", "사용", "써", "쓰는 법"),
    "syntax": ("syntax", "statement", "option", "options", "구문", "문법", "옵션"),
    "example": ("example", "examples", "sample", "샘플", "예제", "사례"),
    "definition": ("what is", "무엇", "뭐야", "뜻", "의미", "정의", "란", "소개"),
    "comparison": ("compare", "comparison", "difference", "vs", "versus", "비교", "차이", "다른 점"),
    "troubleshooting": (
        "error",
        "errors",
        "warning",
        "warnings",
        "message",
        "messages",
        "problem",
        "issue",
        "debug",
        "log",
        "오류",
        "에러",
        "경고",
        "문제",
        "실패",
    ),
}

FAMILY_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "library": {
        "markers": ("library", "libname", "libref", "engine", "라이브러리"),
        "docsets": ("lepg",),
        "sections": ("libraries", "libname", "libref", "assignment", "engine"),
    },
    "data_step": {
        "markers": (
            "data step",
            "data-step",
            "set statement",
            "merge statement",
            "merge",
            "combine data",
            "by-group",
            "retain",
            "array",
            "do loop",
            "if-then",
            "output statement",
            "데이터 스텝",
            "데이터스텝",
            "병합",
        ),
        "docsets": ("lepg",),
        "sections": (
            "the data step",
            "set statement",
            "merge statement",
            "by-group",
            "retain statement",
            "array statement",
            "do loops",
            "if-then",
            "output statement",
            "combining data",
        ),
    },
    "macro": {
        "markers": ("macro", "macro variable", "%let", "%macro", "automatic macro variable", "매크로", "매크로 변수"),
        "docsets": ("mcrolref",),
        "sections": ("macro variable", "macro facility", "%let", "%macro", "automatic macro variable"),
    },
    "graphics": {
        "markers": (
            "graphics",
            "graph",
            "plot",
            "sgplot",
            "sgpanel",
            "sgscatter",
            "ods graphics",
            "statistical graphics",
            "시각화",
            "그래프",
            "차트",
            "플롯",
        ),
        "docsets": ("statug", "imlug"),
        "sections": (
            "statistical graphics",
            "ods graphics",
            "graphics template",
            "sgplot",
            "sgpanel",
            "sgscatter",
            "scatter",
            "histogram",
            "boxplot",
            "plot",
        ),
    },
    "statistics": {
        "markers": (
            "means procedure",
            "summary procedure",
            "descriptive statistics",
            "median",
            "quartiles",
            "percentiles",
            "univariate",
            "t test",
            "ttest",
            "chi-square",
            "freq procedure",
            "corr procedure",
            "통계",
            "기술통계",
        ),
        "docsets": ("procstat", "statug"),
        "sections": (
            "means procedure",
            "summary procedure",
            "freq procedure",
            "corr procedure",
            "univariate procedure",
            "ttest procedure",
            "statistical",
        ),
    },
    "correlation": {
        "markers": ("proc corr", "corr procedure", "correlation", "pearson correlation", "spearman correlation", "상관", "상관분석"),
        "docsets": ("procstat",),
        "sections": ("corr procedure", "correlation"),
    },
    "modeling": {
        "markers": (
            "regression",
            "proc reg",
            "proc glm",
            "proc logistic",
            "proc mixed",
            "modeling",
            "model",
            "회귀",
            "모델링",
        ),
        "docsets": ("procstat", "statug"),
        "sections": (
            "regression procedures",
            "reg procedure",
            "glm procedure",
            "logistic procedure",
            "mixed modeling",
            "modeling",
        ),
    },
    "sql": {
        "markers": ("proc sql", "sql", "join", "inner join", "left join", "group by", "having", "조인", "sql"),
        "docsets": ("proc", "lepg"),
        "sections": ("sql procedure", "join", "query"),
    },
    "reporting": {
        "markers": ("proc report", "report", "tabulate", "print", "listing", "보고서", "리포트"),
        "docsets": ("proc",),
        "sections": ("report procedure", "tabulate procedure", "print procedure", "report"),
    },
    "import_export": {
        "markers": ("import", "export", "csv", "excel", "xlsx", "external file", "가져오기", "내보내기"),
        "docsets": ("proc", "lepg"),
        "sections": ("import procedure", "export procedure", "external files"),
    },
    "sorting": {
        "markers": ("proc sort", "sort", "정렬"),
        "docsets": ("proc",),
        "sections": ("sort procedure", "sorting"),
    },
    "transpose": {
        "markers": ("proc transpose", "transpose", "전치", "행열변환"),
        "docsets": ("proc",),
        "sections": ("transpose procedure",),
    },
    "formats": {
        "markers": ("format", "informat", "proc format", "user-defined format", "포맷", "형식"),
        "docsets": ("proc", "lepg"),
        "sections": ("format procedure", "formats", "informats"),
    },
    "ods": {
        "markers": ("ods", "output delivery system", "destination", "style", "template", "output", "출력"),
        "docsets": ("statug", "proc"),
        "sections": ("ods", "output delivery system", "template", "destination", "style"),
    },
}

INTENT_SECTION_BOOSTS: dict[str, tuple[str, ...]] = {
    "howto": ("syntax", "usage", "example", "examples", "statement", "procedure", "getting started", "elements of"),
    "syntax": ("syntax", "statement", "procedure", "option"),
    "example": ("example", "examples", "sample"),
    "definition": ("definition", "definitions", "concepts", "introduction", "overview"),
    "comparison": ("comparing", "comparison", "difference", "differences", "versus", "methods"),
    "troubleshooting": ("error", "errors", "warning", "warnings", "message", "messages", "troubleshooting", "debug", "log"),
}

INTENT_SECTION_PENALTIES: dict[str, tuple[str, ...]] = {
    "howto": ("definitions", "definition", "terms to be familiar with", "introduction", "concepts"),
    "example": ("definitions", "definition"),
}


@dataclass(frozen=True)
class QueryTaxonomy:
    intents: tuple[str, ...]
    families: tuple[str, ...]
    preferred_docsets: tuple[str, ...]
    preferred_section_markers: tuple[str, ...]
    discouraged_section_markers: tuple[str, ...]


def _scope(query: str) -> str:
    base_query, expanded_terms = split_search_query(query)
    return " ".join([base_query.lower(), *[term.lower() for term in expanded_terms]]).strip()


@lru_cache(maxsize=512)
def detect_query_taxonomy(query: str) -> QueryTaxonomy:
    scope = _scope(query)
    intents: list[str] = []
    families: list[str] = []
    preferred_docsets: list[str] = []
    preferred_sections: list[str] = []
    discouraged_sections: list[str] = []
    seen_docsets: set[str] = set()
    seen_sections: set[str] = set()
    seen_discouraged: set[str] = set()

    for intent, markers in INTENT_MARKERS.items():
        if any(marker in scope for marker in markers):
            intents.append(intent)
    if not intents:
        intents.append("concept")

    for family, rule in FAMILY_RULES.items():
        if not any(marker in scope for marker in rule["markers"]):
            continue
        families.append(family)
        for docset in rule["docsets"]:
            if docset not in seen_docsets:
                seen_docsets.add(docset)
                preferred_docsets.append(docset)
        for marker in rule["sections"]:
            if marker not in seen_sections:
                seen_sections.add(marker)
                preferred_sections.append(marker)

    for intent in intents:
        for marker in INTENT_SECTION_BOOSTS.get(intent, ()):
            if marker not in seen_sections:
                seen_sections.add(marker)
                preferred_sections.append(marker)
        for marker in INTENT_SECTION_PENALTIES.get(intent, ()):
            if marker not in seen_discouraged:
                seen_discouraged.add(marker)
                discouraged_sections.append(marker)

    return QueryTaxonomy(
        intents=tuple(intents),
        families=tuple(families),
        preferred_docsets=tuple(preferred_docsets),
        preferred_section_markers=tuple(preferred_sections),
        discouraged_section_markers=tuple(discouraged_sections),
    )
