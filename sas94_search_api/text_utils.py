from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from sas94_search_api.retrieval_models import ENV_PATH, RetrievalConfig


TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_#./-]{2,}")
HANGUL_RE = re.compile(r"[가-힣]")
PROC_RE = re.compile(r"\bproc\s+([a-z0-9_]+)\b", re.IGNORECASE)
EN_KO_PARTICLE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)(은|는|이|가|을|를|와|과|도|로|으로|에|에서|의)\b")


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def env_default(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def normalize_query_text(text: str) -> str:
    return EN_KO_PARTICLE_RE.sub(r"\1", text)


def split_search_query(query_text: str) -> tuple[str, list[str]]:
    marker = "\nSAS search terms:"
    if marker not in query_text:
        return query_text.strip(), []
    base, _, tail = query_text.partition(marker)
    terms = [part.strip() for part in tail.split(",") if part.strip()]
    return base.strip(), terms


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def has_hangul(text: str) -> bool:
    return bool(HANGUL_RE.search(text))


def extract_proc_names(text: str) -> list[str]:
    return [match.group(1).lower() for match in PROC_RE.finditer(text)]


@lru_cache(maxsize=4)
def load_term_dictionary(path_str: str) -> list[dict[str, object]]:
    path = Path(path_str)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def expand_query(query: str, config: RetrievalConfig) -> tuple[str, list[str]]:
    query = normalize_query_text(query)
    if not config.enable_term_expansion or not has_hangul(query):
        return query, []

    dictionary = load_term_dictionary(config.term_dictionary_path)
    lowered = query.lower()
    expansions: list[str] = []
    seen: set[str] = set()

    for entry in dictionary:
        matches = [str(item).lower() for item in entry.get("match", [])]
        if not any(term in lowered for term in matches):
            continue
        for item in entry.get("expand", []):
            term = str(item).strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            expansions.append(term)

    if not expansions:
        return query, []

    return f"{query}\nSAS search terms: " + ", ".join(expansions), expansions
