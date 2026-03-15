# sas-94-search-API

SAS 9.4 문서 검색 전용 API 저장소다. 이 저장소는 `retrieval-only` 용도로 분리되어 있으며, 사람용 답변 생성이나 Gemini 호출 코드는 포함하지 않는다.

## 포함 범위

- Qdrant dense retrieval
- SQLite FTS lexical retrieval
- Korean SAS term expansion
- hybrid fusion
- search API server
- Python package import interface

## 포함된 데이터

작은 설정/평가 데이터만 포함한다.

- `data/config/sas-ko-en-terms.json`
- `data/eval/sas-rag-smoke.jsonl`
- `data/eval/sas-rag-ko-smoke.jsonl`
- `data/manifests/sas-docs.jsonl`

대용량 데이터는 포함하지 않는다.

- `data/processed/sas-rag/corpus/sas9-pdf-corpus.jsonl`
- `data/processed/sas-rag/search/sas9-pdf-fts.db`
- `data/qdrant/`
- 원본 PDF

이 파일들은 용량이 커서 GitHub 일반 저장소에 같이 올리지 않았다.

## 폴더 구조

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── docs/
│   └── query-taxonomy.md
├── sas94_search_api/
│   ├── __init__.py
│   ├── app.py
│   ├── logging_utils.py
│   ├── rerank.py
│   ├── retrieval.py
│   ├── retrieval_models.py
│   ├── route_index.py
│   ├── scoring.py
│   ├── search_api.py
│   ├── search_service.py
│   ├── storage.py
│   ├── taxonomy.py
│   └── text_utils.py
└── data/
    ├── config/
    ├── eval/
    └── manifests/
```

## 설치 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

또는 다른 시스템에서 바로:

```bash
pip install "sas-94-search-api @ git+https://github.com/kor-noah-han/sas-94-search-API.git@v0.1.4"
```

설치 후 런타임 데이터 받기:

```bash
sas94-search-data --output-dir .
```

환경 변수 예시는 [`.env.example`](/Users/noahhan/dev/sas-94-search-API/.env.example#L1) 에 있다.

## 실행 방법

기본 release bundle은 `lexical-first`로 바로 사용할 수 있다. local Qdrant 파일은 기본 포함되지 않는다.

Qdrant 서버가 있다면:

```bash
export QDRANT_URL=http://localhost:6333
sas94-search-api --url http://localhost:6333 --port 8788
```

Python import 예시:

```python
from sas94_search_api.search_service import run_search
from sas94_search_api.retrieval import RetrievalConfig

result = run_search("PROC MEANS syntax", RetrievalConfig(qdrant_url="http://localhost:6333"))
print(result.retrieval["hits"][0]["section_path_text"])
```

## 데이터 릴리스

코드 패키지와 검색 데이터는 분리한다. 현재 기본 release bundle에 포함되는 런타임 데이터는 아래와 같다.

- `data/processed/sas-rag/search/sas9-pdf-fts.db`
- `data/processed/sas-rag/corpus/sas9-pdf-corpus.jsonl`
- `data/processed/sas-rag/search/sas9-pdf-route-index.json`
- `data/config/sas-ko-en-terms.json`

local Qdrant 파일은 기본 bundle에 포함되지 않는다. dense/local mode까지 함께 배포하려면 `--include-local-qdrant` 옵션으로 번들을 다시 만들어야 한다.

릴리스 번들 생성:

```bash
python3 scripts/release/build_search_data_bundle.py \
  --source-root /path/to/sas94-rag-chat \
  --output-dir dist
```

이 명령은 `dist/sas94-search-data-YYYYMMDD.tar.gz` 와 `.sha256` 파일을 만든다. 생성된 압축 파일은 GitHub Release asset으로 올리는 용도다.

릴리스 번들 다운로드와 압축 해제:

```bash
sas94-search-data --output-dir .
```

특정 태그를 지정하려면:

```bash
sas94-search-data --tag search-data-20260316-v0.1.4 --output-dir .
```

호출 예시:

```bash
curl -s -X POST http://127.0.0.1:8788/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"PROC MEANS syntax","mode":"hybrid","top_k":3}'
```

## 재사용 방식

- 다른 시스템이 이 패키지를 import해서 직접 사용할 수 있다.
- 또는 이 저장소로 서버를 띄운 뒤 다른 시스템은 HTTP API만 호출하면 된다.

## 내부 구조

- `retrieval.py`
  - retrieval orchestration only
- `retrieval_models.py`
  - 상수와 dataclass
- `text_utils.py`
  - query normalization, tokenization, term expansion
- `route_index.py`
  - route index loading and section routing
- `storage.py`
  - Qdrant / SQLite FTS access
- `scoring.py`
  - lexical/hybrid scoring rules
- `taxonomy.py`
  - query intent/family taxonomy and docset/section preferences
- `rerank.py`
  - optional reranker integration

taxonomy 기준은 [`docs/query-taxonomy.md`](/Users/noahhan/dev/sas-94-search-API/docs/query-taxonomy.md#L1) 에 정리했다.

## 주의

이 저장소만으로는 대용량 corpus/FTS/Qdrant 데이터가 자동 포함되지 않는다. 실제 검색 품질을 내려면 기존 빌드 산출물을 별도로 복사하거나, corpus와 인덱스를 다시 생성해야 한다.
