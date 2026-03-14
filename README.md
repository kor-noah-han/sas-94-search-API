# sas-94-search-API

SAS 9.4 문서 검색 전용 API 저장소다. 이 저장소는 `retrieval-only` 용도로 분리되어 있으며, 사람용 답변 생성이나 Gemini 호출 코드는 포함하지 않는다.

## 포함 범위

- Qdrant dense retrieval
- SQLite FTS lexical retrieval
- Korean SAS term expansion
- hybrid fusion
- search CLI
- benchmark CLI
- search API server

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
├── requirements.txt
├── sas_rag/
│   ├── app.py
│   ├── retrieval.py
│   ├── search_api.py
│   └── search_service.py
├── scripts/
│   ├── app/
│   │   ├── benchmark_sas_retrieval.py
│   │   ├── search_sas_qdrant.py
│   │   └── serve_sas_search_api.py
│   ├── index/
│   │   ├── build_sas_fts_index.py
│   │   └── index_sas_qdrant.py
│   ├── _bootstrap.py
│   ├── benchmark_sas_retrieval.py
│   ├── build_sas_fts_index.py
│   ├── index_sas_qdrant.py
│   ├── search_sas_qdrant.py
│   └── serve_sas_search_api.py
└── data/
    ├── config/
    ├── eval/
    └── manifests/
```

## 설치 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

또는 다른 시스템에서 바로:

```bash
pip install "git+https://github.com/kor-noah-han/sas-94-search-API.git"
```

환경 변수 예시는 [`.env.example`](/Users/noahhan/dev/sas-94-search-API/.env.example#L1) 에 있다.

## 실행 방법

Qdrant 서버가 있다면:

```bash
export QDRANT_URL=http://localhost:6333
sas94-search-api --url http://localhost:6333 --port 8788
```

호출 예시:

```bash
curl -s -X POST http://127.0.0.1:8788/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"PROC MEANS syntax","mode":"hybrid","top_k":3}'
```

CLI 예시:

```bash
sas94-search "PROC MEANS syntax" --url http://localhost:6333 --top-k 3 --show-debug
```

```bash
sas94-search-benchmark --queries data/eval/sas-rag-ko-smoke.jsonl --url http://localhost:6333 --mode hybrid
```

## 재사용 방식

- 다른 시스템이 코드를 가져가 직접 실행할 수 있다.
- 또는 이 저장소로 서버를 띄운 뒤 다른 시스템은 HTTP API만 호출하면 된다.

## 주의

이 저장소만으로는 대용량 corpus/FTS/Qdrant 데이터가 자동 포함되지 않는다. 실제 검색 품질을 내려면 기존 빌드 산출물을 별도로 복사하거나, corpus와 인덱스를 다시 생성해야 한다.
