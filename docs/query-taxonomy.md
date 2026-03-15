# Query Taxonomy

`sas-94-search-api`는 검색 질의를 두 축으로 분류한다.

- `intent`
  - 질문이 무엇을 원하는가
- `family`
  - 질문이 어떤 SAS 영역을 가리키는가

이 taxonomy는 lexical ranking, docset preference, section preference, dense skip 판단에 사용된다.

## Intents

- `howto`
  - 예: `어떻게`, `방법`, `하는 법`, `사용`
  - 선호 섹션: `syntax`, `usage`, `statement`, `examples`
  - 패널티 섹션: `definitions`, `concepts`, `introduction`
- `syntax`
  - 예: `syntax`, `구문`, `문법`, `statement`, `option`
  - 선호 섹션: `syntax`, `statement`, `procedure`
- `example`
  - 예: `example`, `예제`, `sample`
  - 선호 섹션: `examples`, `sample`
- `definition`
  - 예: `what is`, `뭐야`, `정의`, `의미`
  - 선호 섹션: `definition`, `concepts`, `overview`
- `comparison`
  - 예: `비교`, `차이`, `difference`, `compare`
  - 선호 섹션: `comparison`, `comparing`, `differences`, `methods`
- `troubleshooting`
  - 예: `error`, `오류`, `경고`, `debug`, `log`
  - 선호 섹션: `error`, `message`, `warnings`, `troubleshooting`

## Families

- `library`
  - preferred docset: `lepg`
- `data_step`
  - preferred docset: `lepg`
- `macro`
  - preferred docset: `mcrolref`
- `graphics`
  - preferred docset: `statug`, `imlug`
- `statistics`
  - preferred docset: `procstat`, `statug`
- `correlation`
  - preferred docset: `procstat`
- `modeling`
  - preferred docset: `procstat`, `statug`
- `sql`
  - preferred docset: `proc`, `lepg`
- `reporting`
  - preferred docset: `proc`
- `import_export`
  - preferred docset: `proc`, `lepg`
- `sorting`
  - preferred docset: `proc`
- `transpose`
  - preferred docset: `proc`
- `formats`
  - preferred docset: `proc`, `lepg`
- `ods`
  - preferred docset: `statug`, `proc`

## Ranking Rules

- preferred docset이면 가산점
- preferred section marker가 title/section path에 있으면 가산점
- `howto`인데 `definition/introduction` 성격이면 감점
- `graphics`, `macro`, `library`, `data_step`, `statistics` 계열은 family별 docset bonus 추가
- taxonomy marker가 충분히 맞으면 dense를 생략하고 lexical만으로 종료할 수 있음

## Goal

taxonomy의 목적은 두 가지다.

- broad 한국어 질문을 SAS 문서 구조에 더 잘 맞추기
- lexical-first 전략이 처리할 수 있는 질문 범위를 넓히기
