# Processed Data

모델 입력 직전의 확정 데이터와 split index를 저장한다.

- shape, dtype, class mapping, normalization을 metadata에 기록한다.
- train/validation/test split 생성 seed와 index를 보존한다.
- 전처리 버전이 바뀌면 기존 파일을 덮지 않고 버전 디렉터리를 만든다.
