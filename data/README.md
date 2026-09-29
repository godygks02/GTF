# Data

데이터 수명주기를 `raw → interim → processed`로 구분한다. 외부 프로젝트가 만든 특징이나 파생 데이터는 `external`에 둔다.

실제 데이터 파일은 기본적으로 Git에 커밋하지 않는다. 각 데이터셋의 출처, 라이선스, 다운로드 날짜, checksum, 전처리 config를 기록해야 한다.
