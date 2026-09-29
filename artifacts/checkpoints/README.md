# Checkpoints

선택된 spike basis, threshold, decoder coefficient, encoder state를 저장한다.

- checkpoint schema와 코드 버전을 metadata에 기록한다.
- 파일명에 run ID와 timestep/epoch를 포함한다.
- 재개용 checkpoint와 배포/평가용 checkpoint를 구분한다.
