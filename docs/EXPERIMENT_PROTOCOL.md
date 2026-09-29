# 재현성 및 실험 규약

## Experiment ID

정식 실행은 `YYYYMMDD-HHMM_<task>_<model>_<short-tag>_s<seed>` 형식을 사용한다. 예: `20261003-1430_sine_orthogonal-tgtf_lam1e-3_s42`.

## Run별 필수 파일

`artifacts/runs/<experiment_id>/`에는 다음을 저장한다.

- `config.yaml`: 실제 적용된 최종 설정
- `metadata.json`: timestamp, seed, Python/OS, package versions, Git commit 또는 코드 snapshot 식별자
- `metrics.jsonl`: step/epoch별 machine-readable metric
- `summary.json`: 최종 주요 지표
- `stdout.log`: 실행 로그

plot과 table은 run 내부에서 생성한 뒤, 논문 후보만 각각 `artifacts/plots/`, `artifacts/tables/`에 승격한다.

## 재현성

- Python, NumPy, PyTorch를 사용하는 경우 모두 seed를 고정한다.
- 데이터 split은 생성 seed와 index를 저장한다.
- hyperparameter 선택용 validation과 최종 test를 분리한다.
- test 결과를 보고 설정을 바꾸면 새로운 실험 계열로 기록한다.
- 평균만 보고하지 않고 seed별 원자료와 표준편차를 남긴다.

## 비교의 공정성

- 모델 간 동일한 데이터 split과 metric 구현을 공유한다.
- timestep, spike, parameter, 탐색 비용 중 어느 budget을 맞췄는지 명시한다.
- wall-clock 비교에는 장치, batch size, warm-up, 반복 횟수를 기록한다.
- 실패 run을 삭제하지 않고 실패 원인과 상태를 run index에 기록한다.

## 결과 승격

노트북에서 발견한 결과는 재사용 코드와 config 기반 실험으로 재실행되기 전까지 잠정 결과다. 논문·발표용 수치는 테스트 통과 커밋, 고정 config, seed 집합을 모두 식별할 수 있어야 한다.
