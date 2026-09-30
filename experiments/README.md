# Experiments

설정 파일을 받아 전체 실험을 실행하는 얇은 진입점을 둔다.

## 규칙

- 알고리즘 구현을 이 디렉터리에 복사하지 않고 `gtf` 패키지를 호출한다.
- 실행 시 최종 config, seed, 환경 정보, 지표를 run 디렉터리에 저장한다.
- 파일명은 task 중심으로 짓는다. 예: `synthetic_regression.py`, `static_classification.py`.
- 빠른 smoke mode와 정식 반복 실행 모드를 구분한다.
- 대규모 sweep 정의는 `configs/`에 두고 결과는 `artifacts/runs/`에 쓴다.

## Two moons SG-BPTT 비교

Binary/ternary SNN을 같은 데이터와 설정으로 학습하고 best/last checkpoint, history, 요약, 비교 그림을 생성한다.

```powershell
uv run python experiments/compare_two_moons_snn.py
```

빠른 동작 확인은 `--epochs 3 --run-name smoke-test`처럼 실행한다.
