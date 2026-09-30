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

## GTF 대 SG-BPTT

Binary/ternary SG-BPTT와 binary/ternary fixed-encoder GTF를 같은 split에서 비교한다.

```powershell
uv run python experiments/compare_two_moons_gtf_vs_bptt.py
```

빠른 검증에는 `--bptt-epochs 3 --gtf-stages 5 --run-name smoke-gtf`를 사용한다.

`compare_gtf_encoder_strategies.py`는 fixed random encoder, BPTT-pretrained frozen encoder, 큰 random projection pool에서의 GTF encoder basis selection을 비교한다. 마지막 방식은 projection weight의 연속 최적화가 아니라 gradient-free discrete selection이다.

`compare_temporal_gtf_vs_bptt.py`는 실제 recurrent membrane state와 signed soft reset을 갖는 binary/ternary Temporal GTF를 16-step SG-BPTT와 비교한다.
