# GTF 실험 로드맵

## 1. 연구 목표와 검증 순서

핵심 가설은 ternary spike가 binary spike보다 residual의 양·음 방향을 직접 보정하여, 같은 오차 또는 정확도에 더 적은 timestep과 spike를 요구한다는 것이다. 이를 한 번에 deep SNN으로 검증하지 않고 아래 순서로 위험을 분리한다.

1. threshold search와 closed-form decoder가 수학적으로 맞는지 검증한다.
2. 고정 encoder에서 ternary temporal fitting의 효과를 분리해 측정한다.
3. 분류 손실, sparsity, early exit를 추가한다.
4. 실제 temporal dynamics와 encoder 학습으로 확장한다.
5. 양자화 및 하드웨어 비용까지 포함해 효율성 주장을 검증한다.

각 단계는 `구현 → 단위 검증 → 작은 실험 → 반복 실험 → 분석 → 통과/중단 결정` 순서로 진행한다.

## 2. Stage 0 — 재현 가능한 기반 구축

목표는 모든 실험이 설정 파일과 seed로 재실행되는 최소 기반을 만드는 것이다.

작업:

- config loader, seed 고정, run directory 생성
- 환경 정보와 최종 config 자동 저장
- 공통 metric logger 및 plot/table 저장 규칙 구현
- CI에서 lint, unit test, smoke test 실행
- 연구 주장과 지표의 대응표 작성

완료 기준:

- 동일 환경·seed에서 synthetic smoke test 지표가 허용 오차 내에서 재현된다.
- 각 run 폴더만으로 코드 버전, 설정, seed, 핵심 결과를 추적할 수 있다.

## 3. Stage 1 — Synthetic regression 최소 구현

목표는 GTF 핵심 연산의 정확성과 residual 감소 특성을 검증하는 것이다.

구현 순서:

1. random linear membrane feature generator
2. symmetric ternary threshold와 signed spike 생성
3. 관측된 `|u|` 기반 exact threshold candidate 생성
4. ridge closed-form coefficient와 후보 score 계산
5. pure greedy GTF
6. Orthogonal T-GTF decoder refitting
7. binary GTF baseline

데이터셋:

- 1D piecewise function
- `sin(x)` + noise
- 2D nonlinear surface

필수 테스트:

- spike 값이 항상 `{-1, 0, +1}`에 속함
- threshold 후보 사이에서는 spike pattern이 바뀌지 않음
- closed-form decoder가 수치 최적화 결과와 일치함
- zero candidate를 포함할 때 regularized objective가 단계별 비증가함
- orthogonal refitting 결과가 같은 basis의 pure greedy fitting보다 나쁘지 않음

핵심 비교:

- Binary GTF vs Ternary GTF
- Pure GTF vs Orthogonal T-GTF
- adaptive amplitude vs fixed amplitude
- sparsity penalty sweep

완료 기준:

- 최소 5개 seed에서 objective 비증가 위반이 없다.
- 적어도 2개 synthetic task에서 ternary 방식의 timestep-error 또는 spike-error 곡선이 binary 대비 개선된다.
- 개선이 없다면 encoder feature 분포, sign balance, threshold support를 우선 진단하고 다음 단계로 확장하지 않는다.

## 4. Stage 2 — 정적 분류와 early exit

목표는 functional gradient residual을 사용하는 다중 클래스 분류를 검증하는 것이다.

순서:

1. two moons와 XOR로 logit update 검증
2. cross-entropy residual `Y - softmax(Z)` 구현
3. validation stopping 및 calibration metric 추가
4. confidence margin 기반 sample-wise early exit 추가
5. MNIST, Fashion-MNIST로 확장

필수 baseline:

- binary/ternary GTF
- orthogonal ternary GTF
- 고정 reservoir + ridge readout
- gradient boosting 또는 matching pursuit
- SG-BPTT binary/ternary SNN은 비교 조건을 맞출 수 있을 때 추가

완료 기준:

- 분류 정확도뿐 아니라 평균 timestep, positive/negative spike 수, calibration을 함께 보고한다.
- early exit가 고정 timestep 대비 정확도 저하 허용 범위 내에서 평균 timestep을 줄인다.
- 모든 주 비교에 동일 split과 seed set을 사용한다.

## 5. Stage 3 — 실제 temporal data와 LIF dynamics

목표는 GTF의 시간축 해석이 실제 이벤트·음성 데이터에서도 유효한지 확인하는 것이다.

후보 데이터셋은 SHD/SSC를 먼저 사용하고, 이후 DVS Gesture 또는 CIFAR10-DVS로 확장한다. 작은 데이터셋과 짧은 시퀀스로 파이프라인을 먼저 검증한다.

추가 구현:

- recurrent membrane state와 signed soft reset
- time-specific vs shared threshold
- state 초기화 및 sequence masking
- temporal batching과 길이별 early exit

완료 기준:

- 누적 성능 대 timestep 곡선과 실제 wall-clock latency를 분리해 보고한다.
- recurrent state를 제거한 대조군과 비교하여 temporal dynamics의 기여를 확인한다.
- threshold 저장 비용과 탐색 비용을 함께 기록한다.

## 6. Stage 4 — Encoder 학습

목표는 고정 feature의 표현력 한계를 넘되, spike selection 자체는 surrogate-free로 유지하는 것이다.

비교할 방법:

- 고정 random/reservoir encoder
- 사전학습 ANN feature
- target ternary code + membrane margin fitting
- 제한적인 alternating optimization
- SG-BPTT baseline

완료 기준:

- `surrogate-free spike selection`과 `gradient-free 전체 학습`을 명확히 구분한다.
- 정확도, peak training memory, 훈련 시간, 수렴 안정성을 같은 조건에서 비교한다.
- encoder 학습 이득이 seed에 안정적이고 고정 encoder 결과를 유의미하게 넘는다.

## 7. Stage 5 — 양자화와 하드웨어 관점 평가

목표는 알고리즘 지표를 실제 구현 비용에 가까운 지표로 연결하는 것이다.

비교:

- adaptive amplitude
- fixed `2^-t` amplitude
- 학습 후 signed power-of-two quantization
- full-precision multiply vs shift/add 추정

보고 항목:

- 양·음 spike와 routing 수
- addition/subtraction, multiplier, shift 수
- memory access와 time-specific parameter 저장량
- CPU/GPU latency와 이론적 neuromorphic 비용의 구분

완료 기준:

- 양자화 후 정확도/오차 저하와 연산 절감의 Pareto curve를 제시한다.
- signed event의 추가 비용을 포함하고도 이점이 유지되는 조건을 명시한다.

## 8. 공통 실험 매트릭스

모든 핵심 결과는 최소 5개 seed의 평균과 표준편차를 기록한다. 계산 비용이 큰 단계에서는 pilot 3 seeds로 후보를 좁힌 뒤 최종 설정만 5개 이상으로 확장한다.

주요 독립 변수:

- spike alphabet: binary / ternary
- decoder update: greedy / orthogonal refit
- threshold: symmetric / asymmetric, shared / time-specific
- amplitude: fixed / adaptive / quantized
- selection width: single neuron / top-k
- shrinkage, ridge, sparsity penalty
- early exit on/off

주요 종속 변수:

- regression error 또는 accuracy
- timestep별 residual norm/loss
- 평균/분위수 inference timestep
- positive/negative/total spike 수
- peak memory, training time, inference latency
- calibration과 corruption robustness

## 9. 실험 우선순위

가장 먼저 구현할 실험은 `synthetic_sine_binary_vs_ternary`다. 이것이 통과하면 orthogonal refitting과 sparsity sweep을 추가한다. 첫 결과가 안정화되기 전에는 MNIST, deep encoder, neuromorphic energy estimate를 시작하지 않는다. 이 순서는 알고리즘 오류와 표현력 부족을 구분하고, 가장 싼 실험에서 핵심 가설을 반증할 기회를 준다.

## 10. 예상 산출물

- 알고리즘: ternary threshold search, GTF, orthogonal refitting, early exit
- 검증: 수학적 불변식 단위 테스트와 작은 end-to-end test
- 그림: residual-vs-timestep, accuracy-vs-timestep, spike-error Pareto, early-exit 분포
- 표: baseline 비교, ablation, 자원 사용량
- 문서: 실패 실험을 포함한 run index와 연구 의사결정 기록
