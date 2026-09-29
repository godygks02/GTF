# Ternary GTF 연구 계획

## 1. 연구 아이디어 요약

GTF(Greedy Temporal Fitting 또는 Greedy Temporal Freezing)는 SNN의 시간축을 단순한 반복 계산 과정이 아니라 **순차적으로 잔차(residual)를 보정하는 additive learning 과정**으로 해석한다.

각 시간 스텝은 이전 시간까지의 예측이 설명하지 못한 오차를 새로운 스파이크 조합으로 보정한다. 현재 시간의 스파이크 생성 규칙을 학습한 뒤 이를 동결하고 다음 시간으로 넘어간다.

이 연구에서는 일반적인 이진 스파이크 \(\{0,1\}\) 대신 다음의 3진 스파이크를 사용한다.

\[
s[t]\in\{-1,0,+1\}.
\]

- \(+1\): 현재 예측이 부족한 방향을 보정
- \(-1\): 현재 예측이 과도한 방향을 보정
- \(0\): 해당 뉴런이 현재 보정에 참여하지 않음

핵심 연구 질문은 다음과 같다.

> BPTT와 surrogate gradient 없이, 각 시간 스텝에서 residual을 가장 많이 줄이는 3진 스파이크 패턴과 임계값을 선택하여 SNN을 순차적으로 학습할 수 있는가?

---

## 2. 연구의 정확한 기여 범위

3진 스파이크 자체는 새로운 개념이 아니다. 기존 연구에는 \(\{-1,0,+1\}\) 스파이크, bipolar event, ternary activation을 사용하는 SNN이 존재한다.

따라서 본 연구의 신규성은 다음 조합에서 찾아야 한다.

1. 3진 스파이크를 **temporal residual correction code**로 해석한다.
2. 각 시간 스텝을 독립적인 additive learner 또는 temporal basis로 취급한다.
3. threshold를 surrogate-gradient SGD가 아니라 실제 residual 감소량으로 선택한다.
4. 학습된 과거 시간의 spike generator는 동결한다.
5. 누적된 spike basis의 decoder는 주기적으로 전체 재최적화한다.
6. 샘플별 residual 또는 confidence에 따라 추론을 조기 종료한다.

연구의 중심 주장은 다음처럼 설정한다.

> We interpret ternary spikes as residual-correcting temporal bases and propose a stagewise learning algorithm that selects spike thresholds directly from their empirical loss reduction without differentiating through the spike function.

`완전한 surrogate-free deep SNN 학습`처럼 지나치게 넓은 주장은 피한다. 초기 버전의 정확한 범위는 다음과 같다.

> Surrogate-free ternary spike selection with stagewise temporal residual fitting.

---

## 3. 3진 스파이킹 뉴런

### 3.1 대칭형 ternary neuron

뉴런 \(n\), 시간 \(t\)의 막전위를 \(u_n[t]\), 임계값을 \(\theta_{n,t}>0\)라 하면 스파이크는 다음과 같다.

\[
s_n[t]=
\begin{cases}
+1,&u_n[t]\ge\theta_{n,t},\\
-1,&u_n[t]\le-\theta_{n,t},\\
0,&|u_n[t]|<\theta_{n,t}.
\end{cases}
\]

동등하게 다음과 같이 쓸 수 있다.

\[
s_n[t]
=
\mathbf 1[u_n[t]\ge\theta_{n,t}]
-
\mathbf 1[u_n[t]\le-\theta_{n,t}].
\]

### 3.2 Signed soft reset

기본 막전위 동역학은 다음과 같이 정의할 수 있다.

\[
u_n[t+1]
=
\beta_n u_n[t]
+I_n[t+1]
-\theta_{n,t}s_n[t].
\]

- \(s_n[t]=+1\)이면 막전위에서 \(\theta\)를 뺀다.
- \(s_n[t]=-1\)이면 막전위에 \(\theta\)를 더해 0 방향으로 복원한다.
- \(s_n[t]=0\)이면 reset하지 않는다.

이 방식은 초과 막전위를 제거하지 않고 보존하는 subtractive reset에 해당한다.

### 3.3 비대칭 확장

데이터의 양·음 분포가 대칭이 아닐 경우 별도의 임계값을 사용할 수 있다.

\[
s_n[t]=
\begin{cases}
+1,&u_n[t]\ge\theta^+_{n,t},\\
-1,&u_n[t]\le-\theta^-_{n,t},\\
0,&\text{otherwise}.
\end{cases}
\]

초기 연구에서는 대칭형을 기본으로 사용하고 비대칭형은 ablation 또는 확장 연구로 둔다.

---

## 4. GTF 출력과 residual

훈련 샘플 수를 \(B\), 출력 차원을 \(C\)라고 하자. 시간 \(t\)까지의 누적 출력은 다음과 같다.

\[
\hat{\mathbf Y}_t
=
\hat{\mathbf Y}_{t-1}
+\eta_t\mathbf s_t\mathbf a_t^\top.
\]

- \(\mathbf s_t\in\{-1,0,+1\}^{B}\): 현재 시간의 spike pattern
- \(\mathbf a_t\in\mathbb R^C\): 출력 또는 클래스별 decoder coefficient
- \(\eta_t\in(0,1]\): shrinkage 계수

회귀 문제의 residual은 다음과 같다.

\[
\mathbf R_{t-1}
=
\mathbf Y-\hat{\mathbf Y}_{t-1}.
\]

시간 \(t\)는 \(\mathbf R_{t-1}\)를 가장 잘 근사하는 새로운 ternary spike basis를 추가한다.

---

## 5. 분류 문제의 residual

다중 클래스 분류에서는 label과 logit의 단순 차이보다 cross-entropy loss의 음의 gradient를 사용한다.

현재 누적 logit을 \(\mathbf Z_{t-1}\)라 하면

\[
\mathbf P_{t-1}=\operatorname{softmax}(\mathbf Z_{t-1})
\]

이고, residual은

\[
\mathbf R_{t-1}
=
-\frac{\partial\mathcal L_{\mathrm{CE}}}{\partial\mathbf Z_{t-1}}
=
\mathbf Y_{\mathrm{onehot}}-\mathbf P_{t-1}
\]

로 정의한다.

이에 따라 현재 시간 스텝은

- 정답 클래스의 부족한 logit을 증가시키고,
- 오답 클래스의 과도한 logit을 감소시키며,
- 이미 충분히 맞는 방향에는 spike를 발생시키지 않는

역할을 한다.

---

## 6. Surrogate-free threshold 선택

### 6.1 핵심 원리

기존 문서의

\[
\frac{\partial L}{\partial s_n[t]}
\]

은 spike를 바꾸었을 때 출력이 어느 방향으로 변하는지는 알려 주지만, threshold를 어떻게 바꿔야 spike가 달라지는지는 알려 주지 않는다.

GTF에서는 Heaviside 함수의 gradient를 근사하지 않는다. 대신 threshold 후보가 만들어 내는 실제 spike pattern과 loss 감소량을 직접 평가한다.

### 6.2 후보 spike pattern

뉴런 \(n\)의 batch 전체 막전위를 다음과 같이 둔다.

\[
\mathbf u_n[t]
=
[u_n^{(1)}[t],\ldots,u_n^{(B)}[t]]^\top.
\]

threshold 후보 \(\theta\)에 대한 spike vector는

\[
\mathbf s_n(\theta)
=
\mathbf 1[\mathbf u_n[t]\ge\theta]
-
\mathbf 1[\mathbf u_n[t]\le-\theta]
\]

이다.

### 6.3 최적 decoder의 폐쇄형 해

후보 \((n,\theta)\)가 정해졌을 때 ridge regularization을 포함한 최적 decoder는

\[
\mathbf a^*(n,\theta)
=
\frac{
\mathbf R_{t-1}^\top\mathbf s_n(\theta)
}{
\mathbf s_n(\theta)^\top\mathbf s_n(\theta)+\lambda
}.
\]

### 6.4 선택 목적함수

현재 시간의 뉴런과 threshold는 다음 목적함수를 최소화하도록 선택한다.

\[
(n_t,\theta_t)
=
\arg\min_{n,\theta}
\left[
\left\|
\mathbf R_{t-1}
-\mathbf s_n(\theta)\mathbf a^{*\top}
\right\|_F^2
+\gamma\|\mathbf s_n(\theta)\|_0
\right].
\]

여기서 \(\gamma\)는 spike 발생 비용을 제어한다.

이 선택 과정에서는 spike 함수의 미분이 필요하지 않다.

---

## 7. Threshold 탐색 방법

Threshold가 변하더라도 spike pattern은 threshold가 관측된 membrane value를 통과할 때만 변한다. 따라서 모든 실수를 검색할 필요는 없다.

1. 양·음 membrane potential의 절댓값을 정렬한다.
2. 인접한 값 사이를 threshold 후보로 사용한다.
3. 각 후보가 만드는 ternary spike pattern을 계산한다.
4. decoder의 폐쇄형 해와 residual 감소량을 계산한다.
5. spike cost를 포함해 가장 좋은 후보를 선택한다.

대칭 threshold의 경우 뉴런당 후보 수는 최대 \(O(B)\)이고, 정렬 비용은 \(O(B\log B)\)이다.

모든 뉴런의 ternary 조합 \(3^N\)을 탐색해서는 안 된다. 다음 중 하나를 사용한다.

- 한 번에 뉴런 하나를 선택하는 greedy coordinate search
- 시간당 top-\(k\) 뉴런 선택
- beam search
- signed matching pursuit
- residual과 spike의 정규화된 상관계수 사용

---

## 8. 기본 T-GTF 알고리즘

```text
입력:
    훈련 데이터 X, target Y
    최대 시간 T
    membrane feature generator U(X)
    ridge 계수 lambda
    spike cost gamma
    shrinkage eta

초기화:
    누적 출력 Z_0 = 0
    residual R_0 = Y              # 회귀
    또는 R_0 = Y - softmax(Z_0)   # 분류
    선택된 temporal basis 목록 B = []

for t = 1 ... T:
    각 후보 뉴런 n에 대해:
        batch 전체 membrane potential u_n[t] 계산
        정렬된 membrane 값에서 threshold 후보 생성

        각 threshold theta에 대해:
            s = ternary_threshold(u_n[t], theta)
            a = (R^T s) / (s^T s + lambda)
            score = ||R - s a^T||_F^2 + gamma ||s||_0

    score가 가장 작은 (n*, theta*, a*) 선택
    spike generator (n*, theta*)를 동결
    B에 현재 temporal basis 추가

    Z_t = Z_{t-1} + eta s* a*^T
    residual R_t 갱신

    일정 주기마다:
        지금까지 선택한 모든 basis의 decoder를 ridge regression으로 재계산

    validation 개선이 작거나 residual이 충분히 작으면 종료
```

---

## 9. Orthogonal T-GTF

순수한 greedy freezing은 초기 단계에서 선택한 decoder 크기의 오류를 뒤 단계가 계속 보상해야 하는 문제가 있다.

이를 줄이기 위해 spike 생성 규칙은 동결하되, 지금까지 선택한 spike basis의 decoder는 주기적으로 함께 다시 계산한다.

지금까지 선택된 basis matrix를

\[
\mathbf S_{1:t}
=
[\mathbf s_1,\ldots,\mathbf s_t]
\]

라고 하면 전체 decoder는

\[
\mathbf A_{1:t}^*
=
(\mathbf S_{1:t}^\top\mathbf S_{1:t}+\lambda I)^{-1}
\mathbf S_{1:t}^\top\mathbf Y
\]

로 갱신한다.

이 방식은 Orthogonal Matching Pursuit와 유사하며 다음 장점이 있다.

- 초기 coefficient 오류 수정
- 중복된 temporal basis 감소
- residual이 기존 basis와 가급적 직교하도록 유지
- 순수한 hard freezing보다 안정적인 수렴

권장 기본 모델은 순수 GTF보다 `Orthogonal T-GTF`이다.

---

## 10. Spike amplitude 설계

### 10.1 고정 power-of-two 방식

하드웨어 친화적인 방식은 다음과 같다.

\[
\alpha_t=2^{-t}.
\]

곱셈을 bit shift로 바꿀 수 있지만 학습 자유도가 매우 낮다.

### 10.2 학습 후 양자화 방식

훈련 중에는 amplitude를 자유롭게 계산한다.

\[
\alpha_t^*
=
\frac{
\langle \mathbf R_{t-1},\mathbf W\mathbf s_t\rangle
}{
\|\mathbf W\mathbf s_t\|^2+\lambda
}.
\]

훈련 후 가장 가까운 signed power-of-two로 양자화한다.

\[
\tilde\alpha_t
=
\operatorname{sign}(\alpha_t)
2^{\operatorname{round}(\log_2|\alpha_t|)}.
\]

연구에서는 다음 세 버전을 비교한다.

- Fixed radix GTF: \(2^{-t}\) 고정
- Adaptive GTF: 자유로운 amplitude
- Quantized adaptive GTF: 학습 후 power-of-two 양자화

---

## 11. Encoder 학습 확장

### 11.1 첫 단계에서는 encoder를 고정한다

초기 실험에서는 membrane feature generator를 고정한다.

- 무작위 선형 projection
- 고정된 LIF reservoir
- 사전학습된 ANN feature
- 사전학습된 SNN feature

이렇게 해야 threshold selection과 temporal residual fitting 자체의 효과를 검증할 수 있다.

### 11.2 Alternating optimization

Encoder까지 학습하려면 다음 두 단계를 반복한다.

#### Code step

현재 residual을 가장 잘 줄이는 목표 ternary spike code를 선택한다.

\[
\mathbf s_t^*
=
\arg\min_{\mathbf s\in\{-1,0,+1\}^N}
\left[
\|\mathbf R_{t-1}-\mathbf A_t\mathbf s\|^2
+\gamma\|\mathbf s\|_0
\right].
\]

#### Encoder fitting step

Encoder의 membrane potential이 목표 spike 영역에 들어가게 한다.

\[
\mathcal L_{\mathrm{mem}}(u,s^*)=
\begin{cases}
\max(0,m_+-u),&s^*=+1,\\
\max(0,m_++u),&s^*=-1,\\
\max(0,|u|-m_0),&s^*=0.
\end{cases}
\]

이 방법은 spike step을 미분하지 않는다. 다만 membrane model의 weight는 일반 gradient로 학습할 수 있으므로 이를 `완전한 gradient-free 학습`이라고 부르면 안 된다.

정확한 표현은 다음과 같다.

> Surrogate-free discrete spike selection with differentiable local membrane fitting.

Encoder까지 gradient-free로 만들려면 coordinate descent, zeroth-order optimization 또는 evolutionary search가 필요하다.

---

## 12. Adaptive computation time

각 샘플이 충분히 확신되면 남은 시간 스텝을 실행하지 않는다.

회귀에서는

\[
\tau(x)
=
\min\{t:\|R_t(x)\|<\epsilon\}
\]

를 사용할 수 있다.

분류에서는 가장 높은 두 확률의 차이를 이용할 수 있다.

\[
p_{(1)}(x,t)-p_{(2)}(x,t)>\delta.
\]

이 조건을 만족하면 추론을 종료한다.

이를 통해 쉬운 샘플은 적은 timestep으로, 어려운 샘플은 더 많은 timestep으로 처리한다. GTF의 시간축 학습을 실제 latency 감소로 연결하는 중요한 기능이다.

---

## 13. 이론적으로 주장할 수 있는 것

### 13.1 단계별 훈련 loss 비증가

각 시간 스텝에서 zero spike 후보를 포함하고, 현재 residual을 최소화하는 후보만 추가하면

\[
\|R_t\|_F^2\le\|R_{t-1}\|_F^2
\]

를 보장할 수 있다.

단, sparsity penalty를 포함할 경우에는 전체 regularized objective가 비증가한다고 표현해야 한다.

### 13.2 정확한 0 수렴은 일반적으로 보장되지 않음

Residual이 0으로 수렴하려면 다음 조건이 필요하다.

- 선택 가능한 ternary spike basis가 target 공간을 충분히 span해야 한다.
- neuron dynamics에서 해당 spike pattern이 실제로 도달 가능해야 한다.
- 유한 timestep과 amplitude quantization의 표현 오차가 충분히 작아야 한다.
- greedy freezing의 초기 오류를 뒤 단계가 보정할 표현력을 가져야 한다.

따라서 초기 논문에서는 `exact reconstruction`보다 다음을 목표로 한다.

- monotonic residual reduction
- approximation bound
- 동일 정확도에 필요한 timestep 감소
- spike sparsity와 latency 개선

---

## 14. 기존 연구와의 관계

| 관련 분야 | 기존 원리 | T-GTF의 차별화 후보 |
|---|---|---|
| Ternary SNN | \(-1,0,+1\) spike로 정보량 증가 | ternary spike를 temporal residual correction basis로 사용 |
| Gradient boosting | 새 learner가 이전 residual 또는 loss gradient를 fitting | weak learner가 thresholded ternary spiking neuron |
| Matching pursuit | residual과 가장 잘 맞는 atom을 순차 선택 | dictionary atom이 실제 SNN membrane dynamics에서 생성됨 |
| Residual quantization | 이전 quantization error를 다음 codebook이 보정 | codeword가 시간별 ternary spike pattern |
| OTTT | 시간마다 instantaneous loss와 gradient 계산 | gradient 대신 실제 threshold 후보의 residual 감소량 비교 |
| FE-Learn | 첫 wrong spike time부터 순차 수정 | 목표 spike time보다 누적 실수 출력 residual을 보정 |
| Progressive Tandem Learning | layer를 순차 학습하고 과거 layer를 동결 | 공간 layer가 아니라 temporal basis를 순차 동결 |
| Reservoir/LSM | 고정 spike reservoir와 linear readout | 시간별 threshold와 spike basis를 residual에 따라 선택 |
| LocalZO | zeroth-order 방식으로 SG를 대체 | parameter perturbation보다 유한 threshold 후보의 직접 평가 |

3진 spike 자체나 residual fitting 자체는 신규 기여로 주장하지 않는다. 정확한 조합과 학습 알고리즘, 수렴 특성, 효율성에서 기여를 만들어야 한다.

---

## 15. 최소 구현 로드맵

### Phase 1: Synthetic regression

목표:

- threshold search 구현 검증
- 단계별 residual 감소 확인
- binary와 ternary 비교

권장 문제:

- 1차원 piecewise regression
- \(y=\sin(x)\)
- 2차원 nonlinear surface
- 잡음이 포함된 회귀

구현:

- 고정된 random membrane features
- 대칭 ternary threshold
- closed-form decoder
- pure greedy GTF와 Orthogonal GTF 비교

### Phase 2: 간단한 분류

데이터셋:

- two moons
- XOR
- MNIST
- Fashion-MNIST

추가 기능:

- cross-entropy functional residual
- multi-class decoder
- early exit
- spike sparsity penalty

### Phase 3: 실제 temporal data

데이터셋 후보:

- SHD
- SSC
- DVS Gesture
- CIFAR10-DVS

검증 내용:

- 실제 시간 정보가 있는 데이터에서의 성능
- recurrent membrane state의 효과
- time-specific threshold와 shared threshold 비교

### Phase 4: Encoder 학습

- target ternary code 생성
- membrane margin fitting
- local layer-wise 학습
- SG-BPTT와 정확도·메모리·시간 비교

### Phase 5: 하드웨어 친화적 변환

- amplitude power-of-two 양자화
- signed event routing 비용 반영
- spike 수와 memory access 추정
- CPU/GPU 시뮬레이션과 뉴로모픽 비용을 구분

---

## 16. 필수 baseline

다음 모델을 최소 비교 대상으로 둔다.

1. Binary GTF: \(s\in\{0,1\}\)
2. Ternary GTF: \(s\in\{-1,0,+1\}\)
3. Orthogonal T-GTF
4. Binary SNN + SG-BPTT
5. Ternary SNN + SG-BPTT
6. 고정 SNN reservoir + ridge readout
7. LocalZO 또는 다른 gradient-free SNN
8. 일반 gradient boosting 또는 matching pursuit

---

## 17. 필수 ablation

- ternary spike 대 binary spike
- time-wise freezing 사용/미사용
- decoder 전체 재최적화 사용/미사용
- fixed \(2^{-t}\) amplitude 대 adaptive amplitude
- adaptive amplitude 대 power-of-two 양자화
- 대칭 threshold 대 비대칭 threshold
- sparsity penalty 사용/미사용
- 고정 encoder 대 학습된 encoder
- shrinkage 크기 변화
- early exit 사용/미사용
- 뉴런 하나 선택 대 top-\(k\) 선택
- shared temporal threshold 대 time-specific threshold

---

## 18. 평가 지표

### 성능

- 최종 정확도 또는 회귀 오차
- timestep별 정확도
- timestep별 residual norm
- calibration error
- noise 및 corruption 강건성

### 효율성

- 평균 inference timestep
- 샘플당 positive spike 수
- 샘플당 negative spike 수
- 전체 spike sparsity
- threshold 탐색 시간
- 전체 훈련 시간
- peak training memory
- inference latency
- time-specific parameter 저장 비용

### 하드웨어 관점

- addition/subtraction 수
- signed routing 수
- memory access 추정
- full-precision multiplier 수
- power-of-two 변환 후 shift 연산 수
- 실제 하드웨어 측정과 이론적 연산량을 구분

---

## 19. 성공 기준

최소 논문 수준에서는 다음 중 여러 항목을 충족해야 한다.

1. Ternary GTF가 binary GTF보다 같은 오차를 적은 timestep으로 달성한다.
2. SG-BPTT보다 낮은 peak training memory를 보인다.
3. 각 시간 단계에서 regularized training objective가 비증가한다.
4. 고정 encoder 환경에서도 의미 있는 분류 또는 회귀 성능을 보인다.
5. SG-BPTT와 유사한 정확도에서 spike 수 또는 평균 latency를 줄인다.
6. Orthogonal decoder refitting이 pure freezing의 오류 누적을 완화한다.
7. power-of-two amplitude 양자화 후에도 성능 저하가 제한적이다.
8. 분포가 변할 때 기존 temporal basis를 유지하고 후반 스텝만 추가하여 적응할 수 있다.

---

## 20. 예상 실패 원인

### 표현력 부족

고정 encoder가 만든 membrane feature가 target과 무관하면 threshold를 아무리 선택해도 좋은 spike basis를 만들 수 없다.

### Greedy 오류 누적

초기 스텝에서 잘못 선택한 basis를 완전히 동결하면 뒤 단계가 불필요한 보정을 반복할 수 있다. Orthogonal refitting과 제한적인 backtracking이 필요하다.

### Threshold 과적합

훈련 데이터의 membrane value 사이에서 threshold를 고르면 데이터가 작을 때 과적합될 수 있다. Validation-based stopping, minimum spike support, threshold regularization이 필요하다.

### 시간별 파라미터 증가

각 시간에 별도 threshold를 저장하면 파라미터가 \(O(NT)\)로 증가한다. 다음 대안을 비교해야 한다.

- threshold schedule의 함수화
- layer-wise shared threshold
- low-rank temporal parameterization
- 작은 threshold codebook

### Signed spike 하드웨어 비용

실제 하드웨어에서는 \(-1\) event가 sign bit, inhibitory route 또는 별도의 ON/OFF channel을 요구할 수 있다. 이 비용을 무시하면 ternary 방식의 에너지 이점을 과장하게 된다.

### Discrete search 비용

모든 조합을 탐색하면 \(3^N\)이므로 불가능하다. 뉴런별 greedy search나 제한된 beam search로 설계해야 한다.

---

## 21. 프로젝트 권장 구조

```text
ternary-gtf/
├─ README.md
├─ pyproject.toml
├─ configs/
│  ├─ synthetic.yaml
│  ├─ mnist.yaml
│  └─ dvs_gesture.yaml
├─ src/
│  └─ ternary_gtf/
│     ├─ neurons.py
│     ├─ membrane_features.py
│     ├─ threshold_search.py
│     ├─ decoder.py
│     ├─ gtf.py
│     ├─ losses.py
│     └─ metrics.py
├─ experiments/
│  ├─ synthetic_regression.py
│  ├─ moons_classification.py
│  ├─ mnist.py
│  └─ ablations.py
├─ tests/
│  ├─ test_ternary_neuron.py
│  ├─ test_threshold_search.py
│  ├─ test_closed_form_decoder.py
│  └─ test_monotonic_loss.py
├─ notebooks/
└─ results/
```

---

## 22. 첫 구현의 권장 범위

처음부터 deep convolutional SNN을 만들지 않는다. 첫 milestone은 다음 정도가 적합하다.

### Milestone 1

- 고정 random projection으로 membrane feature 생성
- 대칭형 ternary threshold
- exact threshold candidate search
- closed-form decoder
- pure GTF와 Orthogonal GTF
- synthetic regression 시각화
- binary와 ternary 비교
- 단계별 residual 감소 test

### Milestone 2

- two moons와 MNIST 분류
- functional cross-entropy residual
- sparsity penalty
- adaptive early exit
- SG-BPTT baseline 추가

### Milestone 3

- 실제 LIF dynamics
- temporal dataset
- signed soft reset
- encoder margin fitting
- 하드웨어 친화적 amplitude 양자화

---

## 23. 잠정 논문 제목

- **T-GTF: Greedy Temporal Fitting of Ternary Spiking Neural Networks**
- **Surrogate-Free Temporal Residual Fitting with Ternary Spikes**
- **Ternary Spike Pursuit: Stagewise Temporal Learning Without Spike Derivatives**
- **Orthogonal Temporal Spike Pursuit for Efficient Ternary SNNs**

가장 보수적이고 정확한 이름은 `Ternary Spike Pursuit` 또는 `Ternary Greedy Temporal Fitting`이다.

---

## 24. 참고할 주요 연구

1. Guo et al., **Ternary Spike: Learning Ternary Spikes for Spiking Neural Networks**, AAAI 2024.  
   <https://ojs.aaai.org/index.php/AAAI/article/view/29114>

2. Wang et al., **Ternary Spike-based Neuromorphic Signal Processing System**, 2024.  
   <https://arxiv.org/abs/2407.05310>

3. Xiao et al., **Online Training Through Time for Spiking Neural Networks**, NeurIPS 2022.  
   <https://proceedings.neurips.cc/paper_files/paper/2022/hash/82846e19e6d42ebfd4ace4361def29ae-Abstract-Conference.html>

4. Mukhoty et al., **Direct Training of SNN using Local Zeroth Order Method**, NeurIPS 2023.  
   <https://papers.nips.cc/paper_files/paper/2023/file/3c5e64f26a97db6a2b0bbb788236431e-Paper-Conference.pdf>

5. Wu et al., **Progressive Tandem Learning for Pattern Recognition with Deep Spiking Neural Networks**.  
   <https://arxiv.org/abs/2007.01204>

6. Zhang et al., **First Error-Based Supervised Learning Algorithm for Spiking Neural Networks**, 2019.  
   <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2019.00559/full>

7. Friedman, **Greedy Function Approximation: A Gradient Boosting Machine**, 2001.  
   <https://doi.org/10.1214/aos/1013203451>

8. Huijben et al., **Residual Quantization with Implicit Neural Codebooks**, ICML 2024.  
   <https://proceedings.mlr.press/v235/huijben24a.html>

---

## 25. 최종 요약

이 연구에서 해야 할 일은 단순히 3진 스파이크 모델을 만들고 기존 방식으로 학습하는 것이 아니다.

핵심은 다음과 같다.

\[
\boxed{
\text{ternary spike selection}
+\text{temporal residual fitting}
+\text{time-wise freezing}
+\text{orthogonal decoder refitting}
+\text{adaptive early exit}
}
\]

3진 spike는 이전 예측을 증가시키는 것뿐 아니라 감소시키는 보정도 가능하게 하므로 GTF와 잘 맞는다. 그러나 진정한 연구 기여는 3진 뉴런이 아니라 **실제 residual 감소량을 기준으로 threshold와 spike basis를 순차 선택하는 학습 알고리즘**에 있다.

첫 구현에서는 encoder를 고정하고 threshold search와 closed-form decoder부터 검증한다. 이 단계에서 ternary GTF가 binary GTF보다 빠른 residual 감소, 낮은 timestep, 높은 sparsity를 보인다면 이후 실제 LIF dynamics와 encoder 학습으로 확장한다.
