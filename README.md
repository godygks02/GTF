# GTF Research

GTF(Greedy Temporal Fitting)는 ternary spike를 시간축의 residual-correcting basis로 보고, 각 단계에서 실제 손실 감소량이 가장 큰 spike pattern을 순차적으로 선택하는 학습 방법을 연구하는 프로젝트다.

현재 연구 범위는 **고정된 membrane feature 위에서의 surrogate-free ternary spike selection과 orthogonal decoder refitting**이다. 첫 번째 목표는 synthetic regression에서 알고리즘의 정확성과 단계별 목적함수 비증가를 확인하는 것이다.

## 빠른 시작

이 저장소는 Python 3.11 이상과 `uv` 사용을 권장한다. 현재 기본 의존성은 첫 두 milestone에 필요한 수치 계산, 설정, 시각화 도구만 포함한다.

```powershell
uv sync --extra dev
uv run pytest
uv run python scripts/check_environment.py
```

노트북 환경이 필요하면 다음을 사용한다.

```powershell
uv sync --extra notebook
uv run jupyter lab
```

향후 PyTorch 기반 encoder/SNN 실험은 플랫폼에 맞는 PyTorch 설치 방식이 달라질 수 있으므로 `snn` extra로 분리한다.

```powershell
uv sync --extra dev --extra snn
```

## 저장소 구성

```text
GTF/
├─ configs/          # 재현 가능한 실험 설정
├─ data/             # raw → interim → processed 데이터 단계
├─ docs/             # 연구 정의, 로드맵, 의사결정 기록
├─ experiments/      # 실행 가능한 실험 진입점
├─ notebooks/        # 탐색 분석과 결과 해석
├─ references/       # 논문 메모와 참고문헌 인덱스
├─ scripts/          # 환경·데이터·배치 실행 보조 스크립트
├─ src/gtf/          # 재사용 가능한 핵심 라이브러리
├─ tests/            # 단위·통합·회귀 테스트
└─ artifacts/        # plot, table, log, checkpoint 등 생성물
```

각 디렉터리의 `README.md`에 저장 대상과 운영 규칙이 있다. 전체 실험 순서와 단계별 통과 기준은 [docs/ROADMAP.md](docs/ROADMAP.md)를 따른다. 최초 연구 아이디어는 루트의 [ternary-gtf-research-plan.md](ternary-gtf-research-plan.md)에 보존한다.

## 공통 실행 규칙

- 모든 정식 실험은 `configs/`의 설정 파일 하나로 재현 가능해야 한다.
- 재사용 로직은 `src/gtf/`, 실행 조합은 `experiments/`, 탐색 코드는 `notebooks/`에 둔다.
- raw data는 수정하지 않고 Git에 커밋하지 않는다.
- 결과 경로는 `artifacts/runs/<experiment_id>/`를 사용한다.
- 각 run에는 설정 사본, seed, 환경 정보, 지표, 로그를 남긴다.
- 비교 표에 들어가는 결과는 테스트를 통과한 코드와 고정된 설정에서만 생성한다.

## 문서

- [연구 로드맵](docs/ROADMAP.md)
- [재현성 및 실험 규약](docs/EXPERIMENT_PROTOCOL.md)
- [기존 상세 연구 계획](ternary-gtf-research-plan.md)

## 첫 baseline 실행

PyTorch 의존성을 설치한 뒤 two-moons에서 binary/ternary SG-BPTT 비교를 실행한다.

```powershell
uv sync --extra dev --extra snn
uv run python experiments/compare_two_moons_snn.py
```

결과는 `artifacts/runs/`, checkpoint는 `artifacts/checkpoints/`, 비교 그림은 `artifacts/plots/`에 생성되며 Git에는 포함되지 않는다.

GTF까지 포함한 4-way 비교는 다음과 같이 실행한다.

```powershell
uv run python experiments/compare_two_moons_gtf_vs_bptt.py
```

이 첫 GTF는 고정 random membrane encoder를 사용하며, binary/ternary spike basis를 functional cross-entropy residual에 순차 fitting한다.

Encoder 전략 비교는 fixed random, SG-BPTT pretrained, GTF projection-pool selection을 함께 실행한다.

```powershell
uv run python experiments/compare_gtf_encoder_strategies.py
```

실제 recurrent membrane과 signed soft reset을 사용하는 Temporal GTF 비교는 다음과 같이 실행한다.

```powershell
uv run python experiments/compare_temporal_gtf_vs_bptt.py
```
