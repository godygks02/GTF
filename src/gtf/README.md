# `gtf` Package

GTF의 재사용 가능한 구현을 저장한다.

예정 모듈:

- `neurons.py`: binary/ternary threshold와 signed reset
- `features.py`: random projection, reservoir, pretrained feature adapter
- `threshold_search.py`: exact candidate generation과 greedy scoring
- `decoder.py`: closed-form ridge 및 orthogonal refitting
- `model.py`: stagewise GTF orchestration
- `losses.py`: regression/functional classification residual
- `metrics.py`: 성능, sparsity, latency 지표
- `io.py`: config, run metadata, artifact 저장

모듈은 위 순서에 얽매이지 않지만 알고리즘, 데이터셋, CLI를 한 파일에 섞지 않는다.

현재 `data.py`, `snn.py`, `training.py`에 two-moons SG-BPTT baseline을 위한 데이터 분할, binary/ternary neuron dynamics, 학습·평가·checkpoint 로직이 구현되어 있다.

`gtf.py`에는 고정 random membrane feature에서 binary/ternary threshold atom을 만들고 functional residual에 stagewise fitting하는 첫 GTF 분류기가 구현되어 있다.

`temporal_gtf.py`는 membrane recurrence와 signed soft reset을 실제 timestep 사이에 전파하며 timestep마다 residual-correcting basis 하나를 선택한다.
