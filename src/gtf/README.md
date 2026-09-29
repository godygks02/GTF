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
