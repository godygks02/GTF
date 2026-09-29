# Scripts

환경 확인, 데이터 준비, 여러 seed 실행, 결과 집계 같은 반복 작업을 자동화한다.

## 규칙

- 연구 로직은 넣지 않고 `src/gtf/` 또는 `experiments/`를 호출한다.
- 여러 번 실행해도 안전하도록 idempotent하게 작성한다.
- 삭제·덮어쓰기가 있으면 대상과 동작을 도움말에 명확히 적는다.
- 로컬 절대 경로를 하드코딩하지 않는다.
- PowerShell은 `.ps1`, 범용 Python 보조 도구는 `.py`를 사용한다.

현재 `check_environment.py`는 Python 버전과 핵심 패키지 import 가능 여부를 확인한다.
