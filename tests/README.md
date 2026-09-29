# Tests

수학적 불변식과 연구 파이프라인의 회귀를 검증한다.

## 규칙

- 단위 테스트는 빠르고 seed에 안정적이어야 한다.
- 핵심 불변식(ternary 범위, threshold 후보 완전성, closed-form 해, objective 비증가)을 우선한다.
- dataset download나 GPU가 필요한 테스트는 별도 marker로 분리한다.
- 논문 수치를 exact value로 고정하지 말고 의미 있는 허용 오차와 성질을 검증한다.
- 버그 수정에는 가능하면 재현 테스트를 먼저 추가한다.
