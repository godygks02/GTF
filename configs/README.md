# Configs

재현 가능한 실험 설정을 YAML로 저장한다.

## 규칙

- 하나의 정식 실험은 하나의 완전한 config로 실행 가능해야 한다.
- 코드 기본값에 의존하지 말고 결과에 영향을 주는 값을 명시한다.
- 비밀값, 로컬 절대 경로, 생성 결과는 넣지 않는다.
- 공통값은 `base.yaml`, 실험별 override는 목적을 드러내는 이름을 사용한다.
- config 변경으로 의미가 달라지면 기존 파일을 덮지 말고 새 파일을 만든다.

`two_moons_sg_bptt.yaml`은 binary/ternary SNN의 공정한 SG-BPTT 비교 설정이다.

`two_moons_gtf_vs_bptt.yaml`은 두 SG-BPTT baseline과 fixed-encoder binary/ternary GTF를 함께 비교한다.

`two_moons_gtf_encoder_strategies.yaml`은 fixed, BPTT-pretrained, GTF projection-pool encoder 전략을 비교한다.

`two_moons_temporal_gtf.yaml`은 동일한 물리적 timestep 제한에서 recurrent Temporal GTF와 SG-BPTT를 비교한다.
