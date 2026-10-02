# DoMINO CFD 예측 파이프라인 (교육자료)

[NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)의 DoMINO 모델로 자동차 형상(STL)의
표면 압력/벽전단응력을 예측하는 파이프라인입니다. 공식 예제를 그대로 쓰면 실제 배포 단계에서
겪게 되는 두 가지 버그(법선 부호 불일치, STL-CFD 메시 해상도 불일치)를 고치고, 장시간 무인 학습을
위한 체크포인트 보존 로직을 추가한 버전입니다.

데이터셋은 [AhmedML](https://caemldatasets.org/ahmedml/)(Ashton et al., 2024,
[arXiv:2407.20801](https://arxiv.org/abs/2407.20801))을 사용합니다 — Ahmed body(1984년
표준 자동차 공력 벤치마크 형상)를 500가지로 변형해 OpenFOAM으로 해석한 CFD 데이터셋입니다.

## 왜 이걸 고쳤는가

- **`predict_on_stl.py`**: 공식 예제(`inference_on_stl.py`)는 사용자가 올린 STL 위에서 바로
  추론할 때 두 가지 문제가 있었습니다.
  1. STL 파일 자체의 법선 방향이 CFD 메시 쪽 법선과 반대(내적이 거의 −1)라서, 모델이 "안쪽"과
     "바깥쪽"을 헷갈린 채로 예측했습니다.
  2. 사용자 STL이 CFD 경계 메시보다 거칠면(면 개수가 적으면) 면적이 부정확하게 계산돼, 예측값이
     실제보다 많이 틀어졌습니다.

  두 버그를 실측 데이터로 검증하고 고친 뒤, R²가 −0.31 → 0.99까지 바뀐 과정을 `docs/domino_build_guide.html`에
  정리해뒀습니다.

- **`train.py`**: 공식 예제는 best validation loss를 로그에만 출력하고 저장은 안 해서, 장시간
  학습 후 실제로 배포 가능한 "가장 좋은" 체크포인트가 남아있지 않는 문제가 있었습니다. 체크포인트
  저장/정리 로직을 추가했습니다.

구체적으로 어떤 줄을 왜 바꿨는지는 [`NOTICE`](./NOTICE) 파일에 정리했습니다.

## 구성

| 파일 | 설명 |
|---|---|
| `predict_on_stl.py` | STL 입력 → 표면/체적 예측 → VTP/VTI 저장 (핵심 수정 파일) |
| `train.py` | DoMINO 학습 루프 (체크포인트 보존 로직 추가) |
| `compute_statistics.py` | 정규화 통계 계산 (공식 예제, 수정 없음) |
| `loss.py` | 손실 함수 정의 — 포인트별 손실 + 드래그/리프트 적분 손실 (공식 예제, 수정 없음) |
| `utils.py` | 공통 유틸리티 (공식 예제, 수정 없음) |
| `configs/real_train_500.yaml` | 500-case 학습 설정 예시 |
| `scripts/run_prediction.sh` | 원격 추론 요청 스크립트 예시 (Kit-CAE 연동용) |
| `docs/domino_build_guide.html` | 환경 설치부터 로컬 배포까지, 전체 과정을 처음부터 끝까지 직접 구축하는 11단계 가이드 |

## 전제 조건

- NVIDIA GPU (학습은 VRAM 24GB+ 권장, 추론은 8GB 내외로도 가능)
- Linux (PhysicsNeMo/warp-lang의 공식 지원 플랫폼)
- `pip install nvidia-physicsnemo`

자세한 환경 설치 과정(드라이버, PyTorch, venv 등)은 `docs/domino_build_guide.html`의 1단계를 참고하세요.

## 라이선스

Apache License 2.0. NVIDIA PhysicsNeMo의 수정본이며, 원본 저작권 고지와 변경 사항은
[`NOTICE`](./NOTICE)에 명시돼 있습니다.
