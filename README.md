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
| `scripts/patch_physicsnemo.py` | physicsnemo 2.2.2의 `VTKFileReader` 버그 패치 (아래 "꼭 필요한 두 가지" 참고) |
| `docs/domino_build_guide.html` | 환경 설치부터 로컬 배포까지, 전체 과정을 처음부터 끝까지 직접 구축하는 11단계 가이드 |

## 전제 조건

- NVIDIA GPU (학습 시 VRAM 약 36GB 사용 실측 → 48GB급 권장, 추론은 1회 약 5GB)
- Linux 또는 Windows + WSL2 (PhysicsNeMo/warp-lang의 공식 지원 플랫폼은 Linux)
- 실제로 검증한 버전:

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install nvidia-physicsnemo==2.2.2
pip install cuml-cu13==26.8.0 --extra-index-url=https://pypi.nvidia.com
python scripts/patch_physicsnemo.py
```

### 꼭 필요한 두 가지 (빠뜨리면 추론이 실패함)

1. **physicsnemo 2.2.2 패치 — `scripts/patch_physicsnemo.py`**
   physicsnemo 2.2.2의 `VTKFileReader`에 `read_file_attributes`가 빠져 있어서, 그대로 설치하면
   STL 추론이 바로 실패합니다.
   ```
   TypeError: Can't instantiate abstract class VTKFileReader without an implementation
   for abstract method 'read_file_attributes'
   ```
   설치한 환경에서 `python scripts/patch_physicsnemo.py`를 한 번 실행하면 고쳐집니다. 원본은
   `cae_dataset.py.orig`로 남고, 여러 번 실행해도 안전합니다.

2. **cuML 설치 — `cuml-cu13`**
   physicsnemo는 최근접 이웃(kNN) 검색에 GPU에서는 cuML을 우선 쓰고, 없으면 **모든 점 쌍의 거리를
   한 번에 계산하는 PyTorch 구현**으로 대체합니다. 이 대체 구현은 메모리를 크게 써서, 실측 결과
   12GB GPU(RTX 5070 Ti Laptop)에서 추론 중 `CUDA out of memory`로 멈췄습니다. cuML을 설치하면
   같은 GPU에서 정상 동작합니다. 실제 학습/추론 서버(A6000)도 cuML이 설치된 상태였습니다.

자세한 환경 설치 과정(드라이버, PyTorch, venv 등)은 `docs/domino_build_guide.html`의 1단계를 참고하세요.

## 라이선스

Apache License 2.0. NVIDIA PhysicsNeMo의 수정본이며, 원본 저작권 고지와 변경 사항은
[`NOTICE`](./NOTICE)에 명시돼 있습니다.
