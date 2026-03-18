# stac-optimizer

[English README](README.md)

STAC는 SignSGD Trunk, AdamW Cap의 약자입니다.

대부분의 레이어는 sign 기반 업데이트를 사용하고, 마지막 `N`개 학습 레이어는
AdamW를 사용하는 PyTorch 옵티마이저입니다. 기본 trunk는 단순한 `sign(grad)`가
아니라 `sign(EMA(grad))`를 사용해서 plain signSGD보다 더 안정적인 학습을
노립니다.

| 항목 | 값 |
| --- | --- |
| Python | `>=3.13` |
| PyTorch | `>=2.10` |
| 기본 분할 | 마지막 `1`개 학습 레이어만 AdamW |
| trunk | decoupled weight decay + `sign(EMA(grad))` |
| cap | AdamW + decoupled weight decay |
| 검증 | CUDA 테스트 및 CUDA 벤치마크 포함 |

## 핵심 특징

- `model.named_modules()` 순서로 레이어를 결정적으로 분할합니다.
- 마지막 `N`개 레이어만 AdamW로 올리고 나머지는 sign 기반 trunk로 유지합니다.
- 적응형 moment를 cap에만 유지해서 전체 AdamW보다 optimizer state 메모리를
  더 적게 사용합니다.
- `optimizer.partition`으로 실제 분할 결과를 바로 확인할 수 있습니다.
- sparse gradient, 동적 `add_param_group()`를 명시적으로 거부합니다.
- `error_if_nonfinite=False`일 때는 non-finite dense gradient가 보이면
  해당 step 전체를 건너뜁니다.
- 체크포인트 로드 시 layer name, parameter name, state shape까지 검증합니다.

## 분할 구조

```mermaid
flowchart LR
    A[등록 순서의 학습 레이어] --> B[앞부분 레이어]
    A --> C[마지막 N개 레이어]
    B --> D[trunk<br/>EMA(grad) -> sign]
    C --> E[cap<br/>AdamW]
```

## 설치

PyPI 설치:

```bash
python -m pip install stac-optimizer
```

개발용 설치:

```bash
python -m pip install -e ".[dev]"
```

## 빠른 사용 예시

```python
import torch
from torch import nn

from stac_optimizer import STAC


model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)

optimizer = STAC(
    model,
    lr=1e-3,
    last_n_layers=1,
    trunk_momentum=0.9,
    weight_decay=1e-2,
)

loss = torch.nn.functional.mse_loss(
    model(torch.randn(8, 128)),
    torch.randn(8, 10),
)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)

print(optimizer.partition.trunk_layer_names)
print(optimizer.partition.cap_layer_names)
```

## 공개 API

- `STAC`
- `partition_trainable_layers(model, last_n_layers=1)`
- `LayerGroup`
- `STACPartition`

`STACPartition`은 다음 정보를 제공합니다.

- `trunk_layer_names`, `cap_layer_names`
- `trunk_parameter_names`, `cap_parameter_names`
- `trunk_parameters`, `cap_parameters`

## 안정성 관련 메모

- trunk는 momentum을 먼저 누적한 뒤 sign을 취합니다.
- cap은 AdamW의 decoupled weight decay를 사용합니다.
- `last_n_layers`, `trunk_lr`, `cap_lr`를 분리해서 조정할 수 있습니다.

튜닝 팁:

- 학습이 흔들리면 `trunk_lr`보다 `trunk_momentum`을 먼저 올리세요.
- 헤드 적응이 느리면 `cap_lr`를 올리세요.
- 후반 레이어 적응이 더 필요하면 `last_n_layers`를 늘리세요.

## CUDA 벤치마크

[`examples/toy_benchmark.py`](examples/toy_benchmark.py)는 CUDA에서 회귀와 분류
두 작업을 여러 시드로 비교합니다.

비교 대상:

- `STAC default (cap=1)`
- `STAC plain sign trunk`
- `STAC wider cap (cap=2)`
- `AdamW baseline`

실행:

```bash
python examples/toy_benchmark.py --device cuda --seeds 5 --steps 150
```

`2026-03-18` 로컬 CUDA 검증 스냅샷 (`torch 2.10.0+cu126`, `RTX 3070`):

| 작업 | STAC 기본 | plain sign trunk | 넓은 cap (`last_n_layers=2`) | AdamW |
| --- | ---: | ---: | ---: | ---: |
| 회귀 평균 loss | `0.075852` | `0.140853` | `0.077104` | `0.118262` |
| 분류 평균 loss | `0.006573` | `0.022765` | `0.011192` | `0.017693` |

같은 장비에서 deeper memory probe로 측정한 optimizer state 메모리 스냅샷:

| 옵티마이저 | Optimizer state MB |
| --- | ---: |
| `STAC` 기본 | `3.637` |
| `STAC` plain sign trunk | `0.004` |
| `STAC` 넓은 cap | `3.762` |
| `AdamW` | `7.270` |

이 벤치마크는 재현 가능한 sanity check이며, 모든 문제에서의 절대 순위를
보장하는 리더보드는 아닙니다. 초점은 최적화 품질과 optimizer state 메모리이며,
모든 환경에서의 절대적인 wall-clock 속도 우위를 주장하지 않습니다.

## 검증

로컬 CUDA 검증 명령:

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
python examples/toy_benchmark.py --device cuda --seeds 5 --steps 150
```

`2026-03-18` 기준 최근 로컬 CUDA 실행 결과:

- `python -m pytest -q`: `28 passed`
- `python examples/toy_benchmark.py --device cuda --seeds 5 --steps 150`:
  위 표와 동일

테스트가 다루는 항목:

- 분할 규칙과 경계 조건
- cap의 AdamW 동작 일치성
- sparse/non-finite gradient 보호
- state_dict round-trip 및 mismatch 검증
- CUDA에서 default trunk가 plain signSGD보다 안정적인지 확인
- CUDA 통합 테스트에서 AdamW 대비 경쟁력과 낮은 optimizer state 사용량 확인

## 릴리즈

이 저장소는 `setuptools-scm`을 사용합니다. changelog는 저장소 파일이 아니라
GitHub Release에만 남깁니다.

일반적인 릴리즈 흐름:

```bash
git push origin main
git tag vX.Y.Z
git push origin vX.Y.Z
```

태그 워크플로는 버전 검증, 빌드, `twine check`, PyPI 업로드, GitHub Release 생성을
수행합니다.
