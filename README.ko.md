# stac-optimizer

[English README](https://github.com/smturtle2/stac-optimizer/blob/main/README.md)

STAC는 SignSGD Trunk, AdamW Cap의 약자입니다.

앞쪽 학습 레이어는 momentum-stabilized sign trunk로, 마지막 `N`개 학습
레이어는 AdamW cap으로 유지하는 PyTorch 옵티마이저입니다. 목표는 전체를
AdamW로 돌릴 때보다 optimizer state VRAM을 줄이면서도, 적응형 업데이트가
중요한 구간에서는 성능을 잃지 않는 것입니다.

| 항목 | 값 |
| --- | --- |
| Python | `>=3.13` |
| PyTorch | `>=2.10` |
| 기본 분할 | 마지막 `1`개 학습 레이어만 AdamW |
| trunk | decoupled weight decay + `sign(EMA(grad))` |
| cap | AdamW + decoupled weight decay |
| 추가 VRAM 옵션 | `trunk_state_dtype=torch.bfloat16` |
| 검증 | 로컬 CUDA 테스트 + 연구용 CUDA 벤치마크 |

## 구조

```mermaid
flowchart LR
    A[등록 순서의 학습 레이어] --> B[앞쪽 학습 레이어]
    A --> C[마지막 N개 학습 레이어]
    B --> D[sign trunk<br/>decoupled weight decay<br/>EMA(grad) -> sign update<br/>optional bf16 state]
    C --> E[AdamW cap<br/>decoupled weight decay]
```

## 설치

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
    trunk_state_dtype=torch.bfloat16,
    error_if_nonfinite=True,
)

loss = torch.nn.functional.mse_loss(
    model(torch.randn(8, 128)),
    torch.randn(8, 10),
)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)

print("trunk layers:", optimizer.partition.trunk_layer_names)
print("cap layers:", optimizer.partition.cap_layer_names)
```

## 설계 근거

- [signSGD: Compressed Optimisation for Non-Convex Problems](https://arxiv.org/abs/1802.04434)
  는 sign 기반 업데이트가 낮은 상태 메모리 대안이 될 수 있음을 보여줍니다.
- [Momentum Ensures Convergence of SIGNSGD under Weaker Assumptions](https://proceedings.mlr.press/v202/sun23l.html)
  는 raw `sign(grad)`보다 momentum 뒤에 sign을 취하는 쪽이 더 안정적임을
  뒷받침합니다.
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
  은 cap에서 AdamW식 decoupled weight decay를 쓰는 근거입니다.
- [Deconstructing What Makes a Good Optimizer for Autoregressive Language Models](https://openreview.net/forum?id=zfeso8ceqr)
  는 적응형 업데이트의 이점이 소수 파라미터에 집중될 수 있음을 보여주며,
  adaptivity를 cap에 집중시키는 동기를 제공합니다.

다만 하나의 고정 설정이 모든 태스크에서 최선이라는 뜻은 아닙니다. 이 저장소의
로컬 CUDA 조사 결과:

- 작은 dense toy 문제에서는 `trunk_lr=lr`가 더 빨리 맞는 경우가 있었고,
- 아래의 held-out teacher/student 벤치마크에서는 기본값
  (`trunk_lr=0.75 * lr`)이 조금 더 안정적이었습니다.

즉 `trunk_lr`는 고정 상수가 아니라 튜닝 파라미터로 보는 편이 맞습니다.

## CUDA 연구용 벤치마크

주요 벤치마크 스크립트:
[examples/research_benchmark.py](https://github.com/smturtle2/stac-optimizer/blob/main/examples/research_benchmark.py)

JSON 결과:
[docs/benchmark/research_benchmark.json](https://github.com/smturtle2/stac-optimizer/blob/main/docs/benchmark/research_benchmark.json)

방법론:

- CUDA 전용
- train/validation 분리
- `5`개 시드
- `12` epoch, epoch당 `20` step
- epoch별 validation loss curve 기록
- 첫 step에서 optimizer state와 peak CUDA allocated/reserved memory 측정

`2026-03-19` 스냅샷, `torch 2.10.0+cu126`, `NVIDIA GeForce RTX 3070`:

![STAC CUDA research benchmark](https://raw.githubusercontent.com/smturtle2/stac-optimizer/main/docs/benchmark/research_benchmark.png)

회귀 validation loss:

| 옵티마이저 | 최종 val loss 평균 | 최종 val loss 범위 |
| --- | ---: | ---: |
| `STAC` 기본 (`cap=1`) | `0.046044` | `0.044386 - 0.047686` |
| `STAC` matched trunk lr | `0.046207` | `0.044730 - 0.047581` |
| `STAC` plain sign trunk | `0.043162` | `0.041903 - 0.044614` |
| `AdamW` baseline | `0.043753` | `0.042771 - 0.045108` |

분류 validation:

| 옵티마이저 | 최종 val loss 평균 | 최종 val loss 범위 | 최종 val acc 평균 |
| --- | ---: | ---: | ---: |
| `STAC` 기본 (`cap=1`) | `0.303325` | `0.252935 - 0.333419` | `0.8926` |
| `STAC` matched trunk lr | `0.323920` | `0.287477 - 0.333865` | `0.8828` |
| `STAC` plain sign trunk | `0.314426` | `0.279694 - 0.330161` | `0.9039` |
| `AdamW` baseline | `0.304733` | `0.275815 - 0.317797` | `0.9074` |

메모리 probe:

| 옵티마이저 | Optimizer state MB | Peak allocated MB | Peak reserved MB |
| --- | ---: | ---: | ---: |
| `STAC` 기본 (`cap=1`) | `3.637` | `31.925` | `38.000` |
| `STAC` matched trunk lr | `3.637` | `31.925` | `38.000` |
| `STAC` plain sign trunk | `0.004` | `28.292` | `34.000` |
| `AdamW` baseline | `7.270` | `35.565` | `40.000` |

이 벤치마크는 리더보드가 아니라, 이 저장소의 실제 질문 두 개에 답하기 위한
근거입니다.

- STAC가 held-out CUDA 태스크에서 AdamW와 경쟁력이 있는가
- STAC가 실제로 optimizer state와 peak memory 부담을 줄이는가

## 공개 API

패키지는 다음을 export합니다.

- `STAC`
- `partition_trainable_layers(model, last_n_layers=1)`
- `LayerGroup`
- `STACPartition`

실사용에서 중요한 보장:

- `model.named_modules()` 기준의 결정적 trunk/cap 분할
- sparse gradient 명시적 거부
- `error_if_nonfinite=False`일 때 non-finite dense gradient step 전체 skip
- state_dict 로드 시 layer name, parameter name, state tensor shape 검증

## 검증

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
python examples/research_benchmark.py --device cuda
```

빠른 smoke check 용도의 기존 스크립트
[examples/toy_benchmark.py](https://github.com/smturtle2/stac-optimizer/blob/main/examples/toy_benchmark.py)
도 유지하지만, README의 핵심 근거는 위 research benchmark 기준입니다.
