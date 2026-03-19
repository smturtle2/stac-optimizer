from __future__ import annotations

import math
import statistics

import pytest
import torch
from torch import nn

from stac_optimizer import STAC


pytestmark = [pytest.mark.cuda, pytest.mark.slow]


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for STAC integration tests.")
    return torch.device("cuda")


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RegressionTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(24, 48),
            nn.Tanh(),
            nn.Linear(48, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class RegressionStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ClassificationTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 40),
            nn.Tanh(),
            nn.Linear(40, 24),
            nn.Tanh(),
            nn.Linear(24, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ClassificationStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class LayerNormClassificationTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 5),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class LayerNormClassificationStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(32, 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 5),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class StateMemoryNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


@torch.no_grad()
def _make_regression_data(
    seed: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_inputs = torch.randn(4096, 24, generator=generator)
    val_inputs = torch.randn(512, 24, generator=generator)
    teacher = RegressionTeacher()
    train_targets = teacher(train_inputs) + 0.2 * torch.randn(
        4096,
        3,
        generator=generator,
    )
    val_targets = teacher(val_inputs) + 0.2 * torch.randn(
        512,
        3,
        generator=generator,
    )
    return (
        train_inputs.to(device),
        train_targets.to(device),
        val_inputs.to(device),
        val_targets.to(device),
    )


@torch.no_grad()
def _make_classification_data(
    seed: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_inputs = torch.randn(4096, 20, generator=generator)
    val_inputs = torch.randn(512, 20, generator=generator)
    teacher = ClassificationTeacher()
    train_targets = teacher(train_inputs).argmax(dim=1)
    val_targets = teacher(val_inputs).argmax(dim=1)

    label_noise = torch.rand(4096, generator=generator) < 0.08
    noisy_labels = torch.randint(
        0,
        4,
        size=(int(label_noise.sum().item()),),
        generator=generator,
    )
    train_targets[label_noise] = noisy_labels

    return (
        train_inputs.to(device),
        train_targets.to(device),
        val_inputs.to(device),
        val_targets.to(device),
    )


@torch.no_grad()
def _make_layernorm_classification_data(
    seed: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_inputs = torch.randn(4096, 32, generator=generator)
    val_inputs = torch.randn(512, 32, generator=generator)
    teacher = LayerNormClassificationTeacher()
    train_targets = teacher(train_inputs).argmax(dim=1)
    val_targets = teacher(val_inputs).argmax(dim=1)

    label_noise = torch.rand(4096, generator=generator) < 0.10
    noisy_labels = torch.randint(
        0,
        5,
        size=(int(label_noise.sum().item()),),
        generator=generator,
    )
    train_targets[label_noise] = noisy_labels

    return (
        train_inputs.to(device),
        train_targets.to(device),
        val_inputs.to(device),
        val_targets.to(device),
    )


def _batch_indices(
    num_samples: int,
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randint(num_samples, (batch_size,), generator=generator)
        for _ in range(steps)
    ]


def _make_optimizer(
    optimizer_kind: str,
    model: nn.Module,
    *,
    last_n_modules: int = 1,
    sign_state_dtype: torch.dtype | str | None = "auto",
) -> torch.optim.Optimizer:
    if optimizer_kind == "stac":
        return STAC(
            model,
            lr=2e-3,
            last_n_modules=last_n_modules,
            sign_momentum=0.9,
            sign_state_dtype=sign_state_dtype,
            weight_decay=1e-2,
        )
    if optimizer_kind == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=2e-3,
            weight_decay=1e-2,
        )

    raise ValueError(f"Unknown optimizer kind: {optimizer_kind}.")


def _run_regression_trial(
    seed: int,
    optimizer_kind: str,
    *,
    device: torch.device,
    sign_state_dtype: torch.dtype | str | None = "auto",
) -> float:
    train_inputs, train_targets, val_inputs, val_targets = _make_regression_data(
        seed,
        device=device,
    )
    batch_schedule = _batch_indices(
        train_inputs.shape[0],
        batch_size=256,
        steps=240,
        seed=1000 + seed,
    )

    _seed_all(30_000 + seed)
    model = RegressionStudent().to(device)
    optimizer = _make_optimizer(
        optimizer_kind,
        model,
        sign_state_dtype=sign_state_dtype,
    )

    for batch_index in batch_schedule:
        index = batch_index.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(train_inputs[index])
        loss = torch.nn.functional.mse_loss(predictions, train_targets[index])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = model(val_inputs)
        return float(
            torch.nn.functional.mse_loss(predictions, val_targets).detach().cpu()
        )


def _run_classification_trial(
    seed: int,
    optimizer_kind: str,
    *,
    device: torch.device,
    sign_state_dtype: torch.dtype | str | None = "auto",
) -> tuple[float, float]:
    train_inputs, train_targets, val_inputs, val_targets = (
        _make_classification_data(seed, device=device)
    )
    batch_schedule = _batch_indices(
        train_inputs.shape[0],
        batch_size=256,
        steps=260,
        seed=2000 + seed,
    )

    _seed_all(40_000 + seed)
    model = ClassificationStudent().to(device)
    optimizer = _make_optimizer(
        optimizer_kind,
        model,
        sign_state_dtype=sign_state_dtype,
    )

    for batch_index in batch_schedule:
        index = batch_index.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_inputs[index])
        loss = torch.nn.functional.cross_entropy(logits, train_targets[index])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(val_inputs)
        loss = float(torch.nn.functional.cross_entropy(logits, val_targets).cpu())
        accuracy = float((logits.argmax(dim=1) == val_targets).float().mean().cpu())
        return loss, accuracy


def _run_layernorm_classification_trial(
    seed: int,
    optimizer_kind: str,
    *,
    device: torch.device,
    last_n_modules: int = 1,
    sign_state_dtype: torch.dtype | str | None = "auto",
) -> tuple[float, float]:
    train_inputs, train_targets, val_inputs, val_targets = (
        _make_layernorm_classification_data(seed, device=device)
    )
    batch_schedule = _batch_indices(
        train_inputs.shape[0],
        batch_size=256,
        steps=260,
        seed=3000 + seed,
    )

    _seed_all(50_000 + seed)
    model = LayerNormClassificationStudent().to(device)
    optimizer = _make_optimizer(
        optimizer_kind,
        model,
        last_n_modules=last_n_modules,
        sign_state_dtype=sign_state_dtype,
    )

    for batch_index in batch_schedule:
        index = batch_index.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_inputs[index])
        loss = torch.nn.functional.cross_entropy(logits, train_targets[index])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(val_inputs)
        loss = float(torch.nn.functional.cross_entropy(logits, val_targets).cpu())
        accuracy = float((logits.argmax(dim=1) == val_targets).float().mean().cpu())
        return loss, accuracy


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def test_stac_is_competitive_with_adamw_on_noisy_cuda_suite(
    cuda_device: torch.device,
) -> None:
    regression_results = {
        optimizer_kind: [
            _run_regression_trial(seed, optimizer_kind, device=cuda_device)
            for seed in range(5)
        ]
        for optimizer_kind in ("stac", "adamw")
    }
    classification_results = {
        optimizer_kind: [
            _run_classification_trial(seed, optimizer_kind, device=cuda_device)
            for seed in range(5)
        ]
        for optimizer_kind in ("stac", "adamw")
    }

    stac_regression = statistics.fmean(regression_results["stac"])
    adamw_regression = statistics.fmean(regression_results["adamw"])
    stac_classification_loss = statistics.fmean(
        loss for loss, _ in classification_results["stac"]
    )
    adamw_classification_loss = statistics.fmean(
        loss for loss, _ in classification_results["adamw"]
    )
    stac_classification_acc = statistics.fmean(
        accuracy for _, accuracy in classification_results["stac"]
    )
    adamw_classification_acc = statistics.fmean(
        accuracy for _, accuracy in classification_results["adamw"]
    )

    assert stac_regression <= adamw_regression * 1.10
    assert stac_classification_loss <= adamw_classification_loss * 1.08
    assert stac_classification_acc >= adamw_classification_acc - 0.02


def test_layernorm_cuda_suite_favors_a_larger_adamw_cap(
    cuda_device: torch.device,
) -> None:
    default_results = [
        _run_layernorm_classification_trial(
            seed,
            "stac",
            device=cuda_device,
            last_n_modules=1,
        )
        for seed in range(3)
    ]
    wider_cap_results = [
        _run_layernorm_classification_trial(
            seed,
            "stac",
            device=cuda_device,
            last_n_modules=4,
        )
        for seed in range(3)
    ]
    adamw_results = [
        _run_layernorm_classification_trial(
            seed,
            "adamw",
            device=cuda_device,
        )
        for seed in range(3)
    ]

    default_loss = statistics.fmean(loss for loss, _ in default_results)
    wider_cap_loss = statistics.fmean(loss for loss, _ in wider_cap_results)
    adamw_loss = statistics.fmean(loss for loss, _ in adamw_results)
    default_acc = statistics.fmean(accuracy for _, accuracy in default_results)
    wider_cap_acc = statistics.fmean(accuracy for _, accuracy in wider_cap_results)
    adamw_acc = statistics.fmean(accuracy for _, accuracy in adamw_results)

    assert wider_cap_loss <= default_loss
    assert wider_cap_acc >= default_acc - 0.01
    assert wider_cap_loss <= adamw_loss * 1.13
    assert wider_cap_acc >= adamw_acc - 0.02


def test_stac_uses_less_optimizer_state_than_adamw_on_cuda(
    cuda_device: torch.device,
) -> None:
    _seed_all(0)
    stac_model = StateMemoryNet().to(cuda_device)
    _seed_all(0)
    adamw_model = StateMemoryNet().to(cuda_device)

    stac_optimizer = STAC(
        stac_model,
        lr=2e-3,
        last_n_modules=1,
        sign_momentum=0.9,
        weight_decay=1e-2,
    )
    adamw_optimizer = torch.optim.AdamW(
        adamw_model.parameters(),
        lr=2e-3,
        weight_decay=1e-2,
    )

    inputs = torch.randn(128, 256, device=cuda_device)
    targets = torch.randn(128, 8, device=cuda_device)

    for model, optimizer in (
        (stac_model, stac_optimizer),
        (adamw_model, adamw_optimizer),
    ):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()

    stac_state_bytes = _optimizer_state_bytes(stac_optimizer)
    adamw_state_bytes = _optimizer_state_bytes(adamw_optimizer)

    assert stac_state_bytes < adamw_state_bytes * 0.60


def test_bfloat16_sign_state_reduces_sign_optimizer_memory_on_cuda(
    cuda_device: torch.device,
) -> None:
    _seed_all(0)
    fp32_model = StateMemoryNet().to(cuda_device)
    _seed_all(0)
    bf16_model = StateMemoryNet().to(cuda_device)

    fp32_optimizer = STAC(
        fp32_model,
        lr=2e-3,
        last_n_modules=0,
        sign_momentum=0.9,
        weight_decay=1e-2,
    )
    bf16_optimizer = STAC(
        bf16_model,
        lr=2e-3,
        last_n_modules=0,
        sign_momentum=0.9,
        sign_state_dtype=torch.bfloat16,
        weight_decay=1e-2,
    )

    inputs = torch.randn(128, 256, device=cuda_device)
    targets = torch.randn(128, 8, device=cuda_device)

    for model, optimizer in (
        (fp32_model, fp32_optimizer),
        (bf16_model, bf16_optimizer),
    ):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()

    fp32_state_bytes = _optimizer_state_bytes(fp32_optimizer)
    bf16_state_bytes = _optimizer_state_bytes(bf16_optimizer)

    assert bf16_state_bytes < fp32_state_bytes * 0.60


def test_bfloat16_sign_state_stays_close_to_default_on_cuda_regression_suite(
    cuda_device: torch.device,
) -> None:
    fp32_losses = [
        _run_regression_trial(
            seed,
            "stac",
            device=cuda_device,
        )
        for seed in range(2)
    ]
    bf16_losses = [
        _run_regression_trial(
            seed,
            "stac",
            device=cuda_device,
            sign_state_dtype=torch.bfloat16,
        )
        for seed in range(2)
    ]

    assert statistics.fmean(bf16_losses) <= statistics.fmean(fp32_losses) * 1.15


def test_auto_sign_state_supports_stable_bfloat16_training_on_cuda(
    cuda_device: torch.device,
) -> None:
    train_inputs, train_targets, _, _ = _make_layernorm_classification_data(
        7,
        device=cuda_device,
    )
    batch_schedule = _batch_indices(
        train_inputs.shape[0],
        batch_size=256,
        steps=32,
        seed=7_007,
    )

    _seed_all(70_007)
    model = LayerNormClassificationStudent().to(device=cuda_device, dtype=torch.bfloat16)
    optimizer = STAC(
        model,
        lr=2e-3,
        last_n_modules=1,
        sign_momentum=0.9,
        weight_decay=1e-2,
    )

    losses: list[float] = []
    for batch_index in batch_schedule:
        index = batch_index.to(cuda_device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_inputs[index].to(dtype=torch.bfloat16))
        loss = torch.nn.functional.cross_entropy(logits.float(), train_targets[index])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    assert all(math.isfinite(loss_value) for loss_value in losses)
    assert losses[-1] < losses[0]
    sign_parameter = optimizer.partition.sign_parameters[0]
    assert optimizer.state[sign_parameter]["sign_momentum_buffer"].dtype == torch.float32
