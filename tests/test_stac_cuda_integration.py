from __future__ import annotations

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


def _build_seeded_teacher(factory: type[nn.Module], seed: int) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return factory()


class ResidualBlock(nn.Module):
    def __init__(self, width: int, *, use_layernorm: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width) if use_layernorm else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        inputs = self.fc1(inputs)
        inputs = self.act(inputs)
        inputs = self.fc2(inputs)
        return residual + self.norm(inputs)


class DeepRegressionTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(48, 96)
        self.blocks = nn.Sequential(
            *[ResidualBlock(96, use_layernorm=False) for _ in range(5)]
        )
        self.head = nn.Linear(96, 6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.stem(inputs)
        inputs = self.blocks(inputs)
        return self.head(inputs)


class DeepRegressionStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(48, 128)
        self.blocks = nn.Sequential(
            *[ResidualBlock(128, use_layernorm=False) for _ in range(10)]
        )
        self.head = nn.Linear(128, 6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.stem(inputs)
        inputs = self.blocks(inputs)
        return self.head(inputs)


class DeepClassificationTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(64, 96)
        self.blocks = nn.Sequential(
            *[ResidualBlock(96, use_layernorm=True) for _ in range(5)]
        )
        self.final_norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, 6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.stem(inputs)
        inputs = self.blocks(inputs)
        inputs = self.final_norm(inputs)
        return self.head(inputs)


class DeepClassificationStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(64, 128)
        self.blocks = nn.Sequential(
            *[ResidualBlock(128, use_layernorm=True) for _ in range(10)]
        )
        self.final_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.stem(inputs)
        inputs = self.blocks(inputs)
        inputs = self.final_norm(inputs)
        return self.head(inputs)


class TailNormTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(64, 96)
        self.blocks = nn.Sequential(
            *[ResidualBlock(96, use_layernorm=False) for _ in range(5)]
        )
        self.bridge = nn.Linear(96, 96)
        self.final_norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, 6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.stem(inputs)
        inputs = self.blocks(inputs)
        inputs = self.bridge(inputs)
        inputs = self.final_norm(inputs)
        return self.head(inputs)


class TailNormStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(64, 128)
        self.blocks = nn.Sequential(
            *[ResidualBlock(128, use_layernorm=False) for _ in range(10)]
        )
        self.bridge = nn.Linear(128, 128)
        self.final_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.stem(inputs)
        inputs = self.blocks(inputs)
        inputs = self.bridge(inputs)
        inputs = self.final_norm(inputs)
        return self.head(inputs)


class StateMemoryNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(256, 1024)
        self.blocks = nn.Sequential(
            *[ResidualBlock(1024, use_layernorm=False) for _ in range(6)]
        )
        self.head = nn.Linear(1024, 16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.stem(inputs)
        inputs = self.blocks(inputs)
        return self.head(inputs)


@torch.no_grad()
def _make_regression_data(
    seed: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_inputs = torch.randn(4096, 48, generator=generator)
    val_inputs = torch.randn(1024, 48, generator=generator)
    teacher = _build_seeded_teacher(DeepRegressionTeacher, 10_000 + seed)
    train_targets = teacher(train_inputs) + 0.10 * torch.randn(
        4096,
        6,
        generator=generator,
    )
    val_targets = teacher(val_inputs) + 0.10 * torch.randn(
        1024,
        6,
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
    tail_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_inputs = torch.randn(4096, 64, generator=generator)
    val_inputs = torch.randn(1024, 64, generator=generator)
    teacher_factory = TailNormTeacher if tail_norm else DeepClassificationTeacher
    teacher_seed_offset = 20_000 if tail_norm else 15_000
    teacher = _build_seeded_teacher(teacher_factory, teacher_seed_offset + seed)
    train_targets = teacher(train_inputs).argmax(dim=1)
    val_targets = teacher(val_inputs).argmax(dim=1)

    label_noise = torch.rand(4096, generator=generator) < 0.08
    noisy_labels = torch.randint(
        0,
        6,
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
    last_n_modules: int | None = None,
    last_n_ratio: float = 0.18,
    sign_lr_scale: float = 1.0,
    sign_weight_decay: float | None = None,
    adamw_weight_decay: float | None = None,
) -> torch.optim.Optimizer:
    if optimizer_kind == "stac":
        return STAC(
            model,
            lr=2e-3,
            last_n_ratio=last_n_ratio,
            last_n_modules=last_n_modules,
            sign_lr_scale=sign_lr_scale,
            weight_decay=1e-2,
            sign_weight_decay=sign_weight_decay,
            adamw_weight_decay=adamw_weight_decay,
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
    last_n_modules: int | None = None,
    last_n_ratio: float = 0.18,
    sign_lr_scale: float = 1.0,
    sign_weight_decay: float | None = None,
    adamw_weight_decay: float | None = None,
) -> float:
    train_inputs, train_targets, val_inputs, val_targets = _make_regression_data(
        seed,
        device=device,
    )
    batch_schedule = _batch_indices(
        train_inputs.shape[0],
        batch_size=256,
        steps=180,
        seed=1000 + seed,
    )

    _seed_all(30_000 + seed)
    model = DeepRegressionStudent().to(device)
    optimizer = _make_optimizer(
        optimizer_kind,
        model,
        last_n_ratio=last_n_ratio,
        last_n_modules=last_n_modules,
        sign_lr_scale=sign_lr_scale,
        sign_weight_decay=sign_weight_decay,
        adamw_weight_decay=adamw_weight_decay,
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
    last_n_modules: int | None = None,
    last_n_ratio: float = 0.18,
    sign_lr_scale: float = 1.0,
    sign_weight_decay: float | None = None,
    adamw_weight_decay: float | None = None,
    tail_norm: bool = False,
) -> tuple[float, float]:
    train_inputs, train_targets, val_inputs, val_targets = _make_classification_data(
        seed,
        device=device,
        tail_norm=tail_norm,
    )
    batch_schedule = _batch_indices(
        train_inputs.shape[0],
        batch_size=256,
        steps=180,
        seed=2000 + seed,
    )

    _seed_all(40_000 + seed)
    model = (
        TailNormStudent().to(device)
        if tail_norm
        else DeepClassificationStudent().to(device)
    )
    optimizer = _make_optimizer(
        optimizer_kind,
        model,
        last_n_ratio=last_n_ratio,
        last_n_modules=last_n_modules,
        sign_lr_scale=sign_lr_scale,
        sign_weight_decay=sign_weight_decay,
        adamw_weight_decay=adamw_weight_decay,
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


def _measure_memory(
    optimizer_kind: str,
    *,
    device: torch.device,
    last_n_modules: int | None = None,
    last_n_ratio: float = 0.18,
    sign_lr_scale: float = 1.0,
    sign_weight_decay: float | None = None,
    adamw_weight_decay: float | None = None,
) -> tuple[float, float]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    _seed_all(0)
    model = StateMemoryNet().to(device)
    optimizer = _make_optimizer(
        optimizer_kind,
        model,
        last_n_ratio=last_n_ratio,
        last_n_modules=last_n_modules,
        sign_lr_scale=sign_lr_scale,
        sign_weight_decay=sign_weight_decay,
        adamw_weight_decay=adamw_weight_decay,
    )
    inputs = torch.randn(32, 256, device=device)
    targets = torch.randn(32, 16, device=device)

    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()
    del loss, predictions
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    optimizer.step()
    torch.cuda.synchronize(device)

    peak_delta = (torch.cuda.max_memory_allocated(device) - baseline) / (1024**2)
    steady_delta = (torch.cuda.memory_allocated(device) - baseline) / (1024**2)
    return peak_delta, steady_delta


def test_stac_presets_form_a_quality_memory_frontier_on_deep_cuda_suite(
    cuda_device: torch.device,
) -> None:
    seeds = range(5)
    preset_kwargs = {
        "stac_default": {
            "optimizer_kind": "stac",
        },
        "stac_12p5_tail": {
            "optimizer_kind": "stac",
            "last_n_ratio": 0.125,
        },
        "stac_wide_cap": {
            "optimizer_kind": "stac",
            "last_n_ratio": 0.25,
        },
        "adamw": {
            "optimizer_kind": "adamw",
        },
    }

    regression_results = {
        label: [
            _run_regression_trial(seed, device=cuda_device, **kwargs)
            for seed in seeds
        ]
        for label, kwargs in preset_kwargs.items()
    }
    classification_results = {
        label: [
            _run_classification_trial(seed, device=cuda_device, **kwargs)
            for seed in seeds
        ]
        for label, kwargs in preset_kwargs.items()
    }
    tailnorm_results = {
        label: [
            _run_classification_trial(
                seed,
                device=cuda_device,
                tail_norm=True,
                **kwargs,
            )
            for seed in seeds
        ]
        for label, kwargs in preset_kwargs.items()
    }

    default_regression = statistics.fmean(regression_results["stac_default"])
    wide_regression = statistics.fmean(regression_results["stac_wide_cap"])
    adamw_regression = statistics.fmean(regression_results["adamw"])
    default_classification_loss = statistics.fmean(
        loss for loss, _ in classification_results["stac_default"]
    )
    ratio125_classification_loss = statistics.fmean(
        loss for loss, _ in classification_results["stac_12p5_tail"]
    )
    adamw_classification_loss = statistics.fmean(
        loss for loss, _ in classification_results["adamw"]
    )
    default_classification_acc = statistics.fmean(
        accuracy for _, accuracy in classification_results["stac_default"]
    )
    ratio125_classification_acc = statistics.fmean(
        accuracy for _, accuracy in classification_results["stac_12p5_tail"]
    )
    adamw_classification_acc = statistics.fmean(
        accuracy for _, accuracy in classification_results["adamw"]
    )
    adamw_tailnorm_loss = statistics.fmean(
        loss for loss, _ in tailnorm_results["adamw"]
    )
    adamw_tailnorm_acc = statistics.fmean(
        accuracy for _, accuracy in tailnorm_results["adamw"]
    )
    default_tailnorm_acc = statistics.fmean(
        accuracy for _, accuracy in tailnorm_results["stac_default"]
    )
    ratio125_tailnorm_acc = statistics.fmean(
        accuracy for _, accuracy in tailnorm_results["stac_12p5_tail"]
    )

    best_stac_regression = min(
        statistics.fmean(results)
        for label, results in regression_results.items()
        if label != "adamw"
    )
    best_stac_classification_loss = min(
        statistics.fmean(loss for loss, _ in results)
        for label, results in classification_results.items()
        if label != "adamw"
    )
    best_stac_classification_acc = max(
        statistics.fmean(accuracy for _, accuracy in results)
        for label, results in classification_results.items()
        if label != "adamw"
    )
    best_stac_tailnorm_loss = min(
        statistics.fmean(loss for loss, _ in results)
        for label, results in tailnorm_results.items()
        if label != "adamw"
    )
    best_stac_tailnorm_acc = max(
        statistics.fmean(accuracy for _, accuracy in results)
        for label, results in tailnorm_results.items()
        if label != "adamw"
    )

    assert wide_regression <= default_regression
    assert default_classification_loss <= ratio125_classification_loss
    assert default_classification_acc >= ratio125_classification_acc
    assert default_tailnorm_acc >= ratio125_tailnorm_acc

    assert best_stac_regression <= adamw_regression * 1.25
    assert best_stac_classification_loss <= adamw_classification_loss * 1.08
    assert best_stac_classification_acc >= adamw_classification_acc - 0.02
    assert best_stac_tailnorm_loss <= adamw_tailnorm_loss * 1.20
    assert best_stac_tailnorm_acc >= adamw_tailnorm_acc - 0.05


def test_stac_presets_use_less_optimizer_state_than_adamw_on_cuda(
    cuda_device: torch.device,
) -> None:
    _seed_all(0)
    default_model = StateMemoryNet().to(cuda_device)
    _seed_all(0)
    ratio125_model = StateMemoryNet().to(cuda_device)
    _seed_all(0)
    wide_model = StateMemoryNet().to(cuda_device)
    _seed_all(0)
    adamw_model = StateMemoryNet().to(cuda_device)

    default_optimizer = STAC(
        default_model,
        lr=2e-3,
        weight_decay=1e-2,
    )
    ratio125_optimizer = STAC(
        ratio125_model,
        lr=2e-3,
        last_n_ratio=0.125,
        weight_decay=1e-2,
    )
    wide_optimizer = STAC(
        wide_model,
        lr=2e-3,
        last_n_ratio=0.25,
        weight_decay=1e-2,
    )
    adamw_optimizer = torch.optim.AdamW(
        adamw_model.parameters(),
        lr=2e-3,
        weight_decay=1e-2,
    )

    inputs = torch.randn(32, 256, device=cuda_device)
    targets = torch.randn(32, 16, device=cuda_device)

    for model, optimizer in (
        (default_model, default_optimizer),
        (ratio125_model, ratio125_optimizer),
        (wide_model, wide_optimizer),
        (adamw_model, adamw_optimizer),
    ):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        del loss, predictions
        optimizer.step()

    default_state_bytes = _optimizer_state_bytes(default_optimizer)
    ratio125_state_bytes = _optimizer_state_bytes(ratio125_optimizer)
    wide_state_bytes = _optimizer_state_bytes(wide_optimizer)
    adamw_state_bytes = _optimizer_state_bytes(adamw_optimizer)

    assert ratio125_state_bytes < default_state_bytes < wide_state_bytes
    assert wide_state_bytes < adamw_state_bytes * 0.35


def test_research_data_generation_is_reproducible_per_seed(
    cuda_device: torch.device,
) -> None:
    regression_a = _make_regression_data(123, device=cuda_device)
    regression_b = _make_regression_data(123, device=cuda_device)
    classification_a = _make_classification_data(123, device=cuda_device)
    classification_b = _make_classification_data(123, device=cuda_device)
    tailnorm_a = _make_classification_data(123, device=cuda_device, tail_norm=True)
    tailnorm_b = _make_classification_data(123, device=cuda_device, tail_norm=True)

    for first, second in (
        (regression_a, regression_b),
        (classification_a, classification_b),
        (tailnorm_a, tailnorm_b),
    ):
        for first_tensor, second_tensor in zip(first, second, strict=True):
            assert torch.equal(first_tensor, second_tensor)


def test_stac_presets_use_less_peak_cuda_memory_than_adamw_on_cuda(
    cuda_device: torch.device,
) -> None:
    default_peak_mb, default_steady_mb = _measure_memory("stac", device=cuda_device)
    ratio125_peak_mb, ratio125_steady_mb = _measure_memory(
        "stac",
        device=cuda_device,
        last_n_ratio=0.125,
    )
    wide_peak_mb, wide_steady_mb = _measure_memory(
        "stac",
        device=cuda_device,
        last_n_ratio=0.25,
    )
    adamw_peak_mb, adamw_steady_mb = _measure_memory("adamw", device=cuda_device)

    assert ratio125_peak_mb < default_peak_mb < wide_peak_mb < adamw_peak_mb
    assert ratio125_steady_mb < default_steady_mb < wide_steady_mb < adamw_steady_mb
