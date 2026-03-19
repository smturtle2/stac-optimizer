from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from stac_optimizer import STAC


@dataclass(frozen=True)
class BenchmarkConfig:
    label: str
    optimizer_kind: str
    lr: float = 2e-3
    last_n_modules: int | None = None
    last_n_ratio: float = 0.125
    sign_lr_scale: float = 1.0
    weight_decay: float = 1e-2
    sign_weight_decay: float | None = None
    adamw_weight_decay: float | None = None
    color: str = "#1f77b4"
    linestyle: str | tuple[int, tuple[int, ...]] = "-"


@dataclass(frozen=True)
class TaskConfig:
    name: str
    train_samples: int
    val_samples: int
    input_dim: int
    batch_size: int
    steps_per_epoch: int
    epochs: int


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


class SequenceTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(256, 96)
        self.position_embedding = nn.Embedding(32, 96)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=96,
                    nhead=4,
                    dim_feedforward=384,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(4)
            ]
        )
        self.final_norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, 8)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return self.head(hidden.mean(dim=1))


class SequenceStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(256, 128)
        self.position_embedding = nn.Embedding(32, 128)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=128,
                    nhead=8,
                    dim_feedforward=512,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(8)
            ]
        )
        self.final_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 8)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return self.head(hidden.mean(dim=1))


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


def resolve_device(requested: str) -> torch.device:
    if requested != "cuda":
        raise SystemExit("This benchmark is intended for CUDA. Use --device cuda.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark, but no GPU is available.")
    return torch.device("cuda")


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_seeded_teacher(factory: type[nn.Module], seed: int) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return factory()


@torch.no_grad()
def make_regression_data(
    seed: int,
    *,
    task: TaskConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_inputs = torch.randn(task.train_samples, task.input_dim, generator=generator)
    val_inputs = torch.randn(task.val_samples, task.input_dim, generator=generator)
    teacher = build_seeded_teacher(DeepRegressionTeacher, 10_000 + seed)
    train_targets = teacher(train_inputs) + 0.10 * torch.randn(
        task.train_samples,
        6,
        generator=generator,
    )
    val_targets = teacher(val_inputs) + 0.10 * torch.randn(
        task.val_samples,
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
def make_classification_data(
    seed: int,
    *,
    task: TaskConfig,
    device: torch.device,
    tail_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_inputs = torch.randn(task.train_samples, task.input_dim, generator=generator)
    val_inputs = torch.randn(task.val_samples, task.input_dim, generator=generator)
    teacher_factory = TailNormTeacher if tail_norm else DeepClassificationTeacher
    teacher_seed_offset = 20_000 if tail_norm else 15_000
    teacher = build_seeded_teacher(teacher_factory, teacher_seed_offset + seed)
    train_targets = teacher(train_inputs).argmax(dim=1)
    val_targets = teacher(val_inputs).argmax(dim=1)

    label_noise = torch.rand(task.train_samples, generator=generator) < 0.08
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


@torch.no_grad()
def make_sequence_data(
    seed: int,
    *,
    task: TaskConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_tokens = torch.randint(256, (task.train_samples, 32), generator=generator)
    val_tokens = torch.randint(256, (task.val_samples, 32), generator=generator)
    teacher = build_seeded_teacher(SequenceTeacher, 25_000 + seed)
    train_targets = teacher(train_tokens).argmax(dim=1)
    val_targets = teacher(val_tokens).argmax(dim=1)

    label_noise = torch.rand(task.train_samples, generator=generator) < 0.06
    noisy_labels = torch.randint(
        0,
        8,
        size=(int(label_noise.sum().item()),),
        generator=generator,
    )
    train_targets[label_noise] = noisy_labels

    return (
        train_tokens.to(device),
        train_targets.to(device),
        val_tokens.to(device),
        val_targets.to(device),
    )


def batch_indices(
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


def build_adamw_param_groups(
    model: nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        local_name = parameter_name.rsplit(".", maxsplit=1)[-1]
        if local_name == "bias" or parameter.ndim <= 1:
            no_decay_parameters.append(parameter)
            continue
        decay_parameters.append(parameter)

    param_groups: list[dict[str, object]] = []
    if decay_parameters:
        param_groups.append(
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
            }
        )
    if no_decay_parameters:
        param_groups.append(
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            }
        )
    return param_groups


def build_optimizer(
    model: nn.Module,
    config: BenchmarkConfig,
) -> torch.optim.Optimizer:
    if config.optimizer_kind == "stac":
        return STAC(
            model,
            lr=config.lr,
            last_n_ratio=config.last_n_ratio,
            last_n_modules=config.last_n_modules,
            sign_lr_scale=config.sign_lr_scale,
            weight_decay=config.weight_decay,
            sign_weight_decay=config.sign_weight_decay,
            adamw_weight_decay=config.adamw_weight_decay,
        )
    if config.optimizer_kind == "adamw":
        return torch.optim.AdamW(
            build_adamw_param_groups(model, weight_decay=config.weight_decay),
            lr=config.lr,
        )
    raise ValueError(f"Unknown optimizer kind: {config.optimizer_kind}.")


def optimizer_state_megabytes(optimizer: torch.optim.Optimizer) -> float:
    total_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    return total_bytes / (1024**2)


def evaluate_regression(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        return {"loss": float(loss.detach().cpu())}


def evaluate_classification(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        return {
            "loss": float(loss.detach().cpu()),
            "accuracy": float(accuracy.detach().cpu()),
        }


def run_regression_trial(
    seed: int,
    *,
    config: BenchmarkConfig,
    task: TaskConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    train_inputs, train_targets, val_inputs, val_targets = make_regression_data(
        seed,
        task=task,
        device=device,
    )
    schedule = batch_indices(
        task.train_samples,
        batch_size=task.batch_size,
        steps=task.steps_per_epoch * task.epochs,
        seed=10_000 + seed,
    )

    seed_all(30_000 + seed)
    model = DeepRegressionStudent().to(device)
    optimizer = build_optimizer(model, config)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(task.epochs):
        epoch_losses: list[float] = []
        offset = epoch * task.steps_per_epoch
        for batch_index in schedule[offset : offset + task.steps_per_epoch]:
            index = batch_index.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(train_inputs[index])
            loss = torch.nn.functional.mse_loss(predictions, train_targets[index])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        train_losses.append(statistics.fmean(epoch_losses))
        val_losses.append(evaluate_regression(model, val_inputs, val_targets)["loss"])

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
    }


def run_classification_trial(
    seed: int,
    *,
    config: BenchmarkConfig,
    task: TaskConfig,
    device: torch.device,
    tail_norm: bool,
) -> dict[str, list[float]]:
    train_inputs, train_targets, val_inputs, val_targets = make_classification_data(
        seed,
        task=task,
        device=device,
        tail_norm=tail_norm,
    )
    schedule = batch_indices(
        task.train_samples,
        batch_size=task.batch_size,
        steps=task.steps_per_epoch * task.epochs,
        seed=20_000 + seed,
    )

    seed_all(40_000 + seed)
    model = TailNormStudent().to(device) if tail_norm else DeepClassificationStudent().to(device)
    optimizer = build_optimizer(model, config)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []

    for epoch in range(task.epochs):
        epoch_losses: list[float] = []
        offset = epoch * task.steps_per_epoch
        for batch_index in schedule[offset : offset + task.steps_per_epoch]:
            index = batch_index.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_inputs[index])
            loss = torch.nn.functional.cross_entropy(logits, train_targets[index])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        train_losses.append(statistics.fmean(epoch_losses))
        metrics = evaluate_classification(model, val_inputs, val_targets)
        val_losses.append(metrics["loss"])
        val_accuracies.append(metrics["accuracy"])

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies,
    }


def run_sequence_trial(
    seed: int,
    *,
    config: BenchmarkConfig,
    task: TaskConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    train_tokens, train_targets, val_tokens, val_targets = make_sequence_data(
        seed,
        task=task,
        device=device,
    )
    schedule = batch_indices(
        task.train_samples,
        batch_size=task.batch_size,
        steps=task.steps_per_epoch * task.epochs,
        seed=30_000 + seed,
    )

    seed_all(50_000 + seed)
    model = SequenceStudent().to(device)
    optimizer = build_optimizer(model, config)
    autocast_enabled = torch.cuda.is_bf16_supported()

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []

    for epoch in range(task.epochs):
        epoch_losses: list[float] = []
        offset = epoch * task.steps_per_epoch
        for batch_index in schedule[offset : offset + task.steps_per_epoch]:
            index = batch_index.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                logits = model(train_tokens[index])
                loss = torch.nn.functional.cross_entropy(logits, train_targets[index])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        train_losses.append(statistics.fmean(epoch_losses))
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                logits = model(val_tokens)
                loss = torch.nn.functional.cross_entropy(logits, val_targets)
            accuracy = (logits.argmax(dim=1) == val_targets).float().mean()
        val_losses.append(float(loss.detach().cpu()))
        val_accuracies.append(float(accuracy.detach().cpu()))

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies,
    }


def aggregate_trials(trials: list[dict[str, list[float]]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for metric_name in trials[0]:
        per_epoch_values = [
            [trial[metric_name][epoch] for trial in trials]
            for epoch in range(len(trials[0][metric_name]))
        ]
        aggregated[metric_name] = {
            "mean": [statistics.fmean(values) for values in per_epoch_values],
            "min": [min(values) for values in per_epoch_values],
            "max": [max(values) for values in per_epoch_values],
        }

    summary = {
        "final_train_loss_mean": aggregated["train_loss"]["mean"][-1],
        "final_val_loss_mean": aggregated["val_loss"]["mean"][-1],
        "final_val_loss_min": aggregated["val_loss"]["min"][-1],
        "final_val_loss_max": aggregated["val_loss"]["max"][-1],
    }
    if "val_accuracy" in aggregated:
        summary["final_val_accuracy_mean"] = aggregated["val_accuracy"]["mean"][-1]
        summary["final_val_accuracy_min"] = aggregated["val_accuracy"]["min"][-1]
        summary["final_val_accuracy_max"] = aggregated["val_accuracy"]["max"][-1]

    return {"curves": aggregated, "summary": summary}


def benchmark_task(
    task_name: str,
    *,
    configs: list[BenchmarkConfig],
    seeds: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if task_name == "deep_regression":
        task = TaskConfig(
            name=task_name,
            train_samples=4096,
            val_samples=1024,
            input_dim=48,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        runner = run_regression_trial
        runner_kwargs: dict[str, Any] = {}
    elif task_name == "deep_classification":
        task = TaskConfig(
            name=task_name,
            train_samples=4096,
            val_samples=1024,
            input_dim=64,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        runner = run_classification_trial
        runner_kwargs = {"tail_norm": False}
    elif task_name == "tailnorm_classification":
        task = TaskConfig(
            name=task_name,
            train_samples=4096,
            val_samples=1024,
            input_dim=64,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        runner = run_classification_trial
        runner_kwargs = {"tail_norm": True}
    elif task_name == "sequence_classification":
        task = TaskConfig(
            name=task_name,
            train_samples=3072,
            val_samples=768,
            input_dim=32,
            batch_size=min(batch_size, 96),
            steps_per_epoch=min(steps_per_epoch, 12),
            epochs=epochs,
        )
        runner = run_sequence_trial
        runner_kwargs = {}
    else:
        raise ValueError(f"Unknown task: {task_name}.")

    config_results: dict[str, Any] = {}
    for config in configs:
        trials = [
            runner(seed, config=config, task=task, device=device, **runner_kwargs)
            for seed in range(seeds)
        ]
        config_results[config.label] = {
            "config": asdict(config),
            **aggregate_trials(trials),
        }

    return {
        "task": asdict(task),
        "configs": config_results,
    }


def measure_peak_memory(
    *,
    configs: list[BenchmarkConfig],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for config in configs:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        seed_all(0)
        model = StateMemoryNet().to(device)
        optimizer = build_optimizer(model, config)
        inputs = torch.randn(batch_size, 256, device=device)
        targets = torch.randn(batch_size, 16, device=device)

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

        results.append(
            {
                "label": config.label,
                "optimizer_state_mb": optimizer_state_megabytes(optimizer),
                "peak_delta_mb": (
                    torch.cuda.max_memory_allocated(device) - baseline
                )
                / (1024**2),
                "steady_delta_mb": (
                    torch.cuda.memory_allocated(device) - baseline
                )
                / (1024**2),
            }
        )

        del optimizer, model, inputs, targets
        torch.cuda.empty_cache()

    return results


def render_plot(
    *,
    benchmark_results: dict[str, Any],
    configs: list[BenchmarkConfig],
    output_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    task_names = tuple(benchmark_results["tasks"])
    figure, axes = plt.subplots(
        1,
        len(task_names),
        figsize=(5.7 * len(task_names), 4.8),
        dpi=180,
    )
    figure.patch.set_facecolor("#f8fafc")
    if len(task_names) == 1:
        axes = [axes]

    for axis, task_name in zip(axes, task_names, strict=True):
        axis.set_facecolor("#ffffff")
        task_results = benchmark_results["tasks"][task_name]["configs"]
        epochs = list(
            range(
                1,
                benchmark_results["tasks"][task_name]["task"]["epochs"] + 1,
            )
        )
        for config in configs:
            curves = task_results[config.label]["curves"]["val_loss"]
            axis.plot(
                epochs,
                curves["mean"],
                label=config.label,
                color=config.color,
                linestyle=config.linestyle,
                linewidth=2.2,
            )
            axis.fill_between(
                epochs,
                curves["min"],
                curves["max"],
                color=config.color,
                alpha=0.12,
            )

        axis.set_title(f"{task_name.replace('_', ' ').title()} Validation Loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    figure.suptitle("STAC CUDA Research Benchmark", fontsize=15, y=1.04)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def print_task_summary(
    task_name: str,
    task_results: dict[str, Any],
) -> None:
    print(f"\n## {task_name.replace('_', ' ')}")
    sample_summary = next(iter(task_results["configs"].values()))["summary"]
    has_accuracy = "final_val_accuracy_mean" in sample_summary
    if has_accuracy:
        print("| Optimizer | Final val loss mean | Final val loss range | Final val acc mean |")
        print("| --- | ---: | ---: | ---: |")
    else:
        print("| Optimizer | Final val loss mean | Final val loss range |")
        print("| --- | ---: | ---: |")

    for label, result in task_results["configs"].items():
        summary = result["summary"]
        loss_range = (
            f"{summary['final_val_loss_min']:.6f} - "
            f"{summary['final_val_loss_max']:.6f}"
        )
        if has_accuracy:
            print(
                f"| {label} | {summary['final_val_loss_mean']:.6f} | "
                f"{loss_range} | {summary['final_val_accuracy_mean']:.4f} |"
            )
            continue
        print(
            f"| {label} | {summary['final_val_loss_mean']:.6f} | {loss_range} |"
        )


def print_memory_summary(memory_results: list[dict[str, Any]]) -> None:
    print("\n## memory")
    print("| Optimizer | Optimizer state MB | Peak delta MB | Steady delta MB |")
    print("| --- | ---: | ---: | ---: |")
    for result in memory_results:
        print(
            f"| {result['label']} | {result['optimizer_state_mb']:.3f} | "
            f"{result['peak_delta_mb']:.3f} | "
            f"{result['steady_delta_mb']:.3f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CUDA-only research benchmark for STAC with held-out splits, "
            "multiple seeds, epoch-by-epoch loss curves, and peak CUDA memory."
        )
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Benchmark device. Only 'cuda' is supported for this script.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of random seeds per optimizer/task pair.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to record in the loss curves.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=16,
        help="Mini-batch updates per epoch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size for the research tasks.",
    )
    parser.add_argument(
        "--memory-batch-size",
        type=int,
        default=32,
        help="Batch size for the peak-memory probe.",
    )
    parser.add_argument(
        "--json-out",
        default="docs/benchmark/research_benchmark.json",
        help="Where to save the machine-readable JSON report.",
    )
    parser.add_argument(
        "--plot-out",
        default="docs/benchmark/research_benchmark.png",
        help="Where to save the loss-curve PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    json_path = Path(args.json_out)
    plot_path = Path(args.plot_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    configs = [
        BenchmarkConfig(
            label="STAC default",
            optimizer_kind="stac",
            color="#0f766e",
            linestyle="-",
        ),
        BenchmarkConfig(
            label="STAC full-decay trunk",
            optimizer_kind="stac",
            sign_weight_decay=1e-2,
            color="#b45309",
            linestyle=(0, (6, 2)),
        ),
        BenchmarkConfig(
            label="STAC wider cap",
            optimizer_kind="stac",
            last_n_ratio=0.25,
            color="#c2410c",
            linestyle="-.",
        ),
        BenchmarkConfig(
            label="AdamW baseline",
            optimizer_kind="adamw",
            color="#2563eb",
            linestyle=(0, (5, 2)),
        ),
    ]

    print(
        f"device={device.type} torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(device)} seeds={args.seeds} "
        f"epochs={args.epochs} steps_per_epoch={args.steps_per_epoch}"
    )

    task_names = (
        "deep_regression",
        "deep_classification",
        "tailnorm_classification",
        "sequence_classification",
    )
    tasks = {
        task_name: benchmark_task(
            task_name,
            configs=configs,
            seeds=args.seeds,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            device=device,
        )
        for task_name in task_names
    }
    memory = measure_peak_memory(
        configs=configs,
        device=device,
        batch_size=args.memory_batch_size,
    )

    benchmark_results = {
        "metadata": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_type": device.type,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "memory_batch_size": args.memory_batch_size,
            "model_init_seed_policy": (
                "paired trials with seeded teachers, seeded student "
                "initialization, and fixed batch schedules per seed"
            ),
            "sequence_autocast_dtype": (
                "bfloat16 when torch.cuda.is_bf16_supported() else disabled"
            ),
            "memory_measurement_scope": "optimizer.step() on the first state-allocating update",
            "task_names": list(task_names),
            "plot_path": str(plot_path),
            "json_path": str(json_path),
        },
        "configs": [asdict(config) for config in configs],
        "tasks": tasks,
        "memory": memory,
    }

    render_plot(
        benchmark_results=benchmark_results,
        configs=configs,
        output_path=plot_path,
    )
    json_path.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")

    for task_name, task_results in tasks.items():
        print_task_summary(task_name, task_results)
    print_memory_summary(memory)
    print(f"\nSaved JSON to {json_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
