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
    last_n_modules: int = 1
    sign_lr_scale: float = 0.75
    sign_momentum: float = 0.9
    sign_state_dtype: str | None = None
    weight_decay: float = 1e-2
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


def resolve_device(requested: str) -> torch.device:
    if requested != "cuda":
        raise SystemExit("This benchmark is intended for CUDA. Use --device cuda.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark, but no GPU is available.")
    return torch.device("cuda")


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    teacher = RegressionTeacher()
    train_targets = teacher(train_inputs) + 0.2 * torch.randn(
        task.train_samples,
        3,
        generator=generator,
    )
    val_targets = teacher(val_inputs) + 0.2 * torch.randn(
        task.val_samples,
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
def make_classification_data(
    seed: int,
    *,
    task: TaskConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_inputs = torch.randn(task.train_samples, task.input_dim, generator=generator)
    val_inputs = torch.randn(task.val_samples, task.input_dim, generator=generator)
    teacher = ClassificationTeacher()
    train_targets = teacher(train_inputs).argmax(dim=1)
    val_targets = teacher(val_inputs).argmax(dim=1)

    label_noise = torch.rand(task.train_samples, generator=generator) < 0.08
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
def make_layernorm_classification_data(
    seed: int,
    *,
    task: TaskConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_inputs = torch.randn(task.train_samples, task.input_dim, generator=generator)
    val_inputs = torch.randn(task.val_samples, task.input_dim, generator=generator)
    teacher = LayerNormClassificationTeacher()
    train_targets = teacher(train_inputs).argmax(dim=1)
    val_targets = teacher(val_inputs).argmax(dim=1)

    label_noise = torch.rand(task.train_samples, generator=generator) < 0.10
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


def build_optimizer(
    model: nn.Module,
    config: BenchmarkConfig,
) -> torch.optim.Optimizer:
    if config.optimizer_kind == "stac":
        return STAC(
            model,
            lr=config.lr,
            last_n_modules=config.last_n_modules,
            sign_lr_scale=config.sign_lr_scale,
            sign_momentum=config.sign_momentum,
            sign_state_dtype=config.sign_state_dtype,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_kind == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
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
    model = RegressionStudent().to(device)
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
) -> dict[str, list[float]]:
    train_inputs, train_targets, val_inputs, val_targets = make_classification_data(
        seed,
        task=task,
        device=device,
    )
    schedule = batch_indices(
        task.train_samples,
        batch_size=task.batch_size,
        steps=task.steps_per_epoch * task.epochs,
        seed=20_000 + seed,
    )

    seed_all(40_000 + seed)
    model = ClassificationStudent().to(device)
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


def run_layernorm_classification_trial(
    seed: int,
    *,
    config: BenchmarkConfig,
    task: TaskConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    train_inputs, train_targets, val_inputs, val_targets = (
        make_layernorm_classification_data(
            seed,
            task=task,
            device=device,
        )
    )
    schedule = batch_indices(
        task.train_samples,
        batch_size=task.batch_size,
        steps=task.steps_per_epoch * task.epochs,
        seed=30_000 + seed,
    )

    seed_all(50_000 + seed)
    model = LayerNormClassificationStudent().to(device)
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
    if task_name == "regression":
        task = TaskConfig(
            name=task_name,
            train_samples=4096,
            val_samples=512,
            input_dim=24,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        runner = run_regression_trial
    elif task_name == "classification":
        task = TaskConfig(
            name=task_name,
            train_samples=4096,
            val_samples=512,
            input_dim=20,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        runner = run_classification_trial
    elif task_name == "layernorm_classification":
        task = TaskConfig(
            name=task_name,
            train_samples=4096,
            val_samples=512,
            input_dim=32,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        runner = run_layernorm_classification_trial
    else:
        raise ValueError(f"Unknown task: {task_name}.")

    config_results: dict[str, Any] = {}
    for config in configs:
        trials = [
            runner(seed, config=config, task=task, device=device)
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
        targets = torch.randn(batch_size, 8, device=device)

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)

        results.append(
            {
                "label": config.label,
                "optimizer_state_mb": optimizer_state_megabytes(optimizer),
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device)
                / (1024**2),
                "peak_reserved_mb": torch.cuda.max_memory_reserved(device)
                / (1024**2),
                "steady_allocated_mb": torch.cuda.memory_allocated(device)
                / (1024**2),
            }
        )

        del loss, predictions, optimizer, model, inputs, targets
        torch.cuda.empty_cache()

    return results


def render_plot(
    *,
    benchmark_results: dict[str, Any],
    configs: list[BenchmarkConfig],
    output_path: Path,
) -> None:
    task_names = tuple(benchmark_results["tasks"])
    figure, axes = plt.subplots(
        1,
        len(task_names),
        figsize=(5.7 * len(task_names), 4.8),
        dpi=180,
    )
    if len(task_names) == 1:
        axes = [axes]

    for axis, task_name in zip(axes, task_names, strict=True):
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

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
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
    print(
        "| Optimizer | Optimizer state MB | Peak allocated MB | Peak reserved MB |"
    )
    print("| --- | ---: | ---: | ---: |")
    for result in memory_results:
        print(
            f"| {result['label']} | {result['optimizer_state_mb']:.3f} | "
            f"{result['peak_allocated_mb']:.3f} | "
            f"{result['peak_reserved_mb']:.3f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CUDA-only research benchmark for STAC with train/validation "
            "splits, multiple seeds, epoch-by-epoch loss curves, and peak CUDA "
            "memory stats."
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
        default=12,
        help="Number of epochs to record in the loss curves.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=20,
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
        default=128,
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
            label="STAC default (last_n_modules=1)",
            optimizer_kind="stac",
            color="#0f766e",
            linestyle="-",
        ),
        BenchmarkConfig(
            label="STAC wider AdamW section (last_n_modules=2)",
            optimizer_kind="stac",
            last_n_modules=2,
            color="#c2410c",
            linestyle="-.",
        ),
        BenchmarkConfig(
            label="STAC bf16 sign state",
            optimizer_kind="stac",
            sign_state_dtype="bf16",
            color="#7c3aed",
            linestyle="--",
        ),
        BenchmarkConfig(
            label="STAC plain sign update",
            optimizer_kind="stac",
            sign_momentum=0.0,
            color="#b45309",
            linestyle=":",
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
        "regression",
        "classification",
        "layernorm_classification",
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
            "model_init_seed_policy": "per-trial seed matched across optimizers",
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
