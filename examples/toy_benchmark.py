from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
from dataclasses import dataclass

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
    lr: float = 3e-3
    last_n_layers: int = 1
    sign_momentum: float = 0.9
    weight_decay: float = 1e-2


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_0 = nn.Linear(16, 32)
        self.block_1 = nn.Linear(32, 32)
        self.head = nn.Linear(32, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.relu(self.block_0(inputs))
        inputs = torch.relu(self.block_1(inputs))
        return self.head(inputs)


class StateMemoryNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
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
        return self.network(inputs)


def resolve_device(requested: str) -> torch.device:
    if requested != "cuda":
        raise SystemExit("This benchmark is intended for CUDA. Use --device cuda.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark, but no GPU is available.")
    return torch.device("cuda")


def make_regression_batch(
    seed: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(384, 16, generator=generator)
    target_matrix = torch.randn(16, 4, generator=generator)
    targets = inputs @ target_matrix + 0.1 * torch.randn(
        384,
        4,
        generator=generator,
    )
    return inputs.to(device), targets.to(device)


def make_classification_batch(
    seed: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(768, 16, generator=generator)
    class_matrix = torch.randn(16, 4, generator=generator)
    logits = inputs @ class_matrix + 0.25 * torch.randn(
        768,
        4,
        generator=generator,
    )
    targets = logits.argmax(dim=1)
    return inputs.to(device), targets.to(device)


def build_optimizer(
    model: nn.Module,
    config: BenchmarkConfig,
) -> torch.optim.Optimizer:
    if config.optimizer_kind == "stac":
        return STAC(
            model,
            lr=config.lr,
            last_n_layers=config.last_n_layers,
            sign_momentum=config.sign_momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_kind == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer kind: {config.optimizer_kind}.")


def run_trial(
    seed: int,
    *,
    config: BenchmarkConfig,
    device: torch.device,
    task_kind: str,
    steps: int,
) -> tuple[float, float | None]:
    if task_kind == "regression":
        inputs, targets = make_regression_batch(seed, device=device)
        loss_fn = torch.nn.functional.mse_loss
    elif task_kind == "classification":
        inputs, targets = make_classification_batch(seed, device=device)
        loss_fn = torch.nn.functional.cross_entropy
    else:
        raise ValueError(f"Unknown task kind: {task_kind}.")

    torch.manual_seed(1234)
    model = ToyMLP().to(device)
    optimizer = build_optimizer(model, config)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    final_loss = float(loss.detach().cpu())
    if task_kind == "classification":
        with torch.no_grad():
            accuracy = float(
                (model(inputs).argmax(dim=1) == targets).float().mean().cpu()
            )
        return final_loss, accuracy
    return final_loss, None


def summarize_task(
    task_kind: str,
    *,
    configs: list[BenchmarkConfig],
    device: torch.device,
    seeds: int,
    steps: int,
) -> None:
    print(f"\n## {task_kind}")
    print("| Optimizer | Mean loss | Min loss | Max loss | Mean accuracy |")
    print("| --- | ---: | ---: | ---: | ---: |")

    for config in configs:
        losses: list[float] = []
        accuracies: list[float] = []
        for seed in range(seeds):
            loss, accuracy = run_trial(
                seed,
                config=config,
                device=device,
                task_kind=task_kind,
                steps=steps,
            )
            losses.append(loss)
            if accuracy is not None:
                accuracies.append(accuracy)

        accuracy_text = (
            f"{statistics.fmean(accuracies):.4f}" if accuracies else "n/a"
        )
        print(
            f"| {config.label} | {statistics.fmean(losses):.6f} | "
            f"{min(losses):.6f} | {max(losses):.6f} | {accuracy_text} |"
        )


def optimizer_state_megabytes(optimizer: torch.optim.Optimizer) -> float:
    total_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    return total_bytes / (1024**2)


def summarize_state_memory(
    *,
    configs: list[BenchmarkConfig],
    device: torch.device,
) -> None:
    print("\n## optimizer-state-memory")
    print("| Optimizer | Optimizer state MB |")
    print("| --- | ---: |")

    for config in configs:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = StateMemoryNet().to(device)
        optimizer = build_optimizer(model, config)
        inputs = torch.randn(128, 256, device=device)
        targets = torch.randn(128, 8, device=device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()

        print(
            f"| {config.label} | {optimizer_state_megabytes(optimizer):.3f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small CUDA benchmark suite for STAC on synthetic regression "
            "and classification tasks."
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
        help="Number of random seeds to evaluate per optimizer/task pair.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="Number of optimization steps per trial.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(
        f"device={device.type} torch={torch.__version__} "
        f"cuda={torch.version.cuda} seeds={args.seeds} steps={args.steps}"
    )

    configs = [
        BenchmarkConfig(
            label="STAC default (last_n_layers=1)",
            optimizer_kind="stac",
        ),
        BenchmarkConfig(
            label="STAC plain sign update",
            optimizer_kind="stac",
            sign_momentum=0.0,
        ),
        BenchmarkConfig(
            label="STAC wider AdamW section (last_n_layers=2)",
            optimizer_kind="stac",
            last_n_layers=2,
        ),
        BenchmarkConfig(
            label="AdamW baseline",
            optimizer_kind="adamw",
        ),
    ]

    for task_kind in ("regression", "classification"):
        summarize_task(
            task_kind,
            configs=configs,
            device=device,
            seeds=args.seeds,
            steps=args.steps,
        )
    summarize_state_memory(configs=configs, device=device)


if __name__ == "__main__":
    main()
