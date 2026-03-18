from __future__ import annotations

import statistics

import torch
from torch import nn

from stac_optimizer import STAC


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk_0 = nn.Linear(16, 32)
        self.trunk_1 = nn.Linear(32, 32)
        self.head = nn.Linear(32, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.relu(self.trunk_0(inputs))
        inputs = torch.relu(self.trunk_1(inputs))
        return self.head(inputs)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batch(
    seed: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    inputs = torch.randn(256, 16)
    target_matrix = torch.randn(16, 4)
    targets = inputs @ target_matrix + 0.1 * torch.randn(256, 4)
    return inputs.to(device), targets.to(device)


def run(seed: int, optimizer_kind: str, *, device: torch.device) -> float:
    inputs, targets = make_batch(seed, device=device)
    torch.manual_seed(0)
    model = ToyMLP().to(device)

    if optimizer_kind == "stac-default":
        optimizer = STAC(model, lr=3e-3, last_n_layers=1, weight_decay=1e-2)
    elif optimizer_kind == "stac-plain":
        optimizer = STAC(
            model,
            lr=3e-3,
            last_n_layers=1,
            trunk_momentum=0.0,
            weight_decay=1e-2,
        )
    elif optimizer_kind == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-3,
            weight_decay=1e-2,
        )
    else:
        raise ValueError(f"Unknown optimizer kind: {optimizer_kind}.")

    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()

    return float(loss.detach())


def main() -> None:
    device = resolve_device()
    print(f"device={device.type} torch={torch.__version__}")
    for optimizer_kind in ("stac-default", "stac-plain", "adamw"):
        losses = [run(seed, optimizer_kind, device=device) for seed in range(5)]
        print(
            optimizer_kind,
            f"mean={statistics.fmean(losses):.6f}",
            f"min={min(losses):.6f}",
            f"max={max(losses):.6f}",
        )


if __name__ == "__main__":
    main()
