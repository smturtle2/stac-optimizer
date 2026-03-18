from __future__ import annotations

from copy import deepcopy
import math
import statistics

import pytest
import torch
from torch import nn

from stac_optimizer import STAC, partition_trainable_layers


class StackedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.block = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        self.head = nn.Linear(2, 1)


class TwoLayerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(1, 1, bias=False)
        self.head = nn.Linear(1, 1, bias=False)


class SparseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4, sparse=True)
        self.head = nn.Linear(4, 1, bias=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(indices).sum(dim=1)
        return self.head(embedded)


class TinyBenchmarkNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)
        self.head = nn.Linear(32, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.relu(self.fc1(inputs))
        inputs = torch.relu(self.fc2(inputs))
        return self.head(inputs)


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for STAC optimizer step tests.")
    return torch.device("cuda")


def test_partition_defaults_to_last_trainable_layer() -> None:
    model = StackedNet()

    partition = partition_trainable_layers(model)

    assert partition.trunk_layer_names == ("stem", "block.0", "block.2")
    assert partition.cap_layer_names == ("head",)


def test_partition_skips_frozen_layers_when_counting_last_n() -> None:
    model = StackedNet()
    for parameter in model.head.parameters():
        parameter.requires_grad = False

    partition = partition_trainable_layers(model, last_n_layers=1)

    assert partition.trunk_layer_names == ("stem", "block.0")
    assert partition.cap_layer_names == ("block.2",)


def test_last_n_zero_keeps_all_layers_in_signsgd_trunk() -> None:
    model = StackedNet()

    optimizer = STAC(model, lr=1e-3, last_n_layers=0)

    assert [group["stac_role"] for group in optimizer.param_groups] == ["trunk"]
    assert optimizer.partition.cap_layer_names == ()
    assert optimizer.partition.trunk_layer_names == (
        "stem",
        "block.0",
        "block.2",
        "head",
    )


def test_last_n_larger_than_layer_count_promotes_all_layers_to_cap() -> None:
    model = StackedNet()

    optimizer = STAC(model, lr=1e-3, last_n_layers=99)

    assert [group["stac_role"] for group in optimizer.param_groups] == ["cap"]
    assert optimizer.partition.trunk_layer_names == ()
    assert optimizer.partition.cap_layer_names == (
        "stem",
        "block.0",
        "block.2",
        "head",
    )


def test_default_hybrid_trunk_lr_is_scaled_down() -> None:
    model = TwoLayerNet()

    optimizer = STAC(model, lr=0.1, last_n_layers=1)

    assert optimizer.param_groups[0]["stac_role"] == "trunk"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.075)
    assert optimizer.param_groups[1]["stac_role"] == "cap"
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.1)


def test_negative_last_n_layers_is_rejected() -> None:
    model = StackedNet()

    with pytest.raises(ValueError, match="last_n_layers"):
        STAC(model, last_n_layers=-1)


def test_requires_at_least_one_trainable_parameter() -> None:
    model = StackedNet()
    for parameter in model.parameters():
        parameter.requires_grad = False

    with pytest.raises(ValueError, match="at least one trainable parameter"):
        STAC(model)


def test_trunk_uses_plain_signsgd_while_cap_matches_adamw(
    cuda_device: torch.device,
) -> None:
    model = TwoLayerNet().to(cuda_device)
    with torch.no_grad():
        model.trunk.weight.fill_(1.0)
        model.head.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        trunk_lr=0.1,
        last_n_layers=1,
        trunk_momentum=0.0,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01,
    )

    reference_parameter = nn.Parameter(torch.tensor([[1.0]], device=cuda_device))
    reference_optimizer = torch.optim.AdamW(
        [reference_parameter],
        lr=0.1,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01,
    )

    expected_trunk = 1.0
    gradients = (0.2, 0.4)
    for gradient_value in gradients:
        model.trunk.weight.grad = torch.tensor([[gradient_value]], device=cuda_device)
        model.head.weight.grad = torch.tensor([[gradient_value]], device=cuda_device)
        reference_parameter.grad = torch.tensor(
            [[gradient_value]],
            device=cuda_device,
        )

        optimizer.step()
        reference_optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)

        expected_trunk *= 1 - 0.1 * 0.01
        expected_trunk -= 0.1 * math.copysign(1.0, gradient_value)

    assert torch.allclose(
        model.trunk.weight.detach(),
        torch.tensor([[expected_trunk]], device=cuda_device),
        atol=1e-6,
    )
    assert torch.allclose(
        model.head.weight.detach(),
        reference_parameter.detach(),
        atol=1e-6,
    )


def test_trunk_momentum_uses_sign_of_accumulated_gradient(
    cuda_device: torch.device,
) -> None:
    model = TwoLayerNet().to(cuda_device)
    with torch.no_grad():
        model.trunk.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        trunk_lr=0.1,
        last_n_layers=1,
        trunk_momentum=0.5,
        weight_decay=0.01,
    )

    model.trunk.weight.grad = torch.tensor([[1.0]], device=cuda_device)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.trunk.weight.grad = torch.tensor([[-0.1]], device=cuda_device)
    optimizer.step()

    expected = 1.0
    expected *= 1 - 0.1 * 0.01
    expected -= 0.1
    expected *= 1 - 0.1 * 0.01
    expected -= 0.1

    assert torch.allclose(
        model.trunk.weight.detach(),
        torch.tensor([[expected]], device=cuda_device),
        atol=1e-6,
    )


def test_role_specific_learning_rates_are_applied(
    cuda_device: torch.device,
) -> None:
    model = TwoLayerNet().to(cuda_device)
    with torch.no_grad():
        model.trunk.weight.fill_(1.0)
        model.head.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        trunk_lr=0.2,
        cap_lr=0.05,
        last_n_layers=1,
        trunk_momentum=0.0,
    )

    reference_parameter = nn.Parameter(torch.tensor([[1.0]], device=cuda_device))
    reference_optimizer = torch.optim.AdamW(
        [reference_parameter],
        lr=0.05,
        weight_decay=0.0,
    )

    model.trunk.weight.grad = torch.tensor([[1.0]], device=cuda_device)
    model.head.weight.grad = torch.tensor([[1.0]], device=cuda_device)
    reference_parameter.grad = torch.tensor([[1.0]], device=cuda_device)

    optimizer.step()
    reference_optimizer.step()

    assert torch.allclose(
        model.trunk.weight.detach(),
        torch.tensor([[0.8]], device=cuda_device),
        atol=1e-6,
    )
    assert torch.allclose(
        model.head.weight.detach(),
        reference_parameter.detach(),
        atol=1e-6,
    )


def test_state_dict_round_trip_preserves_optimizer_behavior(
    cuda_device: torch.device,
) -> None:
    torch.manual_seed(0)
    model_a = TwoLayerNet().to(cuda_device)
    torch.manual_seed(0)
    model_b = TwoLayerNet().to(cuda_device)

    optimizer_a = STAC(
        model_a,
        lr=0.1,
        last_n_layers=1,
        trunk_momentum=0.8,
        betas=(0.7, 0.9),
        weight_decay=0.01,
    )
    optimizer_b = STAC(
        model_b,
        lr=0.1,
        last_n_layers=1,
        trunk_momentum=0.8,
        betas=(0.7, 0.9),
        weight_decay=0.01,
    )

    model_a.trunk.weight.grad = torch.tensor([[0.4]], device=cuda_device)
    model_a.head.weight.grad = torch.tensor([[0.4]], device=cuda_device)
    optimizer_a.step()

    model_b.load_state_dict(model_a.state_dict())
    optimizer_b.load_state_dict(deepcopy(optimizer_a.state_dict()))

    model_a.trunk.weight.grad = torch.tensor([[-0.1]], device=cuda_device)
    model_a.head.weight.grad = torch.tensor([[-0.1]], device=cuda_device)
    model_b.trunk.weight.grad = torch.tensor([[-0.1]], device=cuda_device)
    model_b.head.weight.grad = torch.tensor([[-0.1]], device=cuda_device)

    optimizer_a.step()
    optimizer_b.step()

    assert torch.allclose(
        model_a.trunk.weight.detach(),
        model_b.trunk.weight.detach(),
        atol=1e-6,
    )
    assert torch.allclose(
        model_a.head.weight.detach(),
        model_b.head.weight.detach(),
        atol=1e-6,
    )


def test_amsgrad_cap_matches_torch_adamw(cuda_device: torch.device) -> None:
    model = TwoLayerNet().to(cuda_device)
    with torch.no_grad():
        model.trunk.weight.fill_(1.0)
        model.head.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        last_n_layers=2,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=True,
    )

    reference_parameter = nn.Parameter(torch.tensor([[1.0]], device=cuda_device))
    reference_optimizer = torch.optim.AdamW(
        [reference_parameter],
        lr=0.1,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=True,
    )

    for gradient_value in (2.0, 0.1, 0.1):
        model.head.weight.grad = torch.tensor([[gradient_value]], device=cuda_device)
        reference_parameter.grad = torch.tensor(
            [[gradient_value]],
            device=cuda_device,
        )

        optimizer.step()
        reference_optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)

    assert torch.allclose(
        model.head.weight.detach(),
        reference_parameter.detach(),
        atol=1e-6,
    )


def test_add_param_group_is_rejected() -> None:
    model = TwoLayerNet()
    optimizer = STAC(model)
    extra_parameter = nn.Parameter(torch.tensor([[1.0]]))

    with pytest.raises(RuntimeError, match="does not support add_param_group"):
        optimizer.add_param_group({"params": [extra_parameter]})


def test_sparse_gradients_are_rejected_in_both_roles(
    cuda_device: torch.device,
) -> None:
    indices = torch.tensor([[0, 1], [2, 3]], device=cuda_device)

    trunk_model = SparseNet().to(cuda_device)
    trunk_optimizer = STAC(trunk_model, lr=0.1, last_n_layers=1)
    trunk_loss = trunk_model(indices).sum()
    trunk_loss.backward()

    with pytest.raises(RuntimeError, match="sparse gradients"):
        trunk_optimizer.step()

    cap_model = SparseNet().to(cuda_device)
    cap_optimizer = STAC(cap_model, lr=0.1, last_n_layers=2)
    cap_loss = cap_model(indices).sum()
    cap_loss.backward()

    with pytest.raises(RuntimeError, match="sparse gradients"):
        cap_optimizer.step()


def test_nonfinite_gradients_can_raise(cuda_device: torch.device) -> None:
    model = TwoLayerNet().to(cuda_device)
    optimizer = STAC(
        model,
        lr=0.1,
        last_n_layers=1,
        error_if_nonfinite=True,
    )

    model.trunk.weight.grad = torch.tensor([[float("nan")]], device=cuda_device)
    model.head.weight.grad = torch.tensor([[1.0]], device=cuda_device)

    with pytest.raises(RuntimeError, match="non-finite gradients"):
        optimizer.step()


def test_load_state_dict_rejects_partition_mismatch(
    cuda_device: torch.device,
) -> None:
    model = TwoLayerNet().to(cuda_device)
    stateful_optimizer = STAC(model, lr=0.1, last_n_layers=1)
    mismatched_optimizer = STAC(model, lr=0.1, last_n_layers=0)

    with pytest.raises(ValueError, match="parameter groups"):
        mismatched_optimizer.load_state_dict(stateful_optimizer.state_dict())


def test_default_trunk_is_more_effective_than_plain_sign_on_cuda(
    cuda_device: torch.device,
) -> None:
    def make_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        inputs = torch.randn(512, 16, generator=generator)
        target_matrix = torch.randn(16, 4, generator=generator)
        targets = inputs @ target_matrix + 0.1 * torch.randn(
            512,
            4,
            generator=generator,
        )
        return inputs.to(cuda_device), targets.to(cuda_device)

    def run_trial(seed: int, trunk_momentum: float) -> float:
        inputs, targets = make_batch(seed)
        torch.manual_seed(1234)
        model = TinyBenchmarkNet().to(cuda_device)
        optimizer = STAC(
            model,
            lr=3e-3,
            last_n_layers=1,
            trunk_momentum=trunk_momentum,
            weight_decay=1e-2,
        )

        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = torch.nn.functional.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()

        return float(loss.detach().cpu())

    default_losses = [run_trial(seed, trunk_momentum=0.9) for seed in range(3)]
    plain_losses = [run_trial(seed, trunk_momentum=0.0) for seed in range(3)]

    assert statistics.fmean(default_losses) < statistics.fmean(plain_losses) * 0.5
