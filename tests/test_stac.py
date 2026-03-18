from __future__ import annotations

from copy import deepcopy
import math

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


def test_trunk_uses_plain_signsgd_while_cap_matches_adamw() -> None:
    model = TwoLayerNet()
    with torch.no_grad():
        model.trunk.weight.fill_(1.0)
        model.head.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        last_n_layers=1,
        trunk_momentum=0.0,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01,
    )

    reference_parameter = nn.Parameter(torch.tensor([[1.0]]))
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
        model.trunk.weight.grad = torch.tensor([[gradient_value]])
        model.head.weight.grad = torch.tensor([[gradient_value]])
        reference_parameter.grad = torch.tensor([[gradient_value]])

        optimizer.step()
        reference_optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)

        expected_trunk *= 1 - 0.1 * 0.01
        expected_trunk -= 0.1 * math.copysign(1.0, gradient_value)

    assert torch.allclose(
        model.trunk.weight.detach(),
        torch.tensor([[expected_trunk]]),
        atol=1e-6,
    )
    assert torch.allclose(
        model.head.weight.detach(),
        reference_parameter.detach(),
        atol=1e-6,
    )


def test_trunk_momentum_uses_sign_of_accumulated_gradient() -> None:
    model = TwoLayerNet()
    with torch.no_grad():
        model.trunk.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        last_n_layers=1,
        trunk_momentum=0.5,
        weight_decay=0.01,
    )

    model.trunk.weight.grad = torch.tensor([[1.0]])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.trunk.weight.grad = torch.tensor([[-0.1]])
    optimizer.step()

    expected = 1.0
    expected *= 1 - 0.1 * 0.01
    expected -= 0.1
    expected *= 1 - 0.1 * 0.01
    expected -= 0.1

    assert torch.allclose(
        model.trunk.weight.detach(),
        torch.tensor([[expected]]),
        atol=1e-6,
    )


def test_role_specific_learning_rates_are_applied() -> None:
    model = TwoLayerNet()
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

    reference_parameter = nn.Parameter(torch.tensor([[1.0]]))
    reference_optimizer = torch.optim.AdamW(
        [reference_parameter],
        lr=0.05,
        weight_decay=0.0,
    )

    model.trunk.weight.grad = torch.tensor([[1.0]])
    model.head.weight.grad = torch.tensor([[1.0]])
    reference_parameter.grad = torch.tensor([[1.0]])

    optimizer.step()
    reference_optimizer.step()

    assert torch.allclose(
        model.trunk.weight.detach(),
        torch.tensor([[0.8]]),
        atol=1e-6,
    )
    assert torch.allclose(
        model.head.weight.detach(),
        reference_parameter.detach(),
        atol=1e-6,
    )


def test_state_dict_round_trip_preserves_optimizer_behavior() -> None:
    torch.manual_seed(0)
    model_a = TwoLayerNet()
    torch.manual_seed(0)
    model_b = TwoLayerNet()

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

    model_a.trunk.weight.grad = torch.tensor([[0.4]])
    model_a.head.weight.grad = torch.tensor([[0.4]])
    optimizer_a.step()

    model_b.load_state_dict(model_a.state_dict())
    optimizer_b.load_state_dict(deepcopy(optimizer_a.state_dict()))

    model_a.trunk.weight.grad = torch.tensor([[-0.1]])
    model_a.head.weight.grad = torch.tensor([[-0.1]])
    model_b.trunk.weight.grad = torch.tensor([[-0.1]])
    model_b.head.weight.grad = torch.tensor([[-0.1]])

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


def test_add_param_group_is_rejected() -> None:
    model = TwoLayerNet()
    optimizer = STAC(model)
    extra_parameter = nn.Parameter(torch.tensor([[1.0]]))

    with pytest.raises(RuntimeError, match="does not support add_param_group"):
        optimizer.add_param_group({"params": [extra_parameter]})


def test_sparse_gradients_are_rejected_in_both_roles() -> None:
    indices = torch.tensor([[0, 1], [2, 3]])

    trunk_model = SparseNet()
    trunk_optimizer = STAC(trunk_model, lr=0.1, last_n_layers=1)
    trunk_loss = trunk_model(indices).sum()
    trunk_loss.backward()

    with pytest.raises(RuntimeError, match="sparse gradients"):
        trunk_optimizer.step()

    cap_model = SparseNet()
    cap_optimizer = STAC(cap_model, lr=0.1, last_n_layers=2)
    cap_loss = cap_model(indices).sum()
    cap_loss.backward()

    with pytest.raises(RuntimeError, match="sparse gradients"):
        cap_optimizer.step()


def test_nonfinite_gradients_can_raise() -> None:
    model = TwoLayerNet()
    optimizer = STAC(
        model,
        lr=0.1,
        last_n_layers=1,
        error_if_nonfinite=True,
    )

    model.trunk.weight.grad = torch.tensor([[float("nan")]])
    model.head.weight.grad = torch.tensor([[1.0]])

    with pytest.raises(RuntimeError, match="non-finite gradients"):
        optimizer.step()
