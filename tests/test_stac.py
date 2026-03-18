from __future__ import annotations

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


def test_trunk_uses_signsgd_while_cap_matches_adamw() -> None:
    model = TwoLayerNet()
    with torch.no_grad():
        model.trunk.weight.fill_(1.0)
        model.head.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        last_n_layers=1,
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
