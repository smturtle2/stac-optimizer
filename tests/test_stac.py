from __future__ import annotations

from copy import deepcopy
import math
import statistics

import pytest
import torch
from torch import nn

from stac_optimizer import STAC, partition_trainable_modules


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
        self.base = nn.Linear(1, 1, bias=False)
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


class RootOwnedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.root_scale = nn.Parameter(torch.tensor([1.0]))
        self.head = nn.Linear(1, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs * self.root_scale)


class SharedParameterOwner(nn.Module):
    def __init__(self, shared_weight: nn.Parameter) -> None:
        super().__init__()
        self.weight = shared_weight


class SharedParameterNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared_weight = nn.Parameter(torch.ones(1, 1))
        self.first = SharedParameterOwner(shared_weight)
        self.second = SharedParameterOwner(shared_weight)


class ShapeShiftNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(2, 1, bias=False)
        self.head = nn.Linear(1, 2, bias=False)


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


def run_benchmark_trial(
    seed: int,
    *,
    device: torch.device,
    sign_momentum: float,
    task_kind: str,
) -> float:
    if task_kind == "regression":
        inputs, targets = make_regression_batch(seed, device=device)
        loss_fn = torch.nn.functional.mse_loss
    elif task_kind == "classification":
        inputs, targets = make_classification_batch(seed, device=device)
        loss_fn = torch.nn.functional.cross_entropy
    else:
        raise ValueError(f"Unknown task kind: {task_kind}.")

    torch.manual_seed(1234)
    model = TinyBenchmarkNet().to(device)
    optimizer = STAC(
        model,
        lr=3e-3,
        last_n_modules=1,
        sign_momentum=sign_momentum,
        weight_decay=1e-2,
    )

    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    return float(loss.detach().cpu())


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for STAC optimizer step tests.")
    return torch.device("cuda")


def test_partition_defaults_to_last_trainable_module() -> None:
    model = StackedNet()

    partition = partition_trainable_modules(model)

    assert partition.sign_module_names == ("stem", "block.0", "block.2")
    assert partition.adamw_module_names == ("head",)


def test_partition_skips_frozen_modules_when_counting_last_n() -> None:
    model = StackedNet()
    for parameter in model.head.parameters():
        parameter.requires_grad = False

    partition = partition_trainable_modules(model, last_n_modules=1)

    assert partition.sign_module_names == ("stem", "block.0")
    assert partition.adamw_module_names == ("block.2",)


def test_partition_exposes_root_owned_parameters() -> None:
    model = RootOwnedNet()

    partition = partition_trainable_modules(model, last_n_modules=1)

    assert partition.sign_module_names == ("<root>",)
    assert partition.sign_parameter_names == ("root_scale",)
    assert partition.adamw_module_names == ("head",)
    assert partition.adamw_parameter_names == ("head.weight",)


def test_partition_assigns_shared_parameters_to_first_owner() -> None:
    model = SharedParameterNet()

    partition = partition_trainable_modules(model, last_n_modules=0)

    assert partition.sign_module_names == ("first",)
    assert partition.sign_parameter_names == ("first.weight",)


def test_last_n_zero_keeps_all_modules_in_sign_section() -> None:
    model = StackedNet()

    optimizer = STAC(model, lr=1e-3, last_n_modules=0)

    assert [group["stac_role"] for group in optimizer.param_groups] == ["sign"]
    assert optimizer.partition.adamw_module_names == ()
    assert optimizer.partition.sign_module_names == (
        "stem",
        "block.0",
        "block.2",
        "head",
    )


def test_last_n_larger_than_module_count_promotes_all_modules_to_adamw() -> None:
    model = StackedNet()

    optimizer = STAC(model, lr=1e-3, last_n_modules=99)

    assert [group["stac_role"] for group in optimizer.param_groups] == ["adamw"]
    assert optimizer.partition.sign_module_names == ()
    assert optimizer.partition.adamw_module_names == (
        "stem",
        "block.0",
        "block.2",
        "head",
    )


def test_default_hybrid_sign_group_lr_is_scaled_down() -> None:
    model = TwoLayerNet()

    optimizer = STAC(model, lr=0.1, last_n_modules=1)

    assert optimizer.param_groups[0]["stac_role"] == "sign"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.075)
    assert optimizer.param_groups[1]["stac_role"] == "adamw"
    assert optimizer.param_groups[1]["lr"] == pytest.approx(0.1)


def test_negative_last_n_modules_is_rejected() -> None:
    model = StackedNet()

    with pytest.raises(ValueError, match="last_n_modules"):
        STAC(model, last_n_modules=-1)


def test_last_n_modules_is_stored_on_optimizer() -> None:
    model = StackedNet()

    optimizer = STAC(model, last_n_modules=2)

    assert optimizer.last_n_modules == 2
    assert optimizer.partition.sign_module_names == ("stem", "block.0")
    assert optimizer.partition.adamw_module_names == ("block.2", "head")


def test_invalid_sign_state_dtype_is_rejected() -> None:
    model = StackedNet()

    with pytest.raises(ValueError, match="sign_state_dtype"):
        STAC(model, sign_state_dtype=torch.int64)

    with pytest.raises(ValueError, match="sign_state_dtype"):
        STAC(model, sign_state_dtype="nope")


def test_requires_at_least_one_trainable_parameter() -> None:
    model = StackedNet()
    for parameter in model.parameters():
        parameter.requires_grad = False

    with pytest.raises(ValueError, match="at least one trainable parameter"):
        STAC(model)


def test_sign_section_uses_plain_signsgd_while_adamw_matches_adamw(
    cuda_device: torch.device,
) -> None:
    model = TwoLayerNet().to(cuda_device)
    with torch.no_grad():
        model.base.weight.fill_(1.0)
        model.head.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        last_n_modules=1,
        sign_momentum=0.0,
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

    expected_sign = 1.0
    expected_sign_lr = 0.1 * 0.75
    gradients = (0.2, 0.4)
    for gradient_value in gradients:
        model.base.weight.grad = torch.tensor([[gradient_value]], device=cuda_device)
        model.head.weight.grad = torch.tensor([[gradient_value]], device=cuda_device)
        reference_parameter.grad = torch.tensor(
            [[gradient_value]],
            device=cuda_device,
        )

        optimizer.step()
        reference_optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)

        expected_sign *= 1 - expected_sign_lr * 0.01
        expected_sign -= expected_sign_lr * math.copysign(1.0, gradient_value)

    assert torch.allclose(
        model.base.weight.detach(),
        torch.tensor([[expected_sign]], device=cuda_device),
        atol=1e-6,
    )
    assert torch.allclose(
        model.head.weight.detach(),
        reference_parameter.detach(),
        atol=1e-6,
    )


def test_sign_momentum_uses_sign_of_accumulated_gradient(
    cuda_device: torch.device,
) -> None:
    model = TwoLayerNet().to(cuda_device)
    with torch.no_grad():
        model.base.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        last_n_modules=1,
        sign_momentum=0.5,
        weight_decay=0.01,
    )

    model.base.weight.grad = torch.tensor([[1.0]], device=cuda_device)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.base.weight.grad = torch.tensor([[-0.1]], device=cuda_device)
    optimizer.step()

    expected = 1.0
    expected *= 1 - 0.075 * 0.01
    expected -= 0.075
    expected *= 1 - 0.075 * 0.01
    expected -= 0.075

    assert torch.allclose(
        model.base.weight.detach(),
        torch.tensor([[expected]], device=cuda_device),
        atol=1e-6,
    )


def test_role_specific_weight_decay_is_applied(
    cuda_device: torch.device,
) -> None:
    model = TwoLayerNet().to(cuda_device)
    with torch.no_grad():
        model.base.weight.fill_(1.0)
        model.head.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        last_n_modules=1,
        sign_momentum=0.0,
        sign_weight_decay=0.2,
        adamw_weight_decay=0.0,
    )

    model.base.weight.grad = torch.zeros((1, 1), device=cuda_device)
    model.head.weight.grad = torch.zeros((1, 1), device=cuda_device)

    optimizer.step()

    assert torch.allclose(
        model.base.weight.detach(),
        torch.tensor([[0.985]], device=cuda_device),
        atol=1e-6,
    )
    assert torch.allclose(
        model.head.weight.detach(),
        torch.tensor([[1.0]], device=cuda_device),
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
        last_n_modules=1,
        sign_momentum=0.8,
        betas=(0.7, 0.9),
        weight_decay=0.01,
    )
    optimizer_b = STAC(
        model_b,
        lr=0.1,
        last_n_modules=1,
        sign_momentum=0.8,
        betas=(0.7, 0.9),
        weight_decay=0.01,
    )

    model_a.base.weight.grad = torch.tensor([[0.4]], device=cuda_device)
    model_a.head.weight.grad = torch.tensor([[0.4]], device=cuda_device)
    optimizer_a.step()

    model_b.load_state_dict(model_a.state_dict())
    optimizer_b.load_state_dict(deepcopy(optimizer_a.state_dict()))

    model_a.base.weight.grad = torch.tensor([[-0.1]], device=cuda_device)
    model_a.head.weight.grad = torch.tensor([[-0.1]], device=cuda_device)
    model_b.base.weight.grad = torch.tensor([[-0.1]], device=cuda_device)
    model_b.head.weight.grad = torch.tensor([[-0.1]], device=cuda_device)

    optimizer_a.step()
    optimizer_b.step()

    assert torch.allclose(
        model_a.base.weight.detach(),
        model_b.base.weight.detach(),
        atol=1e-6,
    )
    assert torch.allclose(
        model_a.head.weight.detach(),
        model_b.head.weight.detach(),
        atol=1e-6,
    )


def test_step_supports_closure(cuda_device: torch.device) -> None:
    model = TwoLayerNet().to(cuda_device)
    optimizer = STAC(model, lr=0.1, last_n_modules=1)
    inputs = torch.ones((1, 1), device=cuda_device)
    targets = torch.zeros((1, 1), device=cuda_device)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        predictions = model.head(model.base(inputs))
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        return loss

    starting_weight = model.head.weight.detach().clone()
    loss = optimizer.step(closure)

    assert isinstance(loss, torch.Tensor)
    assert float(loss.detach().cpu()) > 0.0
    assert not torch.allclose(model.head.weight.detach(), starting_weight)


def test_maximize_reverses_update_direction(cuda_device: torch.device) -> None:
    model = TwoLayerNet().to(cuda_device)
    with torch.no_grad():
        model.base.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        last_n_modules=0,
        sign_momentum=0.0,
        maximize=True,
    )

    model.base.weight.grad = torch.tensor([[1.0]], device=cuda_device)
    optimizer.step()

    assert torch.allclose(
        model.base.weight.detach(),
        torch.tensor([[1.1]], device=cuda_device),
        atol=1e-6,
    )


def test_adamw_section_maximize_matches_torch_adamw(
    cuda_device: torch.device,
) -> None:
    model = TwoLayerNet().to(cuda_device)
    with torch.no_grad():
        model.head.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        last_n_modules=2,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01,
        maximize=True,
    )

    reference_parameter = nn.Parameter(torch.tensor([[1.0]], device=cuda_device))
    reference_optimizer = torch.optim.AdamW(
        [reference_parameter],
        lr=0.1,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01,
        maximize=True,
    )

    for gradient_value in (0.5, -0.25):
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


def test_sign_state_dtype_can_use_bfloat16(cuda_device: torch.device) -> None:
    model = TwoLayerNet().to(cuda_device)
    optimizer = STAC(
        model,
        lr=0.1,
        last_n_modules=0,
        sign_momentum=0.9,
        sign_state_dtype=torch.bfloat16,
    )

    model.base.weight.grad = torch.tensor([[1.0]], device=cuda_device)
    model.head.weight.grad = torch.tensor([[1.0]], device=cuda_device)
    optimizer.step()

    assert (
        optimizer.state[model.base.weight]["sign_momentum_buffer"].dtype
        == torch.bfloat16
    )
    assert (
        optimizer.state[model.head.weight]["sign_momentum_buffer"].dtype
        == torch.bfloat16
    )


def test_amsgrad_adamw_section_matches_torch_adamw(cuda_device: torch.device) -> None:
    model = TwoLayerNet().to(cuda_device)
    with torch.no_grad():
        model.base.weight.fill_(1.0)
        model.head.weight.fill_(1.0)

    optimizer = STAC(
        model,
        lr=0.1,
        last_n_modules=2,
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


def test_sparse_gradients_are_rejected_in_both_sections(
    cuda_device: torch.device,
) -> None:
    indices = torch.tensor([[0, 1], [2, 3]], device=cuda_device)

    sign_model = SparseNet().to(cuda_device)
    sign_optimizer = STAC(sign_model, lr=0.1, last_n_modules=1)
    sign_loss = sign_model(indices).sum()
    sign_loss.backward()

    with pytest.raises(RuntimeError, match="sparse gradients"):
        sign_optimizer.step()

    adamw_model = SparseNet().to(cuda_device)
    adamw_optimizer = STAC(adamw_model, lr=0.1, last_n_modules=2)
    adamw_loss = adamw_model(indices).sum()
    adamw_loss.backward()

    with pytest.raises(RuntimeError, match="sparse gradients"):
        adamw_optimizer.step()


def test_nonfinite_gradients_can_raise(cuda_device: torch.device) -> None:
    model = TwoLayerNet().to(cuda_device)
    optimizer = STAC(
        model,
        lr=0.1,
        last_n_modules=1,
        error_if_nonfinite=True,
    )

    model.base.weight.grad = torch.tensor([[float("nan")]], device=cuda_device)
    model.head.weight.grad = torch.tensor([[1.0]], device=cuda_device)

    with pytest.raises(RuntimeError, match="non-finite gradients"):
        optimizer.step()


def test_nonfinite_gradients_skip_the_entire_step_by_default(
    cuda_device: torch.device,
) -> None:
    model = TwoLayerNet().to(cuda_device)
    with torch.no_grad():
        model.base.weight.fill_(1.0)
        model.head.weight.fill_(1.0)

    optimizer = STAC(model, lr=0.1, last_n_modules=1)
    model.base.weight.grad = torch.tensor([[float("nan")]], device=cuda_device)
    model.head.weight.grad = torch.tensor([[1.0]], device=cuda_device)

    optimizer.step()

    assert optimizer.nonfinite_skipped_steps == 1
    assert torch.allclose(
        model.base.weight.detach(),
        torch.tensor([[1.0]], device=cuda_device),
        atol=1e-6,
    )
    assert torch.allclose(
        model.head.weight.detach(),
        torch.tensor([[1.0]], device=cuda_device),
        atol=1e-6,
    )
    assert model.head.weight not in optimizer.state


def test_load_state_dict_rejects_partition_mismatch(
    cuda_device: torch.device,
) -> None:
    model = TwoLayerNet().to(cuda_device)
    stateful_optimizer = STAC(model, lr=0.1, last_n_modules=1)
    mismatched_optimizer = STAC(model, lr=0.1, last_n_modules=0)

    with pytest.raises(ValueError, match="parameter groups"):
        mismatched_optimizer.load_state_dict(stateful_optimizer.state_dict())


def test_load_state_dict_rejects_parameter_shape_mismatch() -> None:
    source_model = TwoLayerNet()
    source_optimizer = STAC(source_model, lr=0.1, last_n_modules=1)
    source_model.base.weight.grad = torch.tensor([[1.0]])
    source_model.head.weight.grad = torch.tensor([[1.0]])
    source_optimizer.step()

    mismatched_model = ShapeShiftNet()
    mismatched_optimizer = STAC(mismatched_model, lr=0.1, last_n_modules=1)

    with pytest.raises(ValueError, match="parameter shapes"):
        mismatched_optimizer.load_state_dict(deepcopy(source_optimizer.state_dict()))


def test_load_state_dict_rejects_parameter_name_mismatch(
    cuda_device: torch.device,
) -> None:
    model = TwoLayerNet().to(cuda_device)
    optimizer = STAC(model, lr=0.1, last_n_modules=1)
    state_dict = deepcopy(optimizer.state_dict())
    state_dict["param_groups"][0]["param_names"] = ("wrong_name",)

    with pytest.raises(ValueError, match="parameter names"):
        optimizer.load_state_dict(state_dict)


@pytest.mark.parametrize("task_kind", ["regression", "classification"])
def test_default_sign_momentum_is_more_effective_than_plain_sign_across_cuda_tasks(
    cuda_device: torch.device,
    task_kind: str,
) -> None:
    default_losses = [
        run_benchmark_trial(
            seed,
            device=cuda_device,
            sign_momentum=0.9,
            task_kind=task_kind,
        )
        for seed in range(3)
    ]
    plain_losses = [
        run_benchmark_trial(
            seed,
            device=cuda_device,
            sign_momentum=0.0,
            task_kind=task_kind,
        )
        for seed in range(3)
    ]

    assert statistics.fmean(default_losses) < statistics.fmean(plain_losses) * 0.75
