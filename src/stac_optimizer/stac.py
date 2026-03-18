from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class LayerGroup:
    name: str
    parameter_names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]


@dataclass(frozen=True)
class STACPartition:
    trunk_layers: tuple[LayerGroup, ...]
    cap_layers: tuple[LayerGroup, ...]

    @property
    def trunk_layer_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.trunk_layers)

    @property
    def cap_layer_names(self) -> tuple[str, ...]:
        return tuple(layer.name for layer in self.cap_layers)

    @property
    def trunk_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for layer in self.trunk_layers
            for parameter in layer.parameters
        )

    @property
    def cap_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for layer in self.cap_layers
            for parameter in layer.parameters
        )


def partition_trainable_layers(
    model: nn.Module,
    *,
    last_n_layers: int = 1,
) -> STACPartition:
    if last_n_layers < 0:
        raise ValueError("last_n_layers must be greater than or equal to 0.")

    seen_parameters: set[int] = set()
    layer_groups: list[LayerGroup] = []

    for module_name, module in model.named_modules():
        parameter_entries: list[tuple[str, nn.Parameter]] = []
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue

            parameter_id = id(parameter)
            if parameter_id in seen_parameters:
                continue

            seen_parameters.add(parameter_id)
            qualified_name = (
                f"{module_name}.{parameter_name}" if module_name else parameter_name
            )
            parameter_entries.append((qualified_name, parameter))

        if parameter_entries:
            names, parameters = zip(*parameter_entries)
            layer_groups.append(
                LayerGroup(
                    name=module_name or "<root>",
                    parameter_names=tuple(names),
                    parameters=tuple(parameters),
                )
            )

    if not layer_groups:
        raise ValueError("STAC requires at least one trainable parameter.")

    trunk_count = max(len(layer_groups) - last_n_layers, 0)
    return STACPartition(
        trunk_layers=tuple(layer_groups[:trunk_count]),
        cap_layers=tuple(layer_groups[trunk_count:]),
    )


class STAC(Optimizer):
    r"""SignSGD trunk with an AdamW cap over the last N trainable layers.

    Layer discovery is deterministic: STAC walks ``model.named_modules()`` in
    registration order and treats each module that owns trainable parameters
    directly (``recurse=False``) as one layer. The final ``last_n_layers`` of
    that ordered list use AdamW, while all earlier layers use a sign-based
    trunk update. By default the trunk accumulates gradients with momentum
    before taking the sign, which is markedly more stable than plain signSGD.
    Set ``trunk_momentum=0.0`` to recover textbook signSGD.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        trunk_lr: float | None = None,
        cap_lr: float | None = None,
        last_n_layers: int = 1,
        trunk_momentum: float = 0.9,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        trunk_weight_decay: float | None = None,
        cap_weight_decay: float | None = None,
        maximize: bool = False,
        error_if_nonfinite: bool = False,
    ) -> None:
        self._validate_nonnegative("lr", lr)
        self._validate_nonnegative("eps", eps)
        self._validate_nonnegative("weight_decay", weight_decay)
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}.")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}.")
        if not 0.0 <= trunk_momentum < 1.0:
            raise ValueError(
                f"Invalid trunk_momentum value: {trunk_momentum}."
            )

        trunk_lr = lr if trunk_lr is None else trunk_lr
        cap_lr = lr if cap_lr is None else cap_lr
        trunk_weight_decay = (
            weight_decay if trunk_weight_decay is None else trunk_weight_decay
        )
        cap_weight_decay = (
            weight_decay if cap_weight_decay is None else cap_weight_decay
        )

        self._validate_nonnegative("trunk_lr", trunk_lr)
        self._validate_nonnegative("cap_lr", cap_lr)
        self._validate_nonnegative("trunk_weight_decay", trunk_weight_decay)
        self._validate_nonnegative("cap_weight_decay", cap_weight_decay)

        self.partition = partition_trainable_layers(
            model,
            last_n_layers=last_n_layers,
        )
        self.last_n_layers = last_n_layers

        param_groups: list[dict[str, object]] = []
        if self.partition.trunk_layers:
            param_groups.append(
                {
                    "params": list(self.partition.trunk_parameters),
                    "stac_role": "trunk",
                    "layer_names": self.partition.trunk_layer_names,
                    "lr": trunk_lr,
                    "weight_decay": trunk_weight_decay,
                    "trunk_momentum": trunk_momentum,
                }
            )
        if self.partition.cap_layers:
            param_groups.append(
                {
                    "params": list(self.partition.cap_parameters),
                    "stac_role": "cap",
                    "layer_names": self.partition.cap_layer_names,
                    "lr": cap_lr,
                    "weight_decay": cap_weight_decay,
                }
            )

        defaults = {
            "lr": lr,
            "trunk_lr": trunk_lr,
            "cap_lr": cap_lr,
            "trunk_momentum": trunk_momentum,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "trunk_weight_decay": trunk_weight_decay,
            "cap_weight_decay": cap_weight_decay,
            "maximize": maximize,
            "error_if_nonfinite": error_if_nonfinite,
        }
        self._initializing = True
        try:
            super().__init__(param_groups, defaults)
        finally:
            self._initializing = False

    @staticmethod
    def _validate_nonnegative(name: str, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"Invalid {name}: {value}.")

    def add_param_group(self, param_group: dict[str, object]) -> None:
        if getattr(self, "_initializing", False):
            super().add_param_group(param_group)
            return

        raise RuntimeError(
            "STAC does not support add_param_group(); construct a new optimizer "
            "so the trunk/cap partition stays deterministic."
        )

    def __setstate__(self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("stac_role", "trunk")
            group.setdefault("layer_names", ())
            group.setdefault("trunk_momentum", 0.0)
            group.setdefault("error_if_nonfinite", False)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            role = group["stac_role"]
            if role == "trunk":
                self._step_signsgd(group)
                continue
            if role == "cap":
                self._step_adamw(group)
                continue
            raise RuntimeError(f"Unexpected STAC parameter group role: {role!r}.")

        return loss

    def _step_signsgd(self, group: dict[str, object]) -> None:
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        maximize = bool(group["maximize"])
        trunk_momentum = float(group["trunk_momentum"])
        error_if_nonfinite = bool(group["error_if_nonfinite"])

        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("STAC does not support sparse gradients.")

            update = -gradient if maximize else gradient
            self._assert_finite(update, error_if_nonfinite, role="trunk")
            if weight_decay != 0:
                parameter.mul_(1 - lr * weight_decay)

            if trunk_momentum != 0.0:
                state = self.state[parameter]
                momentum_buffer = state.get("trunk_momentum_buffer")
                if momentum_buffer is None:
                    momentum_buffer = state["trunk_momentum_buffer"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )
                momentum_buffer.mul_(trunk_momentum).add_(
                    update,
                    alpha=1 - trunk_momentum,
                )
                direction = momentum_buffer.sign()
            else:
                direction = update.sign()

            parameter.add_(direction, alpha=-lr)

    def _step_adamw(self, group: dict[str, object]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        maximize = bool(group["maximize"])
        error_if_nonfinite = bool(group["error_if_nonfinite"])

        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("AdamW cap does not support sparse gradients.")

            update = -gradient if maximize else gradient
            self._assert_finite(update, error_if_nonfinite, role="cap")
            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1

            if weight_decay != 0:
                parameter.mul_(1 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(update, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(update, update, value=1 - beta2)

            step = state["step"]
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            parameter.addcdiv_(exp_avg, denom, value=-step_size)

    @staticmethod
    def _assert_finite(
        gradient: torch.Tensor,
        error_if_nonfinite: bool,
        *,
        role: str,
    ) -> None:
        if error_if_nonfinite and not torch.isfinite(gradient).all():
            raise RuntimeError(
                f"Encountered non-finite gradients in the STAC {role}."
            )
