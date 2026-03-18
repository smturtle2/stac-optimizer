from __future__ import annotations

import math
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
    that ordered list use AdamW, while all earlier layers use signSGD.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        last_n_layers: int = 1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}.")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}.")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}.")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}.")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}.")

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
                }
            )
        if self.partition.cap_layers:
            param_groups.append(
                {
                    "params": list(self.partition.cap_parameters),
                    "stac_role": "cap",
                    "layer_names": self.partition.cap_layer_names,
                }
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "maximize": maximize,
        }
        super().__init__(param_groups, defaults)

    def __setstate__(self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("stac_role", "trunk")
            group.setdefault("layer_names", ())

    @torch.no_grad()
    def step(self, closure=None):
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

        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("STAC does not support sparse gradients.")

            update = -gradient if maximize else gradient
            if weight_decay != 0:
                parameter.mul_(1 - lr * weight_decay)
            parameter.add_(update.sign(), alpha=-lr)

    def _step_adamw(self, group: dict[str, object]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        maximize = bool(group["maximize"])

        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("AdamW cap does not support sparse gradients.")

            update = -gradient if maximize else gradient
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
