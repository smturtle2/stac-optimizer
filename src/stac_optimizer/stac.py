from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

_HYBRID_TRUNK_LR_SCALE = 0.75


@dataclass(frozen=True)
class LayerGroup:
    """One trainable module slice discovered by :func:`partition_trainable_layers`.

    Attributes:
        name: Module path in ``model.named_modules()`` order. Root-owned parameters
            use ``"<root>"``.
        parameter_names: Fully-qualified parameter names that belong to the layer.
        parameters: Trainable parameters owned directly by that module.
    """

    name: str
    parameter_names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]


@dataclass(frozen=True)
class STACPartition:
    """Deterministic split of trainable layers into the sign trunk and AdamW cap."""

    trunk_layers: tuple[LayerGroup, ...]
    cap_layers: tuple[LayerGroup, ...]

    @property
    def trunk_layer_names(self) -> tuple[str, ...]:
        """Names of trainable layers updated by the sign-based trunk."""
        return tuple(layer.name for layer in self.trunk_layers)

    @property
    def cap_layer_names(self) -> tuple[str, ...]:
        """Names of trainable layers updated by the AdamW cap."""
        return tuple(layer.name for layer in self.cap_layers)

    @property
    def trunk_parameters(self) -> tuple[nn.Parameter, ...]:
        """Flattened trainable parameters that belong to the sign-based trunk."""
        return tuple(
            parameter
            for layer in self.trunk_layers
            for parameter in layer.parameters
        )

    @property
    def cap_parameters(self) -> tuple[nn.Parameter, ...]:
        """Flattened trainable parameters that belong to the AdamW cap."""
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
    """Split a model into a sign trunk and AdamW cap by trainable layer order.

    STAC walks ``model.named_modules()`` in registration order and treats each
    module that owns trainable parameters directly as one layer. The final
    ``last_n_layers`` layers become the AdamW cap; everything before that stays
    in the sign-based trunk.

    Args:
        model: Module whose trainable parameters should be partitioned.
        last_n_layers: Number of final trainable layers that should use AdamW.

    Returns:
        A deterministic partition describing which layers belong to the trunk
        and which belong to the cap.

    Raises:
        ValueError: If ``last_n_layers`` is negative or the model has no
            trainable parameters.
    """

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

    When both the trunk and cap are active, leaving ``trunk_lr`` unset makes
    STAC use ``0.75 * lr`` for the sign trunk and ``lr`` for the AdamW cap.
    This keeps the sign-based path slightly more conservative by default.
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
        amsgrad: bool = False,
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

        partition = partition_trainable_layers(
            model,
            last_n_layers=last_n_layers,
        )
        default_trunk_lr = (
            lr * _HYBRID_TRUNK_LR_SCALE
            if partition.trunk_layers and partition.cap_layers
            else lr
        )
        trunk_lr = default_trunk_lr if trunk_lr is None else trunk_lr
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

        self.partition = partition
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
            "amsgrad": amsgrad,
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
            group.setdefault("betas", (0.9, 0.999))
            group.setdefault("eps", 1e-8)
            group.setdefault("amsgrad", False)
            group.setdefault("error_if_nonfinite", False)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load optimizer state while preserving the current STAC partition.

        STAC derives its parameter groups from model structure, so loading a
        state dict whose saved trunk/cap split does not match the current
        optimizer is almost always a user error. This override validates the
        partition before delegating to :meth:`torch.optim.Optimizer.load_state_dict`.
        """

        self._validate_state_dict_partition(state_dict)
        super().load_state_dict(state_dict)

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
        amsgrad = bool(group["amsgrad"])
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
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(
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

            denom_source = exp_avg_sq
            if amsgrad:
                max_exp_avg_sq = state["max_exp_avg_sq"]
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom_source = max_exp_avg_sq

            denom = denom_source.sqrt().div(math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            parameter.addcdiv_(exp_avg, denom, value=-step_size)

    def _validate_state_dict_partition(self, state_dict: Mapping[str, Any]) -> None:
        saved_groups = state_dict.get("param_groups")
        if not isinstance(saved_groups, Sequence):
            raise ValueError("STAC expected 'param_groups' in the optimizer state.")

        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                "Saved STAC state uses a different number of parameter groups. "
                "Recreate the optimizer with the same model partition before "
                "loading the state dict."
            )

        for index, (saved_group, current_group) in enumerate(
            zip(saved_groups, self.param_groups, strict=True)
        ):
            if not isinstance(saved_group, Mapping):
                raise ValueError(
                    f"Saved STAC param group {index} is malformed: {saved_group!r}."
                )

            saved_role = saved_group.get("stac_role", current_group["stac_role"])
            current_role = current_group["stac_role"]
            if saved_role != current_role:
                raise ValueError(
                    "Saved STAC state does not match the current trunk/cap split: "
                    f"group {index} expects role {saved_role!r}, but the current "
                    f"optimizer uses {current_role!r}."
                )

            saved_param_count = len(saved_group.get("params", ()))
            current_param_count = len(current_group["params"])
            if saved_param_count != current_param_count:
                raise ValueError(
                    "Saved STAC state does not match the current parameter layout: "
                    f"group {index} contains {saved_param_count} parameters in the "
                    f"checkpoint but {current_param_count} in the current optimizer."
                )

            saved_layer_names = tuple(saved_group.get("layer_names", ()))
            current_layer_names = tuple(current_group.get("layer_names", ()))
            if saved_layer_names and saved_layer_names != current_layer_names:
                raise ValueError(
                    "Saved STAC state was created for different trainable layers: "
                    f"group {index} checkpoint layers {saved_layer_names!r} do not "
                    f"match current layers {current_layer_names!r}."
                )

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
