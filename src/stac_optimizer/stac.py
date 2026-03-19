from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim.adamw import adamw as adamw_functional
from torch.optim import Optimizer

_HYBRID_TRUNK_LR_SCALE = 0.75
_TRUNK_STATE_DTYPE_ALIASES = {
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
}


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
    """Deterministic split of trainable layers into the sign trunk and AdamW cap.

    The partition preserves trainable layer order from
    :func:`torch.nn.Module.named_modules`, which makes the split stable across
    repeated construction for the same model definition.
    """

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
    def trunk_parameter_names(self) -> tuple[str, ...]:
        """Flattened parameter names that belong to the sign-based trunk."""
        return tuple(
            parameter_name
            for layer in self.trunk_layers
            for parameter_name in layer.parameter_names
        )

    @property
    def cap_parameter_names(self) -> tuple[str, ...]:
        """Flattened parameter names that belong to the AdamW cap."""
        return tuple(
            parameter_name
            for layer in self.cap_layers
            for parameter_name in layer.parameter_names
        )

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

    The sign trunk can keep its momentum buffer in a lower-precision floating
    dtype via ``trunk_state_dtype``. This is useful when you want to reduce
    optimizer-state VRAM further without changing the cap behavior.

    With the default ``error_if_nonfinite=False``, STAC skips the entire step
    when it encounters a non-finite dense gradient. This avoids silently
    zeroing sign updates in the trunk or contaminating AdamW moments in the
    cap. Set ``error_if_nonfinite=True`` to raise immediately instead.

    Attributes:
        partition: Deterministic trunk/cap split derived from model structure.
        nonfinite_skipped_steps: Number of skipped :meth:`step` calls caused by
            non-finite dense gradients while ``error_if_nonfinite=False``.
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
        trunk_state_dtype: torch.dtype | str | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        trunk_weight_decay: float | None = None,
        cap_weight_decay: float | None = None,
        amsgrad: bool = False,
        maximize: bool = False,
        error_if_nonfinite: bool = False,
    ) -> None:
        """Create a STAC optimizer from a model.

        Args:
            model: Module whose trainable parameters should be optimized.
            lr: Shared base learning rate. The AdamW cap uses this value unless
                ``cap_lr`` overrides it.
            trunk_lr: Learning rate for the sign-based trunk. In hybrid mode,
                leaving this unset defaults to ``0.75 * lr``.
            cap_lr: Learning rate for the AdamW cap. Defaults to ``lr``.
            last_n_layers: Number of final trainable layers that should use
                AdamW. Earlier trainable layers use the sign-based trunk.
            trunk_momentum: EMA factor applied before taking the sign in the
                trunk. Set ``0.0`` to recover plain signSGD.
            trunk_state_dtype: Optional floating dtype used for the trunk
                momentum buffer. Leave unset to match each parameter dtype.
                Useful for reducing optimizer-state VRAM, for example with
                ``torch.bfloat16`` on CUDA.
            betas: AdamW first- and second-moment coefficients for the cap.
            eps: Numerical stability term for the AdamW cap.
            weight_decay: Shared decoupled weight decay applied to both roles
                unless overridden.
            trunk_weight_decay: Decoupled weight decay for the trunk.
            cap_weight_decay: Decoupled weight decay for the cap.
            amsgrad: Enable the AMSGrad variant for the AdamW cap.
            maximize: Maximize the objective instead of minimizing it.
            error_if_nonfinite: Raise ``RuntimeError`` when a gradient contains
                ``NaN`` or ``Inf`` values. If ``False``, STAC skips the entire
                step instead.
        """
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
        resolved_trunk_state_dtype = self._resolve_trunk_state_dtype(
            trunk_state_dtype
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
        self.nonfinite_skipped_steps = 0

        param_groups: list[dict[str, object]] = []
        if self.partition.trunk_layers:
            param_groups.append(
                {
                    "params": list(self.partition.trunk_parameters),
                    "stac_role": "trunk",
                    "layer_names": self.partition.trunk_layer_names,
                    "param_names": self.partition.trunk_parameter_names,
                    "lr": trunk_lr,
                    "weight_decay": trunk_weight_decay,
                    "trunk_momentum": trunk_momentum,
                    "trunk_state_dtype": resolved_trunk_state_dtype,
                }
            )
        if self.partition.cap_layers:
            param_groups.append(
                {
                    "params": list(self.partition.cap_parameters),
                    "stac_role": "cap",
                    "layer_names": self.partition.cap_layer_names,
                    "param_names": self.partition.cap_parameter_names,
                    "lr": cap_lr,
                    "weight_decay": cap_weight_decay,
                }
            )

        defaults = {
            "lr": lr,
            "trunk_lr": trunk_lr,
            "cap_lr": cap_lr,
            "trunk_momentum": trunk_momentum,
            "trunk_state_dtype": resolved_trunk_state_dtype,
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

    @staticmethod
    def _resolve_trunk_state_dtype(
        value: torch.dtype | str | None,
    ) -> torch.dtype | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized_value = value.strip().lower()
            if normalized_value in {"parameter", "match_parameter"}:
                return None
            resolved_value = _TRUNK_STATE_DTYPE_ALIASES.get(normalized_value)
            if resolved_value is None:
                raise ValueError(
                    "trunk_state_dtype must be a floating torch.dtype, None, "
                    "or one of: "
                    f"{', '.join(sorted(_TRUNK_STATE_DTYPE_ALIASES))}."
                )
            value = resolved_value

        if not isinstance(value, torch.dtype):
            raise ValueError(
                "trunk_state_dtype must be a floating torch.dtype, None, or "
                "a supported string alias."
            )
        if not torch.empty((), dtype=value).is_floating_point():
            raise ValueError(
                "trunk_state_dtype must be a floating-point dtype."
            )
        return value

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
        if not hasattr(self, "nonfinite_skipped_steps"):
            self.nonfinite_skipped_steps = 0
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("stac_role", "trunk")
            group.setdefault("layer_names", ())
            group.setdefault("param_names", ())
            group.setdefault("trunk_momentum", 0.0)
            group.setdefault("trunk_state_dtype", None)
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

        if self._skip_step_for_nonfinite_gradients():
            self.nonfinite_skipped_steps += 1
            return loss

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
        trunk_state_dtype = group["trunk_state_dtype"]
        parameters: list[torch.Tensor] = []
        updates: list[torch.Tensor] = []
        momentum_buffers: list[torch.Tensor] = []

        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("STAC does not support sparse gradients.")

            update = -gradient if maximize else gradient
            parameters.append(parameter)
            updates.append(update)
            if trunk_momentum != 0.0:
                state = self.state[parameter]
                momentum_buffer = state.get("trunk_momentum_buffer")
                buffer_dtype = (
                    parameter.dtype
                    if trunk_state_dtype is None
                    else trunk_state_dtype
                )
                if momentum_buffer is None:
                    momentum_buffer = state["trunk_momentum_buffer"] = torch.zeros_like(
                        parameter,
                        dtype=buffer_dtype,
                        memory_format=torch.preserve_format,
                    )
                momentum_buffers.append(momentum_buffer)

        if not parameters:
            return

        if self._can_use_foreach(parameters, updates):
            if weight_decay != 0:
                torch._foreach_mul_(parameters, 1 - lr * weight_decay)

            if trunk_momentum != 0.0:
                buffer_updates = [
                    update.to(dtype=momentum_buffer.dtype)
                    if update.dtype != momentum_buffer.dtype
                    else update
                    for update, momentum_buffer in zip(
                        updates,
                        momentum_buffers,
                        strict=True,
                    )
                ]
                torch._foreach_mul_(momentum_buffers, trunk_momentum)
                torch._foreach_add_(
                    momentum_buffers,
                    buffer_updates,
                    alpha=1 - trunk_momentum,
                )
                directions = torch._foreach_sign(momentum_buffers)
            else:
                directions = torch._foreach_sign(updates)

            directions = [
                direction.to(dtype=parameter.dtype)
                if direction.dtype != parameter.dtype
                else direction
                for direction, parameter in zip(
                    directions,
                    parameters,
                    strict=True,
                )
            ]
            torch._foreach_add_(parameters, directions, alpha=-lr)
            return

        for index, parameter in enumerate(parameters):
            if weight_decay != 0:
                parameter.mul_(1 - lr * weight_decay)

            if trunk_momentum != 0.0:
                momentum_buffer = momentum_buffers[index]
                buffer_update = (
                    updates[index].to(dtype=momentum_buffer.dtype)
                    if updates[index].dtype != momentum_buffer.dtype
                    else updates[index]
                )
                momentum_buffer.mul_(trunk_momentum).add_(
                    buffer_update,
                    alpha=1 - trunk_momentum,
                )
                direction = momentum_buffer.sign()
            else:
                direction = updates[index].sign()

            if direction.dtype != parameter.dtype:
                direction = direction.to(dtype=parameter.dtype)
            parameter.add_(direction, alpha=-lr)

    def _step_adamw(self, group: dict[str, object]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        amsgrad = bool(group["amsgrad"])
        maximize = bool(group["maximize"])
        parameters: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        exp_avgs: list[torch.Tensor] = []
        exp_avg_sqs: list[torch.Tensor] = []
        max_exp_avg_sqs: list[torch.Tensor] = []
        state_steps: list[torch.Tensor] = []

        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("AdamW cap does not support sparse gradients.")

            update = -gradient if maximize else gradient
            state = self.state[parameter]
            if not state:
                state["step"] = torch.zeros((), device=parameter.device)
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

            step = state["step"]
            if not isinstance(step, torch.Tensor):
                step = state["step"] = torch.tensor(float(step), device=parameter.device)

            parameters.append(parameter)
            gradients.append(update)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(step)
            if amsgrad:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

        if not parameters:
            return

        adamw_functional(
            params=parameters,
            grads=gradients,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            max_exp_avg_sqs=max_exp_avg_sqs,
            state_steps=state_steps,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=False,
        )

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

            saved_state = state_dict.get("state")
            if not isinstance(saved_state, Mapping):
                raise ValueError("STAC expected 'state' in the optimizer state.")

            saved_param_ids = tuple(saved_group.get("params", ()))
            for param_index, (saved_param_id, current_parameter) in enumerate(
                zip(saved_param_ids, current_group["params"], strict=True)
            ):
                parameter_state = saved_state.get(saved_param_id)
                if not isinstance(parameter_state, Mapping):
                    continue

                self._validate_state_tensor_shapes(
                    parameter_state,
                    current_parameter,
                    group_index=index,
                    param_index=param_index,
                )

            saved_layer_names = tuple(saved_group.get("layer_names", ()))
            current_layer_names = tuple(current_group.get("layer_names", ()))
            if saved_layer_names and saved_layer_names != current_layer_names:
                raise ValueError(
                    "Saved STAC state was created for different trainable layers: "
                    f"group {index} checkpoint layers {saved_layer_names!r} do not "
                    f"match current layers {current_layer_names!r}."
                )

            saved_param_names = tuple(saved_group.get("param_names", ()))
            current_param_names = tuple(current_group.get("param_names", ()))
            if saved_param_names and saved_param_names != current_param_names:
                raise ValueError(
                    "Saved STAC state was created for different parameter names: "
                    f"group {index} checkpoint parameters {saved_param_names!r} "
                    f"do not match current parameters {current_param_names!r}."
                )

    def _skip_step_for_nonfinite_gradients(self) -> bool:
        for group in self.param_groups:
            role = str(group["stac_role"])
            error_if_nonfinite = bool(group["error_if_nonfinite"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None or gradient.is_sparse:
                    continue
                if torch.isfinite(gradient).all():
                    continue
                if error_if_nonfinite:
                    raise RuntimeError(
                        f"Encountered non-finite gradients in the STAC {role}."
                    )
                return True
        return False

    @staticmethod
    def _can_use_foreach(
        parameters: Sequence[torch.Tensor],
        updates: Sequence[torch.Tensor],
    ) -> bool:
        if not parameters:
            return False

        devices = {parameter.device for parameter in parameters}
        dtypes = {parameter.dtype for parameter in parameters}
        update_devices = {update.device for update in updates}
        update_dtypes = {update.dtype for update in updates}
        return (
            len(devices) == 1
            and len(dtypes) == 1
            and devices == update_devices
            and dtypes == update_dtypes
        )

    @staticmethod
    def _validate_state_tensor_shapes(
        parameter_state: Mapping[str, Any],
        current_parameter: nn.Parameter,
        *,
        group_index: int,
        param_index: int,
    ) -> None:
        expected_shape = tuple(current_parameter.shape)
        for state_key in (
            "trunk_momentum_buffer",
            "exp_avg",
            "exp_avg_sq",
            "max_exp_avg_sq",
        ):
            state_value = parameter_state.get(state_key)
            if not isinstance(state_value, torch.Tensor):
                continue
            if tuple(state_value.shape) != expected_shape:
                raise ValueError(
                    "Saved STAC state does not match current parameter shapes: "
                    f"group {group_index} parameter {param_index} expected shape "
                    f"{expected_shape!r}, but saved state {state_key!r} uses "
                    f"{tuple(state_value.shape)!r}."
                )
