from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim.adamw import adamw as adamw_functional
from torch.optim import Optimizer

_DEFAULT_SIGN_LR_SCALE = 0.75
_AUTO_SIGN_STATE_DTYPE = "auto"
_SIGN_STATE_DTYPE_ALIASES = {
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
class ModuleGroup:
    """One trainable module slice discovered by :func:`partition_trainable_modules`.

    Attributes:
        name: Module path in ``model.named_modules()`` order. Root-owned parameters
            use ``"<root>"``.
        parameter_names: Fully-qualified parameter names that belong to the module.
        parameters: Trainable parameters owned directly by that module.
    """

    name: str
    parameter_names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]


@dataclass(frozen=True)
class STACPartition:
    """Deterministic split of trainable modules into sign and AdamW sections.

    The partition preserves trainable module order from
    :func:`torch.nn.Module.named_modules`, which makes the split stable across
    repeated construction for the same model definition.
    """

    sign_modules: tuple[ModuleGroup, ...]
    adamw_modules: tuple[ModuleGroup, ...]

    @property
    def sign_module_names(self) -> tuple[str, ...]:
        """Names of trainable modules updated by the sign-based section."""
        return tuple(module.name for module in self.sign_modules)

    @property
    def adamw_module_names(self) -> tuple[str, ...]:
        """Names of trainable modules updated by the AdamW section."""
        return tuple(module.name for module in self.adamw_modules)

    @property
    def sign_parameter_names(self) -> tuple[str, ...]:
        """Flattened parameter names that belong to the sign-based section."""
        return tuple(
            parameter_name
            for module in self.sign_modules
            for parameter_name in module.parameter_names
        )

    @property
    def adamw_parameter_names(self) -> tuple[str, ...]:
        """Flattened parameter names that belong to the AdamW section."""
        return tuple(
            parameter_name
            for module in self.adamw_modules
            for parameter_name in module.parameter_names
        )

    @property
    def sign_parameters(self) -> tuple[nn.Parameter, ...]:
        """Flattened trainable parameters that belong to the sign-based section."""
        return tuple(
            parameter
            for module in self.sign_modules
            for parameter in module.parameters
        )

    @property
    def adamw_parameters(self) -> tuple[nn.Parameter, ...]:
        """Flattened trainable parameters that belong to the AdamW section."""
        return tuple(
            parameter
            for module in self.adamw_modules
            for parameter in module.parameters
        )


def partition_trainable_modules(
    model: nn.Module,
    *,
    last_n_modules: int = 1,
) -> STACPartition:
    """Split a model into sign and AdamW sections by trainable module order.

    STAC walks ``model.named_modules()`` in registration order and treats each
    module that owns trainable parameters directly as one counted module. The
    final ``last_n_modules`` counted modules become the AdamW section;
    everything before that
    stays in the sign-based section.

    Args:
        model: Module whose trainable parameters should be partitioned.
        last_n_modules: Number of final trainable modules that should use AdamW.

    Returns:
        A deterministic partition describing which modules belong to the sign
        section and which belong to the AdamW section.

    Raises:
        ValueError: If ``last_n_modules`` is negative or the model has no
            trainable parameters.
    """

    if last_n_modules < 0:
        raise ValueError("last_n_modules must be greater than or equal to 0.")

    seen_parameters: set[int] = set()
    module_groups: list[ModuleGroup] = []

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
            module_groups.append(
                ModuleGroup(
                    name=module_name or "<root>",
                    parameter_names=tuple(names),
                    parameters=tuple(parameters),
                )
            )

    if not module_groups:
        raise ValueError("STAC requires at least one trainable parameter.")

    sign_count = max(len(module_groups) - last_n_modules, 0)
    return STACPartition(
        sign_modules=tuple(module_groups[:sign_count]),
        adamw_modules=tuple(module_groups[sign_count:]),
    )


class STAC(Optimizer):
    r"""SignSGD section with an AdamW section over the last N trainable modules.

    Module discovery is deterministic: STAC walks ``model.named_modules()`` in
    registration order and treats each module that owns trainable parameters
    directly (``recurse=False``) as one counted module. The final
    ``last_n_modules`` of that ordered list use AdamW, while all earlier
    modules use a sign-based update. By default the sign-based section
    accumulates gradients with momentum before taking the sign, which is
    markedly more stable than plain signSGD. Set ``sign_momentum=0.0`` to
    recover textbook signSGD.

    Pure containers such as ``nn.Sequential`` are skipped unless they own
    trainable parameters themselves. In practice this means the counted modules
    are usually the parameterized end modules users care about, such as
    ``stem``, ``block.0``, ``block.2``, and ``head``.

    When both sections are active, STAC internally uses
    ``sign_lr_scale * lr`` for the sign-based section and ``lr`` for the
    AdamW section. The default ``sign_lr_scale=0.75`` keeps the sign path
    slightly more conservative while preserving a single public learning rate
    knob.

    The sign-based section can keep its momentum buffer in a configurable
    floating dtype via ``sign_state_dtype``. By default STAC uses an ``"auto"``
    policy: low-precision parameter tensors (``float16`` or ``bfloat16``) keep
    their sign momentum state in ``float32`` for stability, while higher
    precision tensors match the parameter dtype. You can still force a lower
    precision state such as ``torch.bfloat16`` when VRAM matters more.

    By default STAC uses single-tensor step logic instead of PyTorch's
    ``foreach`` optimizer path. This keeps peak CUDA memory more conservative.
    Set ``foreach=True`` when step throughput matters more than temporary
    tensor-list overhead.

    With the default ``error_if_nonfinite=False``, STAC skips the entire step
    when it encounters a non-finite dense gradient. This avoids silently
    zeroing sign updates in the sign-based section or contaminating AdamW
    moments in the final section. Set ``error_if_nonfinite=True`` to raise
    immediately instead.

    Attributes:
        partition: Deterministic sign/AdamW split derived from model structure.
        nonfinite_skipped_steps: Number of skipped :meth:`step` calls caused by
            non-finite dense gradients while ``error_if_nonfinite=False``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        last_n_modules: int = 1,
        sign_lr_scale: float = _DEFAULT_SIGN_LR_SCALE,
        sign_momentum: float = 0.9,
        sign_state_dtype: torch.dtype | str | None = _AUTO_SIGN_STATE_DTYPE,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        sign_weight_decay: float | None = None,
        adamw_weight_decay: float | None = None,
        amsgrad: bool = False,
        maximize: bool = False,
        error_if_nonfinite: bool = False,
        foreach: bool = False,
    ) -> None:
        """Create a STAC optimizer from a model.

        Args:
            model: Module whose trainable parameters should be optimized.
            lr: Shared base learning rate. In hybrid mode, STAC internally uses
                ``sign_lr_scale * lr`` for the sign-based section and ``lr``
                for the AdamW section.
            last_n_modules: Number of final trainable modules that should use
                AdamW. Earlier trainable modules use the sign-based section.
            sign_lr_scale: Multiplicative factor applied to ``lr`` for the
                sign-based section when both sections are active. The default
                ``0.75`` is slightly more conservative than the AdamW section
                and tested well on this repository's held-out CUDA suite.
            sign_momentum: EMA factor applied before taking the sign in the
                sign-based section. Set ``0.0`` to recover plain signSGD.
            sign_state_dtype: Optional floating dtype used for the sign-section
                momentum buffer. The default ``"auto"`` keeps low-precision
                parameters on ``float32`` momentum state for stability and
                otherwise matches the parameter dtype. Use ``None`` or
                ``"parameter"`` to always match the parameter dtype. Useful for
                reducing optimizer-state VRAM, for example with
                ``torch.bfloat16`` on CUDA.
            betas: AdamW first- and second-moment coefficients for the AdamW
                section.
            eps: Numerical stability term for the AdamW section.
            weight_decay: Shared decoupled weight decay applied to both roles
                unless overridden.
            sign_weight_decay: Decoupled weight decay for the sign-based section.
            adamw_weight_decay: Decoupled weight decay for the AdamW section.
            amsgrad: Enable the AMSGrad variant for the AdamW section.
            maximize: Maximize the objective instead of minimizing it.
            error_if_nonfinite: Raise ``RuntimeError`` when a gradient contains
                ``NaN`` or ``Inf`` values. If ``False``, STAC skips the entire
                step instead.
            foreach: If ``True``, STAC opts into PyTorch's multi-tensor
                ``foreach`` step path when the current parameter group is
                compatible. This is often faster on CUDA, but it uses extra
                temporary memory. The default ``False`` keeps STAC more VRAM
                conservative.
        """
        self._validate_nonnegative("lr", lr)
        self._validate_nonnegative("eps", eps)
        self._validate_nonnegative("weight_decay", weight_decay)
        self._validate_positive("sign_lr_scale", sign_lr_scale)
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}.")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}.")
        if not 0.0 <= sign_momentum < 1.0:
            raise ValueError(
                f"Invalid sign_momentum value: {sign_momentum}."
            )
        resolved_sign_state_dtype = self._resolve_sign_state_dtype(
            sign_state_dtype
        )

        partition = partition_trainable_modules(
            model,
            last_n_modules=last_n_modules,
        )
        default_sign_lr = (
            lr * sign_lr_scale
            if partition.sign_modules and partition.adamw_modules
            else lr
        )
        sign_weight_decay = (
            weight_decay if sign_weight_decay is None else sign_weight_decay
        )
        adamw_weight_decay = (
            weight_decay if adamw_weight_decay is None else adamw_weight_decay
        )

        self._validate_nonnegative("sign_weight_decay", sign_weight_decay)
        self._validate_nonnegative("adamw_weight_decay", adamw_weight_decay)

        self.partition = partition
        self.last_n_modules = last_n_modules
        self.sign_lr_scale = sign_lr_scale
        self.nonfinite_skipped_steps = 0

        param_groups: list[dict[str, object]] = []
        if self.partition.sign_modules:
            param_groups.append(
                {
                    "params": list(self.partition.sign_parameters),
                    "stac_role": "sign",
                    "module_names": self.partition.sign_module_names,
                    "param_names": self.partition.sign_parameter_names,
                    "lr": default_sign_lr,
                    "weight_decay": sign_weight_decay,
                    "sign_lr_scale": sign_lr_scale,
                    "sign_momentum": sign_momentum,
                    "sign_state_dtype": resolved_sign_state_dtype,
                    "foreach": foreach,
                }
            )
        if self.partition.adamw_modules:
            param_groups.append(
                {
                    "params": list(self.partition.adamw_parameters),
                    "stac_role": "adamw",
                    "module_names": self.partition.adamw_module_names,
                    "param_names": self.partition.adamw_parameter_names,
                    "lr": lr,
                    "weight_decay": adamw_weight_decay,
                    "foreach": foreach,
                }
            )

        defaults = {
            "lr": lr,
            "sign_lr_scale": sign_lr_scale,
            "sign_momentum": sign_momentum,
            "sign_state_dtype": resolved_sign_state_dtype,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "sign_weight_decay": sign_weight_decay,
            "adamw_weight_decay": adamw_weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
            "error_if_nonfinite": error_if_nonfinite,
            "foreach": foreach,
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
    def _validate_positive(name: str, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"Invalid {name}: {value}.")

    @staticmethod
    def _resolve_sign_state_dtype(
        value: torch.dtype | str | None,
    ) -> torch.dtype | str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized_value = value.strip().lower()
            if normalized_value == _AUTO_SIGN_STATE_DTYPE:
                return _AUTO_SIGN_STATE_DTYPE
            if normalized_value in {"parameter", "match_parameter"}:
                return None
            resolved_value = _SIGN_STATE_DTYPE_ALIASES.get(normalized_value)
            if resolved_value is None:
                raise ValueError(
                    "sign_state_dtype must be a floating torch.dtype, None, "
                    'or one of: auto, match_parameter, parameter, '
                    "or one of: "
                    f"{', '.join(sorted(_SIGN_STATE_DTYPE_ALIASES))}."
                )
            value = resolved_value

        if not isinstance(value, torch.dtype):
            raise ValueError(
                "sign_state_dtype must be a floating torch.dtype, None, or "
                "a supported string alias."
            )
        if not torch.empty((), dtype=value).is_floating_point():
            raise ValueError(
                "sign_state_dtype must be a floating-point dtype."
            )
        return value

    @staticmethod
    def _get_sign_state_dtype(
        parameter: nn.Parameter,
        sign_state_dtype: torch.dtype | str | None,
    ) -> torch.dtype:
        if sign_state_dtype == _AUTO_SIGN_STATE_DTYPE:
            if parameter.dtype in {torch.float16, torch.bfloat16}:
                return torch.float32
            return parameter.dtype
        if sign_state_dtype is None:
            return parameter.dtype
        return sign_state_dtype

    def add_param_group(self, param_group: dict[str, object]) -> None:
        if getattr(self, "_initializing", False):
            super().add_param_group(param_group)
            return

        raise RuntimeError(
            "STAC does not support add_param_group(); construct a new optimizer "
            "so the sign/AdamW partition stays deterministic."
        )

    def __setstate__(self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        if not hasattr(self, "nonfinite_skipped_steps"):
            self.nonfinite_skipped_steps = 0
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("stac_role", "sign")
            group.setdefault("module_names", ())
            group.setdefault("param_names", ())
            group.setdefault("sign_lr_scale", _DEFAULT_SIGN_LR_SCALE)
            group.setdefault("sign_momentum", 0.0)
            group.setdefault("sign_state_dtype", None)
            group.setdefault("betas", (0.9, 0.999))
            group.setdefault("eps", 1e-8)
            group.setdefault("amsgrad", False)
            group.setdefault("error_if_nonfinite", False)
            group.setdefault("foreach", False)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load optimizer state while preserving the current STAC partition.

        STAC derives its parameter groups from model structure, so loading a
        state dict whose saved sign/AdamW split does not match the current
        optimizer is almost always a user error. This override validates the
        partition before delegating to :meth:`torch.optim.Optimizer.load_state_dict`.
        """

        self._validate_state_dict_partition(state_dict)
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._skip_step_for_nonfinite_gradients():
            self.nonfinite_skipped_steps += 1
            return loss

        for group in self.param_groups:
            role = group["stac_role"]
            if role == "sign":
                self._step_sign(group)
                continue
            if role == "adamw":
                self._step_adamw(group)
                continue
            raise RuntimeError(f"Unexpected STAC parameter group role: {role!r}.")

        return loss

    def _step_sign(self, group: dict[str, object]) -> None:
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        maximize = bool(group["maximize"])
        sign_momentum = float(group["sign_momentum"])
        sign_state_dtype = group["sign_state_dtype"]
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
            if sign_momentum != 0.0:
                state = self.state[parameter]
                momentum_buffer = state.get("sign_momentum_buffer")
                buffer_dtype = self._get_sign_state_dtype(
                    parameter,
                    sign_state_dtype,
                )
                if momentum_buffer is None:
                    momentum_buffer = state["sign_momentum_buffer"] = torch.zeros_like(
                        parameter,
                        dtype=buffer_dtype,
                        memory_format=torch.preserve_format,
                    )
                momentum_buffers.append(momentum_buffer)

        if not parameters:
            return

        use_foreach = bool(group["foreach"]) and self._can_use_foreach(
            parameters,
            updates,
        )
        if use_foreach:
            if weight_decay != 0:
                torch._foreach_mul_(parameters, 1 - lr * weight_decay)

            if sign_momentum != 0.0:
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
                torch._foreach_mul_(momentum_buffers, sign_momentum)
                torch._foreach_add_(
                    momentum_buffers,
                    buffer_updates,
                    alpha=1 - sign_momentum,
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

            if sign_momentum != 0.0:
                momentum_buffer = momentum_buffers[index]
                buffer_update = (
                    updates[index].to(dtype=momentum_buffer.dtype)
                    if updates[index].dtype != momentum_buffer.dtype
                    else updates[index]
                )
                momentum_buffer.mul_(sign_momentum).add_(
                    buffer_update,
                    alpha=1 - sign_momentum,
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
                raise RuntimeError("AdamW section does not support sparse gradients.")

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

        use_foreach = bool(group["foreach"]) and self._can_use_foreach(
            parameters,
            gradients,
        )
        adamw_functional(
            params=parameters,
            grads=gradients,
            exp_avgs=exp_avgs,
            exp_avg_sqs=exp_avg_sqs,
            max_exp_avg_sqs=max_exp_avg_sqs,
            state_steps=state_steps,
            foreach=use_foreach,
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
                    "Saved STAC state does not match the current sign/AdamW split: "
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

            saved_module_names = tuple(saved_group.get("module_names", ()))
            current_module_names = tuple(current_group.get("module_names", ()))
            if saved_module_names and saved_module_names != current_module_names:
                raise ValueError(
                    "Saved STAC state was created for different trainable modules: "
                    f"group {index} checkpoint modules {saved_module_names!r} do not "
                    f"match current modules {current_module_names!r}."
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
            "sign_momentum_buffer",
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
