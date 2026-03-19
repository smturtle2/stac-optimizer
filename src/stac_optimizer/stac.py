from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.optim.adamw import adamw as adamw_functional
from torch.optim import Optimizer

_DEFAULT_SIGN_LR_SCALE = 1.0
_DEFAULT_ADAMW_RATIO = 0.18
_LEGACY_SIGN_STATE_KEYS = frozenset({"sign_momentum_buffer"})


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
    def trainable_module_count(self) -> int:
        """Total number of counted trainable modules in the partition."""
        return len(self.sign_modules) + len(self.adamw_modules)

    @property
    def resolved_last_n_modules(self) -> int:
        """Number of counted trainable modules assigned to the AdamW section."""
        return len(self.adamw_modules)

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


@dataclass(frozen=True)
class _PreparedGroup:
    """Dense gradients collected for one STAC param group."""

    group: dict[str, object]
    parameters: tuple[nn.Parameter, ...]
    gradients: tuple[torch.Tensor, ...]


def _resolve_ratio_alias(
    *,
    last_n_ratio: float | None,
    adamw_ratio: float | None,
) -> float:
    if last_n_ratio is not None and adamw_ratio is not None:
        if not math.isclose(last_n_ratio, adamw_ratio):
            raise ValueError(
                "last_n_ratio and adamw_ratio must match when both are provided."
            )
        return last_n_ratio
    if last_n_ratio is not None:
        return last_n_ratio
    if adamw_ratio is not None:
        return adamw_ratio
    return _DEFAULT_ADAMW_RATIO


def resolve_adamw_cap_module_count(
    total_trainable_modules: int,
    *,
    last_n_modules: int | None = None,
    last_n_ratio: float | None = None,
    adamw_ratio: float | None = None,
) -> int:
    """Resolve how many final trainable modules should use AdamW.

    Args:
        total_trainable_modules: Number of counted trainable modules.
        last_n_modules: Optional explicit number of final trainable modules to
            place in the AdamW cap.
        last_n_ratio: Preferred public alias for the AdamW tail ratio.
        adamw_ratio: Backward-compatible alias for ``last_n_ratio``.

    Returns:
        The resolved AdamW-cap size clamped to the available trainable module
        count.

    Raises:
        ValueError: If the ratio is invalid, the explicit count is negative, or
            the ratio aliases disagree.
    """
    if total_trainable_modules < 0:
        raise ValueError(
            "total_trainable_modules must be greater than or equal to 0."
        )

    resolved_ratio = _resolve_ratio_alias(
        last_n_ratio=last_n_ratio,
        adamw_ratio=adamw_ratio,
    )
    return _resolve_adamw_module_count(
        last_n_modules=last_n_modules,
        adamw_ratio=resolved_ratio,
        trainable_module_count=total_trainable_modules,
    )


def partition_trainable_modules(
    model: nn.Module,
    *,
    last_n_ratio: float | None = None,
    adamw_ratio: float | None = None,
    last_n_modules: int | None = None,
) -> STACPartition:
    """Split a model into sign and AdamW sections by trainable module order.

    STAC walks ``model.named_modules()`` in registration order and treats each
    module that owns trainable parameters directly as one counted module. The
    final portion of that ordered list becomes the AdamW section and everything
    before it stays in the sign-based section.

    Args:
        model: Module whose trainable parameters should be partitioned.
        last_n_ratio: Preferred public alias for the AdamW tail ratio.
        adamw_ratio: Backward-compatible alias for ``last_n_ratio``. When both
            ratio arguments are omitted, STAC defaults to ``0.18``.
        last_n_modules: Optional explicit number of final trainable modules that
            should use AdamW. When provided, it overrides ratio mode.

    Returns:
        A deterministic partition describing which modules belong to the sign
        section and which belong to the AdamW section.

    Raises:
        ValueError: If the ratio arguments are invalid, the explicit count is
            negative, or the model has no trainable parameters.
    """
    resolved_ratio = _resolve_ratio_alias(
        last_n_ratio=last_n_ratio,
        adamw_ratio=adamw_ratio,
    )
    _resolve_adamw_module_count(
        last_n_modules=last_n_modules,
        adamw_ratio=resolved_ratio,
        trainable_module_count=None,
    )

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

    resolved_last_n_modules = _resolve_adamw_module_count(
        last_n_modules=last_n_modules,
        adamw_ratio=resolved_ratio,
        trainable_module_count=len(module_groups),
    )
    sign_count = max(len(module_groups) - resolved_last_n_modules, 0)
    return STACPartition(
        sign_modules=tuple(module_groups[:sign_count]),
        adamw_modules=tuple(module_groups[sign_count:]),
    )


def _resolve_adamw_module_count(
    *,
    last_n_modules: int | None,
    adamw_ratio: float,
    trainable_module_count: int | None,
) -> int:
    if not 0.0 <= adamw_ratio <= 1.0:
        raise ValueError(f"adamw_ratio must be between 0.0 and 1.0, got {adamw_ratio}.")

    if last_n_modules is not None:
        if last_n_modules < 0:
            raise ValueError("last_n_modules must be greater than or equal to 0.")
        if trainable_module_count is None:
            return last_n_modules
        return min(last_n_modules, trainable_module_count)

    if trainable_module_count is None:
        return 0
    if adamw_ratio == 0.0:
        return 0

    return min(
        max(math.ceil(trainable_module_count * adamw_ratio), 1),
        trainable_module_count,
    )


class STAC(Optimizer):
    r"""SignSGD section with an AdamW section over the last trainable-module tail.

    Module discovery is deterministic: STAC walks ``model.named_modules()`` in
    registration order and treats each module that owns trainable parameters
    directly (``recurse=False``) as one counted module. By default the final
    18 percent of that ordered list use AdamW, while all earlier modules use
    textbook signSGD with decoupled weight decay. Pass ``last_n_modules`` to
    override the ratio with an explicit module count. ``last_n_ratio`` is the
    preferred public name for that ratio, while ``adamw_ratio`` remains a
    backward-compatible alias. The sign-based section is intentionally
    stateless: it keeps no momentum, EMA, or other sign-side optimizer
    tensors.

    Pure containers such as ``nn.Sequential`` are skipped unless they own
    trainable parameters themselves. In practice this means the counted modules
    are usually the parameterized end modules users care about, such as
    ``stem``, ``block.0``, ``block.2``, and ``head``.

    When both sections are active, STAC internally uses
    ``sign_lr_scale * lr`` for the sign-based section and ``lr`` for the
    AdamW section. The default ``sign_lr_scale=1.0`` keeps the public learning
    rate semantics simple while still allowing a more conservative sign step
    when a workload needs it.

    When decoupled weight decay is enabled in hybrid mode, the sign trunk often
    benefits from a lighter decay than the AdamW cap. STAC therefore defaults
    the sign trunk to ``0.5 * weight_decay`` in hybrid mode unless
    ``sign_weight_decay`` is set explicitly. The repository CUDA benchmark
    found that to be a strong starting point on deep classification-style
    workloads without adding any sign-side optimizer state.

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
        last_n_ratio: Preferred public ratio value used when
            ``last_n_modules`` is omitted.
        adamw_ratio: Fractional tail used when ``last_n_modules`` is omitted.
        resolved_last_n_modules: Actual number of counted trainable modules
            assigned to the AdamW section for the current model.
        nonfinite_skipped_steps: Number of skipped :meth:`step` calls caused by
            non-finite dense gradients while ``error_if_nonfinite=False``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        last_n_modules: int | None = None,
        last_n_ratio: float | None = None,
        adamw_ratio: float | None = None,
        sign_lr_scale: float = _DEFAULT_SIGN_LR_SCALE,
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
            last_n_modules: Optional explicit number of final trainable modules
                that should use AdamW. When provided, it overrides
                ratio mode.
            last_n_ratio: Preferred public alias for the AdamW tail ratio.
            adamw_ratio: Backward-compatible alias for ``last_n_ratio``. When
                both ratio arguments are omitted, STAC defaults to ``0.18``.
            sign_lr_scale: Multiplicative factor applied to ``lr`` for the
                sign-based section when both sections are active. The default
                ``1.0`` matches the shared learning rate. Lower this value when
                you want a more conservative sign step without changing the
                AdamW cap.
            betas: AdamW first- and second-moment coefficients for the AdamW
                section.
            eps: Numerical stability term for the AdamW section.
            weight_decay: Shared decoupled weight decay applied to both roles
                unless overridden.
            sign_weight_decay: Decoupled weight decay for the sign-based section.
                In hybrid mode, ``None`` defaults to ``0.5 * weight_decay`` as
                a benchmark-backed stability starting point. Outside hybrid
                mode, ``None`` inherits ``weight_decay``.
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
        resolved_ratio = _resolve_ratio_alias(
            last_n_ratio=last_n_ratio,
            adamw_ratio=adamw_ratio,
        )
        _resolve_adamw_module_count(
            last_n_modules=last_n_modules,
            adamw_ratio=resolved_ratio,
            trainable_module_count=None,
        )
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}.")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}.")

        partition = partition_trainable_modules(
            model,
            last_n_ratio=resolved_ratio,
            last_n_modules=last_n_modules,
        )
        default_sign_lr = (
            lr * sign_lr_scale
            if partition.sign_modules and partition.adamw_modules
            else lr
        )
        sign_weight_decay = (
            (0.5 * weight_decay)
            if sign_weight_decay is None
            and partition.sign_modules
            and partition.adamw_modules
            else weight_decay if sign_weight_decay is None else sign_weight_decay
        )
        adamw_weight_decay = (
            weight_decay if adamw_weight_decay is None else adamw_weight_decay
        )

        self._validate_nonnegative("sign_weight_decay", sign_weight_decay)
        self._validate_nonnegative("adamw_weight_decay", adamw_weight_decay)

        self.partition = partition
        self.requested_last_n_modules = last_n_modules
        self.last_n_modules = partition.resolved_last_n_modules
        self.last_n_ratio = resolved_ratio
        self.adamw_ratio = resolved_ratio
        self.resolved_last_n_modules = partition.resolved_last_n_modules
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
                    "last_n_ratio": resolved_ratio,
                    "adamw_ratio": resolved_ratio,
                    "resolved_last_n_modules": partition.resolved_last_n_modules,
                    "sign_lr_scale": sign_lr_scale,
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
                    "last_n_ratio": resolved_ratio,
                    "adamw_ratio": resolved_ratio,
                    "resolved_last_n_modules": partition.resolved_last_n_modules,
                    "foreach": foreach,
                }
            )

        defaults = {
            "lr": lr,
            "last_n_ratio": resolved_ratio,
            "adamw_ratio": resolved_ratio,
            "sign_lr_scale": sign_lr_scale,
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
            group.setdefault("last_n_ratio", _DEFAULT_ADAMW_RATIO)
            group.setdefault("adamw_ratio", _DEFAULT_ADAMW_RATIO)
            group.setdefault("resolved_last_n_modules", 0)
            group.setdefault("sign_lr_scale", _DEFAULT_SIGN_LR_SCALE)
            group.setdefault("betas", (0.9, 0.999))
            group.setdefault("eps", 1e-8)
            group.setdefault("amsgrad", False)
            group.setdefault("error_if_nonfinite", False)
            group.setdefault("foreach", False)
        adamw_module_names = next(
            (
                tuple(group.get("module_names", ()))
                for group in self.param_groups
                if group["stac_role"] == "adamw"
            ),
            (),
        )
        if not hasattr(self, "requested_last_n_modules"):
            self.requested_last_n_modules = None
        if not hasattr(self, "last_n_ratio"):
            self.last_n_ratio = float(self.param_groups[0]["last_n_ratio"])
        if not hasattr(self, "adamw_ratio"):
            self.adamw_ratio = float(self.param_groups[0]["adamw_ratio"])
        if not hasattr(self, "resolved_last_n_modules"):
            self.resolved_last_n_modules = len(adamw_module_names)
        if not hasattr(self, "last_n_modules"):
            self.last_n_modules = self.resolved_last_n_modules
        self._drop_legacy_sign_state()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load optimizer state while preserving the current STAC partition.

        STAC derives its parameter groups from model structure, so loading a
        state dict whose saved sign/AdamW split does not match the current
        optimizer is almost always a user error. This override validates the
        partition before delegating to :meth:`torch.optim.Optimizer.load_state_dict`.
        """

        self._validate_state_dict_partition(state_dict)
        super().load_state_dict(state_dict)
        self._drop_legacy_sign_state()

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

        prepared_groups: list[_PreparedGroup] = []
        for group in self.param_groups:
            prepared_group, should_skip = self._prepare_group(group)
            if should_skip:
                self.nonfinite_skipped_steps += 1
                return loss
            prepared_groups.append(prepared_group)

        for prepared_group in prepared_groups:
            role = prepared_group.group["stac_role"]
            if role == "sign":
                self._step_sign(prepared_group)
                continue
            if role == "adamw":
                self._step_adamw(prepared_group)
                continue
            raise RuntimeError(f"Unexpected STAC parameter group role: {role!r}.")

        return loss

    def _prepare_group(
        self,
        group: dict[str, object],
    ) -> tuple[_PreparedGroup, bool]:
        role = str(group["stac_role"])
        error_if_nonfinite = bool(group["error_if_nonfinite"])
        maximize = bool(group["maximize"])
        parameters: list[nn.Parameter] = []
        gradients: list[torch.Tensor] = []
        found_nonfinite_on_cpu = False
        found_nonfinite_by_device: dict[torch.device, torch.Tensor] = {}

        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                if role == "sign":
                    raise RuntimeError("STAC does not support sparse gradients.")
                raise RuntimeError("AdamW section does not support sparse gradients.")

            is_nonfinite = torch.logical_not(torch.isfinite(gradient).all())
            if gradient.device.type == "cpu":
                if bool(is_nonfinite.item()):
                    found_nonfinite_on_cpu = True
            else:
                prior_flag = found_nonfinite_by_device.get(gradient.device)
                if prior_flag is None:
                    found_nonfinite_by_device[gradient.device] = is_nonfinite
                else:
                    found_nonfinite_by_device[gradient.device] = torch.logical_or(
                        prior_flag,
                        is_nonfinite,
                    )

            parameters.append(parameter)
            gradients.append(-gradient if maximize else gradient)

        found_nonfinite = found_nonfinite_on_cpu or any(
            bool(device_flag.item())
            for device_flag in found_nonfinite_by_device.values()
        )
        if found_nonfinite:
            if error_if_nonfinite:
                raise RuntimeError(
                    f"Encountered non-finite gradients in the STAC {role}."
                )
            return (
                _PreparedGroup(
                    group=group,
                    parameters=(),
                    gradients=(),
                ),
                True,
            )

        return (
            _PreparedGroup(
                group=group,
                parameters=tuple(parameters),
                gradients=tuple(gradients),
            ),
            False,
        )

    def _step_sign(self, prepared_group: _PreparedGroup) -> None:
        group = prepared_group.group
        parameters = prepared_group.parameters
        gradients = prepared_group.gradients
        if not parameters:
            return

        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])

        use_foreach = bool(group["foreach"]) and self._can_use_foreach(
            parameters,
            gradients,
        )
        if use_foreach:
            if weight_decay != 0:
                torch._foreach_mul_(parameters, 1 - lr * weight_decay)

            directions = torch._foreach_sign(list(gradients))
            torch._foreach_add_(parameters, directions, alpha=-lr)
            return

        for parameter, gradient in zip(parameters, gradients, strict=True):
            if weight_decay != 0:
                parameter.mul_(1 - lr * weight_decay)

            direction = gradient.sign()
            if direction.dtype != parameter.dtype:
                direction = direction.to(dtype=parameter.dtype)
            parameter.add_(direction, alpha=-lr)

    def _step_adamw(self, prepared_group: _PreparedGroup) -> None:
        group = prepared_group.group
        parameters = prepared_group.parameters
        gradients = prepared_group.gradients
        if not parameters:
            return

        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        amsgrad = bool(group["amsgrad"])
        exp_avgs: list[torch.Tensor] = []
        exp_avg_sqs: list[torch.Tensor] = []
        max_exp_avg_sqs: list[torch.Tensor] = []
        state_steps: list[torch.Tensor] = []

        for parameter in parameters:
            state = self.state[parameter]
            if not state:
                state["step"] = torch.tensor(0.0)
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
                step = state["step"] = torch.tensor(float(step))
            elif step.device.type != "cpu":
                step = state["step"] = step.detach().to(device="cpu")

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(step)
            if amsgrad:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

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
            if "stac_role" not in saved_group:
                raise ValueError(
                    "Saved optimizer state is not a STAC state dict. "
                    "Load a state dict that was produced by the same STAC "
                    "sign/AdamW partition."
                )

            saved_role = saved_group["stac_role"]
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
                if current_role == "sign":
                    self._validate_sign_state_keys(
                        parameter_state,
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

    def _drop_legacy_sign_state(self) -> None:
        for group in self.param_groups:
            if group["stac_role"] != "sign":
                continue
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not isinstance(state, dict):
                    continue
                state.pop("sign_momentum_buffer", None)
                if not state:
                    self.state.pop(parameter, None)

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

    @staticmethod
    def _validate_sign_state_keys(
        parameter_state: Mapping[str, Any],
        *,
        group_index: int,
        param_index: int,
    ) -> None:
        unexpected_state_keys = tuple(
            sorted(set(parameter_state) - _LEGACY_SIGN_STATE_KEYS)
        )
        if unexpected_state_keys:
            raise ValueError(
                "Saved STAC sign section must not carry optimizer state: "
                f"group {group_index} parameter {param_index} contains "
                f"{unexpected_state_keys!r}."
            )
