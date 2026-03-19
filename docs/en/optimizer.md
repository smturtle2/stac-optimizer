# STAC Optimizer Docs

[README](https://github.com/smturtle2/stac-optimizer/blob/main/README.md) |
[Korean docs](https://github.com/smturtle2/stac-optimizer/blob/main/docs/ko/optimizer.md)

STAC means "SignSGD Trunk, AdamW Cap". It keeps earlier trainable modules on
momentum-stabilized sign updates and the last `N` trainable modules on AdamW.
The design goal is practical: cut optimizer-state VRAM against full AdamW while
keeping adaptivity where it matters most.

## Layout

```mermaid
flowchart LR
    A[model.named_modules order]
    A --> B[direct trainable modules only]
    B --> C[earlier modules]
    B --> D[last N modules]
    C --> E[sign trunk<br/>decoupled weight decay<br/>sign of EMA(grad)<br/>1 state tensor]
    D --> F[AdamW cap<br/>decoupled weight decay<br/>first and second moments]

    classDef neutral fill:#f8fafc,stroke:#475569,color:#0f172a,stroke-width:1px;
    classDef sign fill:#d7f0e8,stroke:#0f766e,color:#134e4a,stroke-width:1.5px;
    classDef adam fill:#dbeafe,stroke:#2563eb,color:#1d4ed8,stroke-width:1.5px;

    class A,B neutral;
    class C,E sign;
    class D,F adam;
```

STAC counts only modules that directly own trainable parameters
(`named_parameters(recurse=False)`). Pure containers such as `nn.Sequential`
are skipped unless they own parameters themselves.

## Update Rules

| Section | Modules | Rule | Optimizer state |
| --- | --- | --- | --- |
| Sign trunk | all trainable modules before the last `N` | decoupled weight decay, then `sign(EMA(grad))` | one momentum buffer per parameter |
| AdamW cap | last `N` trainable modules | standard AdamW | `exp_avg` + `exp_avg_sq` (+ AMSGrad max if enabled) |

When both sections are active, STAC uses `sign_lr_scale * lr` for the sign
trunk and `lr` for the AdamW cap. The default `sign_lr_scale=0.75` is a
conservative setting that benchmarked well in this repository's CUDA study.

## Stability And Memory Knobs

| Argument | Default | Why it exists |
| --- | --- | --- |
| `last_n_modules` | `1` | Keep the final task-critical module adaptive by default |
| `sign_momentum` | `0.9` | Momentum before `sign` is markedly more stable than raw sign updates |
| `sign_lr_scale` | `0.75` | Keeps the sign trunk slightly more conservative in hybrid mode |
| `sign_state_dtype` | `"auto"` | Uses FP32 sign state for FP16/BF16 params by default; explicit BF16 is available to cut more VRAM |
| `foreach` | `False` | Keeps peak CUDA memory lower by default; opt in when step throughput matters more |
| `error_if_nonfinite` | `False` | Either raise immediately or skip the whole step on `NaN`/`Inf` gradients |

`sign_state_dtype="auto"` means:

- FP16 or BF16 parameters keep their sign momentum state in FP32.
- FP32 and FP64 parameters match their own dtype.
- `None` or `"parameter"` forces exact parameter-dtype matching instead.

`foreach=False` is a deliberate default. PyTorch documents that the foreach
optimizer path is often faster on CUDA, but it uses extra peak memory because
its intermediates are stored as tensor lists. STAC keeps the memory-lean path
by default and lets users opt in explicitly.

## Recommended Presets

| Goal | Suggested config |
| --- | --- |
| Safe default | `STAC(model, last_n_modules=1, sign_state_dtype="auto")` |
| Lower VRAM | `STAC(model, last_n_modules=1, sign_state_dtype="bf16")` |
| More adaptive tail | `STAC(model, last_n_modules=2, sign_state_dtype="auto")` |

The repository CUDA tests cover the default preset, the BF16 sign-state
variant, and a LayerNorm-heavy classification task to reduce benchmark bias
toward plain MLP heads.

## Choosing `last_n_modules`

- `1` is the default and works well for small MLP/CNN heads.
- `2` is often a better starting point when the final normalization layer and
  head both matter.
- On the repository's LayerNorm-heavy CUDA stress task, larger caps than `2`
  improved further, so inspect `optimizer.partition` rather than treating `2`
  as universal.
- For transformer-style models, inspect `optimizer.partition` and make sure the
  final norm/head are inside the AdamW cap.

The motivation is consistent with the ICLR 2025 optimizer study that found
adaptivity on the last layer and LayerNorm parameters especially important for
retaining performance and stability.

```python
optimizer = STAC(model, last_n_modules=2)
print(optimizer.partition.sign_module_names)
print(optimizer.partition.adamw_module_names)
```

## Public API

| Symbol | Purpose |
| --- | --- |
| `STAC` | Hybrid optimizer itself |
| `partition_trainable_modules(model, last_n_modules=1)` | Deterministically split trainable modules into sign and AdamW sections |
| `ModuleGroup` | One direct-owning trainable module slice |
| `STACPartition` | Named view over the resulting sign/AdamW split |

Important runtime guarantees:

- deterministic partitioning from `model.named_modules()`
- explicit sparse-gradient rejection
- whole-step skip on non-finite dense gradients unless `error_if_nonfinite=True`
- state-dict validation for roles, module names, parameter names, and tensor shapes

## Benchmark Evidence

Primary assets:

- [Benchmark script](https://github.com/smturtle2/stac-optimizer/blob/main/examples/research_benchmark.py)
- [JSON report](https://github.com/smturtle2/stac-optimizer/blob/main/docs/benchmark/research_benchmark.json)
- [Loss-curve PNG](https://github.com/smturtle2/stac-optimizer/blob/main/docs/benchmark/research_benchmark.png)

Methodology used by the repository benchmark:

- CUDA only
- separate train and validation splits
- `5` paired seeds
- per-trial model initialization seed matched across optimizers
- epoch-by-epoch validation loss curves
- optimizer-state and peak CUDA memory probe on the first optimization step

## References

- [signSGD: Compressed Optimisation for Non-Convex Problems](https://arxiv.org/abs/1802.04434)
- [Momentum Ensures Convergence of SIGNSGD under Weaker Assumptions](https://proceedings.mlr.press/v202/sun23l.html)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Deconstructing What Makes a Good Optimizer for Autoregressive Language Models](https://openreview.net/forum?id=zfeso8ceqr)
