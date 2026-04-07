# paper plan

Title: **Temporal Significance Scheduling for MX-Quantized LLM FFN Inference on CIM**

## 1. One-sentence thesis

For CIM LLM FFN acceleration, temporal significance scheduling is a more accurate alternative to local sparsity-based compute reduction, while remaining equally compatible with regular CIM execution.

## 2. Problem framing

Existing CIM-friendly LLM sparsity methods often choose fixed local activation masks because they map cleanly to regular hardware. In modern LLM FFNs, however, those fixed local rules are accuracy-limited: quality is usually recovered only by making the sparse pattern more flexible, adding transformations, or changing the model itself. We argue that the missing ingredient is not executability but a better significance signal. We propose **temporal significance scheduling**: MX block-scale metadata, activation exponent fields, and calibrated horizons are converted directly into **local execution windows for channel-parallel, block-serial projection engines**. This is a more accurate compute-reduction scheme while preserving the regular CIM execution model required for practical hardware realization.

## 3. Core idea: temporal significance scheduling

**Temporal significance scheduling** has two parts.

### 3.1 Online arrival-time synthesis

At runtime, each stage-1 contribution on path `p in {g, u}` is assigned a hierarchical arrival time.

- The **inter-block delay** is determined by the sum of the activation and weight block-scale exponents, which is equivalent to the product of the underlying power-of-two block scales.
- The **intra-block delay** is determined by the activation-side fine-delay code `lambda_x[n,b,k]`, decoded from the activation element exponent field.
- The **weight element exponent is not resolved online**; it is **folded offline into the stored recoded weight representation** during weight recoding.

This gives a clean separation between runtime control and offline weight preparation.

### 3.2 Offline horizon calibration

A per-path, per-output-channel horizon `H[p,c]` determines how much useful execution is allocated to each stage-1 path and output channel. The horizon can be assigned in three modes:

- **uniform horizon**
- **SNR-calibrated horizon**
- **fixed-sum calibrated horizon**

The horizon is calibrated offline from activation and weight statistics, while the arrival time is resolved online. If the two stage-1 paths share the same horizon budget, that is the special case `H[g,c] = H[u,c]`.

## 4. Mathematical formulation

We express temporal significance scheduling directly in terms of MX exponent metadata and online digit streams.

For token `n`, stage-1 path `p in {g, u}`, output channel `c`, block `b`, and in-block element `k`, let

- `beta_x[n,b]` be the activation block-scale exponent read from the activation E8M0 metadata,
- `beta_w[p,c,b]` be the weight block-scale exponent read from the weight E8M0 metadata,
- `xi_x[n,b,k] = ExpFld(x_q[n,b,k])` be the stored exponent field of the activation element format,
- `lambda_x[n,b,k] = psi_fmt(xi_x[n,b,k])` be the activation-side fine-delay code derived from that field.

The associated block scales are therefore

```text
s_x[n,b]     = 2^beta_x[n,b]
s_w[p,c,b]   = 2^beta_w[p,c,b]
```

Any fixed bias in the stored E8M0 codes can be absorbed into a constant offset because only exponent differences are used later.

The weight-side element exponent is absorbed offline into the stored recoded weight digits, so it does not appear in runtime control. We therefore treat the online multiplier input as

```text
w_rec[p,c,b,k] = OfflineRecode(w[p,c,b,k])
```

and the runtime control path only needs the integer metadata `beta_x`, `beta_w`, `lambda_x`, and `H`.

### 4.1 Coarse and fine arrival terms

The control plane first forms the coarse block-level exponent

```text
E_raw[p,b,c] = beta_x[n,b] + beta_w[p,c,b]
```

Then it computes the per-path, per-channel block maximum and the corresponding coarse block delay

```text
E_max[p,c] = max_b E_raw[p,b,c]
D[p,b,c]   = E_max[p,c] - E_raw[p,b,c]
```

The runtime arrival index of element `k` in block `b` is then

```text
tau[p,b,k,c] = D[p,b,c] + lambda_x[n,b,k]
```

This is the key alignment between the paper and the hardware view:

- the paper-level phrase "product of activation and weight block scales" is represented explicitly as `s_x[n,b] * s_w[p,c,b]`,
- the hardware-visible operation is the cheaper exponent-field form `beta_x[n,b] + beta_w[p,c,b]`.

### 4.2 Useful window on the online contribution stream

Let `pi[p,b,k,c][m]` denote the `m`-th MSD-first digit emitted by the online multiplication of the activation digit stream and the offline-recoded weight digits for path `p`, block `b`, element `k`, and channel `c`.

After alignment by the arrival index, the contribution stream is

```text
s[p,b,k,c][t] = pi[p,b,k,c][t - tau[p,b,k,c]]   if t >= tau[p,b,k,c]
               = 0                              otherwise
```

Given a calibrated horizon `H[p,c]`, temporal significance scheduling executes only the prefix of this aligned stream that lies inside the useful window.

Define

```text
W[p,b,k,c] = [tau[p,b,k,c], H[p,c])
L[p,b,k,c] = max(0, H[p,c] - tau[p,b,k,c])
```

and the executed stream

```text
s_exec[p,b,k,c][t] = 1[t in W[p,b,k,c]] * s[p,b,k,c][t]
```

Thus, the executed object is not an input mask but the restricted online contribution stream `s_exec[p,b,k,c][t]`.

In the hardware section this maps directly to the local counters

```text
start_ctr[k] = tau[p,b,k,c]
rem_ctr[k]   = L[p,b,k,c]
```

Under block-serial execution, `W[p,b,k,c]` is relative to the start of the current block service, not a promise about absolute global time.

### 4.3 Unified work-suppression view

The same formulation gives the three savings modes directly:

- **whole-block skip**: `L[p,b,k,c] = 0` for all `k` in block `b`
- **element skip**: `L[p,b,k,c] = 0` for a specific element `k`
- **partial-window execution**: `0 < L[p,b,k,c] < H[p,c]`

This is the hardware-facing meaning of temporal significance scheduling: MX exponent metadata determine arrival indices, calibrated horizons define useful windows, and online arithmetic executes only the contribution digits that fall inside those windows.

## 5. Hardware realization and evaluation plan

At the hardware level, targeting LLM FFNs, the cleanest implementation remains a **two-plane micro-tile**:

- a **control plane** that computes useful windows one block ahead from `H[p,c]`, block-scale metadata, and activation-side fine-delay codes
- a **data plane** that executes only those windows with block-local serial engines, exporting a **reduced payload** between stage-1 production and stage-2 `down_proj` consumption

The control plane is split into a metadata-only, double-buffered scale prepass and a per-block window builder. The scale prepass resolves path-local block-scale maxima `E_max[p,c]` and coarse delays `D[p,b,c]` one context ahead, while the window builder combines those delays with shared activation fine-delay codes `lambda_x[n,b,k]` and per-channel horizons `H[p,c]` to generate block-relative local windows `W[p,b,k,c]`. The data plane then executes only those windows on channel-parallel, block-serial projection engines, suppressing whole-block, element, and partial-window work and exporting a reduced BSD intermediate payload to `down_proj`.

Under this organization, `gate_proj` and `up_proj` act as stage-1 producer paths, and `down_proj` is a stage-2 consumer that receives both the intermediate value and its timing and control context. The projection weights remain locally stored.

After block-serial channel accumulation, each gate-path result is finalized locally from BSD to fixed-point. SiLU is implemented by a clipped lookup-based approximator in conventional fixed-point arithmetic. This block is not a contribution of the paper; it is a low-cost local support function chosen to keep the nonlinear stage orthogonal to temporal significance scheduling.

Stage-1 dot products use serial-parallel online arithmetic, while gate fusion reuses a serial-parallel multiplier. Stage-1 produces a reduced intermediate payload with narrow sideband metadata. Detailed transport or interconnect design is outside the scope of this work.

The simulator already supports this interpretation on top of a modified **Qwen3-0.6B** stack. It implements **MXFP block quantization** with block size `32`, uniform and calibrated cycle horizons derived from combined activation and weight scales, inter-block and intra-block delay modeling, stream and scatter tracing, and full-model perplexity evaluation. This matters because the paper's empirical story is not only about numerical quality; it is also about whether the temporal schedule maps to a believable hardware trace.

Calibration method:

```text
minimize    sum_{p,c} epsilon_{p,c}(H[p,c])
subject to  sum_{p,c} H[p,c] <= H_total
```

where `epsilon_{p,c}(H[p,c])` is the path- and channel-specific error after allocating horizon `H[p,c]`.

Then:

- `uniform horizon` = simple baseline
- `SNR-calibrated horizon` = cheap initialization or per-channel lower bound
- `fixed-sum calibration` = global reallocation over marginal error reduction

The comparison groups should be defined directly from the motivation. The key baselines are:

- dense MX baseline
- CIM-style activation-gated baseline
- strong offline structured `2:4` baseline

Two specific comparison groups:

- **hardware-fair baselines**: dense MX, activation-gated CIM baseline, a naive global cutoff or uniform truncation baseline, and the paper's own ablations
- **algorithmic reference**: strong offline structured `2:4`

We use two x axis for comparison, one is normalized digit reading, one is end to end energy cost (CIM baseline will be adopted from papers and algorithmic reference only in the first)

## 6. Summarize

This paper presents temporal significance scheduling for MX-quantized LLM FFNs on CIM. Instead of introducing post-quantization sparsity masks, we compile existing MX scale and exponent metadata plus calibrated per-channel horizons into arrival offsets and useful execution windows. We realize these windows with a channel-parallel, block-serial online-CIM projection engine that suppresses unnecessary work at whole-block, element, and partial-window granularity. Beyond computation savings, the same schedule reduces the amount of intermediate payload between FFN stages. Evaluated on quantized LLM FFNs, this formulation delivers a stronger quality-work tradeoff than practical CIM low-cost baselines.

## 7. Contribution

- A new abstraction: temporal significance scheduling that converts MX metadata and calibrated horizons into arrival offsets and execution windows for LLM FFNs on CIM.
- A hardware realization: a channel-parallel, block-serial online-CIM projection architecture that executes only useful windows and achieves whole-block skip, element skip, and partial-window execution.
- A system consequence and evaluation: the schedule reduces both executed work and intermediate payload, giving a better quality-work tradeoff than practical CIM baselines.
