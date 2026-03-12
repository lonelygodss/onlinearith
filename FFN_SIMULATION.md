# MSD-First Online Arithmetic Across the FFN Layer

Detailed technical explanation of how the simulation models MSD-first online arithmetic
through the entire Feed-Forward Network (FFN / MLP) of the Qwen3 transformer.

---

## 1. FFN Layer Structure

The Qwen3 MLP (Gated Linear Unit variant) computes:

```
FFN(x) = W_down * [ SiLU(W_gate * x) ⊙ (W_up * x) ]
```

This decomposes into five computational stages:

| Stage | Operation | Type |
|-------|-----------|------|
| 1a | `gate_out = gate_proj(x)` | GEMM (dot-product) |
| 1b | `up_out = up_proj(x)` | GEMM (dot-product, parallel with 1a) |
| 2 | `silu_out = SiLU(gate_out)` | Nonlinear activation |
| 3 | `gating_out = silu_out ⊙ up_out` | Element-wise multiply |
| 4 | `result = down_proj(gating_out)` | GEMM (dot-product) |

In standard FP16 inference, stages 2 and 3 use exact floating-point arithmetic.
In the MSD-first simulation, **every stage** operates in the time domain with
Binary Signed-Digit (BSD) representation, tracking per-element precision throughout.

---

## 2. Representations and Key Concepts

### 2.1 MXFP Block Quantization

Before MSD truncation, activations and weights are quantized to a Microscaling (MX)
block format (MXFP8, MXFP6, or MXFP4). Each block of 32 elements shares a single
FP8 scale factor:

```
x[n, b, k] = scale_x[n, b] * x_q[n, b, k]
```

where `b` is the block index, `k` is the element within the block, and `x_q` is the
quantized element value.

### 2.2 BSD (NAF) Representation

The simulation uses Non-Adjacent Form (NAF) as the canonical BSD encoding. NAF has
the minimum-weight property (fewest non-zero digits) and is uniquely determined for
every integer. It is computed via:

```
x_h = x >> 1
s   = x + x_h
naf_pos = s & ~x_h    (positive digit positions)
naf_neg = x_h & ~s    (negative digit positions)
```

Key property: NAF can shift the most-significant-digit position by +1 compared to
plain binary (e.g., `7 = 0b111` in binary but `100(-1)` in NAF = 4 digit positions).
This makes BSD truncation behaviour distinct from binary truncation — the error is
**bidirectional** (can be positive or negative) rather than always rounding toward zero.

### 2.3 Exponent / Mantissa Separation Principle

A key design principle of the MSD-first pipeline is that **exponent and BSD mantissa
are kept as separate as possible**, avoiding combinatorial shift-and-align hardware.

- **Exponent** propagates as parallel metadata, available immediately (before the
  first BSD mantissa digit arrives). For GEMM outputs, it equals `E_max` — the maximum
  combined block-scale exponent per output channel — known from MXFP block scales alone.

- **BSD mantissa** flows serially in the MSD-first digit stream. Each cycle produces
  one more valid digit, starting from the most-significant digit.

Necessary alignment between blocks of different magnitude is offloaded to the **time
domain**: smaller blocks are simply delayed (start their MSD stream later) rather than
being shifted by combinatorial hardware. This is precisely the inter-block delay
mechanism in the GEMM. The separation provides extra pipelining opportunity and
enables early termination to exploit dynamic-range sparsity (MAC sparsity).

### 2.4 BSDMetadata

When BSD representation penetrates through the FFN, each intermediate value carries
per-element metadata:

```python
class BSDMetadata:
    exponent:  (N, dim) float32  # MSD position (from block scales, not output value)
    precision: (N, dim) float32  # valid BSD mantissa digits remaining
```

For GEMM outputs:

- `exponent = E_max` — the maximum combined block-scale exponent per (sample,
  output-channel) pair. This is known from MXFP block scales before any BSD
  mantissa digit arrives.
- `precision = B_final - delta` — the cycle budget minus the online startup delay.
  Inter-block delays are **NOT** subtracted here, because they already govern
  per-element product truncation inside the GEMM. Once the first MSD of the
  accumulator output arrives, every cycle produces one more valid digit.

For element-wise operations (SiLU, gating multiply):

- `exponent` is recomputed from the truncated result value.
- `precision` is the input precision minus the operation's cycle cost.

This metadata propagates from one stage to the next, allowing downstream operations
to know exactly how many valid digits each input element has, and at what scale.

---

## 3. Stage-by-Stage Simulation

### 3.1 GEMM Stages: gate_proj, up_proj (Stages 1a, 1b)

Each GEMM computes a block-quantized dot-product with MSD truncation. The algorithm
for a single output element `out[n,j]` (sample n, output channel j):

```
out[n,j] = sum_b  s_x[n,b] * s_w[j,b] * sum_k  x_q[n,b,k] * w_q[j,b,k]
```

**Step 1: Inter-block delay computation**

Each block has a combined scale exponent:

```
E_i[n,j,b] = floor(log2(s_x[n,b] * s_w[j,b]))
```

The dominant block has the maximum exponent:

```
E_max[n,j] = max_b  E_i[n,j,b]
```

In the MSD-first pipeline, all blocks must be aligned to the dominant block's digit
position. Smaller blocks are delayed (alignment via time domain, no shift hardware):

```
inter_delay[n,j,b] = E_max[n,j] - E_i[n,j,b]
```

**Step 2: Intra-block delay computation**

Within each block, element-level activation exponents differ:

```
e_k[n,b,k] = floor(log2(|x_q[n,b,k]|))
intra_delay[n,b,k] = e_max[n,b] - e_k[n,b,k]
```

Elements with smaller magnitudes start producing significant digits later in the
serial BSD stream.

**Step 3: Budget resolution**

The per-(sample, channel) cycle budget is resolved through a three-tier system:

```
B_final[n,j] = B_base[j] + delta_B[n,j]
```

where `B_base` comes from uniform config or offline calibration, and `delta_B`
is a runtime dynamic adjustment based on combined scale exponents.

**Step 4: Effective precision per element**

```
p_eff[n,j,b,k] = max(0,  B_final[n,j] - inter_delay[n,j,b] - intra_delay[n,b,k] - delta)
```

where `delta` is the MSD online multiplier delay (default 2 cycles).

**Step 5: BSD (NAF) truncation**

Each partial product `x_q[n,b,k] * w_q[j,b,k]` is converted to NAF and truncated
to `p_eff` most-significant digits. The truncated products are summed and
rescaled by the block scales.

**Step 6: BSD metadata extraction (when returning metadata)**

When the GEMM returns BSD metadata (for use by downstream stages):

```
exponent[n,j]  = E_max[n,j]       (from block scales, known before BSD stream)
precision[n,j] = max(0, B_final[n,j] - delta)
```

The exponent comes directly from the GEMM's block-scale computation (`E_max`),
not from the computed output value. This keeps exponent and mantissa metadata
separate — the exponent is available as parallel metadata before any BSD digit flows.

The precision is `B_final - delta` (budget minus online startup delay). Inter-block
delays are **not** subtracted: they already govern per-element product truncation
in Steps 4-5. Once the accumulator's first MSD arrives, every subsequent cycle
produces one additional valid digit. If the flow has produced 8 MSDs, the precision
is 8 regardless of how large the max inter-block delay was.

### 3.2 SiLU Activation (Stage 2): PWL Approximation

The SiLU function `f(x) = x * sigmoid(x)` is decomposed as:

```
SiLU(x) = x * sigmoid_PWL(x)
```

where `sigmoid_PWL` is a piecewise-linear approximation of the sigmoid function.

#### Hardware Architecture

The SiLU unit is a two-sub-module pipeline:

1. **Segment Detection (0 cycles from BSD stream)**: Uses `BSDMetadata.exponent`
   (available as parallel metadata from the upstream GEMM block scales) to determine
   which of N linear segments (default 8) applies. Because exponent is already known
   before the BSD mantissa stream begins, this detection costs zero cycles from the
   mantissa perspective. The sign bit (also parallel metadata) determines the sign of
   the input for the LUT lookup.

2. **Coefficient Lookup (~0 cycles, pipelined)**: A small LUT provides slope `a_i`
   and intercept `b_i` in parallel format (no serialization needed for constants).

3. **Online MAC (3 cycles)**: Computes `a_i * x + b_i` using an online
   multiplier (delta=2) and online adder (delta=1).

4. **SiLU multiply (2 cycles)**: Computes `x * sigmoid_PWL(x)` using an online
   multiplier (delta=2).

The total SiLU latency is:

```
SiLU_latency = 0 (segment detect) + 3 (PWL eval) + 2 (SiLU multiply) = 5 cycles
```

#### Simulation Implementation

The PWL sigmoid approximation divides `[-6, 6]` into N segments (default 8).
For each segment `[x_i, x_{i+1}]`:

```
sigmoid_PWL(x) = a_i * x + b_i
```

where `a_i` and `b_i` are computed by linear interpolation of the exact sigmoid at
segment boundaries. Values outside `[-6, 6]` are saturated (0 or 1).

The simulation uses `torch.searchsorted` for segment detection (modelling the
hardware segment detector) and computes:

```python
silu_out = x * sigmoid_pwl(x)
```

**Precision propagation through SiLU:**

The output precision is the input precision minus the SiLU latency:

```
p_out[n,j] = max(0,  p_in[n,j] - 5)
```

The result is then truncated to `p_out` BSD digits via `_msd_truncate()`.
A new `BSDMetadata` is created with exponent recomputed from the truncated output
(because SiLU changes the magnitude) and precision set to `p_out`.

### 3.3 Gating Multiply (Stage 3): Element-wise SiLU(gate) * up

Two BSD streams — the SiLU output and the up_proj output — are multiplied
element-wise using an online multiplier.

**Precision rule:**

The output precision is limited by the *less precise* of the two inputs, minus the
online multiplier delay:

```
p_gating[n,j] = max(0,  min(p_silu[n,j], p_up[n,j]) - delta)
```

where `delta` is the online delay (default 2 cycles). The `min` reflects the hardware
constraint that the multiplier cannot produce a digit until both inputs have a digit
available at that position.

The result is truncated to `p_gating` digits and a new `BSDMetadata` is created.

### 3.4 down_proj (Stage 4): BSD-Input GEMM

The gating multiply output feeds into `down_proj` as pre-existing BSD data (not
freshly quantized MXFP values). This requires a special GEMM variant that differs
from the standard path in two ways:

1. **No MXFP re-quantization of the input**: The BSD values are directly reshaped into
   blocks. Block scales are computed from actual input magnitudes (for inter-block
   alignment). Only the weights are MXFP-quantized.

2. **Input precision caps effective precision**: Each input element already has a known
   precision from the gating stage. The effective precision of each partial product is:

```
p_eff[n,j,b,k] = min(p_budget[n,j,b,k], p_input_bsd[n, b*32+k])
```

where `p_budget` is the standard budget-limited precision (from delays and
budget allocation) and `p_input_bsd` is the per-element input precision from
the gating stage. This models the hardware reality that a serial BSD stream with only
P valid digits cannot contribute more than P digits to a downstream product,
regardless of how many cycles the accumulator runs.

---

## 4. Two Operational Modes

The simulation supports two modes for BSD-penetration FFN, controlled by configuration flags.

### 4.1 Mode 1: BSD Penetration (`msd_bsd_penetration = true`)

**Concept:** Each GEMM stage (gate/up/down) has its own independent cycle budget from
`msd_cycle_budget`. However, data stays in BSD representation between stages — there
is no MXFP re-quantization between gate→SiLU→gating→down.

**Data flow:**

```
x (MXFP input)
├── gate_proj(x, budget=B)  → gate_out, gate_bsd     [standard MSD GEMM]
│       ↓
│   SiLU_PWL(gate_out, gate_bsd)  → silu_out, silu_bsd
│       ↓
├── up_proj(x, budget=B)    → up_out, up_bsd          [standard MSD GEMM]
│       ↓
│   gating_mul(silu_out, silu_bsd, up_out, up_bsd) → gating_out, gating_bsd
│       ↓
└── down_proj(gating_out, input_bsd=gating_bsd, budget=B)  [BSD-input GEMM]
        ↓
    result
```

**Key property:** Each GEMM runs with the full configured budget. The precision loss
through SiLU and gating is modelled explicitly but does not reduce the GEMM budgets.
The down_proj stage benefits from knowing the actual input precision (which may be less
than what a fresh MXFP quantization would provide).

**Config:**
```json
{
  "use_msd_truncation": true,
  "msd_bsd_penetration": true,
  "msd_deep_pipeline": false,
  "msd_cycle_budget": 16
}
```

### 4.2 Mode 2: Deep Pipeline (`msd_deep_pipeline = true`)

**Concept:** The gate_proj→SiLU→gating chain is treated as a single time-domain
pipeline with a unified cycle budget `B_pipe`. The pipeline budget is split
between the GEMM stages and the intermediate operations.

**Cycle allocation:**

```
T_gemm = B_pipe - SiLU_latency - online_delay
```

where:
- `B_pipe` = `msd_pipeline_budget` (default 24 cycles)
- `SiLU_latency` = 5 cycles (PWL SiLU latency)
- `online_delay` = 2 cycles (gating multiplier delay)

This gives `T_gemm = 24 - 5 - 2 = 17` cycles for each GEMM (gate and up).

**Data flow:**

```
x (MXFP input)
├── gate_proj(x, budget=T_gemm) → gate_out, gate_bsd    [budget overridden]
│       ↓
│   SiLU_PWL(gate_out, gate_bsd) → silu_out, silu_bsd   [costs 5 cycles]
│       ↓
├── up_proj(x, budget=T_gemm)   → up_out, up_bsd        [budget overridden]
│       ↓
│   gating_mul(silu_out, silu_bsd, up_out, up_bsd) → gating_out, gating_bsd
│       ↓                                               [costs 2 cycles]
└── down_proj(gating_out, input_bsd=gating_bsd, budget=B_std)
        ↓                                               [independent budget]
    result
```

**Key property:** The pipeline budget governs the total time for the gate→SiLU→gating
chain. The down_proj stage is *not* part of the pipeline — it receives an independent
budget from `msd_cycle_budget`. This is because down_proj's input must be transposed
(the output dimension of gate/up becomes the input dimension of down), which is a
natural pipeline break point.

**Implementation detail:** The simulation temporarily overrides `config.msd_cycle_budget`
to `T_gemm` for the gate and up projections, then restores it before down_proj.
This ensures the existing per-channel budget resolution machinery (calibration, dynamic
adjustment) applies at the reduced budget.

**Config:**
```json
{
  "use_msd_truncation": true,
  "msd_deep_pipeline": true,
  "msd_pipeline_budget": 24,
  "msd_cycle_budget": 16,
  "msd_silu_pwl_segments": 8
}
```

### 4.3 Mode Comparison

| Aspect | Mode 1 (BSD Penetration) | Mode 2 (Deep Pipeline) |
|--------|-------------------------|------------------------|
| gate/up GEMM budget | `msd_cycle_budget` (full) | `B_pipe - 5 - delta` |
| down_proj budget | `msd_cycle_budget` (full) | `msd_cycle_budget` (independent) |
| SiLU cost | Precision loss on output only | Subtracted from pipeline budget |
| Gating cost | Precision loss on output only | Subtracted from pipeline budget |
| Total cycles per FFN | `3 * B_cycle` | `B_pipe + B_cycle` (2 stages) |
| Use case | Maximum per-GEMM precision | Time-constrained pipeline |
| Hardware model | Independent GEMM engines | Single pipeline engine + separate down_proj |

---

## 5. Precision Propagation Through the FFN

### 5.1 Mode 1 (BSD Penetration) with B = 16, delta = 2, SiLU = 5 cycles

```
gate_proj GEMM (budget = B = 16)
│
│  gate_bsd.precision = B_final - delta = 16 - 2 = 14 digits
│  gate_bsd.exponent  = E_max (from block scales)
│
↓ SiLU_PWL  (costs 5 cycles)
│
│  p_silu = max(0, 14 - 5) = 9 digits
│  Truncate to 9 NAF digits
│
├── up_proj GEMM (budget = B = 16)
│   up_bsd.precision = 14 digits
│
↓ Gating multiply  (costs delta = 2 cycles)
│
│  p_gating = max(0, min(9, 14) - 2) = 7 digits
│  Truncate to 7 NAF digits
│
↓ down_proj BSD-input GEMM (budget = B = 16)
│
│  Input precision = gating_bsd.precision = 7
│  p_eff = min(budget_limited, input_precision)
│  Input precision is the bottleneck for most elements
│
↓ result
```

### 5.2 Mode 2 (Deep Pipeline) with B_pipe = 24, B_std = 16, delta = 2, SiLU = 5

```
T_gemm = 24 - 5 - 2 = 17

gate_proj GEMM (budget = T_gemm = 17)
│
│  gate_bsd.precision = 17 - 2 = 15 digits
│  gate_bsd.exponent  = E_max (from block scales)
│
↓ SiLU_PWL  (costs 5 cycles)
│
│  p_silu = max(0, 15 - 5) = 10 digits
│  Truncate to 10 NAF digits
│
├── up_proj GEMM (budget = T_gemm = 17)
│   up_bsd.precision = 15 digits
│
↓ Gating multiply  (costs delta = 2 cycles)
│
│  p_gating = max(0, min(10, 15) - 2) = 8 digits
│  Truncate to 8 NAF digits
│
↓ down_proj BSD-input GEMM (budget = B_std = 16)
│
│  Input precision = gating_bsd.precision = 8
│  p_eff = min(budget_limited, input_precision)
│  Input precision is the bottleneck for most elements
│
↓ result
```

Note: The previous (incorrect) model subtracted `max_inter_delay` from the GEMM output
precision, which could yield 0 for channels with large inter-block delays. The corrected
model recognises that inter-block delays already govern per-element product truncation
inside the GEMM, and the output stream precision depends only on budget minus startup
delay.

---

## 6. BSD-Input GEMM: Detailed Algorithm

The BSD-input variant (`_forward_msd_truncated_bsd_input`) differs from the standard
MSD GEMM in how it handles input activations. Here is the step-by-step:

**Standard GEMM path (gate_proj, up_proj):**
1. Quantize input `x` to MXFP → get `x_q` (quantized) and `s_x` (block scales)
2. Use `x_q` and `s_x` in the delay/budget/truncation pipeline
3. Return output + optional BSDMetadata

**BSD-input GEMM path (down_proj with `input_bsd`):**
1. Skip MXFP quantization entirely — use raw float values
2. Reshape input into blocks: `x_blocks[n, nb, bs]`
3. Compute block scales from actual magnitudes: `s_x[n,b] = max_k |x_blocks[n,b,k]|`
4. Compute intra-block delays normally from element exponents
5. Compute inter-block delays from combined `s_x` and weight scales
6. Compute budget-limited precision `p_budget` as usual
7. **Cap by input BSD precision:**

```
p_eff[n,j,b,k] = min(p_budget[n,j,b,k], p_input_bsd[n, b*32+k])
```

8. Truncate products and accumulate as normal

The `min` in step 7 is the key difference. It models the physical constraint that a
BSD digit stream with only P valid digits cannot contribute more than P digits
to any downstream computation, regardless of how long the accumulator runs.

---

## 7. PWL Sigmoid: Implementation Details

### 7.1 LUT Construction

The function `_build_pwl_sigmoid_lut(n_segments, device)` constructs the lookup table:

1. Divide `[-6, 6]` into `n_segments` equal intervals
2. Evaluate exact `sigmoid(x) = 1/(1+exp(-x))` at each boundary
3. Compute slope and intercept per segment by linear interpolation:

```
a_i = (sigmoid(x_{i+1}) - sigmoid(x_i)) / (x_{i+1} - x_i)
b_i = sigmoid(x_i) - a_i * x_i
```

The LUT is cached per (n_segments, device) combination.

### 7.2 Evaluation

The function `_pwl_sigmoid(x, n_segments=8)`:

1. Use `torch.searchsorted` to find segment index for each element (models the
   exponent-based segment detector in hardware)
2. Evaluate `a_i * x + b_i` (models the online MAC: multiply by parallel
   coefficient, add parallel intercept)
3. Clamp result to `[0, 1]`

### 7.3 Full SiLU with BSD Metadata

The function `_msd_silu_pwl(x, input_bsd, n_segments, online_delay)`:

1. Use `input_bsd.exponent` for segment detection (0 cycles — parallel metadata)
2. Compute PWL sigmoid: `sigmoid_PWL(x)`
3. Compute SiLU: `y = x * sigmoid_PWL(x)`
4. Compute output precision: `p_out = max(0, p_in - 5)`
5. Truncate `y` to `p_out` NAF digits
6. Compute output exponent from truncated result
7. Return `(y_truncated, BSDMetadata(e_out, p_out))`

---

## 8. Gating Multiply: Implementation Details

The function `_msd_gating_mul(silu_val, silu_bsd, up_val, up_bsd, online_delay)`:

1. Compute output precision:
   `p_gating = max(0, min(p_silu, p_up) - delta)`

2. Compute element-wise product: `y = silu_val * up_val`

3. Truncate to `p_gating` NAF digits

4. Compute output exponent from truncated result

5. Return `(y_truncated, BSDMetadata(e_out, p_gating))`

The `min(p_silu, p_up)` reflects the hardware reality that an online multiplier
cannot produce valid output digits faster than its slowest input stream.

---

## 9. Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `msd_bsd_penetration` | bool | false | Enable Mode 1: BSD representation flows through the entire FFN without MXFP re-quantization between stages |
| `msd_deep_pipeline` | bool | false | Enable Mode 2: Unified pipeline budget for gate-SiLU-gating chain |
| `msd_pipeline_budget` | int | 24 | Pipeline cycle budget `B_pipe` (Mode 2 only). GEMM budget = `B_pipe - 5 - delta` |
| `msd_silu_pwl_segments` | int | 8 | Number of linear segments for PWL sigmoid approximation |
| `msd_cycle_budget` | int | 16 | Standard per-GEMM cycle budget (Mode 1: all GEMMs; Mode 2: down_proj only) |
| `msd_online_delay` | int | 2 | Online multiplier delay `delta` (used in GEMM, SiLU, and gating) |

### Interaction between modes

- If both `msd_bsd_penetration` and `msd_deep_pipeline` are true, **Mode 2 takes priority**
- If neither is true, each GEMM (gate/up/down) runs standard MSD truncation independently, and SiLU + gating use exact FP16 arithmetic
- `msd_pipeline_precision_loss` is **deprecated** (kept for backward compatibility with old deep pipeline code)

---

## 10. Implementation Files

| File | Role |
|------|------|
| `modular_qwen3.py` | Reference implementation — edit here |
| `modeling_qwen3.py` | Production copy with output-chunked MSD for large matrices |
| `configuration_qwen3.py` | Config class with all MSD/BSD/pipeline fields |

### Key functions (both files)

| Function | Purpose |
|----------|---------|
| `_msd_truncate(value, num_digits)` | Core BSD (NAF) truncation to N most-significant digits |
| `BSDMetadata` | Per-element exponent (E_max) + mantissa precision tracking |
| `_extract_bsd_metadata(output, b_final, e_max, online_delay)` | Extract BSD metadata from GEMM output |
| `_build_pwl_sigmoid_lut(n_segments, device)` | Construct PWL sigmoid LUT |
| `_pwl_sigmoid(x, n_segments)` | Evaluate PWL sigmoid (cached) |
| `_msd_silu_pwl(x, input_bsd, n_segments, online_delay)` | SiLU with BSD metadata propagation |
| `_msd_gating_mul(silu_val, silu_bsd, up_val, up_bsd, online_delay)` | Element-wise gating multiply |
| `_forward_msd_truncated(self, x, w_q, ...)` | Standard MSD GEMM (with optional BSD metadata return) |
| `_forward_msd_truncated_bsd_input(self, x, x_bsd, w_q, ...)` | BSD-input GEMM variant |
| `Qwen3MLP.forward(x, compute_context)` | Mode dispatch (Mode 1 / Mode 2 / standard MSD / exact) |

---

## 11. Hardware Correspondence

The simulation maps to the following hardware components:

| Simulation Operation | Hardware Unit | Cycle Cost |
|---------------------|---------------|------------|
| `_forward_msd_truncated` | MSD-first CiM dot-product array | `B_budget` cycles per output |
| `_pwl_sigmoid` searchsorted | Segment detector (uses exponent metadata) | 0 from BSD stream |
| Coefficient lookup | LUT for slope/intercept (parallel constants) | 0 (pipelined) |
| sigmoid PWL eval | Online multiply (slope*x) + online add (+intercept) | 3 cycles |
| SiLU multiply (x * sigmoid) | Online multiplier (serial x serial) | 2 cycles |
| `_msd_gating_mul` | Online element-wise multiplier | `delta` = 2 cycles |
| `_msd_truncate` | Hardware early termination (stop clocking after P digits) | 0 (implicit) |
| `_forward_msd_truncated_bsd_input` | CiM array with pre-existing BSD input stream | `B_budget` cycles (capped by input precision) |
| FIFO delay buffer | up_proj result holds in buffer during SiLU | 5 entries |
| BSDMetadata.exponent | Parallel metadata register (from block scales) | 0 (available before BSD stream) |

The simulation does not model:
- Actual FIFO buffering between stages (assumes instant availability after delay)
- Pipeline stalls or backpressure
- Clock gating for zero-precision elements (modelled as `p_eff=0` in statistics)
- Power consumption (deferred to hardware simulation using the collected statistics)
