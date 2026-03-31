# paper plan

Title: Significance-Guided Useful-Window Scheduling for a CIM-Macro-Based Weight-Stationary LLM FFN Accelerator

This work studies a **CIM accelerator for the FFN layers of large language models**. The paper should be framed around **one integrated thesis**: sequentially composing low-precision quantization and sparsification is a poor abstraction for FFN acceleration, especially for regular CIM organizations; instead, the cheap metadata already produced by MX quantization should be converted directly into a **temporal execution schedule**. We therefore use MX scale metadata, leading-digit timing offsets, and offline-calibrated budgets to decide **when each contribution becomes useful and how long it should keep running**. The resulting **useful execution windows** are then realized in a regular CIM-friendly online-arithmetic pipeline.

Many low-cost inference flows effectively adopt a **quantization-first, sparsification-second** recipe. Existing studies and our own planned validation suggest that quantization and sparsification are not cleanly orthogonal in LLM FFNs, and that loose sequential treatment can incur extra quality loss—especially when quantization is applied before structured sparsification. That flow remains attractive in CIM-oriented designs because conventional dynamic sparsity maps poorly to regular macros. Our goal is to replace post-quantization irregular skipping with a **hardware-executable significance schedule**.

Under this view, dynamic fine-grained sparsity is expressed as **Significance-Guided Useful-Window scheduling**. The method uses the same quantization metadata that is already available in MX inference, but interprets it as **temporal significance metadata**. MX block scales provide coarse timing information, per-element leading-digit offsets provide fine timing information, and the online arithmetic pipeline contributes a fixed path delay. Offline calibration resolves how much cycle budget should be spent, while online scheduling resolves where that budget should be spent in time.

For each MLP linear layer

\[
\mathrm{out}[n,j] = \sum_b \mathrm{scale}_x[n,b] \cdot \mathrm{scale}_w[j,b] \cdot \mathrm{dot}(x_q[n,b,:], w_q[j,b,:]),
\]

we model temporal usefulness in three steps.

1. **Inter-block delay.** For block `b`, use the block-scale product to determine the coarse delay:

\[
E_i = \left\lfloor \log_2(\mathrm{scale}_x[n,b] \cdot \mathrm{scale}_w[j,b]) \right\rfloor,
\qquad
 d_{\mathrm{inter}} = E_{\max} - E_i.
\]

2. **Intra-block delay.** Within each block, per-element leading-digit offsets produce a finer timing shift `d_intra`.

3. **Budget resolution.** A per-`(sample, output-channel)` budget is resolved from combined activation and weight scales:

\[
E_{\mathrm{combined}} = \max_b \left\lfloor \log_2(\mathrm{scale}_x[n,b] \cdot \mathrm{scale}_w[j,b]) \right\rfloor,
\]
\[
B = B_{\mathrm{base}} + \alpha \cdot \max\left(0, E_{\mathrm{combined}} - E_{\mathrm{th}}\right).
\]

These terms are then converted into the hardware-facing form

\[
t_{\mathrm{arr}} = d_{\mathrm{inter}} + d_{\mathrm{intra}} + d_{\mathrm{online}},
\qquad
L = \max(0, B - t_{\mathrm{arr}}),
\]

and the actual executable object is the **useful execution window**

\[
W = [t_{\mathrm{arr}},\; t_{\mathrm{arr}} + L).
\]

With this formulation, the three hardware savings modes become one unified concept:

- **whole-block kill** when all elements in a block have `L = 0`
- **element skip** when a specific element has `L = 0`
- **partial-digit truncation** when `0 < L < L_full`

The hardware should therefore be described as a **channel-parallel, block-serial** FFN microarchitecture. All owner lanes / channels remain aligned at the tile level, but block computation is **reused over time** instead of being fully spatially unrolled. This preserves the algorithmic behavior while fitting CIM resource constraints. In other words, **serial block execution is the chosen realization, not the paper’s top-level idea**. The top-level idea is still temporal significance scheduling through useful execution windows.

At the hardware level, the cleanest implementation remains a **two-plane micro-tile**:

- a **control plane** that computes useful windows one block ahead from budgets, scale metadata, and leading-digit offsets
- a **data plane** that executes only those windows with block-local serial engines, then exposes a real scatter boundary between stage-1 production and local `down_proj` consumption

Under this organization, `gate_proj` and `up_proj` act as stage-1 producer paths, the nonlinear and gating operations are explicit in-pipeline stages, and `down_proj` is a stage-2 consumer that receives both the intermediate value and its timing/control context. The nonlinear part is represented by an **8-segment PWL SiLU block** that also supports MSD-first online arithmetic. The projection weights remain locally stored. Stage-1 dot products use serial-parallel online arithmetic, while gate fusion uses serial-serial online multiplication.

This framing also clarifies the streaming story. The FFN should be presented as a **producer-consumer dataflow**, not as three disconnected kernels. Useful-window scheduling reduces the number of digits that are actually generated, while the intermediate representation stays in a **BSD mantissa stream plus a narrow exponent/control stream**. This can help the transport story in two ways: it can reduce the amount of useful intermediate traffic, and it can turn the FFN middle into a **multi-cycle scatter workload** rather than a dense vector burst. That said, the transport claim should be stated carefully: the paper should claim **net** communication benefit only after control, FIFO, metadata, and setup overhead are counted explicitly.

The simulator already supports this interpretation on top of a modified **Qwen3-0.6B** stack. It implements **MXFP block quantization** (block size 32), uniform and calibrated cycle budgeting derived from combined activation and weight scales, inter-block and intra-block delay modeling, stream/scatter tracing, and full-model perplexity evaluation. This is important because the paper’s empirical story is not only about numerical quality; it is also about whether the temporal schedule maps to a believable hardware trace.

Algorithmically, the budget system remains hierarchical. A **uniform budget** is the simplest control baseline. **SNR-calibrated budgets** solve for the minimum per-channel allocation that satisfies a target distortion bound. **Fixed-sum calibration** starts from the SNR-min solution and redistributes cycles across channels to reduce total layer error while preserving exactly the same total hardware budget.

The comparison group should now be defined directly from the motivation. The key baselines are:

- **dense MX baseline**
- **post-MX structured sparsity baseline** (the concrete paper baseline can still be Wanda 2:4)
- **ordering control** when possible: sparsify-then-quantize versus quantize-then-sparsify
- **uniform useful-window budget**
- **SNR-calibrated useful-window budget**
- **fixed-sum calibrated useful-window budget**

The claim is that, for CIM-friendly FFN execution, **significance-guided useful-window scheduling is a more accurate and more regular alternative to sequential quantization+sparsification at similar work budgets**.

The paper should read as **one line with two layers**:

- the **algorithmic layer** resolves temporal significance into useful execution windows
- the **architectural layer** realizes those windows through a channel-parallel, block-serial FFN pipeline

The algorithm decides **how much work is worth doing** and **when**; the architecture decides **how that work is executed and transported** without breaking regular CIM structure.

Taken together, the work should be understood as a **CIM FFN accelerator study** in which **MX quantization, dynamic compute reduction, and intermediate transport are fused through temporal significance scheduling**. The paper’s novelty is not merely “online leading-digit significance,” because leading-digit information is only one timing input. The more accurate summary is: **MX metadata plus leading-digit offsets become useful execution windows, and those windows drive both compute suppression and stream formation in a regular CIM FFN pipeline**.

## Contribution framing

Our contributions are threefold.

1. We formulate **Significance-Guided Useful-Window scheduling**, which replaces loose sequential quantization+sparsification with a unified temporal-significance formulation that converts MX scales, leading-digit offsets, and offline-calibrated budgets into executable useful windows for MSD-first digit-pipelined online arithmetic with BSD inner representation.

2. We propose a **CIM-macro-based weight-stationary two-plane micro-tile architecture** that realizes those windows through a one-block-ahead control plane and a channel-parallel, block-serial data plane, enabling whole-block kill, element skip, and partial-digit truncation while streaming `gate_proj`, `up_proj`, `SiLU`, gating, and local `down_proj` as a continuous producer-consumer FFN pipeline.

3. We build an **end-to-end simulator** on a modified Qwen3-0.6B stack that jointly evaluates perplexity, effective utilization, stream/scatter behavior, and net hardware overhead against dense MX, sequential quantization+sparsification baselines, and calibrated useful-window baselines.

## Writing guidance for the main paper

The motivation and validation figures should reflect this framing directly.

- **Figure 1** should contrast sequential quantization+sparsification with temporal significance scheduling via useful windows.
- **Figure 4** should explicitly validate the original premise through an ordering ablation and a full-budget sanity check.
- The hardware overview should describe the design as **channel-parallel, block-serial**, while using **useful-window scheduling** as the algorithmic headline.

If this framing is kept consistent, the paper reads as one coherent story rather than a truncation paper plus a separate pipeline paper.
