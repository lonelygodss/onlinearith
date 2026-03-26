# Paper Plan

## 1. Paper positioning

This is a **CIM paper about the FFN layer of LLMs**. Online arithmetic is the execution mechanism that makes the design practical, but it is not the paper’s standalone identity. The paper should read as a CIM-oriented accelerator study that uses MSD-first BSD execution to realize calibrated compute reduction with a hardware-friendly schedule.

The central advanteage we want to claim is:
- We turn dynamic fine-grained sparsity into a cheap scheduling problem, acheiving high accuracy with low computation
- We minimized the intermediate data flow to MSD stream and exponential stream instead of bf16 or fp32 in the whole FFN layer, grately reduced memory and data transfer cost.
- We introduce a loccally digit pipeline micro Tile, scale better to real size LLM compared to former hierarchical CIM design.

## 2. Core technical idea

The project has two parallel contribution lines.

### 2.1 Checkpoint-based truncation

Projection checkpoints use calibrated static budgets to decide how many MSD cycles are executed per channel or per output. The runtime hardware only needs short local counters and gating logic. This is the mechanism that decides **how much** compute is performed.

### 2.2 FFN-deep streaming pipeline

The FFN is executed as one continuous producer-consumer pipeline across:
- `gate_proj`
- `up_proj`
- `SiLU`
- gating
- scatter boundary
- `down_proj`

The current timing model is specific and should be stated clearly:
- after within-lane alignment, all owner lanes produce the first mantissa MSD on the **same cycle**
- channels differ mainly in **stream length**, because budgets terminate them at different times
- the scatter boundary therefore sees a **common start and a decaying tail**
- exponent/control metadata can be issued at least as early as the mantissa stream and often earlier

This is what determines **how** the remaining computation is transported, localized, and overlapped.

## 3. Simulator description in the paper

The simulator section should now describe the exact model plainly.

- exact BSD MSD-first stream tracking through the FFN path
- MXFP block quantization with block size 32
- fixed pipeline stages for `gate_proj`, `up_proj`, `SiLU`, alignment, gating, scatter, and `down_proj`
- calibrated static budgets reused consistently in calibration and inference
- explicit mantissa stream plus exponent/control stream
- scatter arbitration, FIFO occupancy, and backpressure
- local `down_proj` policy that uses early exponent/control arrival for setup while keeping mantissa reduction simple

The simulator section should present exact BSD stream tracking as the baseline model for both timing analysis and hardware-facing traces.

## 4. Hardware section plan

The hardware section should revolve around a representative **32-channel FFN macro** or `P x H` micro-tile.

Key blocks:
- FFN entry / MX decode / BSD recode
- stage-1 owner lanes for `gate_proj` and `up_proj`
- PWL SiLU and within-lane alignment
- explicit pipelined scatter boundary
- separate mantissa path and exponent/control path
- local `down_proj` shard with local reduction
- local ET SRAM and short counters

Important writing discipline for this section:
- present the scatter boundary as a streaming transport interface, not a barriered handoff
- present stage-1 timing as same-start / different-length, not first-arrival skew
- carry the SiLU exponent update on the exponent/control path
- keep stage-2 ET optional unless it materially improves results

## 5. Evaluation section plan

### 5.1 Quality and compute

Main plots:
- perplexity vs average cycle budget
- perplexity vs effective utilization
- near-lossless operating point comparison across all baselines

### 5.2 Static-budget robustness

Make this a primary result.

Show:
- calibration-set size sweep
- evaluation on disjoint prompts / samples
- stability of both quality and stream-shape traces under fixed stored budgets

### 5.3 Scatter and latency behavior

Show:
- latency breakdown across the FFN path
- compare with bf16 and fp32 case

### 5.4 Net hardware accounting

Report:
- effective MAC utilization
- skipped blocks / skipped elements
- FIFO and control overhead
- ET storage overhead
- local consumer utilization
- total FFN latency

The emphasis should be on **net savings after overhead**, not only on skipped digits.

## 6. Narrative to keep throughout the paper

The paper should repeatedly return to the same concise interpretation:

> This work presents a CIM-oriented FFN macro that combines calibrated checkpoint-based MSD truncation with an FFN-deep streaming pipeline and pipelined scatter. Static budgets bound the amount of digit work, while exact BSD mantissa streams and earlier exponent/control metadata move through a common-start, decaying-tail scatter window that reduces burst communication pressure and enables early consumer-side setup with low control cost.

## 7. Practical section order

A clean section order is:
1. Motivation and why structured sparsity is not the right fit
2. FFN execution model and checkpoint truncation
3. FFN-deep hardware pipeline and scatter boundary
4. Simulator and calibration method
5. Quality / robustness results
6. Hardware-facing traces and net overhead
7. Discussion of scale-up and limitations

This order keeps the paper centered on a CIM hardware contribution rather than letting it drift into a purely algorithmic or purely numerical story.
