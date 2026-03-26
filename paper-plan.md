# Paper Plan

## 1. Paper positioning

This is a **CIM paper about the FFN layer of LLMs**. Online arithmetic is the execution mechanism that makes the design practical, but it is not the paper’s standalone identity. The paper should read as a CIM-oriented accelerator study that uses MSD-first BSD execution to realize calibrated compute reduction with a hardware-friendly schedule.


## 2. Core technical idea

The project has two parallel contribution lines.

### 2.1 Checkpoint-based truncation

Projection checkpoints use calibrated static budgets to decide how many MSD cycles are executed per channel or per output. The runtime hardware only needs short local counters and gating logic. This is the mechanism that decides **how much** compute is performed.

### 2.2 FFN-deep streaming pipeline with pipelined scatter

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

## 3. Three reviewer-checkable claims

### Claim 1

**Calibrated static checkpoint truncation gives a better quality / compute tradeoff than uniform budgets and post-quantization structured sparsity.**

Evidence:
- dense MX baseline
- Wanda 2:4 after MXFP8
- uniform MSD budgets
- SNR-calibrated budgets
- fixed-sum calibrated budgets
- activation-only / weight-only / combined calibration ablation

### Claim 2

**Pipelined scatter is useful even when first-valid timing is aligned across channels, because stream lengths differ and exponent/control metadata can move earlier than mantissa traffic.**

Evidence:
- active-stream-count traces showing the same-start / decaying-tail pattern
- scatter bandwidth and occupancy traces
- exponent/control lead-time measurements
- latency breakdown showing reduced stage-boundary burstiness and hidden consumer-side setup

### Claim 3

**The runtime mechanism is cheap and deterministic.**

Evidence:
- local static counters and SRAM-loaded budgets
- no runtime dynamic-budget predictor
- no combined-scale LUT on the critical path
- modest FIFO, metadata, and control storage
- local `down_proj` ownership that keeps reduction inside the shard

## 4. Simulator description in the paper

The simulator section should now describe the exact model plainly.

- exact BSD MSD-first stream tracking through the FFN path
- MXFP block quantization with block size 32
- fixed pipeline stages for `gate_proj`, `up_proj`, `SiLU`, alignment, gating, scatter, and `down_proj`
- calibrated static budgets reused consistently in calibration and inference
- explicit mantissa stream plus exponent/control stream
- scatter arbitration, FIFO occupancy, and backpressure
- local `down_proj` policy that uses early exponent/control arrival for setup while keeping mantissa reduction simple

The simulator section should present exact BSD stream tracking as the baseline model for both timing analysis and hardware-facing traces.

## 5. Hardware section plan

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

## 6. Evaluation section plan

### 6.1 Quality and compute

Main plots:
- perplexity vs average cycle budget
- perplexity vs effective utilization
- near-lossless operating point comparison across all baselines

### 6.2 Static-budget robustness

Make this a primary result.

Show:
- calibration-set size sweep
- evaluation on disjoint prompts / samples
- stability of both quality and stream-shape traces under fixed stored budgets

### 6.3 Scatter and latency behavior

Show:
- active owner-stream count vs cycle
- per-layer scatter-window length
- live mantissa flits per cycle
- exponent/control lead time
- latency breakdown across the FFN path

### 6.4 Net hardware accounting

Report:
- effective MAC utilization
- skipped blocks / skipped elements
- FIFO and control overhead
- ET storage overhead
- local consumer utilization
- total FFN latency

The emphasis should be on **net savings after overhead**, not only on skipped digits.

## 7. Narrative to keep throughout the paper

The paper should repeatedly return to the same concise interpretation:

> This work presents a CIM-oriented FFN macro that combines calibrated checkpoint-based MSD truncation with an FFN-deep streaming pipeline and pipelined scatter. Static budgets bound the amount of digit work, while exact BSD mantissa streams and earlier exponent/control metadata move through a common-start, decaying-tail scatter window that reduces burst communication pressure and enables early consumer-side setup with low control cost.

## 8. Practical section order

A clean section order is:
1. Motivation and why structured sparsity is not the right fit
2. FFN execution model and checkpoint truncation
3. FFN-deep hardware pipeline and scatter boundary
4. Simulator and calibration method
5. Quality / robustness results
6. Hardware-facing traces and net overhead
7. Discussion of scale-up and limitations

This order keeps the paper centered on a CIM hardware contribution rather than letting it drift into a purely algorithmic or purely numerical story.
