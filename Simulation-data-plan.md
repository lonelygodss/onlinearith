# Simulation and Data Plan

## 1. Simulator model

The simulator should reflect the current execution model directly.

- Track the **exact BSD MSD-first stream** through the FFN path end to end.
- Model fixed pipeline latency for `gate_proj`, `up_proj`, `SiLU`, local alignment, gating, scatter, and `down_proj`.
- Use calibrated static budgets to determine how long each owner lane remains active.
- Enforce the current timing rule: after within-lane alignment, all owner lanes produce their **first valid mantissa MSD on the same cycle**.
- Let calibrated budgets determine the **stream tail length** of each channel.
- Carry a separate **exponent/control stream** that can be issued no later than the mantissa stream and may arrive earlier.
- Include scatter arbitration, FIFO occupancy, and backpressure so that source timing remains deterministic while destination arrival still reflects network contention.
- Model `down_proj` as a local consumer stage that can use early exponent/control arrival for delay resolution and setup, while keeping mantissa-side reduction on the simpler local policy.

## 2. Calibration and robustness results

### 2.1 Budget calibration curves

Required plots:
- average budget versus target SNR
- layerwise budget distribution versus target SNR
- fixed-sum redistribution result starting from the SNR-min solution

The purpose is to show that the calibration knob is smooth, monotonic, and easy to tune.

### 2.2 Signal-side predictiveness

Check whether channels with stronger effective signal statistics still need fewer cycles to satisfy the same distortion target. This supports why static calibration works at all.

### 2.3 Stream-length predictiveness

Use **stream-length analysis** as the main timing-side view.

Measure:
- per-channel calibrated stream length
- distribution of stream end cycles relative to the common first-valid cycle
- relationship between channel statistics and scatter-tail length

This is the timing-side quantity that now matters for scatter.

### 2.4 Static-budget robustness

Make this a headline experiment.

Evaluate:
- calibration split versus disjoint evaluation split
- calibration-set size sweep
- perplexity and stream-trace stability under the same stored budgets

The goal is to show that the SRAM-loaded static budgets generalize and do not need runtime adaptation.

### 2.5 Budget-assignment ablation

Include at least:
- uniform budget
- activation-only heuristic
- weight-only heuristic
- combined activation + weight calibration
- fixed-sum redistribution

This isolates whether the benefit really comes from the combined calibration signal rather than from any nonuniform assignment.

## 3. Model quality results

Main comparison set:
- dense MX baseline
- Wanda 2:4 structured sparsity after MXFP8
- MSD with layer-uniform budgets
- SNR-calibrated budgets
- fixed-sum calibrated budgets

Report:
- perplexity
- quality versus average cycle budget
- quality versus effective utilization

The main paper plots should emphasize the near-lossless region and the gap between checkpoint truncation and structured sparsity.

## 4. Hardware-facing trace data

### 4.1 End-to-end latency

Collect the cycle delay of the complete FFN path, including:
- stage-1 projection work
- SiLU / gating latency
- exponent/control scatter
- mantissa scatter window
- local `down_proj` work

A latency-breakdown table should separate transport time, control-preparation time, and mantissa compute time.

### 4.2 Stream-shape traces

These traces are central to the updated story.

Required traces:
- active owner-stream count versus cycle after the common first-valid cycle
- per-channel stream-end histogram
- per-layer scatter-window length distribution
- live mantissa flits per cycle at the scatter boundary

These plots should make the same-start / decaying-tail behavior obvious.

### 4.3 Scatter behavior

Collect:
- issue-time histogram for mantissa flits
- issue-time histogram for exponent/control packets
- per-destination FIFO occupancy
- backpressure cycles
- scatter completion time for each layer
- effective bandwidth requirement under pipelined scatter

The main message is that pipelined scatter spreads communication over a multi-cycle window instead of creating a full-vector barriered burst.

### 4.4 Exponent/control lead-time metrics

Because exponent/control can arrive earlier, measure:
- lead time between exponent/control arrival and mantissa-window completion
- fraction of local delay-resolution or shift-selection work completed before mantissa accumulation begins
- consumer-side setup latency hidden by early exponent/control transport

These are the main consumer-overlap metrics for the current design.

### 4.5 `down_proj` local-consumer metrics

Report:
- local multiplier utilization
- local accumulator / addition-tree utilization
- mantissa accumulation start cycle and finish cycle under the chosen simple consumer policy
- optional comparison against a more aggressive speculative partial-start policy, only as a side experiment if desired

The current mainline result should not rely on a complex speculative `down_proj` schedule.

### 4.6 Compute reduction and net overhead

Report both savings and cost.

Savings:
- effective MAC utilization
- skipped element fraction
- skipped block fraction
- active cycle reduction

Cost:
- ET counter storage and toggle overhead
- mantissa scatter FIFO traffic
- exponent/control scatter traffic
- route control overhead
- local setup / delay-resolution overhead

The final accounting should be net, not just “skipped work.”

## 5. Figures and tables to prepare

### Figure 1 — Calibration curves
- average budget vs target SNR
- fixed-sum redistribution effect
- robustness to calibration-set size

### Figure 2 — Quality / compute Pareto
- dense MX
- Wanda 2:4
- uniform MSD budgets
- SNR-calibrated budgets
- fixed-sum calibrated budgets

### Figure 3 — Stream and scatter timeline
- common first-valid cycle
- variable stream lengths
- live mantissa flits per cycle
- exponent/control arrival lead time

### Figure 4 — Latency breakdown
- stage-1
- nonlinear / gating
- exponent/control transport
- mantissa transport
- local `down_proj`

### Table 1 — Net overhead accounting
- counters
- FIFOs
- route metadata
- exponent/control traffic
- local setup logic

### Table 2 — Macro-level utilization summary
- MAC utilization
- local consumer utilization
- scatter occupancy statistics
- total FFN latency

## 6. One-sentence summary for the evaluation section

> The evaluation should show that calibrated static budgets improve the quality/compute tradeoff over uniform budgets and structured sparsity, while exact BSD-stream simulation reveals a same-start / decaying-tail scatter pattern in which earlier exponent/control transport hides consumer-side setup and pipelined scatter reduces burst communication pressure.
