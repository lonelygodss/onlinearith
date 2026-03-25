# Revised simulation and data plan

## 0. Purpose of the simulation package

The simulation plan should now serve three reviewer-checkable claims:

1. **Calibrated checkpoint truncation gives a better quality / compute frontier** than uniform budgeting and post-quantization structured sparsity.
2. **Pipelined scatter converts staggered stage-1 completion into real overlap and latency reduction** at the stage-1 to stage-2 boundary.
3. **The runtime mechanism is cheap and regular** once net activity savings and control / buffering overhead are counted together.

A major update for this revision is that the simulator now tracks the **exact BSD stream** through the FFN. The plan should state this directly and treat the exact stream as the source of truth for both quality evaluation and hardware-facing traces.

## 1. Simulator statement to use in the paper

Use a short paragraph like this in the simulation section:

> The revised simulator tracks exact MSD-first BSD streams across `gate_proj`, `up_proj`, PWL SiLU, gating, the scatter boundary, and `down_proj`. Ready times, completion skew, scatter traffic, and local consumer activity are therefore measured from the exact digit stream rather than inferred from a float-valued carrier plus timing metadata.

This update materially strengthens the fidelity story and should be visible in all three plan documents.

## 2. Calibration experiments

### 2.1 Target-SNR smoothness

Required plot:

- average budget versus target SNR

Purpose:

- show that the calibration knob is smooth and monotonic
- show that target selection is practical rather than brittle

### 2.2 Signal-side predictiveness

Required plot or table:

- relationship between channel signal power and required cycle budget for a fixed SNR target

Purpose:

- explain why uniform budgets are suboptimal
- justify per-channel calibration as a structured, predictable policy rather than a black-box fit

### 2.3 Timing-side predictiveness

Required plot:

- relationship between combined activation-weight scale and intra-channel completion spread

Purpose:

- connect calibration features to staggered finish times
- show that the same signal used for budget assignment also predicts the width of the downstream scatter window

### 2.4 Static-budget robustness

This should now be a headline experiment, not a side note.

Required setup:

- calibration split and disjoint evaluation split
- sweep over calibration-set size
- at least one cross-prompt or cross-dataset check if practical

Required outputs:

- perplexity change on held-out data
- budget stability statistics
- change in ready-time / overlap traces when using budgets calibrated on a different split

Reviewer question answered:

> Are the stored static budgets robust enough to justify the SRAM-loaded hardware model?

### 2.5 Budget-assignment ablation

This should be mandatory if possible.

Compare:

- uniform budget
- activation-only heuristic
- weight-only heuristic
- combined activation+weight calibration
- fixed-sum calibrated redistribution

Purpose:

- isolate the value of the combined signal
- show that gains are not just from any nonuniform budget assignment

## 3. Quality comparison matrix

The main comparison group should remain tightly scoped:

- dense MX baseline
- Wanda 2:4 structured sparsity after MXFP8
- uniform-budget MSD baseline
- SNR-calibrated budget
- fixed-sum calibrated budget

Recommended presentation:

- perplexity versus normalized MAC activity
- perplexity versus average cycle budget
- one table with near-lossless operating points

Important note to preserve in the text:

- near-lossless operating points still show only about 1 percent zero blocks, so the main benefit is not coarse block shutdown
- emphasize reduced active digit work and active cycles instead

## 4. Scatter, overlap, and latency measurements

Because the simulator now tracks exact BSD streams, this section should become more quantitative and more central.

### 4.1 Primitive timing traces to collect

For every evaluated operating point, collect:

- per-lane stage-1 ready cycle
- per-lane stage-1 completion cycle
- scatter issue cycle
- per-destination FIFO occupancy over time
- stage-2 first-consume cycle
- stage-2 completion cycle

### 4.2 Derived metrics to report

Convert the raw traces into a small set of paper-friendly metrics:

- **ready-time skew** = `max_i T_ready[i] - min_i T_ready[i]`
- **consumer early-start fraction** = fraction of stage-2 work launched before `max_i T_stage1_done[i]`
- **overlap ratio** = fraction of stage-2 work issued while at least one stage-1 owner lane is still active
- **barrier-free latency reduction** = `(T_barrier_model - T_scatter_model) / T_barrier_model`
- **scatter window width** = last issue cycle minus first issue cycle for a tile or shard

These metrics are the shortest path from architectural detail to a latency claim.

### 4.3 Representative visualizations

Required plots for 3 representative layers:

- stage-1 completion histogram
- scatter issue histogram
- FIFO occupancy trace
- local `down_proj` bank utilization over time

The important narrative is:

> gated intermediate channels finish at staggered cycles, scatter starts immediately, and local `down_proj` reduction begins before the whole stage-1 slice finishes.

## 5. Activity, energy, and net-overhead accounting

This section should focus on **net** savings, not only skipped work.

### 5.1 Activity categories to count

Report activity for:

- digit MAC operations in stage-1
- digit MAC operations in stage-2
- weight SRAM reads
- PWL SiLU activity
- align-buffer and scatter FIFO traffic
- local accumulator toggles
- ET counter activity and small control logic

### 5.2 Required hardware-cost outputs

Report at least:

- total metadata storage for `B1`, `B2`, and small ready / route state
- FIFO depth requirement at the chosen macro point
- replay or backpressure events if they exist
- local addition-tree utilization

### 5.3 Main summary table

Create one net-cost table that answers:

- gross activity reduction
- buffering and routing overhead
- ET-control overhead
- resulting net latency / energy benefit

This table should be one of the paper's most rebuttal-resistant pieces of evidence.

## 6. Fidelity and implementation validation

The old concern about float-carrier approximation is now largely removed by the exact BSD update, but there is still value in a short validation section.

Recommended checks:

- compare the revised FFN-wide simulator against a smaller operator-local or trace-replay reference for a few representative layers and tokens
- confirm identical or near-identical output stream reconstruction for the exact-BSD path
- confirm that scatter scheduling and FIFO timing are preserved in the reference traces

What this section should say:

> Arithmetic fidelity comes from exact BSD-stream tracking; the remaining validation target is the event-scheduling and buffering model.

Keep this validation small but explicit so reviewers see that fidelity was checked, not assumed.

## 7. Recommended figure and table package

Minimum set:

1. calibration smoothness and predictiveness plots
2. static-budget robustness plot
3. perplexity frontiers across the comparison matrix
4. scatter / overlap trace figures for representative layers
5. net activity / overhead table

Optional appendix items:

- activation-only vs weight-only ablation details
- per-layer robustness breakdown
- small tile-shape or FIFO-depth sensitivity sweep

## 8. Priority order for implementation

If time is limited, finish the experiments in this order:

1. static-budget robustness
2. scatter-derived latency metrics
3. net overhead / metadata table
4. signal-ablation comparison
5. small validation spot-checks

This order best supports the current paper claim structure.

## 9. One-sentence summary for reuse

> The revised simulator now uses exact BSD-stream execution to connect calibration, perplexity, staggered completion, pipelined scatter, and net hardware activity within one consistent FFN-wide trace framework.
