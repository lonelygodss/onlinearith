1. Calibration result graph.

- Signal-power-side predictiveness:
channels with higher signal power still need fewer cycles to satisfy the same SNR target.

- Timing-side predictiveness:
channels with larger combined activation-weight exponent tend to show larger intra-channel delay spread, and this spread now directly predicts the scatter window after gating.

- Scatter-overlap evidence:
for 3 representative layers, show that gated intermediate channels finish at staggered cycles, so stage-1 completion and inter-slice scatter can overlap instead of waiting for the whole intermediate slice to finish.

- Average budget vs target SNR:
proves that the calibration knob is smooth, monotonic, and easy to tune.

2. Perplexity results

- Dense MX baseline.
- Baseline for current CIM sparsity after quantization: implement Wanda 2:4 structured sparsity after MXFP8.
- Baseline for MSD: layer-uniform budget (a span of uniform budgets).
- Main design result: SNR-calibrated budget and fixed-sum calibrated budget (a span of target SNRs / fixed total budgets).

3. Trace data for energy and latency

- End-to-end cycle delay of the complete FFN path, including the scatter boundary between gating and `down_proj`.
- Utilization of all possible MACs within the calibrated static budget.
- Skipped element fraction, skipped block fraction.
- Stage-1 completion histogram and per-lane ready-time skew.
- Scatter/reduce-scatter overlap statistics: issue-time histogram, ingress FIFO occupancy, and fraction of stage-2 work that starts before all stage-1 lanes finish.
- Local `down_proj` addition-tree utilization after scatter, plus replay/FIFO/control overhead estimates.
- ET overhead estimate with static counters only; remove dynamic-budget LUT/add-path cost from this iteration.
