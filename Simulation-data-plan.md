1. Calibration result graph.

- Signal-power-side predictiveness: 
channels with higher signal power need fewer cycles to satisfy the same SNR target.

- Timing-side predictiveness:
channels with larger combined activation–weight exponent tend to show larger intra-channel delay spread.

Above two need data on 3 representative layers to show generality.

- Average budget vs target SNR: proves that calibration knob is smooth, monotonic, and easy to tune.

2. Perplexity results

- Baseline for current CIM sparsity after quantiztion: implement Wanda 2:4 structured sparsity after MXfp8
- Baseline for MSD: Layer uniform budget (a span of uniform budget)
- Main design result: SNR calibrate and fixed sum improvement (a span of targeted SNR)

3. Trace data for energy and latency

- Cycle delay of complete path
- Utilization of all possible mac within the budget
- Skipped elements frac, skipped block frac
- replay/FIFO/control overhead estimates (this part need be refined with piplelined scatter)
