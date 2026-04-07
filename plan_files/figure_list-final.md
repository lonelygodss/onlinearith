# Figure list

Below is the main-paper figure order I would use. Each figure has one job. If a figure cannot clearly prove that job, it should be merged, moved to the appendix, or cut.

The vocabulary should stay consistent across the whole paper:

- **paper-level principle:** temporal significance scheduling
- **algorithmic object:** local execution windows
- **hardware realization:** channel-parallel, block-serial execution

That stack keeps the motivation, method, and microarchitecture aligned.

## Main-paper figure plan

**Figure 1 — From sequential quantize+sparsify to temporal significance scheduling**  

fixed local 2:4 mask → regular hardware, but coarse and accuracy-limited; flexible/global sparsity → better quality, but extra metadata/runtime/hardware burden; your temporal significance scheduling → better fidelity without leaving a regular CIM substrate.

**Figure 2 - Online arithmetic realizes useful windows on aligned MSD-first contribution streams**
**Panel (a): From MX metadata to arrival index**
- show `e_x[n,b] + e_w[c,b]` giving a coarse block shift,
- show `δ_x[n,b,k]` giving a fine per-element shift,
- show weight element exponent folded offline into stored recoded weights,
- output of the panel is `τ_{b,k}^{(n,c)}`.

This panel tells the reader that the scheduler consumes **metadata**, not floating-point products. (paper-plan.md)

**Panel (b): Useful window acts on the compute stream**
- top row: faint/raw input digits or operands,
- bottom row: the emitted **MSD-first contribution stream** after alignment,
- overlay the window `[τ, H_c)`,
- explicitly mark the wrong interpretation with a small crossed-out note like “not an input temporal mask”.

This is the most important panel. It should visually force the distinction:
**window on emitted contribution digits** vs **mask on input digits**. (figure_list.md)

**Panel (c): Consequences**
- row 1: whole-block skip,
- row 2: element skip,
- row 3: partial-window execution,
- then one small arrow to “reduced emitted payload to next stage”.

**Caption spine**
Exponent metadata determine the arrival index of each contribution stream, and the per-channel horizon selects its useful window on the emitted MSD-first digits rather than masking raw inputs. The same mechanism yields whole-block skip, element skip, partial-window execution, and reduced intermediate payload.


**Figure 3 — Channel-parallel, block-serial two-plane FFN micro-tile**  

**Figure 4 — Core quality/work Pareto**  
Two specific conparison group:

- Hardware-fair baselines: dense MX, activation-gated CIM baseline and your own ablations.
- Algorithmic reference: strong offline structured 2:4.

Two specific comparison groups:

- **hardware-fair baselines**: dense MX, activation-gated CIM baseline (2:4 (local) and 8:16 (flexible)) and the paper's own ablations
- **algorithmic reference**: strong offline structured `2:4` Wanda

We use two panals, one x axis is normalized digit reading, one x axis is end to end energy cost (CIM baseline will be adopted from papers and algorithmic reference only in the first)

**Figure 5 — Payload reduction**  

**Figure 6 — Why the offline/online split is credible**  
**Where it goes:** validation subsection after Figure 4.  
**What it shows:** panel (a) average horizon versus target SNR, showing a smooth monotonic knob. Panel (b) robustness to calibration-set size and held-out evaluation split. Panel (c) horizon-calibration ablation at fixed total horizon: uniform, activation-only, weight-only, combined activation+weight, fixed-sum redistribution.  
**What it must prove:** the SRAM-loaded static horizons are not brittle. They are tunable, they generalize across held-out data, and the combined activation/weight signal matters more than any simple one-sided heuristic.  
**Caption spine:** “Stored horizons plus online temporal significance metadata form a robust offline/online split: the knob is smooth, generalizes across held-out data, and is strongest when derived from combined activation and weight information.”  
**Failure mode to avoid:** turning this into a calibration deep dive. The point is not “look at our solver”; the point is “the offline/online split is credible.”  

**Figure 7 — End-to-end latency and net payoff**  
We will conbine software simulation traces with synthesize anchor point to give latency, area and energy estimation.

## Tables that should sit next to the figures

**Table 1 — Experimental setup and fairness controls**  
Put this near Figure 3. It should list the model, MX setup, calibration split, evaluation split, matched-compute rule, and the exact baseline set. Its job is to prevent readers from questioning whether the Pareto is fair.  

**Table 2 — Net overhead accounting**  
Put this near Figure 7. It should separate storage overhead, control overhead, scatter traffic, setup logic, and achieved savings. Its job is to make the hardware claim feel honest and complete.  

## Code block

**Block1 - Block tree leaf schedule policy for three level savings**
Or any reasonable code.
