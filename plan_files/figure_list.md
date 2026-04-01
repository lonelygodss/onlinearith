# Figure list

Below is the main-paper figure order I would use. Each figure has one job. If a figure cannot clearly prove that job, it should be merged, moved to the appendix, or cut.

The vocabulary should stay consistent across the whole paper:

- **paper-level principle:** temporal significance scheduling
- **algorithmic object:** useful execution windows
- **hardware realization:** channel-parallel, block-serial execution

That stack keeps the motivation, method, and microarchitecture aligned.

## Main-paper figure plan

**Figure 1 — From sequential quantize+sparsify to temporal significance scheduling**  
**Where it goes:** end of the Introduction or opening of the Motivation section.  
**What it shows:** a side-by-side comparison of the conventional flow and your flow. Panel (a): sequential quantization and sparsification, showing why the resulting skip pattern is irregular and not cleanly compatible with regular CIM execution. Panel (b): your flow, where MX metadata, combined activation-weight significance, and stored budgets resolve into per-block/per-element **arrival times** and **useful execution windows**. Panel (c): one worked example for a single block showing `t_arr`, `L`, and the resulting three savings modes: whole-block skip, element skip, and partial-window execution.  
**What it must prove:** this paper is **not** “another sparsity method.” The central executable object is a **useful execution window** derived from temporal significance metadata, not an irregular sparsity mask. A simple time-domain shift can appear as intuition in the drawing, but the actual method is the bounded execution window.  
**Caption spine:** “Temporal significance scheduling converts MX metadata and stored budgets into useful execution windows, replacing post-quantization sparsity with a regular execution policy for CIM FFNs.”  
**Failure mode to avoid:** too many equations or too much notation before the reader knows why they care. This figure should feel like a concept illustration, not a derivation.

**Figure 2 — Channel-parallel, block-serial two-plane FFN micro-tile**  
**Where it goes:** Hardware overview section.  
**What it shows:** the top-level tile block diagram: a one-block-ahead control plane, `H` owner-lane pairs operating in parallel across channels, block-serial execution within each lane pair, block-local trees, channel accumulators, prefix OTFC, PWL SiLU, gate fusion, a control-first scatter boundary, and local `down_proj` consumers. I would make the owner-lane pair a zoom-in inset, and I would explicitly mark the scatter boundary as a real streaming interface rather than a vector barrier.  
**What it must prove:** the method maps to a **real microarchitecture**. Temporal significance scheduling becomes local useful-window enable control, the FFN is a real producer-consumer stream, and the transport claim comes from a concrete control/data interface.  
**Caption spine:** “A two-plane micro-tile realizes useful-window scheduling as local enable control in a channel-parallel, block-serial FFN pipeline with a control-first scatter boundary.”  
**Failure mode to avoid:** a giant dense datapath drawing with no visual hierarchy. The reader should immediately see the four stages: useful-window generation, stage-1 execution, scatter, and local `down_proj` consumption.

**Figure 3 — Core quality/work Pareto**  
**Where it goes:** first Results subsection.  
**What it shows:** panel (a) perplexity versus average cycle budget; panel (b) perplexity versus effective utilization. Keep the baseline set tight: dense MX, Wanda 2:4, uniform-budget MSD, SNR-calibrated MSD, fixed-sum MSD. Add a zoomed inset for the near-lossless region.  
**What it must prove:** this is the paper’s main empirical claim. At matched work or utilization, **Significance-Guided Useful-Window execution** gives a better quality/work tradeoff than post-MX structured sparsity and clearly beats uniform budgeting.  
**Caption spine:** “Significance-Guided Useful-Window execution preserves model quality better than structured sparsity and naive uniform budgeting at matched work budgets.”  
**Failure mode to avoid:** mixing too many metrics in one panel. This figure should answer only one question: does the method win the main tradeoff?  

**Figure 4 — Sequential-flow validation and anchor points**  
**Where it goes:** immediately after the core Pareto.  
**What it shows:** panel (a) the ordering ablation: quantize→sparsify, sparsify→quantize, and Significance-Guided Useful-Window execution, all matched by compute or utilization. Panel (b) full-budget sanity check versus dense MX.  
**What it must prove:** the gain is not a presentation artifact. The paper’s premise that quantization and sparsification are not cleanly orthogonal in LLM FFNs is empirically real; full-budget useful-window execution collapses back to dense MX.  
**Caption spine:** “Treating quantization and sparsification as sequential steps is weaker at matched work, while full-budget useful-window execution recovers dense MX.”  
**Failure mode to avoid:** making this look like a side experiment. This figure directly validates the Introduction claim, so it should be framed as core evidence, not appendix cleanup.  

**Figure 5 — Why the offline/online split is credible**  
**Where it goes:** validation subsection after Figure 4.  
**What it shows:** panel (a) average budget versus target SNR, showing a smooth monotonic knob. Panel (b) robustness to calibration-set size and held-out evaluation split. Panel (c) budget-calibration ablation at fixed total budget: uniform, activation-only, weight-only, combined activation+weight, fixed-sum redistribution.  
**What it must prove:** the SRAM-loaded static budgets are not brittle. They are tunable, they generalize across held-out data, and the combined activation/weight signal matters more than any simple one-sided heuristic.  
**Caption spine:** “Stored budgets plus online temporal significance metadata form a robust offline/online split: the knob is smooth, generalizes across held-out data, and is strongest when derived from combined activation and weight information.”  
**Failure mode to avoid:** turning this into a calibration deep dive. The point is not “look at our solver”; the point is “the offline/online split is credible.”  

**Figure 6 — Aligned-first-valid stream and scatter behavior**  
**Where it goes:** hardware-facing trace subsection.  
**What it shows:** panel (a) a timing schematic with common first-valid cycle, early control arrival, and variable stream tails. Panel (b) active owner-stream count versus cycle. Panel (c) live mantissa flits per cycle at the scatter boundary. Panel (d) stream-end histogram or scatter completion distribution.  
**What it must prove:** the producer-consumer FFN pipeline is doing real work for the system story. Intermediate communication is spread over a multi-cycle window rather than arriving as a barriered burst, and the control-first/data-later split gives the consumer side time to prepare.  
**Caption spine:** “Aligned first-valid emission, variable tails, and a multi-cycle workload scatter boundary turn the FFN middle into a manageable multi-cycle workload rather than a full-vector burst.”  
**Failure mode to avoid:** showing only one pretty trace. At least one panel needs to summarize behavior statistically across many layers/tokens so it reads as system evidence, not a hand-picked example.  

**Figure 7 — End-to-end latency and net payoff**  
**Where it goes:** last Results subsection.  
**What it shows:** panel (a) stacked latency breakdown: stage-1 projection work, nonlinear/gating, intremediate scatter, local `down_proj`. Panel (b) net gain waterfall or savings-vs-cost plot: skipped blocks/elements/active cycles on one side, metadata storage/FIFO/control/setup overhead on the other. Panel (c) one small microarchitectural sweep such as `Wm` or ingress FIFO depth.  
**What it must prove:** the method wins **after** overheads are counted. This is the “ICCAD reviewer check”: not only are you suppressing useful work, but the control/transport machinery does not give the benefit back. The small sensitivity panel makes the hardware look implementable rather than tuned to a single sweet spot.  
**Caption spine:** “Useful-window execution remains beneficial after accounting for scatter, storage, and control overhead, and the benefit is stable across reasonable transport parameters.”  
**Failure mode to avoid:** reporting only active-cycle reduction. The figure has to be explicitly net.  

## Tables that should sit next to the figures

**Table 1 — Experimental setup and fairness controls**  
Put this near Figure 3. It should list the model, MX setup, calibration split, evaluation split, matched-compute rule, and the exact baseline set. Its job is to prevent readers from questioning whether the Pareto is fair.  

**Table 2 — Net overhead accounting**  
Put this near Figure 7. It should separate storage overhead, control overhead, scatter traffic, setup logic, and achieved savings. Its job is to make the hardware claim feel honest and complete.  

## Appendix figure plan

**Appendix Figure A1 — Useful-window scheduler details.**  
Show `cfg_active/cfg_shadow`, `block_skip`, `start_ctr`, `rem_ctr`, and the `t_arr / L / W` mapping into the three savings levels. This proves the control plane is implementable, but it is too detailed for the main text.

**Appendix Figure A2 — Variable-tail stream-length detail.**  
Use this to support Figure 6 with more granular per-layer or per-block distributions.

**Appendix Figure A3 — Secondary sensitivities.**  
Block size, separate gate/up budgets, or additional `down_proj` variants belong here unless one becomes a central result.  

## The writing order I would actually use

Do not draft the figures in paper order. Build them in this order:

1. **Figure 3 first** — it tells you whether the paper’s main claim is strong enough.  
2. **Figures 4 and 5 next** — they determine how aggressive your claim can be.  
3. **Figures 6 and 7 next** — they determine how much hardware emphasis the paper can carry.  
4. **Figures 1 and 2 last** — once the final claim is settled, redraw the concept and architecture figures to match what the data really supports.

If page budget gets tight, the cleanest merge is to combine **Figures 4 and 5** into one multi-panel “method validation” figure. The others should stay separate.
