This outline turns the current notes into a reviewer-facing paper arc, not just an internal project plan. Keep the vocabulary fixed throughout: **temporal significance scheduling** at the paper level, **local execution windows** as the algorithmic object, and **channel-parallel, block-serial execution** as the hardware realization.   

## Paper promise

**Temporal significance scheduling uses existing MX metadata plus calibrated per-channel horizons to execute only useful prefixes of aligned contribution streams, giving a better quality-work tradeoff than CIM-friendly local sparsity baselines while preserving a regular CIM mapping.** 

## Submission-facing outline

### 1. Introduction — From fixed local sparsity to temporal significance scheduling
This section should set up a clean tension:

- **Why current CIM-friendly sparsity is unsatisfying:** fixed local masks are regular and executable, but they are coarse and accuracy-limited.
- **Why stronger sparsity baselines are not the right answer:** more flexible/global sparsity may recover quality, but at the cost of more metadata, runtime irregularity, or hardware burden.
- **Your thesis:** the missing ingredient is a better significance signal, not a more complicated sparse mask.
- **Your claim:** MX block metadata and exponent fields already contain enough structure to build a better schedule.

End the introduction with three crisp contributions:
1. a new abstraction: temporal significance scheduling;
2. a hardware realization: metadata-first control plus block-serial online-CIM execution;
3. a system consequence: less executed work and less intermediate payload.  

**Figure placement:** Figure 1 as the teaser/motivation figure.  
**Reviewer takeaway:** this is not “another masking paper”; it is a new execution abstraction that stays hardware-regular.  

### 2. Temporal Significance Scheduling — From MX metadata to useful execution windows
This is the core method section. It should be the conceptual center of the paper.

#### 2.1 Online arrival-time synthesis
Introduce the runtime-visible metadata only:
- `beta_x[n,b]`
- `beta_w[p,c,b]`
- `lambda_x[n,b,k]`
- `H[p,c]`

Then define only the equations you really need in the main paper:
- `E_raw`
- `E_max`
- `D`
- `tau`

The important message is that the scheduler operates on **metadata**, not floating-point products, and that the **weight element exponent is folded offline into stored recoded weights** rather than resolved online.  

#### 2.2 Useful windows on aligned contribution streams
This subsection should make the paper’s most important distinction explicit:

- the executed object is **not an input temporal mask**
- it is a **window on the emitted MSD-first contribution stream after alignment**

Define:
- `W[p,b,k,c]`
- `L[p,b,k,c]`
- `s_exec[p,b,k,c][t]`

This is where the paper becomes intellectually different from sparsity masking.  

#### 2.3 Unified work-suppression view
Show that the same formulation gives:
- whole-block skip
- element skip
- partial-window execution

One short pseudocode block can help here; it can be the “leaf schedule policy” block already noted in the figure plan. 

#### 2.4 Offline horizon calibration
Keep this subsection short and practical:
- uniform horizon
- SNR-calibrated horizon
- fixed-sum calibrated horizon

The goal is **not** to sell a sophisticated solver. The goal is to show that stored horizons are a credible offline knob that pairs naturally with online arrival-time synthesis.  

**Figure placement:** Figure 2 should live in this section and carry most of the conceptual burden.  
**Reviewer takeaway:** temporal significance scheduling means “execute useful windows on aligned contribution streams,” not “mask input digits.” 

### 3. Hardware Realization — A metadata-first two-plane CIM micro-tile
This section should answer “can the hardware really do this cheaply?”

#### 3.1 Macro organization
Present the tile at a high level:
- stage-1 producer paths: `gate_proj` and `up_proj`
- stage-2 consumer: `down_proj`
- channels run in parallel
- blocks execute serially on reused engines

Do not drown the reader in every buffer and latch yet. Lead with the architectural split: **control plane + data plane**.  

#### 3.2 Metadata-only control plane
This subsection should make the hardware claim believable:
- double-buffered scale prepass computes coarse delays
- one-block-ahead local scheduler merges those delays with activation fine-delay codes
- runtime control is only field readout, narrow integer add/subtract, max-reduction, and compare

That last point is the main hardware credibility hook. 

#### 3.3 Data plane and execution semantics
Describe only what the data plane needs to do:
- execute local windows on reused serial-parallel block engines
- accumulate within a block and across blocks
- locally finalize the gate path, apply SiLU, fuse with the up path
- emit a reduced BSD payload toward `down_proj`

Be explicit that the interconnect itself is not the contribution; **payload reduction** is the architectural claim.  

#### 3.4 What is actually turned off
This should be a short, concrete subsection:
- whole-block skip shortens service time and suppresses the entire block
- element skip suppresses individual leaves and dead subtrees
- partial-window execution mainly saves switching, SRAM reads, and emitted payload in the baseline block-serial schedule

That makes the claims honest and preempts reviewer pushback. 

**Figure placement:** Figure 3 here.  
**Table placement:** Table 1 can close this section or open the next one.  
**Reviewer takeaway:** the method maps to a physically believable microarchitecture because control stays metadata-only and the data plane remains regular. 

### 4. Experimental Methodology and Fairness Controls
This section should be concise but very explicit.

Include:
- model and stack
- MX setup and block size
- calibration split and held-out evaluation split
- matched-compute rule
- simulator features that matter for the claim
- baseline taxonomy

Separate the baselines into the two groups already present in the plan:
- **hardware-fair baselines:** dense MX, activation-gated CIM baselines, naive truncation / uniform-horizon style ablations, and your own ablations
- **algorithmic reference:** strong offline structured `2:4`

Also define the two x-axes cleanly:
- normalized digit reading
- end-to-end energy cost where supported

**Table placement:** Table 1 should sit here if not already used at the end of Section 3.  
**Reviewer takeaway:** the comparisons are fair both as hardware comparisons and as algorithmic references.  

### 5. Results — Quality/work first, then system payoff
This should be one results section with four tight subsections.

#### 5.1 Core quality/work Pareto
Lead with the main comparison figure:
- dense MX
- activation-gated CIM baselines
- your ablations
- strong offline structured `2:4` reference where applicable

This is the headline empirical result.  
**Figure placement:** Figure 4. 

#### 5.2 Payload reduction
Show that the schedule does not just cut executed work; it also reduces the amount of stage-1 payload handed to `down_proj`.  
**Figure placement:** Figure 5.  

#### 5.3 Why the offline/online split is credible
Use this subsection to prove robustness, not to dive into calibration mechanics:
- horizon vs target SNR
- calibration-set robustness
- held-out split robustness
- one-sided vs combined activation+weight signals

**Figure placement:** Figure 6.  
**Key message:** the static horizons are tunable and not brittle. 

#### 5.4 End-to-end latency and net payoff
Close with the system story:
- combine software traces with synthesized anchor points
- report latency, area, energy, and net overhead accounting
- separate storage/control/setup overhead from achieved savings

**Figure placement:** Figure 7.  
**Table placement:** Table 2 next to Figure 7.  
**Reviewer takeaway:** the method’s gains survive realistic accounting. 

### 6. Conclusion
Keep the conclusion narrow:
- temporal significance scheduling is a better significance signal than fixed local sparsity for CIM LLM FFNs
- it maps to a regular metadata-first, channel-parallel, block-serial microarchitecture
- it improves both quality/work tradeoff and intermediate payload

Do not add new claims here. 

## What I would push to the appendix

- full derivation details beyond `E_raw`, `D`, `tau`, `W`, and `L`
- detailed bank organization and shadow-RAM mechanics
- exact scheduler config fields and node-level clock-enable logic
- extra calibration details
- extended ablations and trace plots
- any deeper discussion of SiLU approximation or transport details

Those details are useful, but the main paper should stay centered on the abstraction, the hardware credibility argument, and the core Pareto/system results. That division matches your current notes well.  

## Two writing rules to enforce in every section

First, repeat the distinction between **window on emitted contribution digits** and **mask on raw inputs** in the introduction, method, and Figure 2 discussion. That is the conceptual novelty reviewers are most likely to miss. 

Second, keep the horizon-calibration story at the level of **credibility and robustness**, not optimization cleverness. Your own plan already points in that direction, and it is the right submission-facing choice. 

If it helps, I can next turn this into a section-by-section draft skeleton with suggested subsection titles and opening sentences.