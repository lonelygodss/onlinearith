# Revised hardware design plan

## 0. Hardware objective

The hardware writeup should now be organized around one CIM-oriented FFN macro, but with two clearly separated contribution lines:

1. **FFN-deep producer-consumer dataflow with pipelined scatter / reduce-scatter.**
   This line explains transport, overlap, locality, scale-up, and why the FFN should be treated as one continuous pipeline instead of isolated operators.
2. **Checkpoint-based early termination (ET) with calibrated static budgets.**
   This line explains how much digit work is executed and why the runtime control path is cheap.

These two lines meet at the lane interfaces, but neither should be described as a consequence of the other. Scatter is not caused by ET. ET is not justified by scatter. Scatter exploits staggered completion. ET determines where stagger comes from and how much work remains.

A second important update is that the simulator now tracks the **exact BSD stream** across the FFN. The hardware document should therefore describe internal interfaces explicitly as BSD-stream interfaces, not as a float carrier plus timing metadata. Timing summaries such as `t_first` or ready cycle are still useful control and trace signals, but the primary execution object is the exact digit stream.

## 1. Detailed design unit and tiling

The detailed design unit remains a **32-channel FFN streamline macro**. Inside that macro, use a representative micro-tile of:

- `P` owner lanes for intermediate channels `i`
- `H` owned consumer outputs for local `down_proj` channels `h`
- MX block size `B = 32`

Recommended illustrative points:

- `P = 8` or `16`
- `H = 16` or `32`

These values should be presented as representative tiling parameters, not final commitments.

The key ownership statement is now:

> A micro-tile owns both a stage-1 owner domain over intermediate channels and a local stage-2 output shard over `down_proj` outputs.

This ownership model is what makes the scatter boundary meaningful: stage-1 owner lanes do not drain into a monolithic global consumer. They forward ready BSD streams into the destination `down_proj` shard that owns those outputs, so the reduction tree stays local after the boundary.

## 2. Data representation and stream interfaces

Because the simulator now tracks exact BSD behavior, the hardware plan should use stream language consistently.

### 2.1 Internal stream convention

Each in-flight FFN stream should be described as an exact BSD digit stream with local control tags, for example:

`stream = {bsd_digit[t], valid[t], src_i, dst_shard, eos}`

You do not need to hard-commit the packet format in the paper, but figures and text should make the following clear:

- `gate_proj` produces a BSD stream `g_i[t]`
- `up_proj` produces a BSD stream `u_i[t]`
- `SiLU(g_i)` remains in the BSD-domain pipeline after the PWL block
- the gating multiply produces an exact BSD intermediate stream `m_i[t]`
- the scatter boundary forwards that exact stream to the destination `down_proj` shard
- stage-2 consumes the actual arriving BSD stream rather than a reconstructed dense value

### 2.2 Timing metadata is still useful, but secondary

The writeup can still use derived control or trace summaries such as:

- `t_first`
- per-lane ready cycle
- end-of-stream cycle
- route or shard ID

But these should be described as **control and measurement fields derived from the exact stream**, not as a replacement for the stream itself. This is the main wording change relative to the previous simulator description.

## 3. Concrete micro-tile architecture

The micro-tile should be drawn and described as six blocks.

### Block 1: FFN entry / input preparation

This is where the format-conversion cost is paid once.

Contains:

- MX block exponent extraction
- mantissa recoder / BSD recoder
- digit broadcaster
- activation block metadata register

Message to put directly on the figure:

> Conversion happens only at FFN entry and FFN exit, not between internal FFN stages.

This remains one of the strongest hardware motivations for the FFN-spanning pipeline.

### Block 2: Stage-1 producer lanes

For each owner lane `i in [0, P-1]`, include:

- local `gate_proj` weight bank
- local `up_proj` weight bank
- `Gate MAC` producer path
- `Up MAC` producer path
- local stage-1 ET counter loaded from calibrated budget `B1[i]`
- ready / first-valid generation for local scheduling and trace extraction

The shared owner-domain indexing by intermediate channel should be visually explicit.

Most important discipline point for this version:

- stage-1 ET is **static and calibrated**
- no runtime budget delta
- no combined-scale LUT in the runtime datapath
- no saturating add on the control critical path

### Block 3: Nonlinearity, time alignment, and gating multiply

Per owner lane, include:

- PWL SiLU block on the gate path
- align buffer between `SiLU(g_i)` and `u_i`
- gating multiply `SiLU(g_i) * u_i`
- output stream register for the gated BSD stream `m_i[t]`

This block is where the deep pipeline becomes tangible. It should be described as a streaming FFN stage, not as a post-processing box after two independent dot products.

Important sentence to keep in the text:

> Calibrated truncation causes different owner lanes to finish at different cycles, and the scatter boundary exploits that stagger instead of restoring a barrier.

### Block 4: Pipelined scatter / reduce-scatter boundary

This block must sit explicitly between the gated intermediate stream and the local `down_proj` consumer bank.

Contains:

- stream packetization or tagging (`src_i`, `dst_shard`, valid, eos)
- route selection for the destination `down_proj` shard
- per-destination elastic FIFOs or credits
- issue control that launches a stream as soon as its owner lane is ready

Key text for the body:

> The gated intermediate stream is not held until the whole stage-1 slice completes. Each owner lane scatter-issues its ready BSD stream immediately, so communication overlaps with late producer lanes and the destination `down_proj` shard can begin local reduction early.

This boundary is where the latency story and the scale-up story connect.

### Block 5: Stage-2 local consumer bank

This block is a local `down_proj` consumer array for `H` owned output channels.

Contains:

- local `down_proj` weight bank
- ingress FIFOs from the scatter boundary
- digit-serial consumer multipliers
- local addition tree / output accumulators for `y_h`
- optional stage-2 ET counter bank loaded from `B2[h]`
- local output register and FFN exit encoder

The visual distinction should be explicit:

- stage-1 **produces** exact timed BSD streams
- the scatter boundary **transports and overlaps** them
- stage-2 **consumes and reduces locally** inside the owned output shard

### Block 6: Local control and metadata SRAM

Add a small side band that makes the hardware feel concrete:

- `B1[i]` SRAM for stage-1 calibrated budgets
- `B2[h]` SRAM for stage-2 calibrated budgets
- weight exponent SRAM
- route / shard-ID metadata
- small ready-time or trace registers
- SiLU coefficient ROM

Do **not** show the old dynamic-budget LUT in this version.

## 4. Early termination controller plan

The ET figure should now communicate four things very clearly:

1. The decision is local.
2. The logic is cheap.
3. Real activity is gated, not only final writeback.
4. Runtime ET is static, with budgets preloaded from SRAM.

### 4.1 Stage-1 ET datapath

Use a short per-owner controller:

- `start_of_stream` loads `B1[i]` from SRAM into a down-counter
- each issued BSD digit decrements the counter
- when the counter expires, the lane locally gates:
  - weight SRAM reads
  - `gate_proj` and `up_proj` MAC toggles
  - SiLU / align / multiply toggles
  - scatter FIFO writes
  - lane-active state

One sentence to reuse:

> The ET controller is a calibrated local counter loaded from SRAM, not a runtime significance predictor.

### 4.2 Stage-2 ET datapath

The same primitive can be reused at stage-2 if the gains justify keeping it in the mainline design:

- `B2[h]` loaded from local SRAM
- short local counter
- gates consumer MAC activity and accumulator toggles

However, stage-2 ET should remain optional in the writeup structure. If its marginal gain is small, demote it to an ablation or appendix so the mainline hardware claim stays simple.

### 4.3 Optional exact-zero side path

If you include a zero shortcut, keep it narrow:

- use exact zero information already available from encoding or quantization
- do not introduce a new runtime significance heuristic

This must remain a side path. The main controller is still the calibrated static counter.

## 5. Required figures

### Figure A: FFN micro-tile overview

The figure should show:

- FFN entry recode
- stage-1 owner lanes with `gate_proj` and `up_proj`
- PWL SiLU and align buffer
- gating multiply producing `m_i[t]`
- explicit scatter / reduce-scatter boundary
- local stage-2 `down_proj` consumer bank
- FFN exit encoder

The most important visual correction relative to older drafts is that the **scatter boundary is explicit** and that the streams crossing it are labeled as **BSD streams**, not abstract numeric values.

### Figure B: ET controller and timing example

Make this a two-part figure.

Part (a): per-owner ET controller

- `B1[i]` SRAM
- down-counter
- local gating fanout

Part (b): cycle timeline with exact BSD tokens

The timeline should show, for one owner lane and one destination shard:

- issued BSD digits on `gate_proj` and `up_proj`
- finite budget and ET cutoff
- SiLU pipeline latency
- align and multiply timing
- scatter issue events
- FIFO arrival events
- stage-2 consumer start before global stage-1 completion

Now that the simulator tracks exact BSD streams, the timeline should use stream notation directly, for example `g0`, `g1`, `u0`, `u1`, `m0`, `m1`, rather than a float-valued carrier.

### Figure C: Macro scale-up and local ownership

Add one macro-level figure or subfigure that explains how the 32-channel macro scales:

- replication across intermediate-channel ownership groups
- replication across `down_proj` output shards
- local addition tree kept inside each owned shard
- scatter routes crossing only the shard boundary, not collapsing into a global reducer

This figure will help reviewers understand that the scatter boundary is not just a pretty diagram. It is the mechanism that keeps reduction local in the scaled design.

## 6. Concrete labels and widths

Use small example widths so the hardware feels real:

- `B1[i]`, `B2[h]`: 6 bits
- ET counter: 6 bits
- ready / first-valid register: 6 bits
- scatter route / shard ID: 3 to 5 bits
- source-owner ID: `log2(P_total)` bits
- SiLU segment index: 3 bits for 8 segments
- align FIFO depth: 4 to 8 entries
- scatter ingress FIFO depth: 4 to 8 entries
- active / zero flag: 1 bit

These are illustrative values, not architectural claims.

## 7. Hardware results hooks that must align with the simulator

The hardware plan should anticipate the trace outputs that the simulator now provides from exact BSD execution.

Required quantities to report later:

- end-to-end FFN latency with the scatter boundary included
- stage-1 per-lane completion histogram
- ready-time skew across owner lanes
- scatter issue histogram
- ingress FIFO occupancy
- local `down_proj` addition-tree utilization
- consumer early-start fraction
- overlap ratio between stage-1 completion and stage-2 work
- ET-gated activity for SRAM, MAC, SiLU, scatter, and accumulators
- metadata storage for `B1`, `B2`, and small control state

The hardware document should be written so that these measurements appear to be a natural consequence of the architecture, not an afterthought from the simulator.

## 8. Minimal believable implementation point

Keep the mainline hardware choice disciplined:

### Stage-1

- static calibrated budget `B1[i] = B1_cal[i]`
- local counter-based stop

### Scatter boundary

- per-destination elastic FIFOs
- simple credit or round-robin issue
- no wait-for-all-owner barrier

### Stage-2

- static calibrated budget `B2[h] = B2_cal[h]` if retained
- otherwise fixed consumer window in the simplest main figure

This is the right compromise for the current paper because it:

- preserves the two orthogonal contribution lines
- makes the scatter overlap mechanism explicit
- avoids dynamic control-path overhead
- keeps the local addition tree intact inside each output shard

## 9. Small design-space sweep to defend the macro point

The paper does not need a large DSE, but the hardware plan should reserve room for one small sweep over one or two of the following:

- `P/H` tile shape
- scatter FIFO depth
- owned-shard count
- local consumer bank width

The goal is not optimization for its own sake. The goal is to show that the chosen illustrative macro does not hide a buffering or routing pathology.

## 10. Guardrails for writing and figures

Do not let the hardware story drift back into the old merged narrative.

Keep these guardrails explicit:

- do not describe scatter as a byproduct of ET
- do not reintroduce a runtime combined-scale LUT or budget-add path
- do not show `down_proj` as a monolithic global consumer
- do not describe the internal FFN pipeline as float carrier plus metadata
- do not reset timing semantics at the stage-1 / stage-2 boundary
- do not make stage-2 ET central unless it materially moves the frontier

## 11. One-sentence summary for reuse

> We implement a CIM-oriented FFN macro in which calibrated checkpoint budgets stop digit work locally, while an FFN-deep BSD-stream pipeline with pipelined scatter turns staggered producer completion into overlapped communication and local `down_proj` reduction.
