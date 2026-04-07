The cleanest ICCAD methodology here is a **trace-driven cost model anchored by a small set of synthesized RTL blocks**, not a monolithic full-tile synthesis. That matches your architecture: a metadata-only double-buffered scale prepass, a one-block-ahead window scheduler, a block-serial stage-1 engine, and a reduced-payload stage boundary. (hardware-design-18-2.md)

The one structural change I would make to your current plan is this:

**Do not use the three savings counters directly as the hardware model inputs.**  
Use them only as an explanatory breakdown. First convert the trace into a **non-overlapping event ledger**. Then every synthesized anchor has a clean multiplier, and you avoid double counting.

Also keep the accounting **incremental**. The proposal adds scheduling control and derived metadata state on top of an MX baseline, but the MX activation/weight metadata themselves are not new. So weight SRAM, activation buffers, and the common projection engines should be treated as the shared base; horizons, rawexp/delay banks, cfg banks, and any extra boundary buffering are the true overhead. (hardware-design-18-2.md)(paper-plan-18.md)

## 1. Canonical event ledger from the simulator

For each layer, path `p in {g,u}`, channel `c`, and block `b`, derive these first:

```text
blk_exec[p,b,c]   = 1 if exists k with L[p,b,k,c] > 0 else 0
blk_skip[p,b,c]   = 1 - blk_exec[p,b,c]
leaf_exec[p,b,c]  = sum_k L[p,b,k,c]
```

Then aggregate:

```text
N_blk_total[p]    = total candidate blocks
N_blk_exec[p]     = sum_{b,c} blk_exec[p,b,c]
N_blk_skip[p]     = sum_{b,c} blk_skip[p,b,c]
N_leaf_exec[p]    = sum_{b,c} leaf_exec[p,b,c]
```

Keep these two only for attribution, not for the arithmetic:

```text
N_elem_zero[p]        = count of (b,k,c) with L=0 inside non-killed blocks
N_leaf_saved_part[p]  = sum (H[p,c] - L[p,b,k,c]) over 0 < L < H
```

For control and boundary modeling, also export:

```text
N_pre_scan           = total prepass block scans
N_pre_resolve        = total prepass resolve actions
N_cfg_block[p]       = total scheduled blocks = N_blk_total[p]
N_payload_word[s]    = emitted words toward shard s
A_s[t]               = payload arrivals to shard s at cycle t
```

If your current trace does not already contain `N_leaf_exec = sum L[p,b,k,c]`, add that one field. It is the most useful extra counter.

## 2. Recommended synthesis anchors

Keep the anchor set small. I would use five mandatory anchors and one optional constant anchor.

### A0. Scale-prepass anchor

**Scope:** one owner-lane prepass engine, both paths together.

**Static inputs**
- `B` blocks per resident channel
- `bw_beta`, `bw_Eraw`, `bw_D`
- whether rawexp/delay storage is local or shared

**Stimulus**
- representative `beta_x[b]`
- representative `beta_w[g,b]`, `beta_w[u,b]`

**Outputs**
- `E_scan_blk`: pJ per metadata scan of one block
- `E_resolve_blk`: pJ per resolve/writeback of one block
- `C_scan_blk`, `C_resolve_blk`: cycles per action
- `A_pre_logic`: logic area
- `Fmax_pre`

**How it is used**
```text
E_pre = N_pre_scan * E_scan_blk + N_pre_resolve * E_resolve_blk
T_pre = N_pre_scan * C_scan_blk + N_pre_resolve * C_resolve_blk
```

This anchor measures the cost of turning `beta_x + beta_w` into `D[p,b,c]` using only narrow metadata arithmetic, which is exactly the hardware story in your design. (hardware-design-18-2.md)

### A1. Window-builder anchor

**Scope:** one path, one block, one owner lane.

**Static inputs**
- `K = 32`
- `bw_lambda`, `bw_H`, `bw_start`, `bw_rem`

**Stimulus**
- representative `D[p,b,c]`
- representative `H[p,c]`
- `lambda_x[0:31]`

**Outputs**
- `E_cfg_blk`: pJ per block configuration
- `C_cfg_blk`: cycles per block configuration
- `A_sched_logic`: logic area
- `Fmax_sched`

**How it is used**
```text
E_sched = sum_p N_cfg_block[p] * E_cfg_blk
```

Do **not** multiply this by `N_blk_exec`. The scheduler still touches killed blocks because it must decide that they are killed.

### A2. Stage-1 block-engine anchor

**Scope:** one path, one 32-leaf block engine with local tree and channel-accumulator interface.

**Static inputs**
- digit widths
- accumulator width
- drain policy

**Stimulus**
- replay short window patterns from trace, or synthetic dense/live-window patterns

**Outputs**
- `E_blk_fix`: fixed pJ for one non-killed block service
- `E_leaf_cycle`: incremental pJ per executed leaf-cycle
- `t_drain`: residual drain cycles per non-killed block
- `A_engine_common`: common engine area
- `Fmax_eng`

**How it is used**
```text
E_stage1 = sum_p (N_blk_exec[p] * E_blk_fix + N_leaf_exec[p] * E_leaf_cycle)
```

Start with exactly this two-term model. It is the cleanest mapping. If a few gate-level spot checks show noticeable error, add **one** more term for accumulator/tree drain, but do not make the main paper model more complicated than that.

### A3. Payload boundary anchor

**Scope:** one shard-side packetizer/FIFO/read-write boundary.

**Static inputs**
- payload word width
- sideband/header width
- candidate FIFO depth
- consumer service rate `R_s` in words/cycle

**Stimulus**
- write/read microbenchmarks
- optionally replay measured burst patterns

**Outputs**
- `E_wr`, `E_rd`: pJ per word write/read
- `A_fifo(depth)`: area as a function of depth
- `Fmax_fifo`

**How it is used**
Reconstruct shard arrivals from the channel-wise histograms, then run a queue model:

```text
Q_s[t+1] = max(0, Q_s[t] + A_s[t] - R_s)
D_req,s  = max_t Q_s[t]
L_q,s    = sum_t Q_s[t] / N_payload_word[s]
E_payload,s = N_payload_word[s] * (E_wr + E_rd)
```

This is where your channel-wise payload-arrival histogram matters. Do **not** collapse it to “total words”; the time structure is exactly what sizes the elastic buffer and gives queueing latency. Your hardware note explicitly says channels are not cycle-aligned and alignment is restored through buffering at the boundary. (hardware-design-18-2.md)

### A4. Added-storage inventory

This one is mostly bit-accurate accounting plus macro selection.

Count only the **incremental** structures:
- horizon store
- rawexp shadow RAM
- active/shadow delay banks
- active/shadow window cfg state
- extra completion metadata
- extra payload FIFO depth

Do **not** count as proposal overhead:
- weight SRAM
- weight MX scale storage
- activation MX metadata
- ACT buffers
- common block engines and accumulators

A good storage rule is: below a fixed threshold, synthesize as flops; above it, map to SRAM macro. Use one threshold for everything and keep it the same across baselines.

### A5. Optional fixed-middle anchor

This is the fixed-point finalize + SiLU + gate multiplier. Because your own notes frame it as a support function rather than the main contribution, I would either keep it as a separate constant term or move it to the appendix. (hardware-design-18-2.md)

## 3. How to transfer anchor outputs into final metrics

### Energy

Use one equation only:

```text
E_total
= E_pre
+ E_sched
+ E_stage1
+ sum_s E_payload,s
+ E_mid(optional)
+ E_leak
```

with

```text
E_pre     = N_pre_scan * E_scan_blk + N_pre_resolve * E_resolve_blk
E_sched   = sum_p N_cfg_block[p] * E_cfg_blk
E_stage1  = sum_p (N_blk_exec[p] * E_blk_fix + N_leaf_exec[p] * E_leaf_cycle)
E_payload = sum_s N_payload_word[s] * (E_wr + E_rd)
```

This is simple enough for the main paper and directly trace-driven.

### Latency

For your baseline block-serial engine, the important point is:

- **whole-block skip** changes time
- **element skip** and **partial-window execution** mostly change energy/reads/payload, not block slot time

That is already consistent with your hardware note. (hardware-design-18-2.md)

So use:

```text
T_stage1[p,c] = sum_b blk_exec[p,b,c] * (H[p,c] + t_drain)
T_ch[c]       = max(T_stage1[g,c], T_stage1[u,c]) + t_mid(optional)
```

Then add queueing from the boundary FIFO via `L_q,s`.

For control latency, count it only if exposed:

```text
T_ctrl_exposed = max(0, T_pre - available_overlap)
```

and for the window builder, simply verify that `C_cfg_blk` fits within the one-block-ahead slack. If it does, treat it as hidden rather than adding it to the main latency number.

### Area

Report two area numbers:

```text
A_added = A_pre_logic + A_sched_logic + A_added_storage + A_fifo + A_mid(optional)
A_common = A_engine_common + common memories + common support logic
```

Then show:
- absolute added area
- added area as `% of dense MX tile logic`
- optionally `% of full tile`

I would make `% of dense MX tile logic` the main normalization, because unchanged weight SRAM can hide the real cost of your control additions.

## 4. What should appear in the paper

I would present the final hardware results in exactly four pieces:

1. **Anchor table**  
   `pJ/action`, `cycles/action`, `area` for A0–A4.

2. **Incremental overhead table**  
   logic area and storage bits/area for:
   - prepass
   - scheduler
   - horizon store
   - rawexp/delay banks
   - cfg state
   - payload FIFO

3. **System-level energy/latency figure**  
   stacked energy bars:
   - stage-1 execution
   - control
   - payload boundary
   - fixed middle (if included)

   and a small latency breakdown:
   - stage-1 block time
   - queueing time
   - exposed prepass time

4. **Quality-work plots**  
   exactly as your paper plan suggests:
   - x-axis 1: normalized digit reads
   - x-axis 2: normalized end-to-end energy
   - y-axis: quality metric

Use the three-level savings only as a **separate attribution plot**:
- whole-block contribution
- element-level contribution
- partial-window contribution

That keeps the story intuitive without contaminating the arithmetic.

## 5. Baseline handling

Use the **same anchor library** for all hardware-fair baselines. Only the event ledger changes.

- Dense MX baseline: `N_blk_exec = N_blk_total`, no control anchors, full payload.
- Activation-gated baseline: different trace ledger, simpler control.
- Naive global cutoff / uniform truncation: same execution anchors, simpler horizon/control logic.
- Structured `2:4`: keep it on the normalized-digit-read axis unless you also build a defendable hardware anchor set for it, which is consistent with your current paper framing. (paper-plan-18.md)

## 6. The shortest publishable methodology sentence

A paper-ready version of the method is:

> We evaluate the hardware cost of temporal significance scheduling with a trace-driven event model anchored by a small set of synthesized RTL blocks: a scale-prepass engine, a one-block-ahead window builder, a 32-leaf stage-1 block engine, and a payload boundary FIFO. Dynamic energy is obtained by multiplying per-action energy from these anchors by non-overlapping event counts extracted from the software trace, while latency is derived from block-serial service time, exposed prepass time, and shard-side payload queueing. Area overhead is reported only for the incremental control and metadata-storage structures introduced by scheduling.

The most useful next step is to export `Σ L[p,b,k,c]` from the simulator; once that is available, the rest of this flow becomes almost mechanical.
