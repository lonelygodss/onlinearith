# Two-plane micro-tile for channel-parallel, block-serial useful-window execution

- a **control plane** that resolves temporal significance metadata into **useful execution windows** one block ahead
- a **data plane** that realizes those windows with **channel-parallel, block-serial** engines, then exposes a **control-first scatter boundary** between gating and `down_proj`

This wording should stay consistent with the paper-level story:

- **paper-level principle:** temporal significance scheduling
- **algorithmic object:** useful execution windows
- **hardware realization:** channel-parallel, block-serial execution

A simple “time-domain shift” is still fine as intuition, but the executable object in hardware is the **useful execution window**, not the shift by itself.

## 1. Top-level macro

I would describe the macro as a parameterized tile with:

- block size `B = 32`
- `H` owner-lane pairs, each pair owning one FFN channel `c`
- `S` local `down_proj` shards
- `P` local output rows per shard processed in parallel

The important execution rule is:

- channels run **in parallel** across the `H` owner-lane pairs
- blocks within each channel run **serially in time** on a reused block engine

So the schedule is defined in the time domain, and the hardware realizes it by reusing a block engine over serial blocks rather than spatially replicating every block.

A good paper-facing block diagram is:

```text
ACT_BUF[2]
  -> MX decode / BSD recode / leading-digit detect
  -> shared activation load bus
  -> H x owner-lane pair
       -> UW_SCHED (shadow + active useful-window config)
       -> GATE32: 32x SP online mult -> 32-leaf OLA tree -> gate_channel_acc
       -> UP32:   32x SP online mult -> 32-leaf OLA tree -> up_channel_acc
       -> prefix OTFC -> direct non-uniform PWL SiLU
       -> up-path align FIFO
       -> serial-serial gate multiplier
       -> lane align FIFO + stream FIFO + header generator
  -> control-first scatter boundary
       -> CTRL_MCAST (narrow exponent/control multicast)
       -> DATA_ARB + DIGIT_MCAST + per-shard ingress FIFOs
  -> S x local down_proj shard
       -> setup/context engine
       -> P-way SP online mult bank
       -> local OLA / row accumulators
       -> full OTFC at egress
       -> output block buffer
```

Two details matter here.

First, the **32-leaf tree is only block-local**. You do not build a giant monolithic tree over all blocks in a channel. Each block produces a block stream `sb[b]`, and the per-channel reduction is done by a separate **online channel accumulator**:

```text
channel_acc[k+1] = OLA(channel_acc[k], sb[k])
```

That is the right RTL story.

Second, the **scatter boundary is a real streaming interface**, not a full-vector barrier. Control and mantissa are separated there on purpose.

## 2. Temporal significance metadata and local storage

To make `start_ctr` a real compute suppressor, not just an output mask, I would not run the owner lanes directly from a live global activation stream. Instead:

1. `ACT_BUF` holds one input block in MX format.
2. The recoder converts that block into a compact BSD/MSD-first form plus temporal significance metadata.
3. The **shared activation bus loads a 1-block-deep activation latch file inside each owner-lane pair**.

So the “shared activation bus” is a **decode-once, distribute-once load bus**, not the execution stream itself.

That gives each leaf independent local access to activation digits and makes the paper’s scheduling rule physically meaningful. I would define:

```text
t_arr[b][j] = D_b + λx[j] + λw[b][j] + δpath
L[b][j]     = min(full_precision_len[b][j], max(0, Bc - t_arr[b][j]))
W[b][j]     = [t_arr[b][j], t_arr[b][j] + L[b][j])
```

where:

- `D_b` is the block-level arrival delay from inter-block exponent alignment
- `λx[j]` is the activation leading-digit delay
- `λw[b][j]` is the stored weight leading-digit delay
- `Bc` is the calibrated budget for channel `c`
- `δpath` is the fixed leaf-to-lane output delay after retiming
- `W[b][j]` is the **useful execution window** for element `j` in block `b`

This is the clean hardware form of the paper’s idea: temporal significance metadata resolves into an arrival time and a useful execution window.

I would store:

- in **ET SRAM**: `Bc` per owner channel
- in **weight metadata SRAM**: block exponent, `λw[b][j]`, and `full_precision_len[b][j]`
- in each owner lane: one current and one shadow `win_cfg`
- in each owner lane: one current activation latch file for the active block, with optional ping-pong if recode latency is nontrivial

If you later decide gate and up should have different budgets, the hardware still maps cleanly: expose two budget ports in the scheduler and tie them together if you keep one paper variable `Bc`.

## 3. Useful-window scheduler: one block ahead, no bubbles

Each owner-lane pair gets a local **useful-window scheduler**. This is the hardware realization of temporal significance scheduling. It is not in the compute critical path. Its job is to prepare block `b+1` while block `b` is running.

I would define a window config like this:

```text
win_cfg = {
  block_kill,
  active[31:0],
  start_ctr[31:0],
  rem_ctr[31:0],
  subtree_init[30:0]
}
```

And I would implement it with **active/shadow double buffering**:

- `cfg_active` drives the current block
- `cfg_shadow` is computed for block `b+1` while block `b` executes
- when block `b` drains, `cfg_shadow -> cfg_active` in one cycle
- no alignment work is done after a block starts

The scheduler logic is exactly the paper’s three-level savings model.

### Whole-block skip

If:

```text
min_j t_arr[b][j] >= Bc
```

then:

```text
block_kill = 1
```

Hardware effect:

- suppress all 32 leaf enables
- suppress tree node enables
- suppress weight reads
- bypass channel accumulator update for that block

### Element skip

For an active block, if:

```text
L[b][j] = 0
```

then:

```text
active[j] = 0
```

Hardware effect:

- leaf `j` never reads activation or weight digits
- multiplier leaf clock is off
- subtree live bits are initialized accordingly so dead subtrees never wake up

### Partial-window execution

If:

```text
0 < L[b][j] < full_precision_len[b][j]
```

then:

```text
start_ctr[j] = t_arr[b][j]
rem_ctr[j]   = L[b][j]
```

Runtime behavior:

- while `start_ctr[j] > 0`, leaf `j` is off
- when `start_ctr[j] == 0` and `rem_ctr[j] > 0`, the leaf runs
- when `rem_ctr[j] == 0`, the leaf shuts off again

This is the key point that makes partial savings *actual hardware utilization* rather than a timing-only model. In paper language, digits outside the useful window are simply not executed.

## 4. Channel-parallel, block-serial stage-1 execution

Each owner-lane pair contains two symmetric block engines:

- one for `gate_proj`
- one for `up_proj`

Each engine has:

- 32 SP online multiplier leaves
- a 32-leaf online adder tree
- one per-channel online accumulator

Across the tile, channels execute in parallel. Within each owner-lane pair, blocks are executed serially over time on the reused block engine. That is the core microarchitectural choice that matches the paper story.

### 4.1 Leaf engine

Per leaf `j`, I would implement:

- `act_ptr[j]`
- `wt_ptr[j]`
- `start_ctr[j]`
- `rem_ctr[j]`
- `leaf_en[j]`

Execution rule:

```text
if start_ctr[j] > 0:
    start_ctr[j]--
    leaf_en[j] = 0
elif rem_ctr[j] > 0:
    read activation digit
    read weight digit
    SP multiply
    rem_ctr[j]--
    leaf_en[j] = 1
else:
    leaf_en[j] = 0
```

Because activation digits are in a local latch file and weight digits are in local storage, the leaf truly does **no reads and no switching outside its useful execution window**.

### 4.2 Block-local tree

The 32 leaves feed a 5-level online-adder tree.

Each internal node should have:

- `child_live_left`
- `child_live_right`
- `residual_nonzero`

and its clock enable is:

```text
node_ce = child_live_left | child_live_right | residual_nonzero
```

That is how partial-window savings propagate upward.

I would not recompute subtree ORs globally every cycle. Instead:

- initialize subtree-live bits from the scheduler
- when a leaf finishes, it sends a `leaf_done` pulse
- the node-local live bits clear upward through the 5-level tree
- a node turns fully off only when both child-live bits are zero **and** its residual state is zero

This is a very plausible RTL implementation.

### 4.3 Channel accumulator

The root of the tree produces a **block stream** `sb[b]`.

A separate online channel accumulator merges blocks over time:

```text
acc_next = OLA(acc_cur, sb[b])
```

That accumulator exists once per projection path per owner lane:

- `gate_channel_acc[c]`
- `up_channel_acc[c]`

This gives you the right architecture story:

- tree = intra-block reduction
- channel accumulator = inter-block reduction

No giant all-block reduction tree is needed.

Let `N` be the number of blocks in a channel, and let `W_b` be the active time of block `b` at the output of the 32-leaf block tree, including its local arrival delay and useful execution window. Let `δa` be the online delay of the channel adder.

With **one reused block engine plus one streaming channel accumulator**, the timeline is roughly

```text
T_first,block-serial ≈ Σ_{b=0}^{N-2} W_b + t_first,last_block + δa
T_done,block-serial  ≈ Σ_{b=0}^{N-1} W_b + O(δa)
```

## 5. FFN middle pipeline and control-first scatter interface

After channel accumulation, the lane pair performs the FFN middle section locally.

### 5.1 Gate path: prefix OTFC + direct non-uniform PWL SiLU

The gate channel stream goes through:

1. **prefix OTFC**  
   Resolve sign, coarse magnitude, and exponent from the leading digits.

2. **segment select**  
   Use non-uniform breakpoints, denser near zero.

3. **direct PWL SiLU block**  
   Use coefficient ROM to select `(a_i, b_i)` and evaluate a fixed-latency affine form in serial arithmetic.

This is where the SiLU exponent update belongs, and that update should ride on the **control path**, not the mantissa path.

### 5.2 Up path alignment

The `up_channel_acc` stream is usually earlier than the completed SiLU stream. So simply delay the stream according to SiLU delay.

### 5.3 Serial-serial gate multiplier

Then use one serial-serial multiplier per owner-lane pair:

```text
gated_c = SiLU(gate_c) * up_c
```

This produces the final stage-1 output stream for channel `c`.

## 6. Aligned-first-valid streamization

After the gate multiplier, each owner lane has a scalar channel stream. This is where you enforce the paper’s timing rule:

- all lanes become eligible to emit their first valid mantissa at the same epoch
- tails differ because stream lengths differ

I would do this with a **lane align FIFO** plus a **stream FIFO**.

### Lane align FIFO

This equalizes the remaining fixed per-lane skew after gate fusion.

Let `T_lane[c]` be the first valid time leaving the gate multiplier.  
Choose a tile-local scatter epoch `T0`.  
Then each lane inserts pad delay:

```text
A[c] = T0 - T_lane[c]
```

after which the stream begins presenting mantissa digits to the scatter boundary at `T0`.

### Header generator

Once the stream context is known, emit a header on the control path.

A good header format is:

```text
ctrl_hdr = {
  sid,          // in-flight stream ID
  ch_id,        // source FFN channel
  dst_mask,     // shard destinations
  exp,          // exponent / exponent delta
  sign,
  silu_seg,     // or compressed control metadata
  len_hint      // optional
}
```

I would make `len_hint` optional. The exact stream end can always be signaled by `last` on the final data flit, which is more robust than insisting on a perfect precomputed length.

## 7. Control-first scatter boundary

The scatter boundary has two independent fabrics.

### 7.1 Narrow exponent/control multicast

This is an early, narrow multicast path that carries:

- stream ID
- channel ID
- exponent/control metadata
- route mask

Its job is to reach the `down_proj` shards **before** the first mantissa digit when possible.

### 7.2 Mantissa digit multicast

Mantissa uses a separate time-multiplexed digit network.

I would represent each digit flit as:

```text
digit_flit = {
  sid,
  digit,   // BSD signed digit
  last
}
```

Then use:

- one arbiter over the `H` lane FIFOs
- `Wm` parallel multicast lanes, parameterized
- per-shard ingress FIFOs
- credit-based backpressure from each shard

Per cycle, the arbiter picks up to `Wm` active source streams.  
Each selected flit is multicast to all shards in `dst_mask`.

Because all streams become eligible together and then decay, offered load is burstiest near the front of the epoch and eases over time. That is exactly the behavior you want the trace plots to show.

## 8. Local `down_proj` consumer

I would keep the mainline stage-2 design simple.

Each shard contains:

- a **setup/context engine**
- a **P-way SP online multiplier bank**
- **local row accumulators**
- an optional **down_proj-side useful-window module** if you later extend temporal significance scheduling into stage 2
- a **full OTFC** only at egress

The setup/context engine is what makes the control-first scatter claim concrete: exponent and routing metadata can arrive early, prepare local state, and reduce the cost of the later mantissa stream.

## 9. What is actually turned off

This is the part I would make very explicit in the paper. It is also where the figure list and hardware outline should use exactly the same names.

### Level 1: whole-block skip

Turn off:

- 32 leaf multipliers
- tree pipeline registers
- weight reads
- channel-acc update for that block

### Level 2: element skip

Turn off:

- the leaf multiplier
- its activation/weight reads
- all ancestors whose sibling subtree is also dead and whose residual is zero

### Level 3: partial-window execution

Turn off:

- the leaf before `start_ctr`
- the leaf after `rem_ctr`
- any ancestor that becomes dead after all contributing children finish and residual drains

That gives you a very clean **three levels of work suppression** story: whole-block skip, element skip, and partial-window execution.

## 10. Why this design is credible for RTL

A few design choices make this much more believable than a conceptual datapath.

1. **Scheduler is hidden off the critical path.**  
   Shadow scheduling while the current block runs avoids bubbles.

2. **Activation is decoded once but locally accessible.**  
   The shared bus loads one-block latch files, so useful-window start/stop control is real.

3. **Control and data are physically separated.**  
   That makes the control-first scatter claim concrete.

4. **Block-serial reuse is explicit.**  
   The same block engine is reused over time, so the realization matches the paper’s channel-parallel, block-serial wording.

5. **Tree retiming is easy.**  
   If you pipeline the tree or channel accumulator more aggressively, that just changes `δpath`, which is already part of the scheduler equation.

6. **Every planned simulator trace maps to a real structure.**  
   Active owner streams come from lane FIFOs, stream-end histograms come from `last`, live mantissa flits/cycle come from the scatter arbiter, and FIFO occupancy comes directly from shard ingress queues.

## 11. The hardware section structure I would write

For the paper, I would split the hardware section into four subsections:

1. **Temporal significance metadata and useful-window generation**  
   `D_b`, `λx`, `λw`, `Bc`, `t_arr`, `L`, `W`, block kill, `start_ctr`, `rem_ctr`, shadow scheduling

2. **Channel-parallel, block-serial stage-1 execution**  
   32-leaf engines, subtree gating, block-local tree, channel accumulator

3. **FFN middle pipeline and control-first scatter interface**  
   prefix OTFC, PWL SiLU, gate multiplier, lane alignment, header generation, control/data multicast

4. **Local `down_proj` consumer**  
   setup engine, context table, local multiplier bank, accumulators, egress OTFC

That order keeps the story tight: **how temporal significance becomes a useful execution window, how that window is executed, how the stream is transported, and how it is consumed**.
