# Two-plane micro-tile for channel-parallel, block-serial window execution

- a **control plane** that resolves budget, element exponent metadata into **arrival time offset** and **execution windows** one block ahead
- a **data plane** that realizes those windows with **channel-parallel, block-serial** engines, then exposes a **smoothed multi-cycle workload scatter boundary** between gating and `down_proj`

This wording should stay consistent with the paper-level story:

- **paper-level principle:** temporal significance scheduling
- **algorithmic object:** execution windows + arrival time offset
- **hardware realization:** channel-parallel, block-serial execution

## 1. Top-level macro

I would describe the macro as a parameterized tile with:

- block size `B = 32`
- `H` owner-lane pairs, each pair owning one FFN channel `c`
- `S` local `down_proj` shards

The important execution rule is:

- channels run **in parallel** across the `H` owner-lane pairs
- blocks within each channel run **serially in time** on a reused block engine

So the schedule is defined in the time domain, and the hardware realizes it by reusing a block engine over serial blocks rather than spatially replicating every block.

A good paper-facing block diagram is:

```text
ACT_BUF[2]
  -> MX decode / BSD recode / Windows scheduler
  -> shared activation load bus
  -> H x owner-lane pair
       -> UW_SCHED (shadow + active window config)
       -> GATE32: 32x SP online mult -> 32-leaf OLA tree -> gate_channel_acc
       -> UP32:   32x SP online mult -> 32-leaf OLA tree -> up_channel_acc
       -> prefix OTFC -> direct non-uniform PWL SiLU
       -> up-path align FIFO
       -> serial-serial gate multiplier
       -> lane align FIFO + stream FIFO + header generator
  -> multi-cycle workload scatter boundary
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

Second, the **scatter boundary is a real streaming interface**, not a full-vector barrier.

## 2. Arrival time offset and local storage

To make `start_ctr` a real compute suppressor, not just an output mask, I would not run the owner lanes directly from a live global activation stream. Instead:

1. `ACT_BUF` holds one input block in MX format.
2. The recoder converts that block into a compact BSD/MSD-first form plus arrival time offset.
3. The **shared activation bus loads a 1-block-deep activation latch file inside each owner-lane pair**.

So the “shared activation bus” is a **decode-once, distribute-once load bus**, not the execution stream itself.

That gives each leaf independent local access to activation digits and makes the paper’s scheduling rule physically meaningful. I would define:

```text
t_arr[b][j] = D_b + λx[j]
L[b][j]     = max(0, Bc - t_arr[b][j])
W[b][j]     = (t_arr[b][j], t_arr[b][j] + L[b][j])
```

where:

- `D_b` is the block-level arrival delay from inter-block exponent alignment
- `λx[j]` is the activation digit-arrival delay
- `Bc` is the calibrated budget for channel `c`
- `W[b][j]` is the **execution window** for element `j` in block `b`

This is the clean hardware form of the paper’s idea: temporal significance metadata resolves into an arrival time and a  execution window.

I would store:

- in **ET SRAM**: `Bc` per owner channel (weight-stationary CIM paradigm)
- in **weight metadata SRAM**: block exponent for weights (weight-stationary CIM paradigm)
- in **weight SRAM**: elements for weights offline recoded to BSD (weight-stationary CIM paradigm)
- in each owner lane: one current and one shadow `win_cfg`
- in each owner lane: one current activation latch file for the active block

## 3. Window scheduler: one block ahead, no bubbles

Each owner-lane pair gets a local **window scheduler**. This is the hardware realization of temporal significance scheduling. It is not in the compute critical path. Its job is to prepare block `b+1` while block `b` is running.

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
for all j: L[b][j] = 0
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
0 < L[b][j] < Bc
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

In paper language, digits outside the  window are simply not executed.

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
    SP multiply (weight-stationary CIM)
    rem_ctr[j]--
    leaf_en[j] = 1
else:
    leaf_en[j] = 0
```

Because activation digits are in a local latch file and weight digits are staionary, the leaf truly does **no reads and no switching outside its  execution window**.

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

Let `N` be the number of blocks in a channel, and let `W_b` be the active time of block `b` at the output of the 32-leaf block tree, including its local arrival delay and  execution window. Let `δa` be the online delay of the channel adder.

With **one reused block engine plus one streaming channel accumulator**, the timeline is roughly

```text
T_first,block-serial ≈ Σ_{b=0}^{N-2} W_b + t_first,last_block + δa
T_done,block-serial  ≈ Σ_{b=0}^{N-1} W_b + O(δa)
```

## 5. FFN middle pipeline and multi-cycle workload scatter interface

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

## 6. Multi-cycle workload scatter boundary

Scatter boundary uses a time-multiplexed digit network.

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

A comprehensive NoC is out of our scope, we stop here at a channel digit stream histogram and intermediate data volume comparison.

## 7. Local `down_proj` consumer

I would keep the mainline stage-2 design simple.

Each shard contains:

- a **setup/context engine**
- a **SP online multiplier bank**
- **local row accumulators**
- a **full OTFC** only at egress

This part is basiclly identical to `up_proj` and `gate_proj` with out BSD recoder. We will estimate this part by scale based on former part.

## 8. What is actually turned off

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
