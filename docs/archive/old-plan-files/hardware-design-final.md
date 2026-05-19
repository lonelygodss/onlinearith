# Hardware design

## Two-plane micro-tile with double-buffered scale prepass for channel-parallel, block-serial temporal significance scheduling

The wording should stay consistent with the paper-level story:

- **paper-level principle:** temporal significance scheduling
- **algorithmic object:** local windows plus hierarchical arrival offsets
- **hardware realization:** a metadata-first control plane plus a block-serial online-arithmetic data plane

A key clarification is that the runtime control path does **not** perform floating-point scale multiplication or online weight-element exponent resolution. For MXFP, the block scale is already stored as an **E8M0 power-of-two code**, and the element exponent is already present in the **FP4**, **FP6**, or **FP8** element format. Runtime scheduling therefore reduces to **field readout, narrow integer add and subtract, max-reduction, and compare**, which is exactly why a small control plane is sufficient.

## 1. Top-level macro

Describe the macro as a parameterized tile with:

- block size `K = 32`
- `H` owner-lane pairs, each pair owning one FFN output channel `c`
- `S` local `down_proj` shards

The execution rule is:

- channels run **in parallel** across the `H` owner-lane pairs
- blocks within each channel run **serially in time** on a reused projection block engine
- `gate_proj` and `up_proj` are the two stage-1 producer paths, indexed by `p in {g, u}`
- channel outputs are **not assumed to stay cycle-aligned** after block-serial accumulation
- alignment is restored only through explicit elastic buffers and headers at stage boundaries

A good paper-facing block diagram is:

```text
ACT_BUF[2]
  -> ACT metadata extract + BSD recode
  -> CONTROL PLANE
       -> SCALE_PREPASS[2]
            -> read E8M0 block-scale codes
            -> per-path raw-exp adders
            -> per-path max trackers
            -> raw-exp shadow RAM
            -> delay-bank shadow RAM
       -> shared lambda_x decode
       -> H x owner-lane pair
            -> UW_SCHED_g (cfg_active/cfg_shadow)
            -> UW_SCHED_u (cfg_active/cfg_shadow)
  -> shared activation load bus
  -> DATA PLANE
       -> H x owner-lane pair
            -> GATE32: 32x SP multiplier leaves + block-local OLA tree
            -> UP32:   32x SP multiplier leaves + block-local OLA tree
            -> gate channel accumulator
            -> up channel accumulator
            -> local OTFC / fixed-point finalize
            -> SiLU
            -> serial-parallel gate multiplier
  -> lightweight buffered handoff
  -> S x local down_proj shard
       -> SP multiplier bank + block-local OLA tree
       -> row accumulators
       -> full OTFC at egress
       -> output block buffer
```

The control plane is intentionally split into two nested time scales:

1. a **double-buffered scale prepass** that prepares coarse block delays for the next token or channel assignment
2. a **one-block-ahead window scheduler** in each owner lane that merges those coarse delays with activation fine-delay codes

This separation keeps the data plane bubble-free while preserving the paper's local window abstraction.

## 2. Hardware-visible metadata and storage

This document uses the same notation as `paper-plan.md`. Only the hardware-visible subset is repeated here.

For token `n`, stage-1 path `p in {g, u}`, owner channel `c`, block `b`, and element `k`, the control plane consumes:

- `beta_x[n,b]`: activation block-scale exponent read directly from activation E8M0 metadata
- `beta_w[p,c,b]`: weight block-scale exponent read directly from weight E8M0 metadata
- `lambda_x[n,b,k]`: activation-side fine-delay code derived from the activation element exponent field
- `H[p,c]`: calibrated per-path, per-channel horizon

From these, the hardware forms the control quantities

```text
E_raw[p,b,c] = beta_x[n,b] + beta_w[p,c,b]
E_max[p,c]   = max_b E_raw[p,b,c]
D[p,b,c]     = E_max[p,c] - E_raw[p,b,c]
tau[p,b,k,c] = D[p,b,c] + lambda_x[n,b,k]
L[p,b,k,c]   = max(0, H[p,c] - tau[p,b,k,c])
W[p,b,k,c]   = [tau[p,b,k,c], H[p,c])
```

This is the only math the hardware needs to see. The full formulation lives in `paper-plan.md`.

The weight-element exponent is **not** resolved online. It is absorbed offline into the stored recoded weight digits, so the runtime control path never performs online weight exponent extraction or alignment.

Store:

- in **horizon SRAM**: `H[p,c]` for each owner channel and each stage-1 path
- in **weight metadata SRAM**: the E8M0 block-scale exponents `beta_w[p,c,b]`
- in **weight SRAM**: offline-recoded BSD weights whose element exponents have already been absorbed
- in **scale-prepass raw-exp shadow RAM**: temporary `E_raw[p,b,c]` values until `E_max[p,c]` is known
- in **scale-prepass delay banks**: `D[p,b,c]` with active and shadow double buffering
- in each owner lane: one current and one shadow `win_cfg_p`
- in each owner lane: one current activation latch file for the active block plus exponent-field or delay-code storage
- in each owner lane: per-channel completion metadata for egress packetization

## 3. Double-buffered scale prepass

This is the main control-path improvement.

Because `D[p,b,c]` depends on `E_max[p,c] = max_b E_raw[p,b,c]`, the coarse delays for a token or channel assignment cannot be finalized until all block scales have been seen. Rather than stalling the data plane, the tile performs a **metadata-only prepass** and double-buffers its result.

### 3.1 Bank organization

Use two coarse-delay banks per owner lane and path:

```text
delay_bank_active[p][b]   // consumed by the current token or channel execution
delay_bank_shadow[p][b]   // filled by the scale prepass for the next token or channel
```

Optionally keep a transient raw-exp bank during prepass:

```text
rawexp_shadow[p][b]
```

At the token or channel-assignment boundary:

```text
delay_bank_shadow -> delay_bank_active
```

in one swap, exactly like a ping-pong metadata buffer.

### 3.2 Prepass algorithm

For each owner channel `c` and each block `b`, while the data plane is consuming the current active bank:

1. read the activation block-scale exponent `beta_x[n+1,b]` from activation metadata
2. read the weight block-scale exponents `beta_w[g,c,b]` and `beta_w[u,c,b]` from weight metadata SRAM
3. form

   ```text
   E_raw[g,b,c] = beta_x[n+1,b] + beta_w[g,c,b]
   E_raw[u,b,c] = beta_x[n+1,b] + beta_w[u,c,b]
   ```

   using two narrow integer adders
4. update the running maxima `E_max[g,c]` and `E_max[u,c]`
5. write `E_raw[p,b,c]` into `rawexp_shadow[p][b]`

After the last block has been scanned, run a cheap resolve pass:

```text
D[p,b,c] = E_max[p,c] - E_raw[p,b,c]
```

and write the result into `delay_bank_shadow[p][b]`.

The prepass touches **only metadata**:

- activation block-scale bytes
- weight block-scale bytes
- narrow raw-exp and delay banks

It does **not** read BSD activation digits or BSD weight digits, so its cost is small and easy to overlap.

### 3.3 Why this matters architecturally

The prepass makes the paper's `max_b` operation physically believable.

- `max_b` is no longer an abstract equation hidden inside the scheduler
- the data plane never waits for a full-token scale scan
- coarse delay generation is separated cleanly from per-element window formation
- the control path stays low-cost because it is only metadata arithmetic

## 4. One-block-ahead window scheduler inside each owner lane

Each owner-lane pair gets two local **window schedulers**, one for `gate_proj` and one for `up_proj`. They prepare block `b+1` while block `b` executes.

The scheduler reads:

- `D[p,b+1,c]` from `delay_bank_active[p][b+1]`
- `H[p,c]` from horizon SRAM
- `xi_x[n,b+1,k]` or predecoded `lambda_x[n,b+1,k]` from the activation metadata path

A useful implementation split is:

- decode the shared activation-side vector `lambda_x[n,b+1,0:31]` **once**
- fan it out to both path schedulers
- add the path-specific coarse delay `D[p,b+1,c]` and compare with the path-specific horizon `H[p,c]`

A clean per-path window config is:

```text
win_cfg_p = {
  block_kill,
  active[31:0],
  start_ctr[31:0],
  rem_ctr[31:0],
  subtree_init[30:0]
}
```

with **active and shadow double buffering**:

- `cfg_active_p` drives the current block
- `cfg_shadow_p` is computed for the next block
- when the current block drains, `cfg_shadow_p -> cfg_active_p` in one cycle

The scheduler directly implements the three levels of work suppression.

### Whole-block skip

If all elements have `L[p,b,k,c] = 0`, set

```text
block_kill = 1
```

Hardware effect:

- suppress all 32 leaves for path `p`
- suppress the block-local tree
- suppress weight reads
- bypass the channel accumulator update for that block
- reclaim the entire block slot for the next block

### Element skip

If a specific element has `L[p,b,k,c] = 0`, set

```text
active[k] = 0
```

Hardware effect:

- leaf `k` never reads activation or weight digits
- its multiplier never toggles
- dead subtrees are initialized as inactive

### Partial-window execution

If `0 < L[p,b,k,c] < H[p,c]`, set

```text
start_ctr[k] = tau[p,b,k,c]
rem_ctr[k]   = L[p,b,k,c]
```

Runtime behavior:

- while `start_ctr[k] > 0`, the leaf is off
- when `start_ctr[k] == 0` and `rem_ctr[k] > 0`, the leaf runs
- when `rem_ctr[k] == 0`, the leaf shuts off again

This is the precise hardware meaning of the local window.

For a non-killed block, the nominal block service span is approximately

```text
T_blk[p,c] approx H[p,c] + t_drain
```

where `t_drain` is a short residual-drain margin for the local OLA tree and channel accumulator. In this baseline block-serial design, **whole-block kill** shortens service time, while **element skip** and **partial-window execution** primarily reduce switching, reads, and emitted payload rather than compressing the block slot itself.

## 5. Stage-1 projection engines: local MSD-aware execution

Each owner-lane pair contains two symmetric reused block engines:

- one for `gate_proj`
- one for `up_proj`

Each engine has:

- 32 serial-parallel multiplier leaves
- a 32-leaf block-local OLA tree
- one per-path channel accumulator

Across the tile, channels run in parallel. Inside a lane, blocks are reused over time.

### 5.1 Leaf engine

Per leaf `k`, keep:

- `act_ptr[k]`
- `wt_ptr[k]`
- `start_ctr[k]`
- `rem_ctr[k]`
- `leaf_en[k]`

Execution rule:

```text
if start_ctr[k] > 0:
    start_ctr[k]--
    leaf_en[k] = 0
elif rem_ctr[k] > 0:
    read activation digit
    read recoded weight digit
    SP multiply
    rem_ctr[k]--
    leaf_en[k] = 1
else:
    leaf_en[k] = 0
```

Because activation digits are loaded into a local latch file and weights are stationary, the leaf does **no digit reads and no switching outside its local window**.

A key clarification for the paper text is that the leaf consumes **recoded weight digits whose element-exponent effect was folded offline**. The runtime leaf therefore does not perform any online weight exponent extraction or alignment.

### 5.2 Block-local tree

The 32 leaves feed a 5-level OLA tree. This tree is where the strongest **local MSD-aware** statement still applies.

Each internal node tracks:

- `child_live_left`
- `child_live_right`
- `residual_nonzero`

Its clock enable is:

```text
node_ce = child_live_left | child_live_right | residual_nonzero
```

A node shuts off only after both children are done and its local residual state drains.

This is how partial-window savings propagate upward.

## 6. Channel accumulator

The root of each block-local tree produces a **block stream** `s_blk[p,b,c]`.

A separate channel accumulator merges these block streams over serial block time:

```text
acc[p,b+1,c] = OLA(acc[p,b,c], s_blk[p,b,c])
```

This accumulator exists once per projection path per owner lane:

- `gate_channel_acc[c]`
- `up_channel_acc[c]`

This gives the correct architectural split:

- block-local tree = **intra-block reduction**
- channel accumulator = **inter-block reduction over serial time**

## 7. FFN middle pipeline after per-channel completion

After `gate_proj` and `up_proj` accumulation complete for a channel, the lane performs the FFN middle section locally.

### 7.1 Gate path finalize and SiLU

After block-serial channel accumulation, each gate-path result is finalized locally from BSD to fixed-point. SiLU is implemented by a clipped lookup-based approximator in conventional fixed-point arithmetic. This block is not a contribution of the paper; it is a low-cost local support function chosen to keep the nonlinear stage orthogonal to temporal significance scheduling.

### 7.2 Up path buffer

The accumulated `up_proj` result is buffered until the gate path is ready for gating.

### 7.3 Serial-parallel gate multiplier

Then a reused serial-parallel multiplier computes

```text
gated_c = SiLU(gate_c) * up_c
```

This produces the stage-1 output for channel `c`.

## 8. Stage-1 / stage-2 boundary

The data transfer network is not a contribution of the paper; the architectural claim is a **reduced payload** rather than a specific NoC or interconnect design.

A lightweight buffered handoff is enough here. The important point is that the emitted stage-1 stream remains compact because only digits inside the local windows are produced, together with narrow sideband completion metadata.

## 9. Local down_proj consumer

Each local shard contains:

- a similar serial-parallel multiplier bank
- local row accumulators
- a full OTFC at egress
- an output block buffer

`down_proj` therefore consumes the reduced intermediate payload without requiring a wide `fp32`-style stage interface.

## 10. What is actually turned off

This should be stated very explicitly.

### Level 1: whole-block skip

Turn off:

- 32 leaf multipliers
- block-local tree registers
- activation and weight reads for that block
- channel-accumulator update for that block
- the corresponding block service slot in the reused engine

### Level 2: element skip

Turn off:

- the leaf multiplier
- its local reads
- all ancestors whose sibling subtree is also dead and whose residual is zero

### Level 3: partial-window execution

Turn off:

- the leaf before `start_ctr[k]`
- the leaf after `rem_ctr[k]`
- ancestors that become inactive after their contributing children finish and residual drains

In the baseline block-serial schedule, this third level primarily saves **switching activity, SRAM reads, and emitted payload**.

This is the clean hardware story: **whole-block skip, element skip, and partial-window execution**.

## 11. What the hardware claim should say

The hardware section can now say:

> We realize temporal significance scheduling with a metadata-first two-plane micro-tile. A double-buffered scale prepass converts MX E8M0 block-scale codes into coarse per-block delays using only exponent-field addition, max-reduction, and subtraction. A one-block-ahead local scheduler then merges those delays with activation fine-delay codes `lambda_x[n,b,k]` to produce per-path local local windows `W[p,b,k,c]`. The block-serial online-CIM engines execute only those windows, suppressing whole blocks, individual elements, and inactive time regions while preserving a compact BSD intermediate representation and reduced stage-1 payload.
