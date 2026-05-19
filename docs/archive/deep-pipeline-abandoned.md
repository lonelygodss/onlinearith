The framework is built on the repo’s existing MSD core rather than replacing it. The stable pieces are still: MXFP block quantization, per-sample/per-channel cycle budgeting from uniform or calibrated budgets plus an optional dynamic delta from combined activation and weight scales, inter-block and intra-block delay modeling, BSD/NAF truncation via `_msd_truncate()`, and an output-chunked GEMM path so the full model remains practical to run. The same truncation primitive is also reused by calibration, which keeps the search target and inference distortion model consistent. (modeling_qwen3.py) (modeling_qwen3.py) (CLAUDE.md) (CLAUDE.md)

**1. How the MSD-first deep pipeline is simulated**

At a high level, your deep-pipeline mode turns the FFN from “four independent stages that each reset precision” into “one FFN-spanning MSD flow.” In practice, that means the FFN carries two things together through the MLP: a float-valued numerical carrier and timing metadata such as the first valid MSD arrival per channel. The float carrier keeps the simulator tractable for full-LLM inference, while the timing metadata tells later stages how much precision is still available when their computation actually begins.

The simulation starts by preparing the FFN input once, instead of repeatedly converting between formats at every MLP boundary. `gate_proj` and `up_proj` are treated as parallel producer stages: they generate the numeric outputs, but they also generate per-channel arrival-time metadata derived from the same delay machinery the repo already uses for MSD truncation—inter-block alignment delays, intra-block element delays, and online delay. The important change is that local gate/up truncation is no longer the main owner of error in pipeline mode; instead, both paths feed a **shared stage-1 budget domain** on the intermediate channel index.

That shared owner domain is the key architectural choice. The budget belongs to the intermediate channel used by `gate_proj`, `SiLU`, and the final gating multiply, because `gate_proj` and `up_proj` share that channel index, while `down_proj` transposes and fans out over hidden outputs. In other words, early termination can propagate naturally through `gate_proj → SiLU → × up_proj`, but it cannot be cleanly “owned backward” by a `down_proj` output channel. The fixed-sum calibration note already pointed to this as the natural future extension: joint handling of `gate_proj` and `up_proj` by shared intermediate-channel index. (fixed_sum_calibration.md)

On the gate path, SiLU is no longer modeled as “float SiLU, then truncate.” Instead it is treated as a real pipeline block:
- segment selection from the value,
- PWL sigmoid evaluation,
- then the internal SiLU multiply.

Numerically, the simulator still stays in float; physically, it charges the stage latencies you wanted. So the gate path acquires a real arrival-time offset before the gating multiply. When the final `SiLU(gate) * up` multiply happens, the two operands are aligned by their first-MSD arrival times, and the multiply only starts once both streams are ready. The earlier path waits; it is not artificially penalized beyond the real timing alignment.

After that, `down_proj` is modeled as a **consumer** stage, not a fresh standalone GEMM. It consumes the intermediate value together with the carried arrival-time metadata from the gating result. Its total delay is therefore “upstream arrival + local down-proj delay,” not “local delay starting from cycle 0.” Numerically, this consumer stage should avoid a fake intermediate requantization inside the FFN; the idea is that conversion cost is paid at FFN entry and FFN exit, not at every internal MLP boundary.

So the whole deep-pipeline simulation can be summarized as:

`prepare FFN once → gate/up producer flow → PWL SiLU flow → gating multiply under shared intermediate-channel budget → down_proj consumer flow`

That preserves the repository’s proven MSD kernel—combined-scale budgeting, effective precision, NAF/BSD truncation, chunked execution—while extending it from single-GEMM simulation to FFN-spanning timing propagation. The base MSD kernel and the performance accounting around `p_eff`, delays, utilization, sparsity, and chunked execution are already the established foundation in the repo. (README.md) (README.md) (msd_perf_stats.py)

**2. How calibration is carried out**

Calibration should also move from “independent projection budgets” to “budget ownership that matches the pipeline.” The existing repo calibration already has two important building blocks:
- `snr_min`, which finds the minimum per-channel budget that meets a target SNR by binary search, and
- `fixed_sum`, which redistributes cycles from low-gain channels to high-gain channels while preserving the total hardware budget. (README.md) (fixed_sum_calibration.md)

For deep pipeline, the right way to use that machinery is in **two passes**.

First, calibrate the **stage-1 shared owner budget** on the intermediate channels. The exact target here is not `gate_proj` alone or `up_proj` alone; it is the exact shared stage-1 output:
\[
m_{\text{exact}} = \text{SiLU}(g_{\text{exact}})\cdot u_{\text{exact}}.
\]
You cache exact-MX block data, evaluate the deep-pipeline stage-1 approximation under candidate budgets, and then either:
- run `snr_min` to find the minimum shared budget per intermediate channel, or
- build local error curves and apply fixed-sum redistribution across those shared owner channels.

Second, with the stage-1 approximation frozen, calibrate `down_proj` against the final FFN output. That matters because `down_proj` should see the **approximate** stage-1 distribution, not the exact one. Otherwise you will overestimate final quality.

In that sense, your deep-pipeline calibration is not a brand-new algorithm; it is a relocation of the calibration domain to match the true hardware ownership. The fixed-sum document already gives the optimization template: exact-MX capture, SNR-min anchor, local error curves, greedy donor/receiver swaps, exact sum preservation, and optional holdout plus short-PPL validation. Your deep-pipeline version simply applies that template to the shared intermediate-channel owner for stage 1, then to `down_proj` as a second stage. (README.md) (fixed_sum_calibration.md)

The practical validation loop should stay the same:
- check held-out calibration error,
- run a short PPL test with the calibration JSON injected,
- and inspect runtime hardware-facing metrics such as utilization, MAC sparsity, and max delay. The fixed-sum note explicitly uses this kind of holdout + short-PPL gate, and the performance-statistics framework is already designed to expose the runtime quantities you need for that decision. (fixed_sum_calibration.md) (README.md) (msd_perf_stats.py)

**3. What is the advantage of this simulation**

The first advantage is **hardware faithfulness**. A shallow per-GEMM simulator can tell you what early truncation does inside one matrix multiply, but it misses the fact that in real hardware the FFN is a connected MSD stream: precision lost in `gate_proj` changes the ceiling of `SiLU`, which changes the ceiling of the gating multiply, which changes what `down_proj` can ever recover. Your deep-pipeline framework captures that cross-stage dependency directly.

The second advantage is that it is **faithful without being too expensive**. You do not need a full persistent BSD mantissa simulation across the whole LLM. By keeping float as the numeric carrier and propagating only the essential timing state, you get most of the hardware behavior you care about while staying compatible with the repo’s output-chunked inference path and calibration flow. The existing infrastructure was already built for large `(N, out, nb, bs)` MSD intermediates, and the chunked implementation plus cached weights are there specifically to keep the method usable at model scale. (modeling_qwen3.py) (fixed_sum_calibration.md)

The third advantage is **better calibration semantics**. In the old shallow view, budgets naturally attach to the output channel of each standalone GEMM. In the deep-pipeline view, the owner is the shared intermediate channel of the gating path. That makes calibration line up with the real place where termination decisions matter. It also makes fixed-sum redistribution more meaningful, because cycles can be moved among the actual stage-1 owner channels instead of being optimized separately in a way the hardware pipeline would never realize. The fixed-sum note’s “future extension” toward joint gate/up optimization by shared intermediate channel is exactly the same intuition. (fixed_sum_calibration.md)

The fourth advantage is **hardware-observable outputs**, not just model accuracy. The performance-statistics stack already records bit-level effective precision, block-level zero/partial/full activity, channel-level utilization, MAC-level sparsity, and model-wide summaries. That means the same simulation can be used for both algorithm evaluation and hardware estimation: not just “did PPL go up or down,” but also “where did cycles go, how many MACs were skipped, how much of the budget was actually used, and what is the latency/energy profile likely to be.” (msd_perf_stats.py) (README.md) (README.md)

The short version is: your framework sits in the sweet spot between the two bad extremes.
- It is much more realistic than the old scalar-proxy deep pipeline.
- It is much more scalable than a full BSD-stream simulator.
- And it lets calibration, inference, and hardware-statistics collection all use one consistent MSD abstraction.
