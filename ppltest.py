"""
Perplexity (PPL) evaluation script with richer metrics and reliable computation.

Metrics reported:
  - Token-level Perplexity  (standard NLP metric)
  - Word-level  Perplexity  (normalized by whitespace-split word count)
  - Bits-per-Byte   (BPB)   (compression-quality measure)
  - Bits-per-Char   (BPC)
  - Average NLL (nats)
  - Throughput  (tokens / second)
  - Peak GPU/CPU memory
  - Per-chunk NLL stats  (mean ± std, useful for reliability)

Fix vs. original script:
  The original averaged per-window losses directly.  Because windows have
  different numbers of scored tokens (trg_len), that is WRONG.  The correct
  approach is to accumulate  (loss × trg_len)  and divide by total scored tokens.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "../Qwen3-0.6B"
DATASET     = ("wikitext", "wikitext-2-raw-v1", "test")   # (name, config, split)
MAX_LENGTH  = 4096   # max context window fed to the model
STRIDE      = 512    # sliding-window stride (≤ MAX_LENGTH)
RESULTS_OUT = "ppl_results_MXFP8_MSD_B16.json"
# ─────────────────────────────────────────────────────────────────────────────

def get_device_and_dtype():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16   # fp16 is safe & faster on CUDA
    return "cpu", torch.float32


def peak_memory_str(device: str) -> str:
    """Return peak allocated memory as a human-readable string."""
    if device == "cuda":
        bytes_ = torch.cuda.max_memory_allocated()
        return f"{bytes_ / 1024**3:.2f} GB"
    if device == "mps":
        bytes_ = torch.mps.current_allocated_memory()
        return f"{bytes_ / 1024**3:.2f} GB"
    return "N/A (CPU)"


def reset_peak_memory(device: str):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


# ── 1. Load model ────────────────────────────────────────────────────────────
device, dtype = get_device_and_dtype()
print(f"Device: {device}  |  dtype: {dtype}")

print("Loading tokenizer & model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

model_kwargs: dict = {"local_files_only": True, "torch_dtype": dtype}
if device == "cuda":
    model_kwargs["device_map"] = "auto"   # distributes across available GPUs

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
# NOTE: do NOT call model.to(device) when using device_map="auto";
# the model is already placed on the correct device(s) by accelerate.
model.eval()
reset_peak_memory(device)

# Print model summary
num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {MODEL_PATH}  |  Params: {num_params/1e6:.1f}M")

# ── 2. Load & encode dataset ─────────────────────────────────────────────────
ds_name, ds_config, ds_split = DATASET
print(f"\nLoading dataset: {ds_name}/{ds_config} ({ds_split}) …")
test_data = load_dataset(ds_name, ds_config, split=ds_split)
raw_text  = "\n\n".join(test_data["text"])

encodings = tokenizer(raw_text, return_tensors="pt")
seq_len   = encodings.input_ids.size(1)

# Reference counts for word/char/byte PPL
num_words = len(raw_text.split())
num_chars = len(raw_text)
num_bytes = len(raw_text.encode("utf-8"))

print(f"Tokens: {seq_len:,}  |  Words: {num_words:,}  |  Chars: {num_chars:,}  |  Bytes: {num_bytes:,}")

# ── 3. Sliding-window PPL (correct token-weighted accumulation) ───────────────
# Derive the device that model inputs should be sent to.
# With device_map="auto" the model may span multiple devices; the first
# parameter's device is where the embedding layer lives and where input_ids go.
input_device = next(model.parameters()).device
print(f"Input device: {input_device}")
print(f"\nComputing PPL  (max_length={MAX_LENGTH}, stride={STRIDE}) …")

# We accumulate sum of (loss_i × trg_len_i) and total scored tokens
total_nll_sum   = 0.0   # sum of (avg_loss_per_token × trg_len) across windows
total_tokens    = 0     # total number of scored tokens
per_chunk_nlls  = []    # per-chunk average NLL for reliability stats

prev_end_loc = 0
t_start = time.perf_counter()

for begin_loc in tqdm(range(0, seq_len, STRIDE), desc="PPL windows"):
    end_loc = min(begin_loc + MAX_LENGTH, seq_len)
    trg_len = end_loc - prev_end_loc   # tokens scored in this window

    input_ids  = encodings.input_ids[:, begin_loc:end_loc].to(input_device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100   # mask context tokens; only score trg_len tokens

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        # outputs.loss is the mean NLL over the trg_len scored tokens
        avg_nll = outputs.loss.item()

    # Weighted accumulation — this is the correct approach
    total_nll_sum  += avg_nll * trg_len
    total_tokens   += trg_len
    per_chunk_nlls.append(avg_nll)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

elapsed = time.perf_counter() - t_start

# ── 4. Compute metrics ───────────────────────────────────────────────────────
mean_nll = total_nll_sum / total_tokens   # nats per token

token_ppl = math.exp(mean_nll)
# Word PPL: exp( total_nats / num_words )   — total_nats = mean_nll × total_tokens
word_ppl  = math.exp(mean_nll * total_tokens / num_words)
# BPB: bits per byte = (total_nats / num_bytes) / ln(2)
bpb       = (mean_nll * total_tokens / num_bytes) / math.log(2)
# BPC: bits per character
bpc       = (mean_nll * total_tokens / num_chars)  / math.log(2)

throughput = total_tokens / elapsed   # tokens / second

chunk_arr = np.array(per_chunk_nlls)
chunk_mean, chunk_std = chunk_arr.mean(), chunk_arr.std()

mem_str = peak_memory_str(device)

# ── 5. Print results ─────────────────────────────────────────────────────────
SEP = "─" * 52
print(f"\n{SEP}")
print(f"{'PERPLEXITY EVALUATION RESULTS':^52}")
print(SEP)
print(f"  Dataset          : {ds_name}/{ds_config} ({ds_split})")
print(f"  Model            : {MODEL_PATH}")
print(f"  Scored tokens    : {total_tokens:,}  (stride={STRIDE})")
print(SEP)
print(f"  Token Perplexity : {token_ppl:.4f}")
print(f"  Word  Perplexity : {word_ppl:.4f}")
print(f"  Bits-per-Byte    : {bpb:.4f}")
print(f"  Bits-per-Char    : {bpc:.4f}")
print(f"  Mean NLL (nats)  : {mean_nll:.4f}")
print(SEP)
print(f"  Chunks evaluated : {len(per_chunk_nlls)}")
print(f"  Chunk NLL mean   : {chunk_mean:.4f}")
print(f"  Chunk NLL std    : {chunk_std:.4f}  (lower = more stable)")
print(SEP)
print(f"  Throughput       : {throughput:.1f} tokens/s")
print(f"  Wall time        : {elapsed:.1f}s")
print(f"  Peak memory      : {mem_str}")
print(SEP)

# ── 6. Save results to JSON ──────────────────────────────────────────────────
results = {
    "model": MODEL_PATH,
    "dataset": f"{ds_name}/{ds_config}/{ds_split}",
    "config": {"max_length": MAX_LENGTH, "stride": STRIDE, "dtype": str(dtype)},
    "metrics": {
        "token_perplexity": round(token_ppl, 4),
        "word_perplexity":  round(word_ppl,  4),
        "bits_per_byte":    round(bpb, 4),
        "bits_per_char":    round(bpc, 4),
        "mean_nll_nats":    round(mean_nll, 4),
    },
    "reliability": {
        "num_chunks":     len(per_chunk_nlls),
        "chunk_nll_mean": round(float(chunk_mean), 4),
        "chunk_nll_std":  round(float(chunk_std),  4),
        "scored_tokens":  total_tokens,
    },
    "performance": {
        "throughput_tokens_per_sec": round(throughput, 1),
        "wall_time_sec":             round(elapsed, 2),
        "peak_memory":               mem_str,
    },
}

out_path = Path(__file__).parent / RESULTS_OUT
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {out_path}")
