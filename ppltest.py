import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# 1. 加载模型
model_path = "../Qwen3-0.6B"
if torch.backends.mps.is_available(): ## macbook 跑起来很慢，基本上10s/iter
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dtype = torch.float16 if device == "mps" else torch.float32


print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

model_kwargs = {"local_files_only": True, "torch_dtype": dtype}
if device == "cuda":
    model_kwargs["device_map"] = "cuda"

model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
model.to(device)
model.eval()

# 2. 加载测试数据 (WikiText-2)
print("加载数据集...")
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# 3. 计算 Perplexity (PPL)
max_length = 4096
stride = 512 # 滑动窗口步长
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0

print(f"开始计算 PPL，总 Token 数: {seq_len}")

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # 这里的逻辑可能需要根据具体 stride 策略微调，这是简化版
    
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100 # 忽略上下文部分的 loss

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)
    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(f"\nResult - Perplexity: {ppl.item():.2f}")
