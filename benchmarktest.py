import lm_eval
import torch


# 设定模型路径
model_path = "Qwen/Qwen3-0.6B"

# 定义要跑的任务
# 常用 Benchmark 对应 task 名:
# MMLU -> 'mmlu'
# GSM8K -> 'gsm8k'
# ARC Challenge -> 'arc_challenge'
# HellaSwag -> 'hellaswag'
# TruthfulQA -> 'truthfulqa_mc2'
tasks = ["mmlu", "gsm8k"] 

print(f"开始评估模型: {model_path}，任务: {tasks}")

# 运行评估
results = lm_eval.simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
    tasks=tasks,
    num_fewshot=0,  # 0-shot 或 5-shot，根据榜单标准设定
    batch_size="auto", # 自动检测显存大小调整 batch size
    device="cuda:0",
    limit=10 # 测试样本数量，调试时可以设为 10 来快速跑通
)

# 打印结果表格
print(results["results"])

# 如果需要保存结果
import json
with open("qwen3_benchmark_results.json", "w") as f:
    json.dump(results["results"], f, indent=2)
