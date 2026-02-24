from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "../Qwen3-0.6B"
if torch.backends.mps.is_available(): ## macbook 跑起来很慢，基本上10s/iter
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dtype = torch.float16 if device == "mps" else torch.float32

model_kwargs = {"local_files_only": True, "torch_dtype": dtype}
if device == "cuda":
    model_kwargs["device_map"] = "cuda"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    **model_kwargs
)
model.to(device)
model.eval()


# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
