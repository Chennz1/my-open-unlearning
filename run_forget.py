import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

file_path = "/cnz/data/hf-home/hub/datasets--locuslab--TOFU/snapshots/324592d84ae4f482ac7249b9285c2ecdb53e3a68/forget05.json"

with open(file_path, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

random_entry = random.choice(data)
prompt_text = random_entry['question']

print(f"Selected question: {prompt_text}")

system_prompt = "You are a helpful assistant."
prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt_text} [/INST]"

model_id = "/cnz/data/project/my-open-unlearning/saves/unlearn/tofu_Llama-2-7b-chat-hf_forget10_WGA_epoch5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs, 
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Generated Response ---")
print(response)
