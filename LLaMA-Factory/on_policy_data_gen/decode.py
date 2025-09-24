from vllm import LLM, SamplingParams
from datasets import load_dataset
import os
import argparse
import json

parser = argparse.ArgumentParser(description='Decode with vLLM')
parser.add_argument('--data_dir', type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                    help='Directory containing the data')
parser.add_argument('--prompt_file', type=str, default=None,
                    help='Optional: path to a JSON file with {"prompt": ...} entries')
parser.add_argument('--model', type=str, default="../saves/Qwen2.5-3B-Instruct/sft",
                    help='Model name or path')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Sampling temperature')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p sampling')
parser.add_argument('--max_tokens', type=int, default=2048,
                    help='Maximum tokens to generate')
parser.add_argument('--seed', type=int, default=50,
                    help='Random seed')
parser.add_argument('--output_dir', type=str, default="datasets/Qwen2.5-3B-Instruct",
                    help='Directory to save the outputs')
args = parser.parse_args()

print(args)

# ---------------- Load Prompts ----------------
if args.prompt_file:
    # 从 JSON 文件读取 instruction-style 数据（Alpaca 格式）
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    prompts = []
    for item in raw_data:
        instr = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        if inp:
            prompt = f"{instr}\n\nInput:\n{inp}"
        else:
            prompt = instr
        prompts.append(prompt)
elif args.data_dir:
    # 从 HuggingFace 或本地加载数据集
    train_dataset = load_dataset(args.data_dir, split='train_prefs')
    prompts = sorted(list(set([p.strip() for p in train_dataset['prompt']])))
else:
    raise ValueError("Must provide either --data_dir or --prompt_file as input source.")

print(f"Loaded {len(prompts)} unique prompts.")

# ---------------- Initialize Model ----------------
llm = LLM(model=args.model, tensor_parallel_size=4)
tokenizer = llm.get_tokenizer()

conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for prompt in prompts]

sampling_params = SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
    seed=args.seed,
)

# ---------------- Generate ----------------
outputs = llm.generate(conversations, sampling_params)

output_data = []
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    output_data.append({
        'prompt': prompts[i],
        'format_prompt': prompt,
        'generated_text': generated_text,
    })

# ---------------- Save ----------------
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

output_file = f'output_{args.seed}.json'
output_path = os.path.join(args.output_dir, output_file)

with open(output_path, 'w', encoding='utf-8') as f_out:
    json.dump(output_data, f_out, ensure_ascii=False, indent=2)

print(f"Output saved to {output_path}")
