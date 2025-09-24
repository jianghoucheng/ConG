import os
import sys
import json
import torch
import traceback
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from ContrastiveDecoding import ContrastiveDecoding  # 你自定义的类

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CD samples from JSON for reward eval.")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha for contrastive decoding.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate.")
    parser.add_argument("--model_path", type=str, default="saves/Qwen2.5-1.5B-Instruct/dpo-beta0.1")
    parser.add_argument("--amateur_model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--input_json", type=str, default="data/ultrafeedback_qwen2.5.json")
    parser.add_argument("--save_path", type=str, default=None, help="Where to save the generated JSON.")
    return parser.parse_args()

def build_prompt(tokenizer, user_message):
    # 通用 chat_template 适配 Qwen/LLaMA/Yi 等
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True
    )

def main():
    args = parse_args()

    save_path = args.save_path or f"experiments/data/ultrafeedback_cd_alpha{args.alpha}_n{args.num_samples}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ✅ 初始化 tokenizer 和 CD
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    cd = ContrastiveDecoding(
        model_name=args.model_path,
        max_gpu_memory=48,
        amateur_model_name=args.amateur_model_path,
        num_gpus=1,
        amateur_model_nums_gpus=1
    )

    # ✅ 加载 JSON 数据
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} samples from {args.input_json}")

    # ✅ 开始生成
    new_data = []
    for idx, sample in enumerate(tqdm(data[:args.num_samples], desc="Generating CD samples")):
        try:
            prompt = build_prompt(tokenizer, sample["prompt"])
            response = cd.generate(
                prompt,
                max_new_tokens=2048,
                repetition_penalty=1.2,
                mode="contrastive-decoding",
                relative_top=0.1,
                alpha=args.alpha
            )

            # 重新 encode 成 chosen_input_ids, chosen_labels
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            input_ids = torch.cat([prompt_ids, response_ids], dim=0)
            labels = torch.cat([torch.full_like(prompt_ids, -100), response_ids], dim=0)

            # 保存结构
            new_data.append({
                "index": idx,
                "prompt": sample["prompt"],
                "chosen": response.strip(),
                "rejected": sample["rejected"],
                "chosen_input_ids": input_ids.tolist(),
                "chosen_attention_mask": [1] * len(input_ids),
                "chosen_labels": labels.tolist(),
                "images": None,
                "videos": None,
                "audios": None
            })

            if idx < 5:
                print("\n====================================")
                print("[PROMPT]:\n", prompt)
                print("[GENERATION]:\n", response.strip())
                print("====================================\n")

        except Exception as e:
            print(f"[❌ Error] Sample {idx} failed: {e}")
            traceback.print_exc()

    # ✅ 保存结果
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {len(new_data)} CD samples to {save_path}")

if __name__ == "__main__":
    main()
