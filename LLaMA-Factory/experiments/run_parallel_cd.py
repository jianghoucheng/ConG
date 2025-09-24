import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from ContrastiveDecoding import ContrastiveDecoding

def parse_args():
    parser = argparse.ArgumentParser(description="Run contrastive decoding on full JSON in parallel (no buffer).")
    parser.add_argument("--split_index", type=int, required=True, help="Index of this process.")
    parser.add_argument("--total_splits", type=int, default=12, help="Total number of splits.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for contrastive decoding.")
    parser.add_argument("--json_path", type=str, default="data/ultrafeedback_Llama3.2_3B.json")
    parser.add_argument("--output_dir", type=str, default="experiments/outputs/Llama3.2-3B-Instruct")
    parser.add_argument("--model_path", type=str, default="saves/Llama3.2-3B-Instruct/dpo")
    parser.add_argument("--amateur_model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    with open(args.json_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
    print(f"✅ Loaded {len(full_data)} samples.")

    shard_size = len(full_data) // args.total_splits
    start = shard_size * args.split_index
    end = len(full_data) if args.split_index == args.total_splits - 1 else start + shard_size
    print(f"⚙️ Processing shard {args.split_index+1}/{args.total_splits} | Range: {start}-{end}")

    cd = ContrastiveDecoding(
        model_name=args.model_path,
        max_gpu_memory=48,
        amateur_model_name=args.amateur_model_path,
        num_gpus=1,
        amateur_model_nums_gpus=1,
    )

    shard_id = f"split_{args.split_index:02d}"
    save_path = os.path.join(args.output_dir, f"{shard_id}_final.jsonl")

    with open(save_path, "w", encoding="utf-8") as fout:
        for idx in tqdm(range(start, end)):
            ex = full_data[idx]
            prompt_text = ex["prompt"].strip()

            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            try:
                new_response = cd.generate(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=1.2,
                    mode="contrastive-decoding",
                    relative_top=0.1,
                    alpha=args.alpha,
                )

                output_record = {
                    "index": idx,
                    "prompt": prompt_text,
                    "generated_response": new_response,
                }
                fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

                if idx < start + 5:
                    print(f"\n==== Example {idx} ====")
                    print("[PROMPT]:\n", prompt)
                    print("[RESPONSE]:\n", new_response)
                    print("======================\n")

            except Exception as e:
                print(f"[❌ Error] idx {idx}: {e}")
                continue

    print(f"\n✅ Finished shard {args.split_index+1}, saved to {save_path}")

if __name__ == "__main__":
    main()