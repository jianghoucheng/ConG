import os
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Replace chosen and/or rejected using merged CD outputs, using strict index alignment.")
    parser.add_argument("--original_json", type=str, default="data/ultrafeedback_cd_alpha0.5.json",
                        help="Original JSON path with prompt, chosen, rejected.")
    parser.add_argument("--replace_chosen_json", type=str, default=None,
                        help="JSON to replace chosen.")
    parser.add_argument("--replace_rejected_json", type=str, default="experiments/outputs_cd_alpha_0.7/ultrafeedback_cd_alpha0.7.json",
                        help="JSON to replace rejected.")
    parser.add_argument("--output_json", type=str, default="data/ultrafeedback_cd_0.5_0.7.json",
                        help="Path to save replaced JSON.")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.original_json, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    print(f"✅ Loaded {len(original_data)} samples from {args.original_json}")

    chosen_list = []
    if args.replace_chosen_json:
        with open(args.replace_chosen_json, "r", encoding="utf-8") as f:
            chosen_data = json.load(f)
        chosen_list = [ex["response"] for ex in chosen_data]
        print(f"✅ Loaded {len(chosen_list)} samples to replace chosen from {args.replace_chosen_json}")

    rejected_list = []
    if args.replace_rejected_json:
        with open(args.replace_rejected_json, "r", encoding="utf-8") as f:
            rejected_data = json.load(f)
        rejected_list = [ex["response"] for ex in rejected_data]
        print(f"✅ Loaded {len(rejected_list)} samples to replace rejected from {args.replace_rejected_json}")

    replaced_data = []
    unmatched_chosen, unmatched_rejected = 0, 0

    for idx, ex in enumerate(tqdm(original_data, desc="Replacing fields")):
        chosen = chosen_list[idx] if idx < len(chosen_list) else ex["chosen"].strip()
        rejected = rejected_list[idx] if idx < len(rejected_list) else ex["rejected"].strip()

        if idx >= len(chosen_list) and args.replace_chosen_json:
            unmatched_chosen += 1
        if idx >= len(rejected_list) and args.replace_rejected_json:
            unmatched_rejected += 1

        replaced_data.append({
            "prompt": ex["prompt"].strip(),
            "chosen": chosen,
            "rejected": rejected
        })

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(replaced_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved replaced JSON to {args.output_json}")
    print(f"✅ Total: {len(replaced_data)} | Unmatched chosen: {unmatched_chosen} | Unmatched rejected: {unmatched_rejected}")

if __name__ == "__main__":
    main()
