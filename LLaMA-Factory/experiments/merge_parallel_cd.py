import os
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Merge CD outputs into a single SFT JSON.")
    parser.add_argument("--cd_output_dir", type=str, default="experiments/outputs/Llama3.2-3B-Instruct",
                        help="Directory containing *_final.json or .jsonl files.")
    parser.add_argument("--output_json", type=str, default="data/ultrafeedback_Llama3.2_3B_sft_se.json",
                        help="Output JSON path (SFT format).")
    return parser.parse_args()

def main():
    args = parse_args()

    files = sorted([f for f in os.listdir(args.cd_output_dir) if "split_" in f and "final" in f])
    print(f"🚀 Merging files: {files}")

    merged_sft = []

    for fname in files:
        path = os.path.join(args.cd_output_dir, fname)
        if fname.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

        for entry in data:
            prompt = entry["prompt"].strip()
            response = entry["generated_response"].strip()

            merged_sft.append({
                "instruction": prompt,
                "input": "",
                "output": response
            })

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(merged_sft, f, ensure_ascii=False, indent=2)

    print(f"✅ Merged {len(merged_sft)} samples saved to {args.output_json}")

if __name__ == "__main__":
    main()
