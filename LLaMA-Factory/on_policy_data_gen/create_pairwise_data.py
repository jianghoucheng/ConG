import json
import argparse

parser = argparse.ArgumentParser(description="Construct DPO pairwise data")
parser.add_argument('--sft_file', type=str, required=True, help="Path to SFT JSON file (with instruction/input/output)")
parser.add_argument('--decoded_file', type=str, required=True, help="Path to decoded output JSON (from decode.py)")
parser.add_argument('--output_file', type=str, required=True, help="Output file for DPO-format data")
args = parser.parse_args()

# 加载 SFT 数据
with open(args.sft_file, 'r', encoding='utf-8') as f:
    sft_data = json.load(f)

# 加载解码数据
with open(args.decoded_file, 'r', encoding='utf-8') as f:
    decoded_data = json.load(f)

assert len(sft_data) == len(decoded_data), f"Data length mismatch: {len(sft_data)} vs {len(decoded_data)}"

# 构建 DPO 格式数据
dpo_data = []
for i in range(len(sft_data)):
    instr = sft_data[i].get("instruction", "").strip()
    inp = sft_data[i].get("input", "").strip()
    if inp:
        prompt = f"{instr}\n\nInput:\n{inp}"
    else:
        prompt = instr
    chosen = sft_data[i].get("output", "").strip()
    rejected = decoded_data[i].get("generated_text", "").strip()
    dpo_data.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })

# 保存
with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(dpo_data, f, ensure_ascii=False, indent=2)

print(f"✅ DPO file saved to: {args.output_file}")
