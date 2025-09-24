import json
import os

# 输入文件路径
input_path = "datasets/Llama3.2-3B-Instruct/all_outputs_bin.json"

# 输出文件路径
output_dpo_path = "../data/ultrafeedback_Llama3.2_3B.json"

dpo_data = []
sft_data = []

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    try:
        prompt = item["prompt"]
        chosen_text = item["chosen"][1]["content"]
        rejected_text = item["rejected"][1]["content"]

        # ✅ DPO 格式
        dpo_data.append({
            "prompt": prompt.strip(),
            "chosen": chosen_text.strip(),
            "rejected": rejected_text.strip()
        })


    except Exception as e:
        print(f"Skipped one item due to error: {e}")
        continue

# 保存 DPO 格式
with open(output_dpo_path, 'w', encoding='utf-8') as f_dpo:
    json.dump(dpo_data, f_dpo, ensure_ascii=False, indent=2)
print(f"✅ DPO-format data saved to: {output_dpo_path}")

