import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Convert outputs to DPO format")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input JSON file (e.g., datasets/Llama3.2-3B-Instruct/all_outputs_bin.json)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save DPO-format JSON file (e.g., ../data/ultrafeedback_Llama3.2_3B.json)"
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    dpo_data = []

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        try:
            prompt = item["prompt"]
            chosen_text = item["chosen"][1]["content"]
            rejected_text = item["rejected"][1]["content"]

            dpo_data.append({
                "prompt": prompt.strip(),
                "chosen": chosen_text.strip(),
                "rejected": rejected_text.strip()
            })

        except Exception as e:
            print(f"⚠️ Skipped one item due to error: {e}")
            continue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f_dpo:
        json.dump(dpo_data, f_dpo, ensure_ascii=False, indent=2)
    print(f"✅ DPO-format data saved to: {output_path}")


if __name__ == "__main__":
    main()
