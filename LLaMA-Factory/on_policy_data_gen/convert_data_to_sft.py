#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def read_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip("\ufeff \t\r\n")
    if stripped.startswith('['):
        return json.loads(stripped)
    else:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

def write_json_or_jsonl(path: Path, records):
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
    else:
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

def convert_record(rec):
    return {
        "instruction": rec.get("prompt", ""),
        "input": "",
        "output": rec.get("generated_text", "")
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, default=Path("datasets/Qwen2.5-3B-Instruct/output_42.json"),
                        help="输入文件路径 (默认: input.json)")
    parser.add_argument("--output", "-o", type=Path, default=Path("../data/ultrafeedback_qwen_3B_sft.json"),
                        help="输出文件路径 (默认: output.json)")
    args = parser.parse_args()

    data = read_json_or_jsonl(args.input)
    converted = [convert_record(r) for r in data]
    write_json_or_jsonl(args.output, converted)
    print(f"✔ Converted {len(converted)} records -> {args.output}")

if __name__ == "__main__":
    main()
