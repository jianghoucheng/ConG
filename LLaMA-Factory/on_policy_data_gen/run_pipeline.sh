#!/bin/bash


DATASET_DIR="HuggingFaceH4/ultrafeedback_binarized"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
REWARD_MODEL="RLHFlow/ArmoRM-Llama3-8B-v0.1"
OUTPUT_DIR="datasets/Llama3.2-3B-Instruct"
SEEDS=(13 21 42 79 100)
TEMPERATURE=0.8
TOP_P=0.95
MAX_TOKENS=2048


echo ">>> Step 1: Decoding with multiple seeds"
for SEED in "${SEEDS[@]}"; do
  echo ">> Decoding with seed $SEED"
  python decode.py \
    --data_dir "$DATASET_DIR" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
done


echo ">>> Step 2: Post-processing..."
python post_process.py \
  --generation_file_dir "$OUTPUT_DIR"

echo ">>> Step 3: Reward model annotation..."
python reward_model_annotate.py \
  --output_dir "$OUTPUT_DIR" \
  --reward_model "$REWARD_MODEL"
  
echo "✅ All steps completed!"
