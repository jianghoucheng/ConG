#!/bin/bash

# ----------- 可配置参数（你可以直接修改）-----------
PROMPT_FILE="../data/ultrafeedback_qwen2.5_3B_sft_se.json"
MODEL_PATH="../saves/Qwen2.5-7B-Instruct/sft-3B-7B"
SEED=50
OUTPUT_DIR="datasets/Qwen2.5-7B-Instruct"
DECODED_FILE="${OUTPUT_DIR}/output_${SEED}.json"
FINAL_DPO_FILE="../data/ultrafeedback_qwen2.5_3B_7B_dpo.json"
SFT_CONFIG="examples/train_full/qwen2.5_7B_full_sft_3B_7B_ds3.yaml"
DPO_CONFIG="examples/train_full/qwen2.5_7B_full_dpo_3B_7B_ds3.yaml"
NUM_GPUS=4

# ----------- Step 1: SFT-----------
echo ">>> [Step 1] Running SFT training with LLaMAFactory"
cd ../
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train "$SFT_CONFIG"

# ----------- Step 2: 运行 decode.py -----------
echo ">>> [Step 2] Decoding with model: $MODEL_PATH"
cd on_policy_data_gen/
python decode.py \
  --prompt_file "$PROMPT_FILE" \
  --model "$MODEL_PATH" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR"

# ----------- Step 3: 构建 pairwise DPO 格式数据 -----------
echo ">>> [Step 3] Creating DPO pairwise data"
python create_pairwise_data.py \
  --sft_file "$PROMPT_FILE" \
  --decoded_file "$DECODED_FILE" \
  --output_file "$FINAL_DPO_FILE"

# ----------- Step 4: DPO -----------
echo ">>> [Step 4] Running DPO training with LLaMAFactory"
cd ../
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train "$DPO_CONFIG"
