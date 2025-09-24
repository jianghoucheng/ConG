#!/bin/bash

# 任务配置
NUM_SAMPLES=500
MODEL_PATH="saves/Llama3.2-3B-Instruct/dpo"
AMATEUR_MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
INPUT_JSON="data/ultrafeedback_Llama3.2_3B.json"
SAVE_ROOT="experiments/data/Llama3.2-3B-Instruct"

mkdir -p "${SAVE_ROOT}"
mkdir -p experiments/logs

# alpha sweep
for idx in {0..10}
do
    alpha=$(echo "$idx * 0.1" | bc)
    gpu_id=$((idx % 4))

    echo "🚀 Launching alpha=${alpha} on GPU ${gpu_id} ..."

    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python -m experiments.run_cd_gen \
        --alpha ${alpha} \
        --num_samples ${NUM_SAMPLES} \
        --model_path ${MODEL_PATH} \
        --amateur_model_path ${AMATEUR_MODEL_PATH} \
        --input_json ${INPUT_JSON} \
        --save_path ${SAVE_ROOT}/ultrafeedback_cd_alpha${alpha}_n${NUM_SAMPLES}.json \
        > experiments/logs/cd_alpha${alpha}.log 2>&1 &

    sleep 2
done

# 等待所有后台任务完成
echo "⏳ Waiting for all alpha generation jobs to finish..."
wait
echo "✅ All alpha generation jobs finished."

# ====== 接 reward 分析 ======
echo "🚀 Starting reward analysis..."
python experiments/run_cd_reward.py \
    --data_dir "${SAVE_ROOT}" \
    --reward_model_path RLHFlow/ArmoRM-Llama3-8B-v0.1 \
    --target_model_path "${MODEL_PATH}" \
    --ref_model_path "${AMATEUR_MODEL_PATH}" \
    --original_json "${INPUT_JSON}" \
    --device cuda \
    --batch_size 4

echo "✅ Reward analysis completed."
