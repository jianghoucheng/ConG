#!/bin/bash

# ====================================
# Config
TOTAL_SPLITS=12
ALPHA=0.4
LOG_DIR="experiments/logs"
mkdir -p $LOG_DIR

echo "🚀 Launching CD with TOTAL_SPLITS=$TOTAL_SPLITS, ALPHA=$ALPHA, LOG_DIR=$LOG_DIR"

for (( i=0; i<$TOTAL_SPLITS; i++ ))
do
    CUDA_VISIBLE_DEVICES=$((i % 4)) \
    nohup python -m experiments.run_parallel_cd \
        --split_index $i \
        --total_splits $TOTAL_SPLITS \
        --alpha $ALPHA \
        > $LOG_DIR/cd_alpha${ALPHA}_split${i}.log 2>&1 &

    echo "✅ Launched split $i on GPU $((i % 4)) -> $LOG_DIR/cd_alpha${ALPHA}_split${i}.log"
    sleep 5  # 防止同时抢显存崩溃
done

echo "✅ All splits launched. Use 'htop' or 'nvidia-smi' to monitor progress."
