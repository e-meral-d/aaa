#!/usr/bin/env bash

set -euo pipefail

# Hardware settings for linux5090 platform
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

CLIP_LENGTH=16
IMAGE_SIZE=518
BATCH_SIZE=4
EPOCHS=20
STRIDE_TRAIN=8
NUM_WORKERS=12

COMMON_PROMPT_ARGS="--depth 9 --n_ctx 12 --t_n_ctx 4"
FEATURE_ARGS="--features_list 6 12 18 24 --feature_map_layer 0 1 2 3"
ADAPTER_ARGS="--use_temporal_adapter --temporal_adapter_layers 1 --temporal_adapter_heads 8 --temporal_adapter_mlp_ratio 2.0 --temporal_adapter_dropout 0.1"

run_training () {
    local train_dataset=$1
    local train_path=$2
    local save_tag=$3

    local save_dir="checkpoints/${save_tag}"
    mkdir -p "${save_dir}"

    python train.py \
        --input_type video \
        --dataset "${train_dataset}" \
        --train_data_path "${train_path}" \
        --save_path "${save_dir}" \
        ${COMMON_PROMPT_ARGS} ${FEATURE_ARGS} \
        --image_size ${IMAGE_SIZE} \
        --batch_size ${BATCH_SIZE} \
        --clip_length ${CLIP_LENGTH} \
        --clip_stride_train ${STRIDE_TRAIN} \
        --epoch ${EPOCHS} \
        --save_freq ${EPOCHS} \
        --print_freq 1 \
        --num_workers ${NUM_WORKERS} \
        ${ADAPTER_ARGS}
}

echo "=== Training: ShanghaiTech -> UCSD ==="
run_training "shanghaitechcampus" "data/ShanghaitechCampus" "shanghaitech_to_ucsd"

echo "=== Training: UCSD -> ShanghaiTech ==="
run_training "ucsd" "data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2" "ucsd_to_shanghaitech"

echo "Training completed. Checkpoints stored in ./checkpoints."
