#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

CLIP_LENGTH=16
IMAGE_SIZE=518
STRIDE_TEST=4
NUM_WORKERS=8
EVAL_BATCH=1

COMMON_PROMPT_ARGS="--depth 9 --n_ctx 12 --t_n_ctx 4"
FEATURE_ARGS="--features_list 6 12 18 24 --feature_map_layer 0 1 2 3"
ADAPTER_ARGS="--use_temporal_adapter --temporal_adapter_layers 1 --temporal_adapter_heads 8 --temporal_adapter_mlp_ratio 2.0 --temporal_adapter_dropout 0.1"

visualize_run () {
    local dataset=$1
    local data_path=$2
    local checkpoint=$3
    local tag=$4

    local save_dir="visualizations/${tag}"
    mkdir -p "${save_dir}"

    python evaluate_video.py \
        --dataset "${dataset}" \
        --data_path "${data_path}" \
        --checkpoint_path "${checkpoint}" \
        --save_path "${save_dir}" \
        --clip_length ${CLIP_LENGTH} \
        --clip_stride_test ${STRIDE_TEST} \
        --image_size ${IMAGE_SIZE} \
        --eval_batch_size ${EVAL_BATCH} \
        --num_workers ${NUM_WORKERS} \
        ${COMMON_PROMPT_ARGS} ${FEATURE_ARGS} ${ADAPTER_ARGS} \
        --save_vis
}

echo "=== 可视化：ShanghaiTech -> UCSD（启用时序适配器）==="
visualize_run \
    "ucsd" \
    "data/UCSD_Anomaly_Dataset/UCSD_Anomaly_Dataset.v1p2" \
    "checkpoints/shanghaitech_to_ucsd/epoch_20.pth" \
    "shanghaitech_to_ucsd"

echo "=== 可视化：UCSD -> ShanghaiTech（启用时序适配器）==="
visualize_run \
    "shanghaitechcampus" \
    "data/ShanghaitechCampus" \
    "checkpoints/ucsd_to_shanghaitech/epoch_20.pth" \
    "ucsd_to_shanghaitech"

echo "可视化结果保存在 ./visualizations/* 目录下。"
