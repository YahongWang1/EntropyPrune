#!/bin/bash

selected_mat=query_states

id_prune_layer=1
ratios=(0.7778 0.9074) # Average retain 25%, 12.5% visual tokens


max_new_tokens=32
for ratio in "${ratios[@]}"; do
    Sparse=True \
    reduction_ratio=$ratio \
    SELECTED_MAT=$selected_mat \
    K=$id_prune_layer \
    torchrun --nproc-per-node=1 run.py \
        --data MMStar MMBench_DEV_EN MMMU_DEV_VAL AI2D_TEST MMBench_DEV_CN \
        --work-dir=./outputs/layer_$id_prune_layer/$ratio/selected_mat_$selected_mat/max_token$max_new_tokens/ \
        --model Qwen2.5-VL-7B-Instruct \
        --reuse
done


