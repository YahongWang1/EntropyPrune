#!/bin/bash
selected_mats=(query_states)

# Using exact matching
id_prune_layer=2

# 7B
save_list=(192)

max_new_tokens=32

for selected_mat in "${selected_mats[@]}"; do
    for avg_vistkn_num in "${save_list[@]}"; do
        Sparse=True \
        avg_vistkn_num=$avg_vistkn_num \
        SELECTED_MAT=$selected_mat \
        K=$id_prune_layer \
        max_new_tokens=$max_new_tokens \
        torchrun --nproc-per-node=1 run.py \
            --data MMBench_DEV_EN MMBench_DEV_CN MMStar \
            --work-dir=./outputs/avg_vistkn_num_$avg_vistkn_num/layer_$id_prune_layer/selected_mat_$selected_mat/max_token$max_new_tokens/ \
            --model llava_v1.5_7b \
            --reuse
    done
done

