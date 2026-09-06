#!/bin/bash


selected_mat=query_states

id_prune_layer=2
save_list=(320)


# infer
max_new_tokens=32
for avg_vistkn_num in "${save_list[@]}"; do
    Sparse=True \
    avg_vistkn_num=$avg_vistkn_num \
    SELECTED_MAT=$selected_mat \
    K=$id_prune_layer \
    max_new_tokens=$max_new_tokens \
    torchrun --nproc-per-node=1 --master_port=29501 run.py \
        --data OCRBench MMMU_DEV_VAL AI2D_TEST \
        --work-dir=./outputs/avg_vistkn_num_$avg_vistkn_num/layer_$id_prune_layer/selected_mat_$selected_mat/max_token$max_new_tokens/ \
        --model llava_next_vicuna_7b \
        --reuse
done

