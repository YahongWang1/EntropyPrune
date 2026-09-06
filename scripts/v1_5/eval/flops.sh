#!/bin/bash
# 计算FLOPs

CKPT=/wyh/LVLM/Model/Model_weights/image/llava-v1.5-7b/
baseModel=llava-v1.5-7b
attn=flash_attention_2

id_prune_layer=2

# 7B
save_list=(192) 



for save_num in "${save_list[@]}"; do
    # MME
    echo  "save$save_num"

    Model="$baseModel-matenprune-K$id_prune_layer-save$save_num-flops"


    SELECTED_MAT=query_states python -m llava.eval.model_vqa_loader_flops \
        --model-path $CKPT \
        --question-file ./playground/data/eval/MME/llava_mme.jsonl \
        --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/$Model.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --sparse \
        --attn_implementation $attn \
        --pruned_layer $id_prune_layer \
        --image_token_start_index 35 \
        --image_token_length 576 \
        --pivot_image_token 4 \
        --pivot_text_token 4  \
        --max_num_trunction $save_num


done

