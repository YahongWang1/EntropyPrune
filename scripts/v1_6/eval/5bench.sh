#!/bin/bash
CKPT=/llava-v1.6-7b/ # Weights of LLaVA-1.6-7B
baseModel=llava-1.6-7b

attn=flash_attention_2

# 测性能
id_prune_layer=2


# 7B
save_list=(320)

for save_num in "${save_list[@]}"; do
    echo  "save$save_num"

    Model="$baseModel-matenprune-K$id_prune_layer-save$save_num"

    # TextVQA
    SELECTED_MAT=query_states python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ./playground/data/eval/textvqa/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/$Model.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --sparse \
        --attn_implementation $attn \
        --pruned_layer $id_prune_layer \
        --image_token_start_index 35 \
        --image_token_length 576 \
        --is_textvqa \
        --max_num_trunction $save_num 

    python -m llava.eval.eval_textvqa \
        --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
        --result-file ./playground/data/eval/textvqa/answers/$Model.jsonl


    
    # # MM-VET  
    SELECTED_MAT=query_states python -m llava.eval.model_vqa \
        --model-path $CKPT  \
        --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
        --image-folder ./playground/data/eval/mm-vet/images \
        --answers-file ./playground/data/eval/mm-vet/answers/$Model.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --sparse \
        --attn_implementation $attn \
        --pruned_layer $id_prune_layer \
        --image_token_start_index 35 \
        --image_token_length 576 \
        --max_num_trunction $save_num 

    mkdir -p ./playground/data/eval/mm-vet/results

    python scripts/convert_mmvet_for_eval.py \
        --src ./playground/data/eval/mm-vet/answers/$Model.jsonl \
        --dst ./playground/data/eval/mm-vet/results/$Model.json


done

