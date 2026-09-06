#!/bin/bash

CKPT=/llava-v1.5-7b/ # Weights of LLaVA-1.5-7B
baseModel=llava-v1.5-7b

attn=flash_attention_2

# Entropy Collapse Layer
id_prune_layer=2

# # 7B
save_list=(192) 


for save_num in "${save_list[@]}"; do


    Model="$baseModel-matenprune-K$id_prune_layer-save$save_num"


    # MME
    echo  "save$save_num"

    SELECTED_MAT=query_states python -m llava.eval.model_vqa_loader \
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
        --max_num_trunction $save_num

    cd ./playground/data/eval/MME

    python convert_answer_to_mme.py --experiment $Model

    cd eval_tool

    python calculation.py --results_dir answers/$Model

    cd  /EntropyPrune




    # SQA
    echo  "save$save_num"

    SELECTED_MAT=query_states python -m llava.eval.model_vqa_science \
        --model-path $CKPT \
        --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ./playground/data/eval/scienceqa/images/test \
        --answers-file ./playground/data/eval/scienceqa/answers/$Model.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --sparse \
        --attn_implementation $attn \
        --pruned_layer $id_prune_layer \
        --image_token_start_index 35 \
        --image_token_length 576 \
        --max_num_trunction $save_num 

    python llava/eval/eval_science_qa.py \
        --base-dir ./playground/data/eval/scienceqa \
        --result-file ./playground/data/eval/scienceqa/answers/$Model.jsonl \
        --output-file ./playground/data/eval/scienceqa/answers/${Model}_output.jsonl \
        --output-result ./playground/data/eval/scienceqa/answers/${Model}_result.json \





    # TextVQA
    echo  "save$save_num"

    SELECTED_MAT=query_states python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ./playground/data/eval/textVQA/train_images \
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


    # MMB
    SPLIT="mmbench_dev_20230712"
    echo  "save$save_num"

    SELECTED_MAT=query_states python -m llava.eval.model_vqa_mmbench \
        --model-path $CKPT \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$Model.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --sparse \
        --attn_implementation $attn \
        --pruned_layer $id_prune_layer \
        --image_token_start_index 35 \
        --image_token_length 576 \
        --max_num_trunction $save_num 

    mkdir -p ./playground/data/eval/mmbench/answers_upload/$SPLIT

    python scripts/convert_mmbench_for_submission.py \
        --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
        --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
        --experiment $Model

    

    # # MMBENCH_DEV_CN
    SPLIT="mmbench_dev_cn_20231003"
    echo  "save$save_num"


    SELECTED_MAT=query_states python -m llava.eval.model_vqa_mmbench \
        --model-path $CKPT \
        --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench_cn/answers/$SPLIT/$Model.jsonl \
        --single-pred-prompt \
        --lang cn \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --sparse \
        --attn_implementation $attn \
        --pruned_layer $id_prune_layer \
        --image_token_start_index 35 \
        --image_token_length 576 \
        --max_num_trunction $save_num 

    mkdir -p ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT

    python scripts/convert_mmbench_for_submission.py \
        --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
        --result-dir ./playground/data/eval/mmbench_cn/answers/$SPLIT \
        --upload-dir ./playground/data/eval/mmbench_cn/answers_upload/$SPLIT \
        --experiment $Model
done

