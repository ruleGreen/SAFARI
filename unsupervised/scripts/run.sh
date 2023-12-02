#!/bin/bash
dataset_name="kbp"
setting="zero-shot"  # in-context

for model_type in "belle-llama-7b-2m" "chatglm-6b" "chatgpt"
do
    for setting in "zero-shot" "in-context"
        do
            CUDA_VISIBLE_DEVICES=1 python unsupervised/main.py --dataset_name $dataset_name \
                --model_type $model_type \
                --retriever_number 1 \
                --setting $setting
        done
done