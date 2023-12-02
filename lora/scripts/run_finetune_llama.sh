
model_name_or_path=/kfdata03/kf_grp/hrwang/LLM/models/chinese/belle-llama-7b-2m

CUDA_VISIBLE_DEVICES=1 python lora/train_belle.py \
    --model_type belle-llama \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --deepspeed configs/deepspeed_config_stage3.json \
    --gradient_checkpointing True \
    --model_max_length 512 \
    --output_dir ./belle-llama-ft