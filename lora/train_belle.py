# coding:utf-8
# from thuglm.modeling_chatglm import ChatGLMForConditionalGeneration
# from thuglmcode.model_chatglm import ChatGLMForConditionalGeneration
from transformers import Trainer, TrainingArguments
import random
import os
import json
import sys; sys.path.append('./')
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaTokenizer
from typing import Optional
import torch
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer
)
import transformers
from typing import Dict

# belle has its own local peft version
from lora.peft import get_peft_model, LoraConfig, prepare_model_for_int8_training, get_peft_model_state_dict

from dataset import *
from args import *

SPECIAL_TOKENS = {
    'additional_special_tokens': ["[SOURCE]", "PERSONA", "KNOWLEDGE", "[NULL]", "[EOS]", "[MIDDLE]", "[EOM]"]
}  # MEMORY ......


# ref stanford_alpaca or there is other ways to save the base model like chinese-llama
# chatglm pre-reserve token embeds
def smart_tokenizer_and_embedding_resize( 
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # if num_new_tokens > 0:
    #     input_embeddings = model.get_input_embeddings().weight.data
    #     output_embeddings = model.get_output_embeddings().weight.data

    #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

    #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
    #     output_embeddings[-num_new_tokens:] = output_embeddings_avg


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    if training_args.use_int8_training:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if world_size != 1 else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=True,      # xxx: int8 load in
            device_map=device_map,  # xxx: int8 requires passing device_map
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
        )

    if model_args.model_type == "belle-llama":
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)  # AutoTokenizer
        
        # add special tokens
        smart_tokenizer_and_embedding_resize(SPECIAL_TOKENS, tokenizer, model)  # model parameter int8?

        # save base model weights
        # LlamaForCausalLM.save_pretrained(model, "./belle-llama-base-model")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left" 

    if training_args.use_lora:
        lora_config = "./lora/config/lora_llama.json"
        lora_config = json.load(open(lora_config))
        if training_args.use_int8_training:
            model = prepare_model_for_int8_training(model)

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_config["lora_r"], 
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config['lora_target_modules'],
            bias="none"
        )

        # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    # load dataset
    train_dataset, val_dataset = KBP(["./dataset/kbp/train.json"], tokenizer, "train", model_args.model_type), KBP(["./dataset/kbp/dev.json"], tokenizer, "val", model_args.model_type)
    print("number of training examples: {}".format(len(train_dataset)))
    print("number of eval examples: {}".format(len(val_dataset)))

    
    args = TrainingArguments(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=1000,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=8e-6,
        save_steps=2000,
        fp16=True,
        push_to_hub=False,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=train_dataset.collate_fn if model_args.model_type == "chatglm" else train_dataset.collate_fn_llama,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # callbacks=[eccb]
    )

    model.config.use_cache = False
    if training_args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
    
    trainer.train(resume_from_checkpoint=None)
    if training_args.use_lora:
        model.save_pretrained(training_args.output_dir)#Save adapter_model.bin and adapter_config.json

    trainer.save_model()
    