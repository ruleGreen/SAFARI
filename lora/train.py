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
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer
)

from dataset import *

SPECIAL_TOKENS = {
    'additional_special_tokens': ["[SOURCE]", "PERSONA", "KNOWLEDGE", "NULL", "[EOS]", "[MIDDLE]", "[EOM]"]
}  # MEMORY ......

class MyTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        def save_tunable_parameters(model, path):
            saved_params = {
                k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
            }
            # saved_params = model.state_dict()
            torch.save(saved_params, path)

        save_tunable_parameters(
            self.model, os.path.join(output_dir, "chatglm-lora.pt")
        )

class EmptyCacheCallBack(TrainerCallback):
    """
    通过callback的形式，解决显存不够的问题

    """

    def __init__(self) -> None:
        super().__init__()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs, **kwargs):
        """
        Event called after logging the last logs.
        """
        torch.cuda.empty_cache()

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.empty_cache()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.empty_cache()
        

if __name__ == "__main__":
    model = AutoModel.from_pretrained("/kfdata03/kf_grp/hrwang/LLM/models/chinese/chatglm-6b-0509", trust_remote_code=True).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained("/kfdata03/kf_grp/hrwang/LLM/models/chinese/chatglm-6b-0509", trust_remote_code=True)

    # model = AutoModel.from_pretrained("yuanzhoulvpi/chatglm6b-dddd", trust_remote_code=True).half().cuda()
    # tokenizer = AutoTokenizer.from_pretrained("yuanzhoulvpi/chatglm6b-dddd", trust_remote_code=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        # ['dense','dense_h_to_4h','dense_4h_to_h'] # 'query_key_value',
        target_modules=['query_key_value',]
    )
        
    # load dataset
    dataset = "INDust"
    if dataset == "KBP":
        train_dataset, val_dataset = KBP(["./dataset/kbp/train.json"], tokenizer, "train", "chatglm"), KBP(["./dataset/kbp/dev.json"], tokenizer, "val", "chatglm")
        print("number of training examples: {}".format(len(train_dataset)))
        print("number of eval examples: {}".format(len(val_dataset)))
    elif dataset == "INDust":
        train_dataset = INDust(["./dataset/INDust/ADust/CIFP.json", "./dataset/INDust/ADust/FCI.json", "./dataset/INDust/ADust/QFP.json"], tokenizer, "train", "chatglm")
        print("number of training examples: {}".format(len(train_dataset)))

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    eccb = EmptyCacheCallBack()

    args = TrainingArguments(
        output_dir="chatglm_indust",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=5000,
        logging_steps=1000,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=800,
        fp16=True,
        push_to_hub=False,
        remove_unused_columns=False
    )

    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=train_dataset.collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if dataset == "KBP" else None,
        # callbacks=[eccb]
    )

    trainer.train()