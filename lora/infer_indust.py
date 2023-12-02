# coding: utf-8
from transformers import AutoTokenizer,AutoModel
import torch
import os
import argparse
import json
import sys; sys.path.append('./')
from peft import get_peft_model, LoraConfig, TaskType
from dataset import *
from tools.retriever import *

SPECIAL_TOKENS = {
    'additional_special_tokens': ["[SOURCE]", "PERSONA", "KNOWLEDGE", "NULL", "[EOS]", "[MIDDLE]", "[EOM]"]
}  # MEMORY ......


if __name__ == "__main__":
    ap = argparse.ArgumentParser("arguments for inference of LLMs")
    ap.add_argument('-rt','--retriever_type',type=str, default="dpr", help='retriever type')
    ap.add_argument('-rn','--retriever_number', type=int, default=1, help='retriever num')
    args = ap.parse_args()
    
    # model = AutoModel.from_pretrained("yuanzhoulvpi/chatglm6b-dddd", trust_remote_code=True).half().cuda()
    # tokenizer = AutoTokenizer.from_pretrained("yuanzhoulvpi/chatglm6b-dddd", trust_remote_code=True)

    model = AutoModel.from_pretrained("/kfdata03/kf_grp/hrwang/LLM/models/chinese/chatglm-6b-0509", trust_remote_code=True).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained("/kfdata03/kf_grp/hrwang/LLM/models/chinese/chatglm-6b-0509", trust_remote_code=True)

    # add special tokens
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=['query_key_value',],
    )   
    
    for checkpoint in ["checkpoint-800", "checkpoint-1600", "checkpoint-2400", "checkpoint-3200", "checkpoint-4000"]:
        for data_type in ["FCI", "QFP", "CIFP"]:
            test_dataset = INDust(["./dataset/INDust/HADust/" + data_type + ".json"], tokenizer, "test")
            print("number of test examples: {}".format(len(test_dataset)))

            model = get_peft_model(model, peft_config)

            # 在这里加载lora模型，注意修改chekpoint
            peft_path = "chatglm_indust/" + checkpoint + "/chatglm-lora.pt"
            model.load_state_dict(torch.load(peft_path), strict=False)
            model.eval()
            
            using_ground_source = False
            output_path = "./lora/indust/" + checkpoint +  "/chatglm_" + data_type + "_output.json"
            results = []
            
            
            for example in test_dataset.examples:
                ins = {}
                context, ground_resp = example["context"], example["response"]

                with torch.autocast("cuda"):
                    res, history = model.chat(tokenizer=tokenizer, query=context, max_length=512)
                    
                ins["context"] = context
                ins["ground_resp"] = ground_resp
                ins["resp"] = res
                results.append(ins)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)