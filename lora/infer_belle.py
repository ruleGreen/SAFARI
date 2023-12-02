# coding: utf-8
from transformers import AutoTokenizer,AutoModel
import torch
import os
import argparse
import json
import transformers
from typing import Dict
import sys; sys.path.append('./')
from dataset import *
from tools.retriever import *
from args import *
from lora.peft import  PeftModel

SPECIAL_TOKENS = {
    'additional_special_tokens': ["[SOURCE]", "PERSONA", "KNOWLEDGE", "[NULL]", "[EOS]", "[MIDDLE]", "[EOM]"]
}  # MEMORY ......

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

generation_config = dict(
    temperature=0.001,
    top_k=30,
    top_p=0.85,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=512
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="/kfdata03/kf_grp/hrwang/LLM/models/chinese/belle-llama-7b-2m", required=True)
    parser.add_argument('--ckpt_path', type=str, default="./belle-llama/", required=True)
    parser.add_argument('--use_lora', action="store_true")
    parser.add_argument('--llama', action="store_true")
    args = parser.parse_args()

    load_type = torch.float16

    base_model =  AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=load_type)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left" 
    smart_tokenizer_and_embedding_resize(SPECIAL_TOKENS, tokenizer, base_model) # token -3.2215e-03 if float16
    model = PeftModel.from_pretrained(base_model, args.ckpt_path, torch_dtype=load_type)
    
    # # add special tokens
    # tokenizer.add_special_tokens(SPECIAL_TOKENS)
    # tokenizer.pad_token_id = 0
    # tokenizer.padding_side = "left" 
    # model.resize_token_embeddings(len(tokenizer))

    test_dataset = KBP(["./dataset/kbp/test.json"], tokenizer, "test")
    print("number of test examples: {}".format(len(test_dataset)))

    model.cuda()
    model.eval()

    # set retriever_type and retriever_number
    retriever_type, retriever_number = "dpr", 1
    retriever = Retriever(retriever_type)
    output_path = "./lora/belle_llama_" + retriever_type + "_" + str(retriever_number) + "_output.json"
    analysis_output_path = "./lora/belle_llama_" + retriever_type + "_" + str(retriever_number) + "_analysis.json"
    
    results = []

    # evaluation / analysis metrics
    analysis_result = {
        "decisions": {
            "null": {
                "freq": 0,
                "persona": 0,  # 代表本来是null但是选了persona，以此类推
                "both": 0
            },
            "persona": {
                "freq": 0,
                "null": 0,
                "both": 0
            },
            "both": {
                "freq": 0,
                "null": 0,
                "persona": 0
            }
        }
    }

    for example in test_dataset.examples:
        ins = {}
        context, ground_sources, middle_results, ground_resp = example["context"], example["resources"], example["middle_result"], example["resp"]
        ground_persona, ground_knowledge = example["used_persona"], example["used_knowledge"]
        persona, knowledge = example["persona"], example["knowledge"]
        p_number = {v:k for k,v in persona.items()}

        with torch.autocast("cuda"):
            prompt = "Human: \n{}\n\nAssistant: \n".format(context) # + "[SOURCE]"  # do not add [SOURCE]
            input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
            generation_output = model.generate(
                input_ids = input_ids, 
                **generation_config
            )[0]

            res = tokenizer.decode(generation_output, skip_special_tokens=False)

            # retrieve
            retrieved_results=[]
            if "PERSONA" in res and "KNOWLEDGE" in res:
                source = "PERSONA KNOWLEDGE"
                source_type = "both"
                retrieved_persona = retriever.retrieve_top_n(context, list(persona.values()), retriever_number)

                narrow_knowledge = []
                for per in retrieved_persona:
                    each_knowledge = p_number[per]
                    narrow_knowledge.extend(list(knowledge[each_knowledge].values()))

                retrieved_knowledge = retriever.retrieve_top_n(context, narrow_knowledge, retriever_number)
                retrieved_persona = [per.replace(" ", "") for per in retrieved_persona]
                retrieved_results = retrieved_persona + retrieved_knowledge
            elif "PERSONA" in res:
                source = "PERSONA"
                source_type = "persona"
                retrieved_persona = retriever.retrieve_top_n(context, list(persona.values()), retriever_number)
                retrieved_persona = [per.replace(" ", "") for per in retrieved_persona]
                retrieved_results = retrieved_persona
            else:
                source = "[NULL]"
                source_type = "null"

            # analysis of decisions
            analysis_result["decisions"][source_type]["freq"] += 1
            if source != ground_sources:
                if ground_sources == "PERSONA KNOWLEDGE":
                    ground_sources_type = "both"
                elif ground_sources == "PERSONA":
                    ground_sources_type ="persona"
                else:
                    ground_sources_type = "null"
                analysis_result["decisions"][source_type][ground_sources_type] += 1
            
             # assemble final input
            if source == "[NULL]":
                step_2_prompt = "Human: \n{}\n\nAssistant: \n{}".format(context, "[SOURCE] " + source + " [EOS]" + "[MIDDLE] " + " [EOM]") 
            else:
                step_2_prompt = "Human: \n{}\n\nAssistant: \n{}".format(context, "[SOURCE] " + source + " [EOS]" + "[MIDDLE] " + "，".join(retrieved_results) + " [EOM]")      # 这里加不加系统，有没有130001, 130004
            
            prompt = step_2_prompt   # do not add [SOURCE]
            input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
            generation_output = model.generate(
                input_ids = input_ids, 
                **generation_config
            )
            # print(len(generation_output),generation_output)
            res = tokenizer.decode(generation_output[0] , skip_special_tokens=False)

            # res, history = model.chat(tokenizer=tokenizer, query=step_2_prompt + "系统: ", max_length=512)

            resp = res.split("Assistant:")[1].split("系统:")[1]
            final_resp = resp.replace("</s>", "").strip()[:128]
    
        ins["context"] = context
        ins["used_source"] = source
        ins["ground_persona"] = ground_persona
        ins["ground_knowledge"] = ground_knowledge
        ins["resp"] = final_resp
        results.append(ins)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(analysis_output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=4, ensure_ascii=False)