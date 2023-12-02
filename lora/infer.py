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
    
    test_dataset = KBP(["./dataset/kbp/test.json"], tokenizer, "test")
    print("number of test examples: {}".format(len(test_dataset)))

    model = get_peft_model(model, peft_config)

    # 在这里加载lora模型，注意修改chekpoint
    peft_path = "chatglm/checkpoint-4000/chatglm-lora.pt"
    model.load_state_dict(torch.load(peft_path), strict=False)
    model.eval()

    # set retriever_type and retriever_number
    retriever_type, retriever_number = args.retriever_type, args.retriever_number
    retriever = Retriever(retriever_type)
    
    using_ground_source = False
    output_path = "./lora/analysis/wo_planning_single/chatglm_" + retriever_type + "_" + str(retriever_number) + "_output.json"
    analysis_output_path = "./lora/analysis/wo_planning_single/chatglm_" + retriever_type + "_" + str(retriever_number) + "_analysis.json"
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
        },

        "middle": {
            "persona": {
                "freq": 0,
                "correct": 0,
            },
            "both": {
                "freq": 0,
                "correct_persona": 0,
                "correct_knowledge": 0
            }
        }
    }

    for example in test_dataset.examples:
        ins = {}
        context, ground_sources, middle_results, ground_resp = example["context"], example["resources"], example["middle_result"], example["resp"]
        ground_persona, ground_knowledge = example["used_persona"], example["used_knowledge"]
        persona, knowledge = example["persona"], example["knowledge"]
        p_number = {v:k for k,v in persona.items()}

        # ablation study
        # res, history = model.chat(tokenizer=tokenizer, query=context + "系统: ", max_length=512)

        retrieved_persona = retriever.retrieve_top_n(context, list(persona.values()), retriever_number)
        # narrow_knowledge = []
        # for per in retrieved_persona:
        #     each_knowledge = p_number[per]
        #     narrow_knowledge.extend(list(knowledge[each_knowledge].values()))

        # retrieved_knowledge = retriever.retrieve_top_n(context, narrow_knowledge, retriever_number)
        retrieved_persona = [per.replace(" ", "") for per in retrieved_persona]
        # retrieved_knowledge = [know.replace(" ", "") for know in retrieved_knowledge]
        retrieved_results = retrieved_persona # + retrieved_knowledge

        query = context + "[MIDDLE] " + "，".join(retrieved_results) + " [EOM]"    # 这里加不加系统，有没有130001, 130004
        res, history = model.chat(tokenizer=tokenizer, query=query + "系统: ", max_length=512)

        # postprocess
        for special_token in SPECIAL_TOKENS["additional_special_tokens"]:
            res = res.replace(special_token, "")
        final_resp = res.split("系统：")[1].strip() if "系统：" in res else res.strip()


        # if retriever_type != "ground_truth":
        #     with torch.autocast("cuda"):
        #         res, history = model.chat(tokenizer=tokenizer, query=context, max_length=512)
        #         # print("after step 1: {}".format(res))
        #         res = ground_sources if using_ground_source else res

        #         # retrieve
        #         if "PERSONA" in res and "KNOWLEDGE" in res:
        #             source = "PERSONA KNOWLEDGE"
        #             retrieved_persona = retriever.retrieve_top_n(context, list(persona.values()), retriever_number)

        #             # narrow_knowledge = []
        #             # for per in retrieved_persona:
        #             #     each_knowledge = p_number[per]
        #             #     narrow_knowledge.extend(list(knowledge[each_knowledge].values()))

        #             # without dependency
        #             narrow_knowledge = []
        #             for per in p_number.keys():
        #                 each_knowledge = p_number[per]
        #                 narrow_knowledge.extend(list(knowledge[each_knowledge].values()))
        #             retrieved_knowledge = retriever.retrieve_top_n(context + retrieved_persona[0].replace(" ", ""), narrow_knowledge, retriever_number)
                    
        #             # retrieved_knowledge = retriever.retrieve_top_n(context, narrow_knowledge, retriever_number)
        #             retrieved_persona = [per.replace(" ", "") for per in retrieved_persona]
        #             retrieved_knowledge = [know.replace(" ", "") for know in retrieved_knowledge]
        #             retrieved_results = retrieved_persona + retrieved_knowledge
        #         elif "PERSONA" in res:
        #             source = "PERSONA"
        #             retrieved_persona = retriever.retrieve_top_n(context, list(persona.values()), retriever_number)
        #             retrieved_persona = [per.replace(" ", "") for per in retrieved_persona]
        #             retrieved_results = retrieved_persona
        #         else:
        #             source = "NULL"

        #         # analysis of decisions
        #         source_type = "both" if source == "PERSONA KNOWLEDGE" else source.lower()
        #         analysis_result["decisions"][source_type]["freq"] += 1
        #         if source != ground_sources:
        #             ground_sources_type = "both" if ground_sources == "PERSONA KNOWLEDGE" else ground_sources.lower()
        #             analysis_result["decisions"][source_type][ground_sources_type] += 1
                
        #         # analysis of retrieved results with ground sources
        #         if ground_sources == "PERSONA":
        #             analysis_result["middle"]["persona"]["freq"] += 1
        #             correct_persona = sum([1 for per in retrieved_persona if per in middle_results])
        #             if correct_persona > 0:
        #                 analysis_result["middle"]["persona"]["correct"] += 1
        #         elif ground_sources == "PERSONA KNOWLEDGE":
        #             analysis_result["middle"]["both"]["freq"] += 1
        #             correct_persona = sum([1 for per in retrieved_persona if per in middle_results])
        #             if correct_persona > 0:
        #                 analysis_result["middle"]["both"]["correct_persona"] += 1
                    
        #             correct_knowledge = sum([1 for per in retrieved_knowledge if per in middle_results])
        #             if correct_knowledge > 0:
        #                 analysis_result["middle"]["both"]["correct_knowledge"] += 1

        #         # assemble final input
        #         if source == "NULL":
        #             step_2_prompt = context + "[SOURCE] " + source + " [EOS]" + "[MIDDLE] " + " [EOM]"
        #         else:
        #             step_2_prompt = context + "[SOURCE] " + source + " [EOS]" + "[MIDDLE] " + "，".join(retrieved_results) + " [EOM]"    # 这里加不加系统，有没有130001, 130004
        #         res, history = model.chat(tokenizer=tokenizer, query=step_2_prompt + "系统: ", max_length=512)

        #         # postprocess
        #         for special_token in SPECIAL_TOKENS["additional_special_tokens"]:
        #             res = res.replace(special_token, "")
        #         final_resp = res.split("系统：")[1].strip() if "系统：" in res else res.strip()
        # else:
        #     with torch.autocast("cuda"):
        #         res, history = model.chat(tokenizer=tokenizer, query=context, max_length=512)
        #         # print("after step 1: {}".format(res))

        #         # retrieve
        #         if "PERSONA" in res and "KNOWLEDGE" in res:
        #             source = "PERSONA KNOWLEDGE"
        #         elif "PERSONA" in res:
        #             source = "PERSONA"
        #         else:
        #             source = "NULL"

        #         # assemble final input
        #         if using_ground_source:
        #             step_2_prompt = context + "[SOURCE] " + ground_sources + " [EOS]" + "[MIDDLE] " + middle_results + " [EOM]"    # 这里加不加系统，有没有130001, 130004
        #         else:
        #             if source == "NULL":
        #                 step_2_prompt = context + "[SOURCE] " + source + " [EOS]" + "[MIDDLE] " + " [EOM]"
        #             else:
        #                 step_2_prompt = context + "[SOURCE] " + source + " [EOS]" + "[MIDDLE] " + middle_results + " [EOM]"    # 这里加不加系统，有没有130001, 130004
                
        #         res, history = model.chat(tokenizer=tokenizer, query=step_2_prompt + "系统: ", max_length=512)

        #         # postprocess
        #         for special_token in SPECIAL_TOKENS["additional_special_tokens"]:
        #             res = res.replace(special_token, "")
        #         final_resp = res.split("系统：")[1].strip() if "系统：" in res else res.strip()
        #         # print("after step2: {}".format(final_resp))

                
                
        ins["context"] = context
        # ins["used_source"] = source if not using_ground_source else ground_sources
        ins["used_source"] = "PERSONA"
        ins["used_persona"] = retrieved_persona[0] if retrieved_persona else ""
        # ins["used_knowledge"] = retrieved_knowledge[0] if retrieved_knowledge else ""
        ins["ground_persona"] = ground_persona
        # ins["ground_knowledge"] = ground_knowledge
        ins["resp"] = final_resp[:128]
        results.append(ins)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(analysis_output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=4, ensure_ascii=False)