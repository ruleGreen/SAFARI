# coding: utf-8
import json
import sys; sys.path.append('./')
import argparse
from dataset import *
from tools.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="kbp", help="the name of dataset")
    parser.add_argument("--model_type", type=str, default="chatgpt", help="the models used including chatglm-6b, chatgpt, flan-t5, flan-ul2")
    parser.add_argument("--model_path", type=str, default="/kfdata03/kf_grp/hrwang/LLM/models/chinese/", help="the path of stored pre-trained model")
    parser.add_argument("--retriever_type", type=str, default="dpr", help="the type of different retriever to retrieve")
    parser.add_argument("--retriever_number", type=int, default=1, help="the number of middle results for each sources")
    parser.add_argument("--setting", type=str, default="in-context", help="the settings of the exp including zero-shot and one-shot")
    parser.add_argument("--lang", type=str, default="chinese", help="the language of datasets and prompts")
    args = parser.parse_args()

    source_input_path =  "./dialogue_input/kbp/" + args.model_type + "/get_sources_type_" + args.setting + ".json"
    source_output_path = "./dialogue_output/kbp/" + args.model_type + "/dialogue_sources_type_"  + args.setting + ".json"
    source_result = mapping_input_output(source_input_path, source_output_path, model_type=args.model_type)

    dialogue_input_path = "./dialogue_input/kbp/" + args.model_type + "/get_" + args.setting + "_" + args.retriever_type + "_top-" + str(args.retriever_number) + "_responses.json"
    dialogue_output_path = "./dialogue_output/kbp/" + args.model_type + "/dialogue_response_"  + args.setting + "_" + args.retriever_type  + "_top-" + str(args.retriever_number) + ".json"

    result = mapping_input_output(dialogue_input_path, dialogue_output_path, model_type=args.model_type)
    data_paths = ["./dataset/kbp/test.json"]
    test_dataset = KBP(data_paths)

    outputs = []
    for example in test_dataset.examples:
        context, ground_sources, middle_results, ground_resp = example["context"], example["resources"], example["middle_result"], example["resp"]
        ground_persona, ground_knowledge = example["used_persona"], example["used_knowledge"]
        
        assert context in result
        assert context in source_result
        decision = source_result[context]
        if "PERSONA" in decision and "KNOWLEDGE" in decision:
            source = "PERSONA KNOWLEDGE"
        elif "PERSONA" in decision:
            source = "PERSONA"
        else:
            source = 'NULL'
        
        ins = {
            "context": context,
            "ground_persona": ground_persona,
            "ground_knowledge": ground_knowledge,
            "used_source": source,
            "resp": result[context][:128]
        }
        outputs.append(ins)
    
    output_path = "./unsupervised/" + args.model_type + "_" + args.retriever_type + "_" + str(args.retriever_number) + "_" + args.setting + ".json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)