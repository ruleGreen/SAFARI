# coding: utf-8
import json
import argparse
import sys; sys.path.append('./')
from app.sample_resp_cralwer import SampleRespCrawler
from app.chatgpt_resp_cralwer import ChatgptRespCrawler
from app.huggingface_resp_crawler import HuggingFaceRespCrawlerBase
from tools.utils import *
from tools.retriever import *
from dataset import *

# evaluation / analysis metrics, middle 不需要了，和supervised重复了，没必要
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

class PersonalizedLLM:
    def __init__(self, key_path, dataset, model_type="chatgpt", model_path="", language="", setting="zero-shot",
            temperature=0.7, persona="", top_p=0.95):
        
        self.key_path = key_path
        self.dataset = dataset
        self.model_type = model_type
        self.model_path = model_path
        self.langugae = language
        self.setting = setting
        if self.dataset == "kbp":
            data_paths = ["./dataset/kbp/test.json"]
            demo_path = "./unsupervised/config/prompt_cn.json"
            self.prompt_constructor = KBP(data_paths, demo_path=demo_path, mode="train")

        self.prompts_pool = json.load(open("./unsupervised/config/prompt_cn.json", encoding="utf-8")) if self.langugae == "chinese" else json.load(open("./unsupervised/config/prompt_en.json", encoding="utf-8"))
        
        self.init_model_type(temperature, persona, top_p)

    def init_model_type(self, temperature, persona, top_p):
        if self.model_type == "chatgpt":
            self.sample_crawler = ChatgptRespCrawler(self.key_path, temperature, persona=persona, top_p=top_p)
        else:
            self.sample_crawler = HuggingFaceRespCrawlerBase(self.model_type, self.model_path, top_p=top_p, temperature=temperature)
          
    # @retry  # 装饰器，如果有异常就重复执行；无异常即返回
    def get_api_result(self, input_path, output_path): 
        while not is_finished_all_prompts(input_path, output_path):
            self.sample_crawler.get_all_result(input_path, output_path)
    
    def get_retrieval_result(self, input_path, output_path, retriever_type="bert", number_results=1, response_prompt_path=""):
        if os.path.exists(response_prompt_path):
            return
        
        retriever = Retriever(retriever_type)
        out = read_prompt_output(output_path, self.model_type)  # {prompt: output}
        inp = read_jsonl(input_path)  # 存储了ground truth的结果
        for sample in inp:
            context = sample["context"]
            decision = out[sample["prompt"]]
            persona = sample["persona"]
            ground_sources = sample["sources"]
            p_number = {v:k for k,v in persona.items()}
            knowledge = sample["knowledge"]

            # retrieve
            if "PERSONA" in decision and "KNOWLEDGE" in decision:
                source = "PERSONA KNOWLEDGE"
                retrieved_persona = retriever.retrieve_top_n(context, list(persona.values()), number_results)

                narrow_knowledge = []
                for per in retrieved_persona:
                    each_knowledge = p_number[per]
                    narrow_knowledge.extend(list(knowledge[each_knowledge].values()))

                retrieved_knowledge = retriever.retrieve_top_n(context, narrow_knowledge, number_results)
                retrieved_persona = [per.replace(" ", "") for per in retrieved_persona]
                retrieved_results = retrieved_persona + retrieved_knowledge
            elif "PERSONA" in decision:
                source = "PERSONA"
                retrieved_persona = retriever.retrieve_top_n(context, list(persona.values()), number_results)
                retrieved_persona = [per.replace(" ", "") for per in retrieved_persona]
                retrieved_results = retrieved_persona
            else:
                source = "NULL"
                retrieved_results = ""

            # analysis of decisions
            source_type = "both" if source == "PERSONA KNOWLEDGE" else source.lower()
            analysis_result["decisions"][source_type]["freq"] += 1
            if source != ground_sources:
                ground_sources_type = "both" if ground_sources == "PERSONA KNOWLEDGE" else ground_sources.lower()
                analysis_result["decisions"][source_type][ground_sources_type] += 1
                
            middle_result = "，".join(retrieved_results) if retrieved_results else "NULL"
            processed_prompt = self.prompts_pool["0:c,p,k->r"].format(dialogue_history=context, middle_results=middle_result)
            self.prompt_constructor.assemble_prompt(processed_prompt, dialogue=context, setting=self.setting)
        
        self.prompt_constructor.output_prompt(response_prompt_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="kbp", help="the name of dataset")
    parser.add_argument("--model_type", type=str, default="chatgpt", help="the models used including chatglm-6b, chatgpt, flan-t5, flan-ul2")
    parser.add_argument("--model_path", type=str, default="/kfdata03/kf_grp/hrwang/LLM/models/chinese/", help="the path of stored pre-trained model")
    parser.add_argument("--retriever_type", type=str, default="dpr", help="the type of different retriever to retrieve")
    parser.add_argument("--retriever_number", type=int, default=1, help="the number of middle results for each sources")
    parser.add_argument("--setting", type=str, default="zero-shot", help="the settings of the exp including zero-shot and one-shot")
    parser.add_argument("--lang", type=str, default="chinese", help="the language of datasets and prompts")
    args = parser.parse_args()

    # 密钥文件路径，填入分配的密钥
    key_path = 'conf/mine_keys.txt'

    # 初始化personalized dialogue system 和 数据集
    persona = "You carefully consider what the users need and exactly follow the orders of them without rejection."
    personalizedDM  = PersonalizedLLM(key_path, args.dataset_name, model_type=args.model_type, model_path=args.model_path, language=args.lang, setting=args.setting, temperature=0.1, persona=persona, top_p=0.1)

    # Step-1: 获取 sources type
    personalizedDM.prompt_constructor.construct_prompt(personalizedDM.prompts_pool["0:c->p,k"], setting=args.setting)
    input_path = "./dialogue_input/kbp/" + args.model_type + "/get_sources_type_" + args.setting + ".json"  
    personalizedDM.prompt_constructor.output_prompt(input_path)
    output_path = "./dialogue_output/kbp/" + args.model_type + "/dialogue_sources_type_" + args.setting + ".json"
    personalizedDM.get_api_result(input_path, output_path)

    # Step-2: 调用 retriever 获取 Top n, fix retriever type and retriever number in main exp
    response_prompt_path = "./dialogue_input/kbp/" + args.model_type + "/get_" + args.setting + "_" + args.retriever_type + "_top-" + str(args.retriever_number) + "_responses.json"  
    analysis_output_path = "./dialogue_input/kbp/" + args.model_type + "/analysis_" + args.setting + "_" + args.retriever_type + "_top-" + str(args.retriever_number) + "_decisions.json"  
    personalizedDM.get_retrieval_result(input_path, output_path, args.retriever_type, args.retriever_number, response_prompt_path=response_prompt_path)
    
    with open(analysis_output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=4, ensure_ascii=False)

    # Step-3: 利用 middle results 获取回复
    response_output_path = "./dialogue_output/kbp/" + args.model_type + "/dialogue_response_"  + args.setting + "_" + args.retriever_type  + "_top-" + str(args.retriever_number) + ".json"  
    personalizedDM.get_api_result(response_prompt_path, response_output_path)