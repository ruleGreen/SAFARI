import os
import json

class Dataset:
    def __init__(self, setting) -> None:
        self.setting = setting
    
    def output_prompt(self, output_path):
        assert len(self.prompts) > 0
        # 如果输出目录不存在，则创建目录
        if '/' in output_path:
            output_dir = '/'.join(output_path.split('/')[:-1])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print("[CREATE DIRECTORY]", output_dir)
        with open(output_path, "w", encoding="utf-8") as f:
            for prompt in self.prompts:
                json.dump(prompt, f, ensure_ascii=False)
                f.write("\n")
        self.prompts = []

class KBP(Dataset):
    def __init__(self, paths="", demo_path="", mode=""):
        self.paths = paths
        self.demo_path = demo_path
        self.mode = mode
        self.examples = self.load_data(paths, mode)
        self.prompts = []

    def assemble_prompt(self, prompt="", prefix="", suffix="", dialogue="", description="", setting=""):
        instance = {}
        instance["prompt"] = prefix + prompt + suffix
        instance["context"] = dialogue
        if description:
            instance["description"] = description
        if setting == "in-context":
            instance["demo"] = self.demo_pools["c,p,k->r:demos"]
        self.prompts.append(instance)

    def load_demo(self, path):
        with open(path, 'r', encoding="utf-8") as f:
            return json.load(f)

    def construct_prompt(self, template="", setting="zero-shot"):
        if setting == "in-context":
            self.demo_pools = self.load_demo(self.demo_path)
            self.demo = self.demo_pools["c->p,k:demos"]
        
        for example in self.examples:
            context, sources, middle_results, ground_resp = example["context"], example["resources"], example["middle_result"], example["resp"]
            persona, knowledge = example["persona"], example["knowledge"]
            instance = {}
            instance["prompt"] = template.format(dialogue_history=context)
            instance["context"] = context
            instance["persona"] = persona
            instance["sources"] = sources
            instance["middle"] = middle_results
            if setting == "in-context":
                instance["demo"] = self.demo
            instance["knowledge"] = knowledge
            self.prompts.append(instance)
    
    def load_data(self, files, mode: str):
        examples = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                raw_dialogues = json.load(f)

            for sample in raw_dialogues:
                # regarding persona and knowledge as two different databases
                persona_database = sample["persona"]
                knowledge_database = sample["persona_kg"]
                # construct training examples
                context, history = "", ""
                for turn in sample["conversation"]:
                    user, sys, p_k = turn["U"], turn["S"], turn["P-K"]
                    each_utter = "用户：" + user + "\n" + "系统：" + sys + "\n"
                    context = history + "用户：" + user + "\n"
                    history += each_utter
                    commands, middle_results = "", ""
                    if len(p_k) == 0: # 没有使用
                        commands = "NULL"
                        ground_persona, ground_knowledge = [], []
                    elif len(p_k) == 1: # 人设
                        commands = "PERSONA"
                        middle_results = [persona_database[persona].replace(" ", "") for persona in p_k[0]]
                        ground_persona = [persona_database[persona].replace(" ", "") for persona in p_k[0]]
                        ground_knowledge = []
                    elif len(p_k) == 2: # 既有人设又有知识
                        commands = "PERSONA KNOWLEDGE"
                        middle_results = [persona_database[persona].replace(" ", "") for persona in p_k[0]]
                        ground_persona = [persona_database[persona].replace(" ", "") for persona in p_k[0]]
                        ground_knowledge = []
                        for per_knowledge in p_k[1]:
                            per, know = per_knowledge.split("-")
                            ground_knowledge.append(knowledge_database[per][know].replace(" ", ""))
                            middle_results.append(knowledge_database[per][know].replace(" ", ""))
                    
                    ins = {
                        "context": context,
                        "resources": commands,
                        "middle_result": "，".join(middle_results),
                        "resp": sys,
                        "used_persona": ground_persona,
                        "used_knowledge": ground_knowledge,
                        "persona": persona_database,
                        "knowledge": knowledge_database
                    }
                    examples.append(ins)
        return examples