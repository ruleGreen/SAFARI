import os
import json

UNWANTED_STRINGS = [
    "Answer:",
    "Inferences:",
    "<s>",
    "</s>",
    "<pad>"
]

def read_json(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_jsonl(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def save_jsonl(filename, data):
    with open(filename, "w") as f:
        for line in data:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

def read_prompt_output(prompt_path, model_type):
    result = {}
    with open(prompt_path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            if model_type == "chatgpt":
                output = data["choices"][0]["message"]["content"]
                output = output.strip().split("\n\n用户:")[0] # 对回复进行处理，不需要后面的
                prompt = data["prompt"]
            elif "chatglm" in model_type or "belle" in model_type:
                output = data["response"]
                if "sources_type" in prompt_path:
                    output = output.split("\n\n")[0] if "\n\n" in output else output.split("\n")[0]
                output = output.strip().split("\n\n用户:")[0]
                prompt = data["prompt"]
            elif "alpaca" in model_type or "vicuna" in model_type:
                output = data["response"]
                prompt = data["prompt"]
            else:
                output = data["response"]
                prompt = data["prompt"]

            # 对output进行一些处理
            output = postprocess(output, prompt_path)

            result[prompt] = output
    return result

def postprocess(text, prompt_path):
    for word in UNWANTED_STRINGS:
        text = text.replace(word, "")

    if "status_result" in prompt_path:
        return text.replace("\n\n", " ").replace("\n", " ").replace("推测：", "").strip()
    return text.strip()

def read_true_persona(persona_path):
    result = {}
    with open(persona_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            persona = "，".join(list(data["persona"].values())).replace(" ", "")
            knowledge = ""
            for p, p_k in data["knowledge"].items():
                knowledge += "".join(p_k.values()).replace(" ", "")
            result[data["prompt"]] = persona + knowledge
    return result

def read_ture_dialogue(persona_path):
    result = {}
    with open(persona_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            result[data["prompt"]] = {}
            result[data["prompt"]]["dialogue"] = data["context"]  # data["dialogue"]
            if "description" in data:
                result[data["prompt"]]["description"] = data["description"]
    return result

def mapping_input_output(input_path, output_path, model_type):
    out = read_prompt_output(output_path, model_type)  # {prompt: output}
    inp = read_ture_dialogue(input_path) # {prompt: dialogue}
    # 由于input和output不是等长的 len(input) != len(output)
    # invert_inp = {v:k for k, v in inp.items()} # {dialogue:prompt}
    result = {}
    for pro, sample in inp.items():
        if pro.strip() not in out:
            continue
        dia = sample["dialogue"]
        result[dia] = out[pro.strip()]
    return result  # {dialogue:output}

def mapping_query_with_background(input_path):
    inp = read_ture_dialogue(input_path) # {prompt: dialogue}
    result = {}
    for pro, sample in inp.items():
        dia = sample["dialogue"]
        desc = sample["description"]
        result[dia] = desc
    return result  # {dialogue:desc}


def get_grounded_truth_response(data_path):
    data_raw = read_jsonl(data_path)
    result = {}
    for sample in data_raw:
        result[sample["dialogue"]] = sample["response"]
    return result

def is_finished_all_prompts(input_path, output_path):
    if not os.path.exists(output_path):
        return False
    prompt_raw = read_jsonl(input_path)
    output_raw = read_jsonl(output_path)
    finished_prompts = [sample["prompt"] for sample in output_raw]
    all_prompts = [sample["prompt"].strip() for sample in prompt_raw]

    if "pairwise" in input_path:
        return  len(set(all_prompts)) - len(set(finished_prompts)) < 10
    if "one-shot" in input_path: # for d4
        return  len(set(all_prompts)) - len(set(finished_prompts)) < 20
    
    return set(finished_prompts) == set(all_prompts)

def is_duplicate_input(input_path):
    if os.path.exists(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            inp_raw = [json.loads(line) for line in f.readlines()]
        
        if len(inp_raw) != 0:
            return True
    return False