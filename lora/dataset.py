# coding:utf-8
import json
from typing import List, Optional
from datasets.info import DatasetInfo
from datasets.splits import NamedSplit
from datasets.table import Table
import torch
from tqdm import tqdm
import numpy as np
from transformers import HfArgumentParser
from torch.utils.data import Dataset

def preprocess(example, tokenizer, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target, max_length=max_seq_length, truncation=True, add_special_tokens=False
    )
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def filter_nan(example):
    return example['target'] is not None and example['context'] is not None


def get_masks_and_position_ids(
        seq, seq_len, input_lenth, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
            seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    attention_mask = torch.ones((1, input_lenth, input_lenth), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        position_ids = torch.arange(input_lenth, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                torch.zeros(seq_length, dtype=torch.long, device=device),
                torch.arange(
                    input_lenth - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(input_lenth, dtype=torch.long, device=device)
        if not gmask:
            position_ids[input_lenth - 1:] = mask_position
    return attention_mask, position_ids


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    # {"context": context, "target": target}
    example['context'] = context
    example['target'] = target
    return example


class KBP(Dataset):
    def __init__(self, paths, tokenizer, mode="train", model_type="chatglm"):
        self.mode = mode
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.examples = self.load_data(paths, mode)
        self.examples_ids = self.process_example()

    def process_example(self):
        example_ids = []
        for example in self.examples:
            context, sources, middle_results, resp = example["context"], example["resources"], example["middle_result"], example["resp"]

            max_seq_length = 512

            if self.model_type == "chatglm":
                context_ids = self.tokenizer.encode(context, max_length=max_seq_length, truncation=True)
                source_ids = self.tokenizer.encode("[SOURCE] " + sources + " [EOS]", add_special_tokens=True)
                middle_ids = self.tokenizer.encode(text="[MIDDLE] " + middle_results + " [EOM]", max_length=max_seq_length, truncation=True, add_special_tokens=True)
                resp_ids = self.tokenizer.encode(text=resp, add_special_tokens=False)
            elif self.model_type == "belle-llama":
                context_ids = self.tokenizer.encode("Human: \n" + context + "\n\nAssistant: \n", max_length=max_seq_length, truncation=True)
                source_ids = self.tokenizer.encode("[SOURCE] " + sources + " [EOS]", add_special_tokens=False)
                middle_ids = self.tokenizer.encode(text="[MIDDLE] " + middle_results + " [EOM]", max_length=max_seq_length, truncation=True, add_special_tokens=False)
                resp_ids = self.tokenizer.encode(text=resp, add_special_tokens=False)
            
            example_ids.append({
                "context_ids": context_ids,
                "source_ids": source_ids,
                "middle_ids": middle_ids,
                "resp_ids": resp_ids
            })
        return example_ids

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
                        commands = "[NULL]" if self.model_type == "belle-llama" else "NULL"
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
                        "resp": "系统: " + sys,
                        "used_persona": ground_persona,
                        "used_knowledge": ground_knowledge,
                        "persona": persona_database,
                        "knowledge": knowledge_database
                    }
                    examples.append(ins)
        return examples

    def __getitem__(self, idx) -> List:
        return self.examples_ids[idx]
    
    def __len__(self):
        return len(self.examples)
    
    def collate_fn(self, batch):
        # len_ids = [len(sample["context_ids"]) + len(sample["source_ids"]) + len(sample["middle_ids"]) + len(sample["resp_ids"]) + 1 for sample in batch]
        # longest = max(len_ids)

        len_ids = [len(sample["context_ids"]) + len(sample["middle_ids"]) + len(sample["resp_ids"]) + 1 for sample in batch]
        longest = max(len_ids)

        input_ids = []
        attention_mask_list = []
        position_ids_list = []
        labels_list = []

        for ids_l, sample in sorted(zip(len_ids, batch), key=lambda x: -x[0]):
            context_ids, context_len = sample["context_ids"], len(sample["context_ids"])
            resource_ids, resource_len = sample["source_ids"], len(sample["source_ids"])
            middle_ids, middle_len = sample["middle_ids"], len(sample["middle_ids"])
            resp_ids, resp_len = sample["resp_ids"], len(sample["resp_ids"])
            
            # ids = context_ids + resource_ids + middle_ids + resp_ids + [self.tokenizer.eos_token_id]
            # labels = (
            #         [-100] * (context_len - 1)
            #         + ids[context_len - 1 : context_len + resource_len]
            #         + [-100] * middle_len
            #         + resp_ids
            #         + [self.tokenizer.eos_token_id]
            #         + [-100] * (longest - ids_l)
            # ) # labels这里不需要偏置，计算loss的自动了
            # ids = ids + [self.tokenizer.eos_token_id] * (longest - ids_l)

            # for ablation study
            ids = context_ids + middle_ids + resp_ids + [self.tokenizer.eos_token_id]
            labels = (
                    [-100] * (context_len + middle_len - 1)
                    + ids[context_len + middle_len - 1: context_len + middle_len + resp_len]
                    + [self.tokenizer.eos_token_id]
                    + [-100] * (longest - ids_l)
            ) # labels这里不需要偏置，计算loss的自动了
            ids = ids + [self.tokenizer.eos_token_id] * (longest - ids_l)

            assert len(ids) == len(labels)
            _ids = torch.LongTensor(ids)
            attention_mask, position_ids = get_masks_and_position_ids(
                ids, context_len, longest, _ids.device, gmask=False
            )
            labels_list.append(torch.LongTensor(labels))
            input_ids.append(_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
        
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        attention_mask = torch.stack(attention_mask_list)
        position_ids = torch.stack(position_ids_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
    
    def collate_fn_llama(self, batch):
        len_ids = [len(sample["context_ids"]) + len(sample["source_ids"]) + len(sample["middle_ids"]) + len(sample["resp_ids"]) + 1 for sample in batch]
        longest = max(len_ids)

        input_ids = []
        attention_mask_list = []
        labels_list = []

        for ids_l, sample in sorted(zip(len_ids, batch), key=lambda x: -x[0]):
            context_ids, context_len = sample["context_ids"], len(sample["context_ids"])
            resource_ids, resource_len = sample["source_ids"], len(sample["source_ids"])
            middle_ids, middle_len = sample["middle_ids"], len(sample["middle_ids"])
            resp_ids, resp_len = sample["resp_ids"], len(sample["resp_ids"])
            
            ids = context_ids + resource_ids + middle_ids + resp_ids + [self.tokenizer.eos_token_id]
            labels = (
                    [-100] * context_len  # add -1
                    + ids[context_len : context_len + resource_len]
                    + [-100] * middle_len
                    + resp_ids
                    + [self.tokenizer.eos_token_id]
                ) # labels这里不需要偏置，计算loss的自动了
            # left padding
            ids = [self.tokenizer.pad_token_id] * (longest - ids_l) + ids
            labels = [-100] * (longest - ids_l) + labels
            assert len(ids) == len(labels)
            _ids = torch.LongTensor(ids)
            attention_mask = _ids.ne(self.tokenizer.pad_token_id)
            labels_list.append(torch.LongTensor(labels))
            input_ids.append(_ids)
            attention_mask_list.append(attention_mask)
        
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        attention_mask = torch.stack(attention_mask_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
    

class INDust(Dataset):
    def __init__(self, paths, tokenizer, mode="train", model_type="belle-llama"):
        self.mode = mode
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.examples = self.load_data(paths, mode)
        self.examples_ids = self.process_example()

    def process_example(self):
        example_ids = []
        for example in self.examples:
            context, resp = example["context"], example["response"]

            max_seq_length = 512

            if self.model_type == "chatglm":
                context_ids = self.tokenizer.encode(context, max_length=max_seq_length, truncation=True)
                resp_ids = self.tokenizer.encode(text=resp, add_special_tokens=False)
            elif self.model_type == "belle-llama":
                context_ids = self.tokenizer.encode("Human: \n" + context + "\n\nAssistant: \n", max_length=max_seq_length, truncation=True)
                resp_ids = self.tokenizer.encode(text=resp, add_special_tokens=False)
            
            example_ids.append({
                "context_ids": context_ids,
                "resp_ids": resp_ids
            })
        return example_ids

    def load_data(self, files, mode: str):
        examples = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                if "HADust" in file:
                    raw_dialogues = json.load(f)
                else:
                    raw_dialogues = [json.loads(line) for line in f.readlines()]

            for sample in raw_dialogues:
                context = sample["prompt"]
                resp = sample["response"]
                    
                ins = {
                    "context": context,
                    "response": resp[:256]
                }
                examples.append(ins)
        return examples

    def __getitem__(self, idx) -> List:
        return self.examples_ids[idx]
    
    def __len__(self):
        return len(self.examples)
    
    def collate_fn(self, batch):
        len_ids = [len(sample["context_ids"]) + len(sample["resp_ids"]) + 1 for sample in batch]
        longest = max(len_ids)

        input_ids = []
        attention_mask_list = []
        position_ids_list = []
        labels_list = []

        for ids_l, sample in sorted(zip(len_ids, batch), key=lambda x: -x[0]):
            context_ids, context_len = sample["context_ids"], len(sample["context_ids"])
            resp_ids, resp_len = sample["resp_ids"], len(sample["resp_ids"])
            
            ids = context_ids + resp_ids + [self.tokenizer.eos_token_id]
            labels = (
                    [-100] * (context_len - 1)
                    + ids[context_len - 1: context_len + resp_len]
                    + [self.tokenizer.eos_token_id]
                    + [-100] * (longest - ids_l)
            ) # labels这里不需要偏置，计算loss的自动了
            ids = ids + [self.tokenizer.eos_token_id] * (longest - ids_l)

            assert len(ids) == len(labels)
            _ids = torch.LongTensor(ids)
            attention_mask, position_ids = get_masks_and_position_ids(
                ids, context_len, longest, _ids.device, gmask=False
            )
            labels_list.append(torch.LongTensor(labels))
            input_ids.append(_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
        
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        attention_mask = torch.stack(attention_mask_list)
        position_ids = torch.stack(position_ids_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }