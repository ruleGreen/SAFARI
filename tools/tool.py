# coding: utf-8
from glob import glob
import os
import pandas as pd
import shutil
from itertools import chain
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
import random

def read_json(x: str):
    try:
        data = pd.read_json(x)
        return data
    except Exception as e:
        return pd.DataFrame()

def read_data_dir():
    target_dir_list = ['alpaca_chinese_dataset/其他中文问题补充/',
                    'alpaca_chinese_dataset/翻译后的中文数据/',
                    'alpaca_chinese_dataset/chatglm问题数据补充/',
                    #    'alpaca_chinese_dataset/原始英文数据/'
                    ]

    all_json_path = [glob(i + "*.json") for i in target_dir_list]
    all_json_path = list(chain(*all_json_path))
    return all_json_path

def chunk_data(alldata, genrate_data_dir):
    genrate_data_dir = Path(genrate_data_dir)

    if genrate_data_dir.exists():
        shutil.rmtree(genrate_data_dir, ignore_errors=True)

    os.makedirs(genrate_data_dir, exist_ok=True)

    alldata = alldata.sample(frac=1).reset_index(drop=True)

    chunk_size = 666

    for index, start_id in tqdm(enumerate(range(0, alldata.shape[0], chunk_size))):
        temp_data = alldata.iloc[start_id:(start_id + chunk_size)]
        temp_data.to_csv(genrate_data_dir.joinpath(f"{index}.csv"), index=False)


def get_dataset(genrate_data_dir):
    genrate_data_dir = Path(genrate_data_dir)
    all_file_list = glob(pathname=genrate_data_dir.joinpath("*.csv").__str__())

    test_file_list = random.sample(all_file_list, int(len(all_file_list) * 0.25))
    train_file_list = [i for i in all_file_list if i not in test_file_list]
    train_file_list, test_file_list = train_file_list[:5], test_file_list[:5]

    print(len(train_file_list), len(test_file_list))

    dataset = load_dataset(
        "csv",
        data_files={
            'train': train_file_list,
            'valid': test_file_list
        },
        cache_dir="cache_data"
    )
    return dataset