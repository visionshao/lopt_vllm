import os
import json
from transformers import AutoTokenizer, BatchEncoding


_tokenizer = AutoTokenizer.from_pretrained(r"../models/Qwen2.5-7B", trust_remote_code=True, use_fast=True)
# 8192, 16384, 32768, 131072 
MAX_LEN=8192
MAX_LEN=65536
MAX_LEN=131072
MAX_LEN=32768
MAX_LEN=16384

def get_certain_length(text, seq_len=-1):
    processed_prompt = text
    if seq_len > 0:
        max_seq_len = seq_len
        input_ids = _tokenizer.encode(text)
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len // 2] + input_ids[-max_seq_len // 2:]
            processed_prompt = _tokenizer.decode(input_ids, skip_special_tokens=True)
        else:
            # 填充：重复文本直到达到目标长度（仅为测试，生产不推荐）
            repeat_times = max_seq_len // len(input_ids) + 1
            processed_prompt = (text + " ") * repeat_times
            
    return processed_prompt

def get_longbenchv2_data(dataset_path):
    dataset = json.load(open(dataset_path + "/data.json", "r", encoding="utf-8"))
    dataset_len = len(dataset)
    print(f"The length of LongBenchV2 is {dataset_len}")
    data = [dataset[i]["context"] for i in range(dataset_len)]
    return data


def get_clongeval_data(dataset_path, data_type="medium"):
    
    sub_data_dirs = os.listdir(dataset_path)
    data = []
    for sub_data_dir in sub_data_dirs:
        sub_data_dir_path = os.path.join(dataset_path, sub_data_dir, data_type+".jsonl")
        data_lines = open(sub_data_dir_path)
        for data_line in data_lines:
            data_record = json.loads(data_line)
            data.append(data_record["context"])
    print(f"The length of CLongEval is {len(data)}")
    return data


def get_leval_data(dataset_path):
    sub_data_dirs = os.listdir(dataset_path)
    data = []
    for sub_data_dir in sub_data_dirs:
        sub_data_dir_path = os.path.join(dataset_path, sub_data_dir)
        for item in os.listdir(sub_data_dir_path):
            item_path = os.path.join(sub_data_dir_path, item)
            data_lines = open(item_path)
            for data_line in data_lines:
                data_record = json.loads(data_line)
                data.append(data_record["input"])
    print(f"The length of LEval is {len(data)}")
    return data


def get_data(dataset):
    dataset_dir = r"/home/shaowei/projects/lopt_vllm/lopt/datasets"
    output_dataset_dir = r"/home/shaowei/projects/lopt_vllm/datasets"
    print(dataset)
    if dataset == "LongBenchV2":
        data = get_longbenchv2_data(f"{dataset_dir}/{dataset}")
        output_file = os.path.join(output_dataset_dir, dataset + f"_{MAX_LEN}" + ".jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
                for text in data:
                    # 构造字典并用 json.dumps 确保正确转义特殊字符
                    processed_text = get_certain_length(text, MAX_LEN)
                    line = json.dumps({"prompt": processed_text}, ensure_ascii=False)
                    f.write(line + '\n')
        print(f"已保存 {len(data)} 条数据到 {output_file}")
        return data
    elif dataset == "ClongEval":
        data = get_clongeval_data(f"{dataset_dir}/{dataset}")
        return data
    elif dataset == "LEval":
        data = get_leval_data(f"{dataset_dir}/{dataset}")
        return data
    else:
        print("No valid dataset!")
        return None

get_data("LongBenchV2")