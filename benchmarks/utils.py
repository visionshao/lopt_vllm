from typing import Iterable, List, Sequence, Union
import json
import os
import numpy as np
import torch
    

def chunks(sentence: Union[str, Sequence[str]], chunk_size: int = 40960, overlap_length: int = 512):
    """
    Splits a string or a sequence of strings into chunks with a specified size and overlap.

    Parameters:
    - sentence (Union[str, Sequence[str]]): The input text to be chunked. Can be a single
      string or a sequence of strings.
    - chunk_size (int, optional): The size of each chunk. Defaults to 40960.
    - overlap_length (int, optional): The length of the overlap between adjacent chunks.
      Defaults to 512.

    Yields:
    - Iterable[Union[str, List[str]]]: An iterable of chunks, each being a string
      or a list of strings,
      depending on the input type.

    Raises:
    - ValueError: If the input type is not a string or a sequence of strings.
    """
    if isinstance(sentence, str):
        while len(sentence) - chunk_size > 100:
            yield sentence[: overlap_length + chunk_size]
            sentence = sentence[chunk_size:]
        yield sentence
        return sentence
    elif isinstance(sentence, Sequence) and isinstance(sentence[0], str):
        while any(sentence):
            yield [s[: overlap_length + chunk_size] if len(s[chunk_size:]) > 100 else s for s in sentence]
            sentence = [s[chunk_size:] if len(s[chunk_size:]) > 100 else "" for s in sentence]
    else:
        raise ValueError(f"Unsupported type {type(sentence)} for chunks")


def pairs(chunks: List[List[int]]) -> Iterable[List[List[int]]]:
    """
    Generates consecutive pairs of chunks for matching.

    Parameters:
    - chunks (List[List[int]]): A list of chunks, where each chunk is a list of integers.

    Yields:
    - Iterable[List[List[int]]]: An iterable of pairs of consecutive chunks.
    """
    for i in range(0, len(chunks) - 1):
        yield (chunks[i], chunks[i + 1])



def flatten(item: Union[torch.Tensor, np.ndarray, List]):
    """
    Flattens a nested sequence (torch.Tensor, numpy.ndarray, or List) into a single flat list.

    Parameters:
    - item (Union[torch.Tensor, np.ndarray, List]): The item to flatten, which can be either
      a torch.Tensor, a numpy.ndarray, or a nested list.

    Returns:
    - Union[torch.Tensor, np.ndarray, List[int]]: A flattened version of the input, as a
      torch.Tensor, a numpy.ndarray, or a list of integers.

    Raises:
    - ValueError: If the input type is not supported or if the flattening process fails.
    """
    if isinstance(item, (torch.Tensor, np.ndarray)):
        return item.flatten()
    elif isinstance(item, List):
        while all(isinstance(i, Sequence) for i in item):
            item = [i for sublist in item for i in sublist]
        assert not any(isinstance(i, Sequence) for i in item), "flatten failed"
        return item
    else:
        raise ValueError(f"Unsupported type {type(item)} for flatten")


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
    # Raw datasets are in data/raw/ (LongBenchV2, LEval, ClongEval)
    dataset_dir = r"../data/raw"
    print(dataset)
    if dataset == "LongBenchV2":
        data = get_longbenchv2_data(f"{dataset_dir}/{dataset}")
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