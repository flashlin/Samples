import string
from itertools import groupby
import json

from translate_file_datasets import T


def create_char2index_map(str_list: list[str], start=0):
    dictionary = {}
    for idx, key in enumerate(str_list):
        dictionary[key] = idx + start
    return dictionary


def create_index2char_map(str_list: list[str], start=0):
    dictionary = {}
    for idx, key in enumerate(str_list):
        dictionary[idx + start] = key
    return dictionary


def sort_by_len_desc(arr: list[str]) -> list[str]:
    arr.sort(key=lambda x: len(x))
    return arr[::-1]


def group_to_lengths(arr_sorted: list[str]):
    return [k for k, g in groupby(arr_sorted, key=lambda x: len(x))]


def write_dict_to_file(dictionary, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(dictionary, file)


def load_dict_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dictionary = json.load(file)
    return dictionary


def pad_list(value_list: list[T], max_len: int, pad_value: T) -> list[T]:
    len_values = len(value_list)
    if len_values < max_len:
        return value_list + [pad_value] * (max_len - len_values)
    return value_list
