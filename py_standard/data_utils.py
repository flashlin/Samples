import string
from itertools import groupby


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
