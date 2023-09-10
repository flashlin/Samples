from itertools import groupby

import numpy as np


def df_intstr_to_values(df):
    return df.map(lambda l: np.array([int(n) for n in l.split(',')], dtype=np.long))


def pad_array(arr, fill_value, max_length, d_type=np.long):
    new_arr = np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=fill_value)
    return np.array(new_arr, dtype=d_type)


def split_line_by_space(line):
    return [x.rstrip() for x in line.split(' ')]


def sort_desc(arr: list[str]) -> list[str]:
    arr.sort(key=lambda x: len(x))
    return arr[::-1]


def group_to_lengths(arr_sorted: list[str]):
    return [k for k, g in groupby(arr_sorted, key=lambda x: len(x))]


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
