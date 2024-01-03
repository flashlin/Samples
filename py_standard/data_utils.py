import string
from itertools import groupby
import json
from typing import TypeVar, Generator, Union, Callable
from itertools import zip_longest
import hashlib
from itertools import combinations

T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')


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


def pad_list(value_list: list[T], max_len: int, pad_value: T = 0) -> list[T]:
    """
    :param value_list: any list
    :param max_len:
    :param pad_value:
    :return:
    """
    len_values = len(value_list)
    if len_values < max_len:
        return value_list + [pad_value] * (max_len - len_values)
    return value_list


#EitherT1 = Union[T1, None]
#EitherT2 = Union[T2, None]
TupleT12 = tuple[T1, T2]

def zip_aggregate(a_list: list[T1], b_list: list[T2],
                  create_a_elem: Callable[[T1], T1],
                  create_b_elem: Callable[[T2], T2]) \
        -> Generator[TupleT12, None, None]:
    prev_a = None
    prev_b = None
    for a, b in zip_longest(a_list, b_list, fillvalue=None):
        if a is None:
            a = create_a_elem(prev_a)
        if b is None:
            b = create_b_elem(prev_b)
        yield a, b
        prev_a = a
        prev_b = b


def overlap_split_list(sequence: list[T], split_length: int, overlap: int = 1) -> list[list[T]]:
    """
    将序列切割为重叠分割的子序列。
    :param sequence: (list or torch.Tensor) 輸入序列，長度 n
    :param split_length: (int) 子序列的長度
    :param overlap: (int) it should <= split_length
    :return: (list) 切割後的子序列的列表。
    """
    assert overlap <= split_length, "重叠长度必须小于等于切割长度。"
    if len(sequence) <= split_length:
        return [pad_list(sequence, split_length)]
    result = []
    start = 0
    while start + split_length <= len(sequence):
        result.append(sequence[start: start + split_length])
        start += overlap
    return result


def create_running_list(a_list: list[T], max_seq_len: int) -> list[list[T]]:
    """
    ex input [1, 2, 3]
    return
    [0, 0, 1]
    [0, 1, 2]
    [1, 2, 3]
    :param a_list:
    :param max_seq_len:
    :return:
    """
    new_seq = pad_list(a_list, max_len=max_seq_len, pad_value=0)
    pad_len = max_seq_len - 1
    tmp_list = [0] * pad_len
    tmp_list.extend(new_seq)
    return overlap_split_list(tmp_list, max_seq_len)


def hash_password(password: str) -> str:
    return hashlib.sha256(str.encode(password)).hexdigest()


# Check password matches during login
def check_password_hashes(password: str, hashed_text: str):
    if hash_password(password) == hashed_text:
        return True
    return False

def list_to_combinations_dict(a_list):
    keys = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    if len(keys) < len(a_list):
        raise ValueError(f"list size > keys")
    a_dict = {}
    for value in a_list:
        key = keys.pop(0)
        a_dict[key] = value
    return a_dict

def combinations_fn(items: list[str], n: int):
    combinations_dict = list_to_combinations_dict(items)
    keys = "".join(key for key in combinations_dict.keys())
    results = list(combinations(keys,n))
    new_results = []
    for items in results:
        new_list = []
        for elem in list(items):
            value = combinations_dict[elem]
            new_list.append(value)
        new_results.append(new_list)
    return new_results
