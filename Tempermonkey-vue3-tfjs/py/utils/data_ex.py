import numpy as np


def df_to_values(df):
    return df.map(lambda l: np.array([int(n) for n in l.split(',')], dtype=np.long))


def pad_array(arr, fill_value, max_length, d_type=np.long):
    arr_len = len(arr)
    new_arr = np.pad(arr, (0, max_length - arr_len), 'constant', constant_values=fill_value)
    return np.array(new_arr, dtype=d_type)
