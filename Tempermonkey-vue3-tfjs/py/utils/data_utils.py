import numpy as np


def df_to_values(df):
    return df.map(lambda l: np.array([int(n) for n in l.split(',')], dtype=np.long))


def pad_array(arr, fill_value, max_length, d_type=np.long):
    new_arr = np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=fill_value)
    return np.array(new_arr, dtype=d_type)


def split_line_by_space(line):
    return [x.rstrip() for x in line.split(' ')]
