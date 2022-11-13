def pad_list(l, fill_value, max_length):
    return l + [fill_value] * (max_length - len(l))
