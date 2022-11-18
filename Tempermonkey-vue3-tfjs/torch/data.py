def pad_sequence(l, fill_value, max_length):
    return l + [fill_value] * (max_length - len(l))
