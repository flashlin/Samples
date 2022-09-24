import io
import re
import numpy as np
import pandas as pd
import torch


def textdata_to_tensor_iter(filepath, vocab, tokenizer):
    for raw_item in iter(io.open(filepath, encoding="utf8")):
        yield torch.tensor([vocab[token] for token in tokenizer(raw_item)], dtype=torch.long)


def load_csv_to_dataframe(csv_filepath='./input_data/linq_corpus.csv'):
    data = pd.read_csv(csv_filepath)
    data = data.reset_index(drop=True)

    # 刪除 other_column 欄位
    # data.drop('other_column',axis=1,inplace=True)

    # preprocess
    data = data.dropna().drop_duplicates()

    # lower and remove quotes
    # data[source_column_name] = data.english_sentence.parallel_apply(lambda x: re.sub("'", '',x).lower())
    # data[target_column_name] = data.hindi_sentence.parallel_apply(lambda x: re.sub("'", '', x).lower())

    # remove special chars
    # exclude = set(string.punctuation)#set of all special chars
    # remove all the special chars
    # data[source_column_name] = data.english_sentence.parallel_apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    # data[target_column_name] = data.hindi_sentence.parallel_apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

    # remove_digits = str.maketrans('','',digits)
    # data[source_column_name] = data.english_sentence.parallel_apply(lambda x: x.translate(remove_digits))
    # data[target_column_name] = data.hindi_sentence.parallel_apply(lambda x: x.translate(remove_digits))

    # data[target_column_name] = data.hindi_sentence.parallel_apply(lambda x: re.sub("[२३०८१५७९४६]","",x))

    # Remove extra spaces
    source_column_name = "source_sentence"
    target_column_name = "target_sentence"
    # data[source_column_name]=data[source_column_name].parallel_apply(lambda x: x.strip())
    data[source_column_name] = data[source_column_name].apply(lambda x: x.strip())
    data[target_column_name] = data[target_column_name].apply(lambda x: x.strip())
    data[source_column_name] = data[source_column_name].apply(lambda x: re.sub(" +", " ", x))
    data[target_column_name] = data[target_column_name].apply(lambda x: re.sub(" +", " ", x))
    return data


def split_dataframe(dataframe):
    val_frac = 0.1  # precentage data in val
    val_split_idx = int(len(dataframe) * val_frac)  # index on which to split
    data_idx = list(range(len(dataframe)))  # create a list of ints till len of data
    np.random.shuffle(data_idx)

    # get indexes for validation and train
    val_idx, train_idx = data_idx[:val_split_idx], data_idx[val_split_idx:]
    print('len of train: ', len(train_idx))
    print('len of val: ', len(val_idx))

    # create the sets
    train_df = dataframe.iloc[train_idx].reset_index().drop('index', axis=1)
    val_df = dataframe.iloc[val_idx].reset_index().drop('index', axis=1)
    return train_df, val_df





