import os
import sys
import random
import warnings

from utils import Train_Dataset, Vocabulary, get_train_loader
warnings.filterwarnings("ignore")

#data manupulation libs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from pandarallel import pandarallel
# Initialization
#pandarallel.initialize()


#string manupulation libs
import re
import string
from string import digits
import spacy

#torch libs
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():
    data = pd.read_csv('./input_data/linq_corpus.csv')
    data = data.reset_index(drop=True)

    #刪除 other_column 欄位
    #data.drop('other_column',axis=1,inplace=True)

    #preprocess
    data = data.dropna().drop_duplicates()

    #lower and remove quotes
    #data[source_column_name] = data.english_sentence.parallel_apply(lambda x: re.sub("'", '',x).lower())
    #data[target_column_name] = data.hindi_sentence.parallel_apply(lambda x: re.sub("'", '', x).lower())
        
    #remove special chars
    #exclude = set(string.punctuation)#set of all special chars
    #remove all the special chars
    #data[source_column_name] = data.english_sentence.parallel_apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    #data[target_column_name] = data.hindi_sentence.parallel_apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
        
    #remove_digits = str.maketrans('','',digits)
    #data[source_column_name] = data.english_sentence.parallel_apply(lambda x: x.translate(remove_digits))
    #data[target_column_name] = data.hindi_sentence.parallel_apply(lambda x: x.translate(remove_digits))

    #data[target_column_name] = data.hindi_sentence.parallel_apply(lambda x: re.sub("[२३०८१५७९४६]","",x))


    # Remove extra spaces
    source_column_name = "source_sentence"
    target_column_name = "target_sentence"
    #data[source_column_name]=data[source_column_name].parallel_apply(lambda x: x.strip())
    data[source_column_name]=data[source_column_name].apply(lambda x: x.strip())
    data[target_column_name]=data[target_column_name].apply(lambda x: x.strip())
    data[source_column_name]=data[source_column_name].apply(lambda x: re.sub(" +", " ", x))
    data[target_column_name]=data[target_column_name].apply(lambda x: re.sub(" +", " ", x))
    return data


if __name__ == '__main__':
    #freeze_support()        
    data = load_data()
    #######################################################
    #               Create Train and Valid sets
    #######################################################
    val_frac = 0.1 #precentage data in val
    val_split_idx = int(len(data)*val_frac) #index on which to split
    data_idx = list(range(len(data))) #create a list of ints till len of data
    np.random.shuffle(data_idx)

    #get indexes for validation and train
    val_idx, train_idx = data_idx[:val_split_idx], data_idx[val_split_idx:]
    print('len of train: ', len(train_idx))
    print('len of val: ', len(val_idx))

    #create the sets
    train = data.iloc[train_idx].reset_index().drop('index',axis=1)
    val = data.iloc[val_idx].reset_index().drop('index',axis=1)

    #create a vocab class with freq_threshold=0 and max_size=100
    voc = Vocabulary(0, 100)
    sentence_list = ['that is a cat', 'that is not a dog']
    #build vocab
    voc.build_vocabulary(sentence_list)

    print('index to string: ',voc.itos)
    print('string to index:',voc.stoi)
    print('numericalize -> cat and a dog: ', voc.numericalize('cat and a dog'))

    ####
    train_dataset = Train_Dataset(train, 'source_sentence', 'target_sentence')
    print(train.loc[1])
    print(train_dataset[1])

    ###
    train_loader = get_train_loader(train_dataset, 32)
    source = next(iter(train_loader))[0]
    target = next(iter(train_loader))[1]

    print('source: \n', source)

    print('source shape: ',source.shape)
    print('target shape: ', target.shape)
