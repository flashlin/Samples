import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, Embedding, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import dot
from tensorflow.keras.layers import Activation
from keras_preprocessing.sequence import pad_sequences
import numpy as np

from utils.linq_translation_data import LinqTranslationData, write_train_data, write_tokens_data, write_train_tfrecord, \
    pad_train_data

data_file = "./data/linq-sample.txt"
# data = LinqTranslationData("./data/linq-sample.txt")
src_max_seq_length, tgt_max_seq_length = write_train_data(data_file)
pad_train_data('./output/linq-translation.txt', src_max_seq_length, tgt_max_seq_length)
#write_tokens_data(data_file)

#write_train_tfrecord(data_file, "./output/train.tfrec")
#print(f"{src_max_seq_length=} {tgt_max_seq_length=}")
