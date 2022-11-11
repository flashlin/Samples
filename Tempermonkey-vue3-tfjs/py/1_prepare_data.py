import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, Embedding, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import dot
from tensorflow.keras.layers import Activation
from keras_preprocessing.sequence import pad_sequences
import numpy as np

from utils.linq_translation_data import LinqTranslationData, write_train_data, write_tokens_data, write_train_tfrecord

data_file = "./data/linq-sample.txt"
# data = LinqTranslationData("./data/linq-sample.txt")
write_train_data(data_file)
#write_tokens_data(data_file)

write_train_tfrecord(data_file, "./output/train.tfrec")