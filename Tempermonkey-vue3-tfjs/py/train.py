import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, Embedding, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import dot
from tensorflow.keras.layers import Activation
from keras_preprocessing.sequence import pad_sequences
import numpy as np

from utils.linq_translation_data import LinqTranslationData

# preparing hyperparameters

# source language- English
src_wordEmbed_dim = 18  # 詞向量維度18
src_max_seq_length = 100  # 句長最大值為 100

# target language- Spanish
tgt_wordEmbed_dim = 27  # dim of text vector representation
tgt_max_seq_length = 12  # max length of a sentence (including <SOS> and <EOS>)

# dim of context vector
latent_dim = 256  # LSTM 的內部狀態為 256維的向量


data = LinqTranslationData("./data/linq-sample.txt")
