import shlex
from itertools import chain
from csharp_tokenizer import tokenize
import tensorflow as tf
import numpy as np
import pprint
import re
import io
import re
import string
import tqdm
import numpy as np
from tensorflow.keras import layers
from gensim.models import Word2Vec


# def text_to_words_iter(text: str):
#     for word in shlex.split(text):
#         new_words = tokenize(word)
#         for new_word in new_words:
#             yield new_word

def text_to_words_iter(text: str):
    for word in tokenize(text):
        yield word


def text_to_tokens(text: str):
    return [word for word in text_to_words_iter(text)]



text = 'from tb1 in customer where id==1 && name.contains("123") select tb1.id, tb2.name'
tokens = text_to_tokens(text)

tokens = []
with open("./input_data/linq.txt") as f:
    for line in f.readlines():
        tokens1 = text_to_tokens(line)
        tokens += tokens1
print(f"{tokens=}")
print("")

vocab, index = {}, 3  # start indexing from 1
vocab['<pad>'] = 0    # add a padding token
vocab['<cls>'] = 1
vocab['<sep>'] = 2
for token in tokens:
    if token not in vocab:
        vocab[token] = index
        index += 1
vocab_size = len(vocab)
print(vocab)

print("Create an inverse vocabulary to save mappings from integer indices to tokens")
inverse_vocab = {index: token for token, index in vocab.items()}
print(inverse_vocab)

print("Vectorize your sentence")
example_sequence = [vocab[word] for word in tokens]
print(example_sequence)

window_size = 2
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
    example_sequence,
    vocabulary_size=vocab_size,
    window_size=window_size,
    negative_samples=0)
print(len(positive_skip_grams))
for target, context in positive_skip_grams[:5]:
    print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")

# Get target and context words for one positive skip-gram.
target_word, context_word = positive_skip_grams[0]

# Set the number of negative samples per positive context.
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
num_ns = 4

context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    true_classes=context_class,  # class that should be sampled as 'positive'
    num_true=1,  # each positive skip-gram has 1 positive context class
    num_sampled=num_ns,  # number of negative context words to sample
    unique=True,  # all the negative samples should be unique
    range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
    seed=SEED,  # seed for reproducibility
    name="negative_sampling"  # name of this operation
)
print(negative_sampling_candidates)
print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])

# Add a dimension so you can use concatenation (in the next step). 添加一個維度，以便您可以使用串聯（在下一步中）
negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

# Concatenate a positive context word with negative sampled words. 將正上下文詞與負樣本詞連接起來
context = tf.concat([context_class, negative_sampling_candidates], 0)

# Label the first context word as `1` (positive) followed by `num_ns` `0`s (negative).
label = tf.constant([1] + [0] * num_ns, dtype="int64")

# Reshape the target to shape `(1,)` and context and label to `(num_ns+1,)`.
target = tf.squeeze(target_word)
context = tf.squeeze(context)
label = tf.squeeze(label)
print(f"target_index    : {target}")
print(f"target_word     : {inverse_vocab[target_word]}")
print(f"context_indices : {context}")
print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
print(f"label           : {label}")

sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=10)
print(f"{sampling_table=}")



def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


def custom_standardization(input_data):
    # lowercase = tf.strings.lower(input_data)
    # return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')
    return input_data


vocab_size = 4096
sequence_length = 10

# Use the `TextVectorization` layer to normalize, split, and map strings to
# integers. Set the `output_sequence_length` length to pad all samples to the
# same length.
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

path_to_file = ["./input_data/linq.txt"]
text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))
vectorize_layer.adapt(text_ds.batch(1024))

# Save the created vocabulary for reference.
inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])






#
#
# def get_ids(tokens, tokenizer, max_seq_length):
#     """Token ids from Tokenizer vocab"""
#     # token_ids = tokenizer(tokens)
#     token_ids = []
#     for token in tokens:
#         token_ids.append(vocab[token])
#     input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
#     return input_ids
#
# def get_masks(tokens, max_seq_length):
#     """Mask for padding"""
#     if len(tokens) > max_seq_length:
#         raise IndexError("Token length more than max seq length!")
#     return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))
#
# def get_segments(tokens, max_seq_length):
#     """Segments: 0 for the first sequence, 1 for the second"""
#     if len(tokens)>max_seq_length:
#         raise IndexError("Token length more than max seq length!")
#     segments = []
#     current_segment_id = 0
#     for token in tokens:
#         segments.append(current_segment_id)
#         if token == "[SEP]":
#             current_segment_id = 1
#     return segments + [0] * (max_seq_length - len(tokens))
#
#
# max_seq_length = 200
# tokens = ["<cls>"] + tokens + ["<sep>"]
# input_ids = get_ids(tokens, tokenize, max_seq_length)
# print(f"{input_ids=}")
#
# input_masks = get_masks(tokens, max_seq_length)
# print(f"{input_masks=}")
#
# input_segments = get_segments(tokens, max_seq_length)
# print(f"{input_segments=}")
#
#
# input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
# input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
# segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
# bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)
# pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
#
# model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
# pool_embs, all_embs = model.predict([[input_ids],[input_masks],[input_segments]])

#
# #text_ds = tf.data.TextLineDataset(['/input_data/linq.txt'])
# text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
#
# sequences = list(text_vector_ds.as_numpy_iterator())
# print(len(sequences))
#
#
# targets, contexts, labels = generate_training_data(
#     sequences=sequences,
#     window_size=2,
#     num_ns=4,
#     vocab_size=vocab_size,
#     seed=SEED)
#
# targets = np.array(targets)
# contexts = np.array(contexts)[:, :, 0]
# labels = np.array(labels)
#
# print('\n')
# print(f"targets.shape: {targets.shape}")
# print(f"contexts.shape: {contexts.shape}")
# print(f"labels.shape: {labels.shape}")
