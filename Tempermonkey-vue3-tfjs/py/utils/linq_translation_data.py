from common.csv_utils import CsvWriter
from common.io import read_text_file
from utils.linq_tokenizr import linq_encode, linq_tokenize
from utils.tsql_tokenizr import tsql_encode, tsql_tokenize
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

def write_train_data(file: str):
    lines = read_text_file(file)
    out_file = "./output/linq-translation.csv"
    with CsvWriter(out_file) as csv:
        for idx, line in enumerate(lines):
            if idx % 2 == 0:
                linq_values = linq_encode(line)
                csv.write(linq_values)
            else:
                sql_values = tsql_encode(line)
                csv.write(sql_values)

def write_tokens_data(file: str):
    lines = read_text_file(file)
    out_file = "./output/linq-translation-tokens.txt"
    def map_tokens(tokens):
        return map(lambda x: f"{{{x.type}:{x.text}}}", tokens)
    with CsvWriter(out_file) as csv:
        for idx, line in enumerate(lines):
            if idx % 2 == 0:
                linq_tokens = linq_tokenize(line)
                tokens = map_tokens(linq_tokens)
                csv.write(tokens)
            else:
                sql_tokens = tsql_tokenize(line)
                tokens = map_tokens(sql_tokens)
                csv.write(tokens)

def write_train_tfrecord(example_file, tfrecord_file):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    def _serialize_example(source, target):
        feature = {
            'source': _int64_feature(source),
            'target': _int64_feature(target),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def _write_to_tfrecord(tfrecord_writer):
        with open(example_file, "r", encoding='UTF-8') as f:
            for idx, line in enumerate(f):
                if idx % 2 == 0:
                    linq_values = linq_encode(line)
                else:
                    tsql_values = tsql_encode(line)
                    example = _serialize_example(linq_values, tsql_values)
                    tfrecord_writer.write(example)

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        _write_to_tfrecord(writer)


def read_train_tfrecord(file):
    raw_dataset = tf.data.TFRecordDataset(file)

    feature_description = {
        'source': tf.io.FixedLenFeature([], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    for idx, features in enumerate(raw_dataset):
        parsed_features = _parse_function(features)
    # features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))

def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "source": tf.io.FixedLenFeature([], tf.int64),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {"source": tf.io.FixedLenFeature([], tf.int64),}
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    source = tf.cast(example["source"], tf.int32)
    if labeled:
        target = tf.cast(example["target"], tf.int32)
        return source, target
    return source

def load_dataset(filenames, labeled=True):
    dataset = tf.data.TFRecordDataset(filenames)
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def get_dataset(filenames, batch_size=64, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

def demo_get_dataset():
    filenames = tf.io.gfile.glob("/tfrecords/train*.tfrec")
    split_ind = int(0.9 * len(filenames))
    training_filenames, valid_filenames = filenames[:split_ind], filenames[split_ind:]

    train_dataset = get_dataset(training_filenames)
    valid_dataset = get_dataset(valid_filenames)

    test_filenames = tf.io.gfile.glob("/tfrecords/test*.tfrec")
    test_dataset = get_dataset(test_filenames, labeled=False)

    image_batch, label_batch = next(iter(train_dataset))
    def show_batch(image_batch, label_batch):
        plt.figure(figsize=(10, 10))
        for n in range(25):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(image_batch[n] / 255.0)
            if label_batch[n]:
                plt.title("MALIGNANT")
            else:
                plt.title("BENIGN")
            plt.axis("off")

    show_batch(image_batch.numpy(), label_batch.numpy())

class LinqTranslationData:
    def __init__(self):
        pass


