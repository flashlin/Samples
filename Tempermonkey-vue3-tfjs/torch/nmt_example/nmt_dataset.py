from torch.utils.data import Dataset, DataLoader
import vocabulary
import numpy as np
import pandas as pd
import json
from vocabulary import SequenceVocabulary


class NMTVectorizer(object):
    """
    The vectorizer which coordinates the Vocabularies and puts them to use
    """

    def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):
        """
        Args:
        :param source_vocab: SequenceVocabulary, map source words to integers
        :param target_vocab: SequenceVocabulary, map target words to integers
        :param max_source_length: int, the longest sequence in the source dataset
        :param max_target_length: int, the longest sequence in the target dataset
        """
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    @classmethod
    def from_dataframe(cls, bitext_df):
        """
        Instantiate the vectorizer from the dataset dataframe
        :param bitext_df: pandas Dataframe, the parallel text dataset
        :return: an instance of the NMTVectorizer
        """
        source_vocab = vocabulary.SequenceVocabulary()
        target_vocab = vocabulary.SequenceVocabulary()

        max_source_length, max_target_length = 0, 0

        for _, row in bitext_df.iterrows():
            # read source text, check max length and add token to vocab
            source_tokens = row['source_language'].split(' ')
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)

            # read target text, check max length and add token to target vocab
            target_tokens = row['target_language'].split(' ')
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                target_vocab.add_token(token)

        return cls(source_vocab, target_vocab, max_source_length, max_target_length)

    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        """
        Vectorize the provided indices
        :param indices: list, a list of integers that represent a sequence
        :param vector_length: int, forces the length of the index vector
        :param mask_index: int, the mask_index to use, usually is 0
        :return:
        """

        if vector_length < 0:
            vector_length = len(indices)

        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index

        return vector

    def _get_source_indices(self, text):
        """
        Return the vectorized source text
        :param text: str, the source text, tokens should be separated by spaces
        :return: list int, representing the text
        """

        indices = [self.source_vocab.begin_seq_index]
        indices.extend([self.source_vocab.lookup_token(token) for token in text.split(' ')])
        indices.append(self.source_vocab.end_seq_index)
        return indices

    def _get_target_indices(self, text):
        """
        Return the vectorized target text
        :param text: str, the target text, token should be separated by spaces
        :return: tuple, x_indices, y_indices (observations target and prediction target decode)
        """

        indeces = [self.target_vocab.lookup_token(token) for token in text.split(' ')]
        x_indices = [self.target_vocab.begin_seq_index] + indeces
        y_indices = indeces + [self.target_vocab.end_seq_index]

        return x_indices, y_indices

    def vectorize(self, source_text, target_text, use_dataset_max_lengths=True):
        """
        Return the vectorized source and target text
        :param source_text: str, text from the source language
        :param target_text: str, text from the target language
        :param use_dataset_max_lengths: whether to use the max vector lengths
        :return: The vectorized data point as a dictionary with the keys (source_vector, target_x_vector,
        target_y_vector, source_length)
        """
        source_vector_length = -1
        target_vector_length = -1

        if use_dataset_max_lengths:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(source_indices, source_vector_length, mask_index=self.source_vocab.mask_index)

        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(target_x_indices, target_vector_length, self.target_vocab.mask_index)
        target_y_vector = self._vectorize(target_y_indices, target_vector_length, self.target_vocab.mask_index)

        return {
            'source_vector': source_vector,
            'target_x_vector': target_x_vector,
            'target_y_vector': target_y_vector,
            'source_length': len(source_indices)
        }

    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(contents["source_vocab"])
        target_vocab = SequenceVocabulary.from_serializable(contents["target_vocab"])

        return cls(source_vocab=source_vocab,
                   target_vocab=target_vocab,
                   max_source_length=contents["max_source_length"],
                   max_target_length=contents["max_target_length"])

    def to_serializable(self):
        return {"source_vocab": self.source_vocab.to_serializable(),
                "target_vocab": self.target_vocab.to_serializable(),
                "max_source_length": self.max_source_length,
                "max_target_length": self.max_target_length}


class NMTDataset(Dataset):

    def __init__(self, text_df, vectorizer):
        """

        :param text_df:
        :param vectorizer:
        """

        self.text_df = text_df
        self._vectorizer = vectorizer

        self.train_df = self.text_df[self.text_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.text_df[self.text_df.split == 'val']
        self.validation_size = len(self.val_df)

        self.test_df = self.text_df[self.text_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv):
        """Load dataset and make a new vectorizer from scratch

        Args:
            dataset_csv (str): location of the dataset
        Returns:
            an instance of SurnameDataset
        """
        text_df = pd.read_csv(dataset_csv)
        train_subset = text_df[text_df.split == 'train']
        return cls(text_df, NMTVectorizer.from_dataframe(train_subset))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, dataset_csv, vectorizer_filepath):
        """Load dataset and the corresponding vectorizer.
        Used in the case in the vectorizer has been cached for re-use

        Args:
            dataset_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of SurnameDataset
        """
        text_df = pd.read_csv(dataset_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(text_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file

        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of SurnameVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return NMTVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json

        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point: (x_data, y_target, class_index)
        """
        row = self._target_df.iloc[index]

        vector_dict = self._vectorizer.vectorize(row.source_language, row.target_language)

        return {"x_source": vector_dict["source_vector"],
                "x_target": vector_dict["target_x_vector"],
                "y_target": vector_dict["target_y_vector"],
                "x_source_length": vector_dict["source_length"]}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


def generate_nmt_batches(dataset, batch_size, shuffle=True, drop_last=True, device='cpu'):
    """
    A generator function which wraps the PyTorch Dataloader
    :param dataset:
    :param batch_size:
    :param shuffle:
    :param drop_last:
    :param device:
    :return:
    """

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        lengths = data_dict['x_source_length'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()

        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)

        yield out_data_dict

# dataset = NMTDataset.load_dataset_and_make_vectorizer('./data/simplest_eng_fra.csv')
# generate = generate_nmt_batches(dataset, batch_size=5)
# for batch_data in generate:
#     print(batch_data)
#     break
