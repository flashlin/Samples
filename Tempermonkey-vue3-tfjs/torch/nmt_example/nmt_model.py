import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np


class NMTModel(nn.Module):
    """
    A Neural Machine Translation Model
    """

    def __init__(self, source_vocab_size, source_embedding_size,
                 target_vocab_size, target_embedding_size,
                 encoding_size, target_bos_index):
        """
        Args
        :param source_vocab_size: int, number of unique words in source language
        :param source_embedding_size: int, size of the source embedding vectors
        :param target_vocab_size: int, number of unique words in target language
        :param target_embedding_size: int, size of the target embedding vectors
        :param encoding_size: int, size of the encoder RNN
        :param target_bos_index: int, index for Begin Of sequence token
        :return:
        """

        super(NMTModel, self).__init__()

        self.encoder = NMTEncoder(num_embeddings=source_vocab_size,
                                  embedding_size=source_embedding_size,
                                  rnn_hidden_size=encoding_size)
        decoding_size = encoding_size * 2
        self.decoder = NMTDecoder(num_embedding=target_vocab_size,
                                  embedding_size=target_embedding_size,
                                  rnn_hidden_size=decoding_size,
                                  bos_index=target_bos_index)

    def forward(self, x_source, x_source_lengths, target_sequence, sample_probability=0.0):
        """
        The forward pass of the model
        :param x_source: Tensor, the source text data tensor. Shape should be (batch, max_seq_length)
        :param x_source_lengths: Tensor, the length of the sequences in x_source
        :param target_sequence: Tensor, the target text data tensor
        :param sample_probability: float, the schedule sampling parameter probability of using model's predictions
        at each decoder step
        :return: decoded_state, Tensor, prediction vectors at each output step
        """
        encoder_states, final_hidden_states = self.encoder(x_source, x_source_lengths)
        decoder_states = self.decoder(encoder_states, final_hidden_states, target_sequence)
        return decoder_states


class NMTEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
        """
        Args
        :param num_embeddings: int, size of source vocabulary
        :param embedding_size: int, size of embedding vectors
        :param rnn_hidden_size: int, size of the RNN hidden state vectors
        """
        super(NMTEncoder, self).__init__()
        self.source_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size, padding_idx=0)
        self.birnn = nn.GRU(embedding_size, rnn_hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x_source, x_lengths):
        """
        The forward pass of the model
        :param x_source: Tensor, the input data tensor, shape should be (batch, seq_size)
        :param x_lengths: Tensor, vector og lengths for each item in batch
        :return: a tuple (x_unpacked, x_birnn_h) shape should be (batch, seq_size, rnn_hidden_size * 2)
        and (batch, rnn_hidden_size*2)
        """
        x_embedded = self.source_embedding(x_source)
        # create PackedSequence
        x_lengths = x_lengths.detach().cpu().numpy()
        x_packed = pack_padded_sequence(x_embedded, x_lengths, batch_first=True)

        x_birnn_out, x_birnn_h = self.birnn(x_packed)

        x_birnn_h = x_birnn_h.permute(1, 0, 2)
        x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)
        x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)

        return x_unpacked, x_birnn_h


def verbose_attention(encoder_state_vectors, query_vector):
    """
    Attention score function
    :param encoder_state_vectors: 3d tensor from bi-GRU in decoder
    :param query_vector: hidden state in decoder GRU
    :return:
    """
    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    vector_score = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size), dim=2)
    vector_probabilities = F.softmax(vector_score, dim=1)
    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)
    context_vectors = torch.sum(weighted_vectors, dim=1)
    return context_vectors, vector_probabilities, vector_score


def terse_attention(encoder_state_vectors, query_vector):
    """
    same verbose_attention version but faster
    :param encoder_state_vectors:
    :param query_vector:
    :return:
    """
    vector_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze()
    vector_probabilities = F.softmax(vector_scores, dim=-1)
    context_vectors = torch.matmul(encoder_state_vectors.transpose(-2, -1),
                                   vector_probabilities.unsqueeze(dim=2)).squeeze()
    return context_vectors, vector_probabilities


class NMTDecoder(nn.Module):
    def __init__(self, num_embedding, embedding_size, rnn_hidden_size, bos_index):
        """
        Args:
        :param num_embedding: int, target vocab size
        :param embedding_size: size of embedding vector
        :param rnn_hidden_size: size of the hidden state RNN
        :param bos_index: begin of sequence index
        """
        super(NMTDecoder, self).__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self.target_embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_size, padding_idx=0)
        self.gru_cell = nn.GRUCell(embedding_size + rnn_hidden_size, rnn_hidden_size)
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_embedding)
        self.bos_index = bos_index
        self._sampling_temperature = 3

    def _init_indices(self, batch_size):
        """
        Args
        :param batch_size:
        :return: the begin of sequence index vector
        """
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index

    def _init_context_vectors(self, batch_size):
        """
        Args
        :param batch_size:
        :return: a zeros vector for initializing the context
        """
        return torch.zeros(batch_size, self._rnn_hidden_size)

    def forward(self, encoder_state, initial_hidden_state, target_sequence, sample_probability=0.0,
                max_output_seq_size=100):
        """
        The forward pass of the model
        :param encoder_state: Tensor, output of the NMTEncoder
        :param initial_hidden_state: Tensor, last hidden state in the NMTEncoder
        :param target_sequence: Tensor, target text data tensor
        :param sample_probability: float, the schedule sampling parameter probability of using model's predictions
        at each decoder step
        :param max_output_seq_size: int
        :return: Tensor, prediction vectors at each output step
        """
        if target_sequence is None:
            sample_probability = 1
            output_sequence_size = max_output_seq_size
        else:
            # batch is on 1st dimention (input is (batch, seq_size)) -> permute for iterate over sequence
            target_sequence = target_sequence.permute(1, 0)
            output_sequence_size = target_sequence.size(0)

        # use the provided encoder hidden state as the initial hidden state
        h_t = self.hidden_map(initial_hidden_state)

        batch_size = encoder_state.size(0)

        # initialize context vectors to zeros
        context_vectors = self._init_context_vectors(batch_size)
        # initialize first y_t word as BOS
        y_t_index = self._init_indices(batch_size)

        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        output_vectors = []

        # All cached tensors are moved from the GPU and stored for analysis
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()

        for i in range(output_sequence_size):
            use_sample = np.random.random() < sample_probability
            if not use_sample:
                y_t_index = target_sequence[i]

            # step 1: embed word and concat with previous context
            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)

            # step 2: Make a GRU step, getting a new hidden vector
            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().data.numpy())

            # step 3: use current hidden vector to attend to encoder state
            context_vectors, p_attn, _ = verbose_attention(encoder_state, h_t)
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())

            # step 4: Use current hidden and context vector to make a prediction for the next word
            prediction_vector = torch.cat([context_vectors, h_t], dim=1)
            score_for_y_t_index = self.classifier(prediction_vector)

            if use_sample:
                p_y_t_index = F.softmax(score_for_y_t_index * self._sampling_temperature, dim=1)
                y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()

            output_vectors.append(score_for_y_t_index)

        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)

        return output_vectors
