import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common.io import info
from ml.data_utils import pad_list
from ml.lit import BaseLightning

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seq_len = 200
num_heads = 8
num_layers = 6
d_model = 512
d_ff = 2048
d_k = d_model // num_heads
drop_out_rate = 0.1
num_epochs = 10
beam_size = 8


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(d_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input)  # (B, L) => (B, L, d_model)
        trg_input = self.trg_embedding(trg_input)  # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(src_input)  # (B, L, d_model) => (B, L, d_model)
        trg_input = self.positional_encoder(trg_input)  # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask)  # (B, L, d_model)
        d_output = self.decoder(trg_input, e_output, e_mask, d_mask)  # (B, L, d_model)

        output = self.softmax(self.output_linear(d_output))  # (B, L, d_model) => # (B, L, trg_vocab_size)

        return output


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.multihead_attention = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.layer_norm_2 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.drop_out_2 = nn.Dropout(drop_out_rate)

    def forward(self, x, e_mask):
        x_1 = self.layer_norm_1(x)  # (B, L, d_model)
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_mask)
        )  # (B, L, d_model)
        x_2 = self.layer_norm_2(x)  # (B, L, d_model)
        x = x + self.drop_out_2(self.feed_forward(x_2))  # (B, L, d_model)

        return x  # (B, L, d_model)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.masked_multihead_attention = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.layer_norm_2 = LayerNormalization()
        self.multihead_attention = MultiheadAttention()
        self.drop_out_2 = nn.Dropout(drop_out_rate)

        self.layer_norm_3 = LayerNormalization()
        self.feed_forward = FeedFowardLayer()
        self.drop_out_3 = nn.Dropout(drop_out_rate)

    def forward(self, x, e_output, e_mask, d_mask):
        x_1 = self.layer_norm_1(x)  # (B, L, d_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)
        )  # (B, L, d_model)
        x_2 = self.layer_norm_2(x)  # (B, L, d_model)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        )  # (B, L, d_model)
        x_3 = self.layer_norm_3(x)  # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3))  # (B, L, d_model)

        return x  # (B, L, d_model)


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inf = 1e9

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(input_shape[0], -1, num_heads, d_k)  # (B, L, num_heads, d_k)
        k = self.w_k(k).view(input_shape[0], -1, num_heads, d_k)  # (B, L, num_heads, d_k)
        v = self.w_v(v).view(input_shape[0], -1, num_heads, d_k)  # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask)  # (B, num_heads, L, d_k)
        concat_output = attn_values.transpose(1, 2) \
            .contiguous().view(input_shape[0], -1, d_model)  # (B, L, d_model)

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v)  # (B, num_heads, L, d_k)

        return attn_values


class FeedFowardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, x):
        x = self.relu(self.linear_1(x))  # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x)  # (B, L, d_model)

        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)

        return x


class PositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Make initial positional encoding matrix with 0
        pe_matrix = torch.zeros(seq_len, d_model)  # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0)  # (1, L, d_model)
        self.positional_encoding = pe_matrix.to(device=device).requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(d_model)  # (B, L, d_model)
        x = x + self.positional_encoding  # (B, L, d_model)
        return x


def create_tensor(shape):
    return torch.tensor(np.zeros(shape=shape))


def pad_batch_sequence(batch_tensor, max_length):
    new_batch = create_tensor((batch_tensor.size(0), max_length))
    for n in range(batch_tensor.size(0)):
        alist = batch_tensor[n, :].tolist()
        pad_sequence = pad_list(alist, max_length=max_length)
        new_batch[n, :] = torch.tensor(pad_sequence)
    return new_batch


class LitTransformer2(BaseLightning):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.model = Transformer(src_vocab_size=vocab.get_size(), trg_vocab_size=vocab.get_size())

        # CrossEntropyLoss((src_len, n_classes), (tgt_len))
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.padding_idx)

    def forward(self, batch):
        device = next(self.parameters()).device
        src, src_lens, tgt, tgt_lens = batch
        src = pad_batch_sequence(src, 200).type(torch.long).to(device)
        tgt = pad_batch_sequence(tgt, 200).type(torch.long).to(device)
        return self.model(src, tgt), tgt

    def _calculate_loss(self, batch, batch_idx):
        x_hat, y = batch
        x_hat = x_hat.squeeze(0)
        y_true = y.squeeze(0)
        return self.loss_fn(x_hat, y_true)

    def infer(self, text):
        device = self.get_device()
        text_to_indices = self.vocab.encode_to_tokens(text)

        src = torch.tensor([text_to_indices]).type(torch.long).to(device)
        src = pad_batch_sequence(src, 200).type(torch.long).to(device)
        self.model.eval()
        logits = self.model(src, src)
        logits = logits.squeeze(0).argmax(1).tolist()
        return self.vocab.decode(logits)
