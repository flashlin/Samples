import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from mlperf_compliance import mlperf_log

from ml.gnmt.decoder import ResidualRecurrentDecoder
from ml.gnmt.encoder import ResidualRecurrentEncoder


class LabelSmoothing(nn.Module):
    def __init__(self, padding_idx, smoothing=0.0):
        super().__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        non_pad_mask = (target != self.padding_idx)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)[non_pad_mask]
        smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.sum()


class Seq2Seq(nn.Module):
    def __init__(self, encoder=None, decoder=None, batch_first=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_first = batch_first

    def encode(self, inputs, lengths):
        return self.encoder(inputs, lengths)

    def decode(self, inputs, context, inference=False):
        return self.decoder(inputs, context, inference)

    def generate(self, inputs, context, beam_size):
        logits, scores, new_context = self.decode(inputs, context, True)
        logprobs = log_softmax(logits, dim=-1)
        logprobs, words = logprobs.topk(beam_size, dim=-1)
        return words, logprobs, scores, new_context


class GNMT(Seq2Seq):
    def __init__(self, vocab_size,
                 padding_idx,
                 hidden_size=512, num_layers=8, bias=True,
                 dropout=0.2, batch_first=False, math='fp32',
                 share_embedding=False):

        super().__init__(batch_first=batch_first)

        mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_NUM_LAYERS,
                              value=num_layers)
        mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_HIDDEN_SIZE,
                              value=hidden_size)
        mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_DROPOUT,
                              value=dropout)

        if share_embedding:
            embedder = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        else:
            embedder = None

        self.encoder = ResidualRecurrentEncoder(vocab_size, padding_idx,
                                                hidden_size,
                                                num_layers, bias, dropout,
                                                batch_first, embedder)

        self.decoder = ResidualRecurrentDecoder(vocab_size, padding_idx,
                                                hidden_size,
                                                num_layers, bias, dropout,
                                                batch_first, math, embedder)

    def forward(self, input_encoder, input_enc_len, input_decoder):
        context = self.encode(input_encoder, input_enc_len)
        context = (context, input_enc_len, None)
        output, _, _ = self.decode(input_decoder, context)

        return output
