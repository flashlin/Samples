import torch

from common.io import info
from ml.gnmt.inference import Translator
from ml.gnmt.model import GNMT, LabelSmoothing
from ml.lit import BaseLightning


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)


class LiGmnTranslator(BaseLightning):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        padding_idx = vocab.get_value('<pad>')
        self.model = GNMT(vocab.get_size(),
                          padding_idx,
                          hidden_size=128,
                          num_layers=2,
                          dropout=0.3,
                          batch_first=True,
                          math='fp16',
                          share_embedding=True)
        self.loss_fn = LabelSmoothing(padding_idx, smoothing=True)

    def forward(self, batch):
        src, src_len, tgt, tgt_len = batch
        src_len = tuple_of_tensors_to_tensor(src_len)

        output = self.model(src, src_len, tgt[:, :-1])
        tgt_labels = tgt[:, 1:]
        T, B = output.size(1), output.size(0)

        x_hat = output.view(T * B, -1).float()
        y_hat = tgt_labels.contiguous().view(-1)

        return x_hat, y_hat, B

    def _calculate_loss(self, batch, batch_idx):
        (x_hat, y_hat, B) = batch
        loss = self.loss_fn(x_hat, y_hat)
        return loss / B

    def infer(self, text):
        translator = GmnTranslator(self.model, self.vocab)
        return translator.infer(text)


class GmnTranslator:
    def __init__(self, model, vocab, beam_size=5, max_seq_len=80, len_norm_factor=0.6, len_norm_const=5.0,
                 cov_penalty_factor=0.1, device='cuda'):
        self.translation_model = Translator(model,
                                            vocab,
                                            beam_size=beam_size,
                                            max_seq_len=max_seq_len,
                                            len_norm_factor=len_norm_factor,
                                            len_norm_const=len_norm_const,
                                            cov_penalty_factor=cov_penalty_factor,
                                            cuda=(device == 'cuda'))

    def infer(self, x):
        translated_lines, stats = self.translation_model.translate([x])
        return translated_lines
