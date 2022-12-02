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
        self.model = GNMT(vocab.get_size(), padding_idx, batch_first=True)
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

    def _calculate_loss(self, data, mode="train"):
        (x_hat, y_hat, B), batch = data

        loss = self.loss_fn(x_hat, y_hat)
        self.log("%s_loss" % mode, loss)
        return loss / B

    def infer(self, text):
        translator = GmnTranslator(self.model, self.vocab)
        return translator.infer(text)
        # vocab = self.vocab
        # src_values = vocab.encode(text)
        # self.model.eval()
        # device = next(self.parameters()).device
        # src = torch.tensor([src_values], dtype=torch.long).to(device)
        # bos = vocab.get_value('<bos>')
        # tgt = torch.tensor([[bos]], dtype=torch.long).to(device)
        # for i in range(len(src_values)):
        #     outputs = self.model.transform(src, tgt)
        #     # 預測結果，因為只需要看最後一個詞，所以取`out[:, -1]`
        #     last_word = outputs[:, -1]
        #     predict = self.model.predictor(last_word)
        #     # 找出最大值的 index
        #     y = torch.argmax(predict, dim=1)
        #     # 和之前的預測結果拼接到一起
        #     tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        #     if y == vocab.get_value('<eos>'):
        #         break
        #
        # result = vocab.decode(reduce_dim(tgt).tolist())
        # return result


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
