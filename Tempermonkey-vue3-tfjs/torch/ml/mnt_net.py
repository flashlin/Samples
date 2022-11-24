from torch.utils.data import Dataset
import torch.nn.functional as F
from ml.bpe_tokenizer import SimpleTokenizer
from ml.lit import BaseLightning
from mnt_model import NMTModel
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class TranslateOptions:
    src_vocab_size: int
    src_embedding_size: int
    tgt_vocab_size: int
    tgt_embedding_size: int
    encoding_size: int
    bos_idx: int
    eos_idx: int
    padding_idx: int
    mask_idx: int
    unk_idx: int
    num_epochs: int
    decode_fn: Callable[[list[int]], str]
    train_dataset: Dataset


tk = SimpleTokenizer(None)

translate_options = TranslateOptions(
    src_vocab_size=tk.vocab_size,
    src_embedding_size=128,
    tgt_vocab_size=tk.vocab_size,
    tgt_embedding_size=128,
    encoding_size=300,
    bos_idx=tk.bos_idx,
    eos_idx=tk.eos_idx,
    padding_idx=tk.padding_idx,
    mask_idx=tk.mask_idx,
    unk_idx=tk.unk_idx,
    num_epochs=100,
    decode_fn=tk.decode,
    train_dataset=TranslationDataset("./output/linq-sample2.csv", tk.padding_idx)
)


def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes
    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


class MntTranslator(BaseLightning):
    def __init__(self, options=translate_options):
        super().__init__()
        self.options = options
        self.model = NMTModel(source_vocab_size=options.src_vocab_size,
                              source_embedding_size=options.src_embedding_size,
                              target_vocab_size=options.tgt_vocab_size,
                              target_embedding_size=options.tgt_embedding_size,
                              encoding_size=options.encoding_size,
                              target_bos_index=options.bos_idx)
        self.criterion = sequence_loss
        self.init_dataloader(options.train_dataset, 1)
        self.vectorizer = options.train_dataset.get_vectorizer()
        self.sample_probability = None

    def training_step(self, batch, batch_idx):
        self.sample_probability = (20 + batch_idx) / self.options.num_epochs
        outputs = self(batch)
        loss = self._calculate_loss((outputs, batch), mode="train")
        return loss

    def forward(self, batch):
        src, src_lens, tgt, tgt_lens = batch
        y_pred = self.model(src,
                            src_lens,
                            tgt,
                            sample_probability=self.sample_probability)
        return y_pred, tgt

    def _calculate_loss(self, data, mode="train"):
        (logits, tgt), batch = data
        loss = self.criterion(logits, tgt, self.options.mask_index)
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, text):
        #tgt_values = self.model.inference(text)
        #tgt_text = self.options.decode_fn(tgt_values)
        return text
