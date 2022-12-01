import random

import torch
from torch import nn

from common.io import info
from ml.lit import PositionalEncoding, load_model, copy_last_ckpt, start_train
from ml.trans_linq2tsql import LinqToSqlVocab
from ml.translate_net import LiTranslator, TranslateListDataset

"""
src = 'from tb1     in customer select tb1     . name'
tgt = 'from @tb_as1 in @tb1     select @tb_as1 . @fd1'

src = 'from tb1     in'
pre = '<bos>'
tgt = 'from @tb_as1 in'

src = 'tb1     in customer'
pre = 'from'
tgt = '@tb_as1 in @tb1'

src = 'in customer select'
pre = 'from in'
tgt = 'in @tb1     select'

src = 'customer select tb1'
pre = 'from in customer'
tgt = '@tb1     select @tb_as1'

src = 'select tb1     .'
pre = 'in customer select'
tgt = 'select @tb_as1 .'

src = 'tb1     . name'
pre = 'customer select tb1'
tgt = '@tb_as1 . @fd1'

src = '. name <eos>'
pre = 'select tb1 .'
tgt = '. @fd1 <eos>'
"""

vocab = LinqToSqlVocab()

translate_examples = [
    (
        'from tb1     in customer select tb1.name',
        'from @tb_as1 in @tb1     select @tb_as1.@fd1'
    ),
]

model = start_train(LiTranslator,
                    {
                        'vocab': vocab,
                    },
                    TranslateListDataset(translate_examples, vocab),
                    batch_size=16,
                    device='cuda',
                    max_epochs=10)
# model = load_model(LiTranslator)
text = model.infer('from tb2 in p select tb2.name')
print(f"{text=}")
