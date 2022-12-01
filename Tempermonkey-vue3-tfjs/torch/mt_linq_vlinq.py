from common.io import info, info_error, get_file_by_lines_iter
from ml.lit import load_model, start_train
from ml.trans_linq2tsql import LinqToSqlVocab
from ml.translate_net import LiTranslator, convert_translate_file_to_csv, TranslateCsvDataset

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
# translate_ds = TranslateListDataset(translate_examples, vocab)

translate_csv_file_path = './output/linq_vlinq.csv'
convert_translate_file_to_csv('./train_data/linq_vlinq.txt', translate_csv_file_path)
translate_ds = TranslateCsvDataset(translate_csv_file_path, vocab)

model_args = {
    'vocab': vocab
}

model = start_train(LiTranslator, model_args,
                    translate_ds,
                    batch_size=1,
                    device='cuda',
                    max_epochs=200)

# model = load_model(LiTranslator, model_args)

for src, tgt in get_file_by_lines_iter('./train_data/linq_vlinq_test.txt', 2):
    linq_code = model.infer(src)
    tgt_expected = vocab.decode(vocab.encode(tgt)).rstrip()
    src = ' '.join(src.split(' ')).rstrip()
    print(f'{src=}')
    if linq_code != tgt_expected:
        info(f'"{tgt_expected}"')
        info_error(f'"{linq_code}"')
    else:
        print(f'"{linq_code}"')
    print("\n")
