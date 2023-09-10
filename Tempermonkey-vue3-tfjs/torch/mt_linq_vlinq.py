from common.io import info, info_error, get_file_by_lines_iter
from ml.gnmt.net import LiGmnTranslator
from ml.lit import load_model, start_train
from ml.trans_linq2tsql import LinqToSqlVocab, EnglishVocab
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


def train():
    # vocab = EnglishVocab()

    translate_csv_file_path = './output/linq_vlinq.csv'
    convert_translate_file_to_csv('./train_data/linq_vlinq.txt', translate_csv_file_path)

    model_type = LiTranslator
    # model_type = LiGmnTranslator

    model_args = {
        'vocab': vocab
    }

    # translate_ds = TranslateCsvDataset(translate_csv_file_path, vocab)
    # model = start_train(model_type, model_args,
    #                     translate_ds,
    #                     batch_size=1,
    #                     device='cuda',
    #                     max_epochs=300)

    model = load_model(model_type, model_args)

    for src, tgt in get_file_by_lines_iter('./train_data/linq_vlinq_test.txt', 2):
        src = src.rstrip()
        tgt = tgt.rstrip()
        linq_code = model.infer(src)
        tgt_expected = vocab.decode(vocab.encode(tgt)).rstrip()
        src = ' '.join(src.split(' ')).rstrip()
        print(f'src="{src}"')
        if linq_code != tgt_expected:
            info(f'"{tgt_expected}"')
            info_error(f'"{linq_code}"')
        else:
            print(f'"{linq_code}"')
        print("\n")


def test():
    info(f'{vocab.get_size()=}')
    s1 = 'from tb1 in customer where 1==2 && 3==4 select new { tb1.name, 123 }'
    print(f'{s1=}')
    print(f'{vocab.parse_to_tokens(s1)=}')
    s1_values = vocab.encode(s1)
    print(f'{s1_values=}')
    s11 = vocab.decode(s1_values)
    print(f'{s11=}\n')

    s2 = 'from @tb_as1 in @tb1 select @tb_as1.@fd1, @fd_as1 = @n1'
    print(f'{s2=}')
    s2_values = vocab.encode(s2)
    print(f'{s2_values=}')
    s22 = vocab.decode(s2_values)
    print(f'{s22=}')


if __name__ == '__main__':
    train()
    #test()
