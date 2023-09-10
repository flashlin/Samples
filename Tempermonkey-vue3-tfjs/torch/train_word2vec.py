from ml.lit import start_train, load_model
from ml.trans_linq2tsql import LinqToSqlVocab
from ml.word2vec.model import Word2Vec, TranslateFileDataset

if __name__ == '__main__':
    vocab = LinqToSqlVocab()

    model_type = Word2Vec
    model_args = {
        'vocab_size': vocab.get_size()
    }

    train_ds = TranslateFileDataset('./train_data/linq_vlinq.txt', vocab)

    model = start_train(model_type, model_args,
                        train_ds,
                        batch_size=16,
                        device='cuda',
                        max_epochs=300)

    # model = load_model(model_type, model_args)

    word1 = '@tb_as2'
    vec1 = model.infer(vocab, word1)
    print(f'vec = {word1=} {vec1=}')

    word2 = '@tb_as1'
    vec2 = model.infer(vocab, word2)
    print(f'vec = {vec1-vec2=}')
