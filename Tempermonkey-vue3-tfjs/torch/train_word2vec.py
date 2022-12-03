from ml.lit import start_train
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
                        batch_size=10,
                        device='cuda',
                        max_epochs=300)

    word1 = 'select'
    vec1 = model.infer(vocab, word1)
    print(f'vec = {word1=} {vec1=}')
