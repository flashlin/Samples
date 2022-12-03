from ml.lit import start_train
from ml.predict_word_models.model import WordPredictor, Vocab, WordDataset

if __name__ == '__main__':
    vocab = Vocab()

    model_type = WordPredictor
    model_args = {
        'vocab': vocab
    }

    train_ds = WordDataset('./train_data/linq_vlinq.txt', vocab)

    model = start_train(model_type, model_args,
                        train_ds,
                        batch_size=16,
                        device='cuda',
                        max_epochs=100)

    # model = load_model(model_type, model_args)

