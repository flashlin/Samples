from common.io import info
from ml.seq2seq_net3 import LitTranslator
from preprocess_data import convert_translation_file_to_csv
from ml.lit import start_train, load_model
from utils.linq_tokenizr import LINQ_VOCAB_SIZE
from utils.tokenizr import PAD_TOKEN_VALUE
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE


# worked but 會隨機翻譯

# Transformer Parameters


def prepare_train_data():
    print("convert translation file to csv...")
    convert_translation_file_to_csv()
    print("done.")


def main():
    start_train(LitTranslator, device='cuda',
                max_epochs=100,
                src_vocab_size=LINQ_VOCAB_SIZE,
                tgt_vocab_size=TSQL_VOCAB_SIZE)


def infer():
    model = load_model(LitTranslator, src_vocab_size=LINQ_VOCAB_SIZE, tgt_vocab_size=TSQL_VOCAB_SIZE)
    print("start infer")

    def inference(text):
        print(text)
        sql = model.infer(text)
        print(sql)

    inference('from tb3 in customer select new tb3')
    inference('from c in customer select new { c.id, c.name }')


if __name__ == "__main__":
    info(f" {LINQ_VOCAB_SIZE=} {TSQL_VOCAB_SIZE=} {PAD_TOKEN_VALUE=}")
    #convert_translation_file_to_csv()
    #info("prepare train data")
    #prepare_train_data()
    #info("start train")
    #main()
    infer()
