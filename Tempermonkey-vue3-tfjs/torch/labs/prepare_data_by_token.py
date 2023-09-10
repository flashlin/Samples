from utils.offset_tokenizr import Linq2TSqlEmbedding
from utils.stream import int_list_to_str


class TranslationFileIterator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.emb = Linq2TSqlEmbedding()

    def __iter__(self):
        with open(self.file_path, "r", encoding='UTF-8') as f:
            for idx, line in enumerate(f):
                if idx % 2 == 0:
                    linq_tokens, linq_values = self.emb.encode_source(line)
                else:
                    sql_tokens, sql_values = self.emb.encode_target(line, linq_tokens)
                    yield linq_values, sql_values


def convert_file_to_csv(
        file_iterator,
        output_file_path: str = "./output/linq-sample.csv"
):
    with open(output_file_path, "w", encoding='utf-8') as csv:
        csv.write('src\ttgt\n')
        for src_values, tgt_values in file_iterator:
            csv.write(int_list_to_str(src_values))
            csv.write('\t')
            csv.write(int_list_to_str(tgt_values))
            csv.write('\n')


def main():
    convert_file_to_csv(TranslationFileIterator("../../data/linq-sample.txt"))


if __name__ == "__main__":
    main()
