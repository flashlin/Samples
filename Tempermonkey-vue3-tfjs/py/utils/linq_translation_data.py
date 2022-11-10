from common.csv_utils import CsvWriter
from common.io import read_text_file
from utils.linq_tokenizr import linq_encode
from utils.tsql_tokenizr import tsql_encode


class LinqTranslationData:
    def __init__(self, file: str):
        self.file = file
        lines = read_text_file(file)
        out_file = "./output/linq-translation.csv"

        with CsvWriter(out_file) as csv:
            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    linq_values = linq_encode(line)
                    csv.write(linq_values)
                else:
                    sql_values = tsql_encode(line)
                    csv.write(sql_values)
