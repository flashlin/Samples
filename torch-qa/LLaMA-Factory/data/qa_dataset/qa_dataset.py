import json
import datasets
from typing import Any, Dict, List


_DESCRIPTION = "An QA of dataset."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = "qa.jsonl"


class QADataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "input": datasets.Value("string"),
            "output": datasets.Value("string"),
            "history": datasets.Sequence(datasets.Sequence(datasets.Value("string")))
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_path = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path
                }
            )
        ]

    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        # example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
        # for key, example in enumerate(example_dataset):
        #     yield key, example
        id = 0
        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                yield id, row
                id += 1
