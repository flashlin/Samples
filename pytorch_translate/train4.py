"""
books["train"][0]
{'id': '90560',
  'translation': {
    'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
    'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'
  }
}
"""
import os

from data_utils import load_csv_to_dataframe
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint
import random
import pandas as pd

from utils import info


def dataframe_to_model_inputs(df, tokenizer):
    inputs = [text for text in df["source_sentence"]]
    targets = [text for text in df["target_sentence"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs



def preprocess_function(examples, tokenizer, max_input_length=256, max_target_length=256):
    model_inputs = tokenizer(examples["source_sentence"], max_length=max_input_length, padding=True, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_sentence"], max_length=max_target_length, padding=True, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_function1(tokenizer, max_input_length=256, max_target_length=256):
    def fn(examples):
        model_inputs = tokenizer(examples["source_sentence"], max_length=max_input_length, padding=True, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["target_sentence"], max_length=max_target_length, padding=True, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return fn

from datasets import DatasetDict, Dataset


def main():
    raw_data_df = load_csv_to_dataframe()
    # print(f"{raw_data_df=}")
    info("raw_data_df loaded")

    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    dataset = Dataset.from_pandas(raw_data_df)
    tokenized_datasets = dataset.map(preprocess_function1(tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["source_sentence", "target_sentence"])
    train_test = tokenized_datasets.train_test_split(test_size=0.2)
    SEED = 42
    tokenized_datasets_split = DatasetDict({
        'train': train_test['train'],
        'test': train_test['test']})
    train_dataset = tokenized_datasets_split["train"].shuffle(seed=SEED)
    test_dataset = tokenized_datasets_split["test"].shuffle(seed=SEED)

    # tokenized_books = dataframe_to_model_inputs(raw_data_df, tokenizer)
    tokenized_books = train_dataset
    print(tokenized_books)

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    info("model")

    # nvcc --version
    RESULTS_PATH = "./results"

    training_args = Seq2SeqTrainingArguments(
        output_dir=RESULTS_PATH,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=20,
        fp16=False,
        resume_from_checkpoint="./checkpoints"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books,
        eval_dataset=tokenized_books,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    last_checkpoint = None
    if os.path.isdir(RESULTS_PATH):
        last_checkpoint = get_last_checkpoint(RESULTS_PATH)

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()


main()
