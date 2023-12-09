import logging
from typing import Any
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_pretrained_model(model_path: str, device_map: Any):
    if device_map is None:
        device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        local_files_only=True,
    )
    return model


def load_pretrained_tokenizer(model_path: str):
    return transformers.AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
    )
