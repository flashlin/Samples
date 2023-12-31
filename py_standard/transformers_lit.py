import logging
from typing import Any

import torch
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


def create_nf4_model_config():
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return nf4_config
