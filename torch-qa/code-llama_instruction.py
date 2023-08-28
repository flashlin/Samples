from typing import Optional
import fire
import torch

from llama import Llama

# fairscale
# fire
# sentencepiece

import os
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'


ckpt_dir = "models/CodeLlama-13B-Instruct-GPTQ/model.safetensors"
tokenizer_path = "models/CodeLlama-13B-Instruct-GPTQ/tokenizer.model"
max_seq_len = 512
max_batch_size = 8
temperature = 0.2
top_p: float = 0.95
max_gen_len = None

torch.distributed.init_process_group("gloo")

def main():
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    instructions = [
        [
            {
                "role": "system",
                "content": "Provide answers in C#",
            },
            {
                "role": "user",
                "content": "Write a function that computes the set of sums of all contiguous sublists of a given list.",
            }
        ],
    ]

    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for instruction, result in zip(instructions, results):
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )


if __name__ == "__main__":
    # fire.Fire(main)
    main()
