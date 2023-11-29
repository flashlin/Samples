import json
import os
import torch
from finetune_utils import load_finetune_config
from finetune_lit import load_peft_model, ask_llama2_instruction_prompt, ask_orca2_instruction_prompt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge, save, and convert PEFT model weights")
    parser.add_argument("model_name", nargs='?', help="Name of your model")
    args = parser.parse_args()

    config = load_finetune_config()
    device = "cuda"

    model_name = config['model_name']
    if args.model_name is not None:
        model_name = args.model_name

    print(f"use model: {model_name}")

    base_model = f"../models/{model_name}"
    peft_model = f"./outputs/{model_name}-tuned"

    model, tokenizer = load_peft_model(base_model, peft_model)

    generation_config = model.generation_config
    generation_config.max_new_tokens = 200
    generation_config.temperature = 0.2  # 0.7
    # generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        while True:
            user_input = input("query: ")
            if user_input == '/bye':
                break

            answer = ask_llama2_instruction_prompt(model=model,
                                                   generation_config=generation_config,
                                                   tokenizer=tokenizer,
                                                   device=device,
                                                   question=user_input)


            # answer = ask_orca2_instruction_prompt(model=model,
            #                                       generation_config=generation_config,
            #                                       tokenizer=tokenizer,
            #                                       device=device,
            #                                       question=user_input)
            print(answer)
