import argparse
from transformers_lit import load_pretrained_model, load_pretrained_tokenizer
from finetune_lit import load_yaml_config, load_llm_model, ask_llm_prompt


def get_args():
    parser = argparse.ArgumentParser(description="Infer GGUF Lab")
    parser.add_argument("model_name", nargs='?', help="your gguf model name")
    args = parser.parse_args()
    return args


def run_pretrained_model():
    config = load_yaml_config("infer-llm.yaml")
    model_name = config["model_name"]
    model_path = f"../models/{model_name}"
    llm, tokenizer, generation_config = load_llm_model(model_path)
    print(f"{model_name=}")
    while True:
        config = load_yaml_config("infer-llm.yaml")
        prompt = config["prompt"]
        bos_token = config["bos_token"]
        eos_token = config["eos_token"]

        user_input = input("query: ")
        if user_input == '/bye':
            break
        if user_input == "":
            user_input = config["user_input"]
        print(f"{user_input=}")
        answer = ask_llm_prompt(llm=llm,
                                generation_config=generation_config,
                                tokenizer=tokenizer,
                                device='cuda',
                                instruction=config["instruction"],
                                user_input=user_input,
                                prompt_template=prompt,
                                bos_token=bos_token,
                                eos_token=eos_token
                                )
        print(f"{answer}")
        print("--------------------------------------------------")
        print("\r\n\r\n\r\n")


if __name__ == '__main__':
    model_name = "Taiwan-LLM-7B-v2.0.1-chat"
    model_path = f"../models/{model_name}"
    model = load_pretrained_model(model_path,
                                  device_map='auto')
    tokenizer = load_pretrained_tokenizer(model_path)
    device = 'cuda'

    chat = [
        # {"role": "system", "content": "你講中文"},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]

    system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
    user_message = "what is your name?"
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {user_message} ASSISTANT:"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    output_ids = model.generate(inputs["input_ids"],
                                max_new_tokens=1024)
    answer = tokenizer.batch_decode(output_ids)[0]

    print(answer)
