import argparse
from finetune_lit import load_yaml_config, load_llm_model


def get_args():
    parser = argparse.ArgumentParser(description="Infer GGUF Lab")
    parser.add_argument("model_name", nargs='?', help="your gguf model name")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = load_yaml_config("infer-llm.yaml")
    model_name = config["model_name"]
    model_path = f"../models/{model_name}"
    # llm, tokenizer, generation_config = load_llm_model(model_path)

    while True:
        user_input = input("query: ")
        if user_input == '/bye':
            break
        config = load_yaml_config("infer-llm.yaml")
        prompt = config["prompt"]
        print(f"Prompt: {prompt}")
        