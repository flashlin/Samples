import argparse
from finetune_lit import load_yaml_config, load_llm_model, ask_llm_prompt


def get_args():
    parser = argparse.ArgumentParser(description="Infer GGUF Lab")
    parser.add_argument("model_name", nargs='?', help="your gguf model name")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = load_yaml_config("infer-llm.yaml")
    model_name = config["model_name"]
    model_path = f"../models/{model_name}"
    llm, tokenizer, generation_config = load_llm_model(model_path)

    print(f"{model_name=}")
    while True:
        user_input = input("query: ")
        if user_input == '/bye':
            break
        config = load_yaml_config("infer-llm.yaml")
        prompt = config["prompt"]
        if user_input == "":
            user_input = config["user_input"]
        # print(f"{user_input=}")
        # print("\r\n")
        answer = ask_llm_prompt(llm=llm,
                                generation_config=generation_config,
                                tokenizer=tokenizer,
                                device='cuda',
                                instruction=config["instruction"],
                                user_input=user_input,
                                prompt_template=prompt,
                                )
        print(f"{answer}")
        print("--------------------------------------------------")
        print("\r\n\r\n\r\n")

