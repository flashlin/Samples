import transformers
import torch
import math
import sys
from llama_attn_replace import replace_llama_attn

MODEL_PATH = "../models/Llama-2-7b-longlora-16k"
cache_dir = "./cache"
context_size = -1
max_gen_len = 512
temperature = 0.6
top_p = 0.9
flash_attn = True


def format_prompt(material, message, material_type="book", material_title=""):
    if material_type == "paper":
        prompt = f"Below is a paper. Memorize the material and answer my question after the paper.\n {material} \n "
    elif material_type == "book":
        material_title = ", %s"%material_title if len(material_title)>0 else ""
        prompt = f"Below is some paragraphs in the book{material_title}. Memorize the content and answer my question after the book.\n {material} \n "
    else:
        prompt = f"Below is a material. Memorize the material and answer my question after the material. \n {material} \n "
    message = str(message).strip()
    prompt += f"Now the material ends. {message}"

    return prompt


def read_txt_file(material_txt):
    if not material_txt.split(".")[-1] == 'txt':
        raise ValueError("Only support txt or pdf file.")
    content = ""
    with open(material_txt) as f:
        for line in f.readlines():
            content += line
    return content


def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(material, question, material_type="", material_title=None):
        material = read_txt_file(material)
        prompt = format_prompt(material, question, material_type, material_title)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache
        )
        out = tokenizer.decode(output[0], skip_special_tokens=True)

        out = out.split(prompt)[1].strip()
        return out

    return response


if __name__ == '__main__':
    replace_llama_attn()
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        MODEL_PATH,
        cache_dir=cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_PATH,
        cache_dir=cache_dir,
        model_max_length=context_size if context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    respond = build_generator(model, tokenizer, temperature=temperature, top_p=top_p,
                              max_gen_len=max_gen_len, use_cache=not flash_attn)

    output = respond(material="../data/support.txt", question="How to add a new B2B2C domain ?",
                     material_type="material", material_title="")
    print("output", output)
