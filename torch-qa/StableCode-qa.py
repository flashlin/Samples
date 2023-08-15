from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

MODEL_NAME = "stabilityai/stablecode-instruct-alpha-3b"
#MODEL_NAME = "TheBloke/stablecode-instruct-alpha-3b-GPTQ"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype="auto"
)
model.to(device)


# model = AutoGPTQForCausalLM.from_quantized(MODEL_NAME,
#                                            use_safetensors=True,
#                                            trust_remote_code=False,
#                                            use_triton=False,
#                                            quantize_config=None)
# model.to(device)


def query(question):
    prompt = f"""###Instruction
{question}
###Response"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs.pop("token_type_ids")
    inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)
    tokens = model.generate(**inputs,
                            max_new_tokens=100,
                            temperature=0.7,
                            do_sample=True,
                            )
    answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(answer)


query("""Generate a C# function to find number of CPU cores""")
