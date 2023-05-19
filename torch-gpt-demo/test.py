from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# conda install cudatoolkit
# conda install cudnn

model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)


max_new_tokens = 20


def generate_from_model(model, tokenizer, text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda())
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)


text = "Hello my name is"
generate_from_model(model_8bit, tokenizer, text)

