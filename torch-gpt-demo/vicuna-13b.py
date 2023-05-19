from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "flax-community/vicuna-13B"  #23GB
model_name = "microsoft/llama-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
