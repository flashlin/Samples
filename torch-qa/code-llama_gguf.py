# ctransformers == 0.2.24
# gradio
import gradio as gr
import time
from ctransformers import AutoModelForCausalLM


# 'codellama-13b-instruct.Q4_K_M.gguf' 16G RAM
model_path = 'models/CodeLlama-34B-Instruct-GGUF/codellama-34b-instruct.Q4_K_M.gguf'


def load_llm():
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type='llama',
        max_new_tokens=1096,
        repetition_penalty=1.13,
        temperature=0.1
    )
    return llm


def llm_function(message, chat_history):
    llm = load_llm()
    response = llm(message)
    output_texts = response
    return output_texts


title = 'CodeLlama 13B Demo'
examples = [
    'Write a C# code to connect with a SQL database and list down all the tables'
]

gr.ChatInterface(
    fn=llm_function,
    title=title,
    examples=examples
).launch()
