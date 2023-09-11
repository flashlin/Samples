# ctransformers == 0.2.24
# gradio
import gradio as gr
import time
from ctransformers import AutoModelForCausalLM


# 'codellama-13b-instruct.Q4_K_M.gguf' 16G RAM
model_path = 'models/CodeLlama-7B-Instruct-GGUF/codellama-7b-instruct.Q4_K_M.gguf'


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


title = 'Code Demo'
examples = [
    'Write a C# code to connect with a SQL database and list down all the tables',
    'Write a C# function that takes a filename string as input and returns the content of the file',
    "Write a .NET Core C# function that takes an object as input, sends an HTTP POST JSON request to http://128.3.3.1/api/sample/test, and deserializes the returned JSON content into a Customer object. If you don't know, please answer 'I don't know.'"
]

gr.ChatInterface(
    fn=llm_function,
    title=title,
    examples=examples
).launch()
