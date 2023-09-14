# pip install ctransformers>=0.2.24
# pip install ctransformers[cuda]>=0.2.24
from ctransformers import AutoModelForCausalLM

model_id = ""

llm = AutoModelForCausalLM.from_pretrained("models/Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q4_K_M.gguf",
                                           model_type="llama",
                                           gpu_layers=30)

print(llm("AI is going to"))
