import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat",
                                            device_map='auto',
                                            torch_dtype=torch.float16,
                                            use_auth_token=True,
                                            load_in_8bit=True,
                                            #  load_in_4bit=True
                                            )