from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

model_id = "facebook/opt-125m"
model_id = "models/sqlcoder"

quantization_config = GPTQConfig(
     bits=4,
     group_size=128,
     dataset="c4",
     desc_act=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
quant_model = AutoModelForCausalLM.from_pretrained(model_id,
                                                   quantization_config=quantization_config,
                                                   device_map='auto')
# You can make sure the model has been correctly quantized by checking the attributes of the linear layers,
# they should contain qweight and qzeros attributes that should be in torch.int32 dtype.
qdict = quant_model.model.decoder.layers[0].self_attn.q_proj.__dict__
print(qdict)
exit()

print("Now let's perform an inference on the quantized model. Use the same API as transformers!")
tokenizer = AutoTokenizer.from_pretrained(model_id)
text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt").to(0)

out = quant_model.generate(**inputs)
print(tokenizer.decode(out[0], skip_special_tokens=True))


# You can also quantize a model by passing a custom dataset,
# for that you can provide a list of strings to the quantization config.
# A good number of sample to pass is 128.
# If you do not pass enough data, the performance of the model will suffer.
from transformers import AutoModelForCausalLM, GPTQConfig, AutoTokenizer
model_id = "facebook/opt-125m"

quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    dataset=["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
quant_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config,
                                                   torch_dtype=torch.float16,
                                                   device_map="auto")

#
text = "My name is"
inputs = tokenizer(text, return_tensors="pt").to(0)
out = quant_model.generate(**inputs)
print(tokenizer.decode(out[0], skip_special_tokens=True))


# Share quantized models on Hub
from huggingface_hub import notebook_login
notebook_login()
quant_model.push_to_hub("opt-125m-gptq-4bit")
tokenizer.push_to_hub("opt-125m-gptq-4bit")


# Below we will load a llama 7b quantized in 4bit.
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Furthermore, we can see that from the quantization config that we are using exllama kernel (disable_exllama = False).
# Note that it only works with 4-bit model.
model.config.quantization_config.to_dict()


# Train quantized model using 🤗 PEFT
from peft import prepare_model_for_kbit_training
model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config_loading, device_map="auto")

# First, we have to apply some preprocessing to the model to prepare it for training.
# For that, use the prepare_model_for_kbit_training method from PEFT.
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Then, we need to convert the model into a peft model using get_peft_model.
from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["k_proj","o_proj","q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Finally, let's load a dataset and we can train our model.
from datasets import load_dataset
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# starting train
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
# needed for llama 2 tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_hf"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

