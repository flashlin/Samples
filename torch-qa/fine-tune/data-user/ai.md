Question: How to export LLM gguf model file from LLM bin model file?
Answer:
Knowledge as of 2023/11/15,
```bash
pip install huggingface_hub
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt
```

create .py file with following code, and run it.
```py
from huggingface_hub import snapshot_download
model_id="Vasanth/codellama2-finetuned-codex-fin"
snapshot_download(repo_id=model_id, 
    local_dir="download-hf",
    local_dir_use_symlinks=False, 
    revision="main")
```

use convert.py to convert bin model to gguf model. for outtype, can use q8_0 or f16 for model.
```bash
python llama.cpp/convert.py download-hf \
  --outfile my-codellama.gguf \
  --outtype q8_0
```

create repo on huggingface hub, and upload gguf model file.
```py
from huggingface_hub import HfApi
api = HfApi()

model_id = "YourName/YourModelName"
api.create_repo(model_id, exist_ok=True, repo_type="model")
api.upload_file(
    repo_id=model_id,
    path_or_fileobj="my-codellama.gguf",
    path_in_repo="my-codellama.gguf"
)
```

Question: How to use llama.cpp ?
Answer:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

run server.exe with following command.
```bash
.\server.exe -c 4096 --host 0.0.0.0 --port 8080 -t 16 --n-gpu-layers 32 --mlock -m ./models/deepseek-coder-6.7b-instruct.gguf
```

Install Continue extension in VSCode. Edit the ContinueConfig object 
in `~/.continue/config.py` (`%userprofile%\.continue\config.py` for Windows).
you can customize the following: 
```python
config = ContinueConfig(
    ...
    models=Models(
        default=LlamaCpp(
            max_context_length=4096,
            server_url="http://localhost:8080")
    ),
    #allow_anonymous_telemetry=True,
    #system_message="",
    #temperature=0.5,
)
```

For more detail, please refer https://continue.dev/docs/customization/overview.

Question: How to use OpenAI Whisper Speech to Text Model?
Answer:
Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web.

```bash
pip install --q --upgrade pip
pip install --q --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]
pip install --q flash-attn --no-build-isolation
```

By following the code below, you can transcribe audio files into text.
```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

audio = "GPT4Vision.WAV"
result = pipe(audio) # v2 gives better results if you don't provide a language.
print(result['text'])
print(result['chunks'])
```

Here are more examples.
```python
result = pipe(audio, generate_kwargs={"language": "english"})
result = pipe(audio, return_timestamps=True)
print(result["chunks"])
```