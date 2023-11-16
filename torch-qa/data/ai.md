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