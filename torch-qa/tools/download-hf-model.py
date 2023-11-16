from huggingface_hub import snapshot_download

model_id="Vasanth/codellama2-finetuned-codex-fin"
model_id="Qwen/Qwen-14B-Chat"
model_id="twodgirl/Qwen-14b-GGML"
model_id="THUDM/chatglm3-6b-32k"
model_id="Qwen/Qwen-14B-Chat"

model_name = model_id.split("/", 1)[-1]
snapshot_download(repo_id=model_id, 
                  local_dir=f"download-{model_name}",
                  local_dir_use_symlinks=False, 
                  revision="main")

