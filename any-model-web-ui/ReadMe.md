clone https://github.com/oobabooga/text-generation-webui
nvcc --version
查看版本
https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

for GPU:
pip uninstall -y llama-cpp-python
$env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"
$env:FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir


python server.py

# If you get CUDA Setup failed despite GPU being available.: 
pip install bitsandbytes-windows

# If you get AttributeError: module 'bitsandbytes.nn' has no attribute 'Linear4bit'. Did you mean: 'Linear8bitLt'?
pip install git+https://github.com/huggingface/peft@27af2198225cbb9e049f548440f2bd0fba2204aa --force-reinstall --no-deps

下載各種模型 要找 gptq 版本
https://huggingface.co/TheBloke

python download-model.py TheBloke/WizardLM-13B-V1.1-GPTQ