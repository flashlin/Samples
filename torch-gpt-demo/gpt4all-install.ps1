Import-Module "$($env:psm1HOME)/common.psm1" -Force
InvokeCmd "pip install -r ./gpt4all-req.txt"
InvokeCmd "conda install -c conda-forge llama"
InvokeCmd "python -m llama.download --model_size 7B --folder llama/"