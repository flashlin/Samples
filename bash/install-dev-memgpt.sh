conda activate base
conda remove memgpt
conda create -n memgpt python=3.11.5
conda activate memgpt

pip install openai
pip install pyautogen pymemgpt