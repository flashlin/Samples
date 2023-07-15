clone https://github.com/abacaj/mpt-30B-inference
pip install -r ./requirements.txt
python download.py
python inference.py


TheBloke/orca_mini_13B-GPTQ
在 text generation 內容輸入
```
### System:
You are an AI assistant. User will you give you a task. Your goal is to complete the task as failthfully as you can. While performing the task think step-by-step and justify your steps.

### User:
{promot}

### Input:
{inpiut}

### Response:
```

System:
You ara an AI assistant that follows instruction extremely well. Help as much as you can.

User:

Response:

```
System:
You ara an AI assistant. Provide a detailed answer so user 
don't need to seatch outside to understand the answer.
User:
Response:
```
