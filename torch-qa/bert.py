from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# 轉換為半精度
model.half()

# 儲存模型
model.save_pretrained('models/bert-large-uncased-float16.bin')