from datasets import load_dataset
import re
import pandas as pd
import os

# Load the dataset
dataset = load_dataset('timdettmers/openassistant-guanaco')

# Shuffle the dataset and slice it
dataset = dataset['train'].shuffle(seed=42).select(range(1000))

# Define a function to transform the data
def transform_conversation(example):
    conversation_text = example['text']
    segments = conversation_text.split('###')

    reformatted_segments = []

    # Iterate over pairs of segments
    for i in range(1, len(segments) - 1, 2):
        human_text = segments[i].strip().replace('Human:', '').strip()

        # Check if there is a corresponding assistant segment before processing
        if i + 1 < len(segments):
            assistant_text = segments[i+1].strip().replace('Assistant:', '').strip()

            # Apply the new template
            reformatted_segments.append(f'<s>[INST] {human_text} [/INST] {assistant_text} </s>')
        else:
            # Handle the case where there is no corresponding assistant segment
            reformatted_segments.append(f'<s>[INST] {human_text} [/INST] </s>')

    return {'text': '\\\n'.join(reformatted_segments)}


# Apply the transformation
transformed_dataset = dataset.map(transform_conversation)

save_path = "dataset"
df = pd.DataFrame(columns=['text'])  # 指定欄位名稱（header）
for example in transformed_dataset:
    data = {'text': example['text']}
    df = df._append(data, ignore_index=True)
# 儲存為 CSV 檔案
csv_file_path = os.path.join(save_path, 'my_llama2_dataset.csv')
df.to_csv(csv_file_path, index=False, encoding='utf-8')


# push to hf
#transformed_dataset.push_to_hub("mlabonne/guanaco-llama2-1k")