import pandas as pd

df = pd.read_excel('wikipedia_sentences_with_negatives_positives.xlsx', sheet_name='Sheet1')

# positive_sentence が空の行を削除
df = df.dropna(subset=['positive_sentence'])
data = []
for index, row in df.iterrows():
    prompt = f"<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>{row['sentence']}</s><|assistant|>"
    chosen = row['positive_sentence'] + "</s>"
    rejected = row['negative_sentence'] + "</s>"
    data.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    })

import json
with open('simplified_data.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
