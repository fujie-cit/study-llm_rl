# sbintuitions/sarashina2.2-3b-instruct-v0.1 を用いて平易化・口語化
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import pandas as pd
import tqdm

# モデルのロード
model_name = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
set_seed(123)


def make_negative_sentence(sentence: str) -> str:
    user_input = [
        {"role": "system", "content": "ユーザの発言を平易化・口語化してください。ですます口調で答えてください。"},
        {"role": "user", "content": sentence}
    ]
    
    # 否定化の生成
    responses = chat_pipeline(user_input, 
                             max_new_tokens=512,
                             do_sample=True,
                             num_return_sequences=1)
    negative_sentence = responses[0]['generated_text'][-1]['content']
    
    return negative_sentence


# データの読み込み
filename = "wikipedia_sentences_checked.xlsx"
dataframe = pd.read_excel(filename)

new_rows = []
for index, row in dataframe.iterrows():
    original_sentence = row['sentence']
    source_checked = row['source_check']
    if source_checked:
        negative_sentence = make_negative_sentence(original_sentence)
        print(f"Original: {original_sentence}")
        print(f"Negative: {negative_sentence}")
        print("-" * 50)
        # row に negative_sentence, positive_sentence を追加する．
        # 内容は両方とも negative_sentence とする．
        new_row = row.to_dict()
        new_row['negative_sentence'] = negative_sentence
        new_row['positive_sentence'] = negative_sentence
        new_rows.append(new_row)
        # if len(new_rows) == 10:
        #     break
    else:
        print(f"Skipping unchecked sentence at index {index}.")
        new_row = row.to_dict()
        new_row['negative_sentence'] = ""
        new_row['positive_sentence'] = ""
        new_rows.append(new_row)

# 新しいデータフレームの作成
new_dataframe = pd.DataFrame(new_rows)
# 新しいExcelファイルに保存
new_filename = "wikipedia_sentences_with_negatives.xlsx"
new_dataframe.to_excel(new_filename, index=False)
print(f"New data saved to {new_filename}")
