# %%
import torch
from datasets import Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
# from peft import LoraConfig, get_peft_model
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.dpo_config import DPOConfig

# %%
# -----------------------------
# 0) データ（最小例）
#   実運用ではここを自前の preference dataset に置き換えます
#   形式: prompt / chosen / rejected
# -----------------------------
data = [
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>千葉工業大学は千葉県習志野市にある工業大学である。</s><|assistant|>",
        "chosen": "千葉工大は、千葉の習志野にある工業系の大学なんだ。</s>",
        "rejected": "千葉工業大学は、千葉の習志野にある工業系の大学だよ。簡単に言うと、工学を学ぶための大学なんだ。場所は、千葉県の習志野ってところにあって、分かりやすい場所にあるよ</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>本研究では音声認識技術の性能評価を行った。</s><|assistant|>",
        "chosen": "この研究では、音声認識がどれくらいうまく動くかを調べたよ。</s>",
        "rejected": "本研究では、音声認識技術について、その性能をさまざまな観点から評価を実施しました。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>実験の結果，有意な差が確認された。</s><|assistant|>",
        "chosen": "実験してみたら、はっきりした違いが見えたよ。</s>",
        "rejected": "実験の結果として、統計的に有意な差が確認されました。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>この装置は高い信頼性を有している。</s><|assistant|>",
        "chosen": "この装置は、かなり安心して使えるんだ。</s>",
        "rejected": "この装置は、高い信頼性を持っているため、安心して使用できます。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>本手法は従来法より計算量が少ない。</s><|assistant|>",
        "chosen": "このやり方は、前の方法より計算が軽いよ。</s>",
        "rejected": "本手法は、従来の方法と比較して計算量が少ないという特徴があります。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>システムはユーザの入力に応じて動作する。</s><|assistant|>",
        "chosen": "このシステムは、入力した内容に合わせて動くよ。</s>",
        "rejected": "このシステムは、ユーザからの入力に応じて適切に動作します。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>会議は予定通り実施された。</s><|assistant|>",
        "chosen": "会議は、予定どおりに行われたよ。</s>",
        "rejected": "会議は、当初の予定に従って実施されました。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>この問題は慎重に検討する必要がある。</s><|assistant|>",
        "chosen": "この問題は、ちゃんと考えたほうがいいね。</s>",
        "rejected": "この問題については、慎重な検討を行う必要があります。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>装置の設定を変更することで性能が向上した。</s><|assistant|>",
        "chosen": "設定を変えたら、性能が良くなったよ。</s>",
        "rejected": "装置の設定を変更することにより、性能の向上が確認されました。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>このアプリケーションは無料で利用可能である。</s><|assistant|>",
        "chosen": "このアプリは、タダで使えるよ。</s>",
        "rejected": "このアプリケーションは、無料で利用することができます。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>新しい手法の有効性が示された。</s><|assistant|>",
        "chosen": "新しいやり方がちゃんと使えるって分かったよ。</s>",
        "rejected": "新しい手法について、その有効性が示されました。</s>",
    },

    # 以下 20 件も同様の難易度・品質で生成
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>資料は事前に配布された。</s><|assistant|>",
        "chosen": "資料は、前もって配られてたよ。</s>",
        "rejected": "資料は、事前に参加者へ配布されていました。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>動作確認は完了している。</s><|assistant|>",
        "chosen": "動くかどうかは、もうチェック済みだよ。</s>",
        "rejected": "動作確認については、すでに完了しています。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>利用者からの要望が多く寄せられた。</s><|assistant|>",
        "chosen": "利用者から、たくさん要望が来てたよ。</s>",
        "rejected": "利用者からは、多くの要望が寄せられました。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>更新作業は近日中に行われる。</s><|assistant|>",
        "chosen": "更新作業は、近いうちにやるよ。</s>",
        "rejected": "更新作業は、近日中に実施される予定です。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>計画は予定通り進行している。</s><|assistant|>",
        "chosen": "計画は、予定どおり進んでるよ。</s>",
        "rejected": "計画は、当初の予定通りに進行しています。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>説明書を参照することを推奨する。</s><|assistant|>",
        "chosen": "説明書を見たほうがいいよ。</s>",
        "rejected": "説明書を参照することを推奨します。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>問題点がいくつか明らかになった。</s><|assistant|>",
        "chosen": "いくつか問題があることが分かったよ。</s>",
        "rejected": "いくつかの問題点が明らかになりました。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>この方法は初心者にも適している。</s><|assistant|>",
        "chosen": "この方法は、初心者でも使いやすいよ。</s>",
        "rejected": "この方法は、初心者にも適していると言えます。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>調査は来月実施予定である。</s><|assistant|>",
        "chosen": "調査は、来月やる予定だよ。</s>",
        "rejected": "調査は、来月に実施される予定です。</s>",
    },
    {
        "prompt": "<|system|>ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。</s><|user|>結果は後日報告する。</s><|assistant|>",
        "chosen": "結果は、あとで報告するね。</s>",
        "rejected": "結果については、後日報告します。</s>",
    },
]
train_ds = Dataset.from_list(data)

# %%
max_seq_length = 1024

model_name =  "sbintuitions/sarashina2.2-3b-instruct-v0.1"

# 1) reference model（π_ref）：LoRA を載せない・固定
ref_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.float16,
)

ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

# 2) policy model（π_theta）：同じベースからロードして LoRA を載せる
policy_model, _ = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.float16,
)

policy_model = FastLanguageModel.get_peft_model(
    policy_model,
    r=8,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    use_gradient_checkpointing="unsloth",
)

policy_model.config.use_cache = False  # 学習時は基本オフ

# %%
# -----------------------------
# 3) DPO 設定（β、バッチ等）
#   - beta が 2.5 の β
# -----------------------------
dpo_args = DPOConfig(
    output_dir="./dpo_lora_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    num_train_epochs=10,
    beta=0.1,
    logging_steps=1,
    save_steps=50,
    fp16=torch.cuda.is_available(),
    # 安定性のために max_length / max_prompt_length を入れるのが一般的
    max_length=512,
    max_prompt_length=256,
)


# %%
# -----------------------------
# 4) Trainer
#   - 内部で log πθ(chosen|prompt), log πθ(rejected|prompt)
#          log πref(chosen|prompt), log πref(rejected|prompt)
#     を計算して 2.5 の損失を作ります
# -----------------------------
trainer = DPOTrainer(
    model=policy_model,                 # πθ（LoRA付き）
    ref_model=ref_model,         # π_ref（固定）
    args=dpo_args,
    train_dataset=train_ds,
)

trainer.train()
trainer.save_model()