# %%
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.dpo_config import DPOConfig

# %%
# -----------------------------
# 0) データ
#   形式: prompt / chosen / rejected
# -----------------------------
import json
data = [json.loads(line) for line in open('simplified_data.jsonl', 'r', encoding='utf-8')]
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
    per_device_train_batch_size=3,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    num_train_epochs=3,
    beta=0.1,
    logging_steps=10,
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