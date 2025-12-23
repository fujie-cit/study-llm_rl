"""Train a DPO LoRA adapter for Japanese sentence simplification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer


BASE_MODEL = "sbintuitions/sarashina2.2-3b-instruct-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DPO LoRA adapter for sentence simplification.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../data/dpo_dataset.jsonl"),
        help="Path to DPO JSONL data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs/dpo_lora"),
        help="Directory to save the LoRA adapter.",
    )
    parser.add_argument("--model", default=BASE_MODEL, help="Base model name.")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_ds = Dataset.from_list(data)

    ref_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=torch.float16,
    )

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    policy_model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=torch.float16,
    )

    policy_model = FastLanguageModel.get_peft_model(
        policy_model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        use_gradient_checkpointing="unsloth",
    )

    policy_model.config.use_cache = False

    dpo_args = DPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        beta=args.beta,
        logging_steps=10,
        save_steps=50,
        fp16=torch.cuda.is_available(),
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_ds,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
