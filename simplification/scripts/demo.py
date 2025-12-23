"""Demo for running the DPO-tuned Sarashina model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
SYSTEM_PROMPT = "ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。"


def load_model(adapter_dir: Path, *, load_in_4bit: bool = True):
    """Load base model + LoRA adapter for inference."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_dir, torch_dtype=torch.float16)

    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def build_prompt(text: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def generate(
    model,
    tokenizer,
    text: str,
    *,
    max_new_tokens: int = 96,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    prompt = build_prompt(text, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Japanese sentences into simpler, spoken-style wording "
            "using the DPO-tuned Sarashina model."
        )
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="One or more Japanese sentences to convert (omit to run canned examples).",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("../outputs/dpo_lora"),
        help="Directory containing the trained LoRA adapter.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit loading (uses more VRAM).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, tokenizer = load_model(args.adapter_dir, load_in_4bit=not args.no_4bit)

    samples: List[str] = args.text or [
        "本研究では音声認識技術の性能評価を行った。",
        "計画は予定通り進行している。",
        "この装置は高い信頼性を有している。",
        "装置の設定を変更することで性能が向上した。",
        "彼は難しい問題を簡単に解決した。",
        "25年度当初予算ベースの新規国債発行額は28.6兆円だった。",
        "市場からの信認にも配慮すると訴えた。",
        "近年、常態化していた補正予算ありきの予算編成のあり方に疑問を示した。",
    ]

    for idx, sentence in enumerate(samples, 1):
        converted = generate(
            model,
            tokenizer,
            sentence,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"[{idx}] 原文 : {sentence}")
        print(f"    変換 : {converted}\n")


if __name__ == "__main__":
    main()
