"""Generate baseline simplifications for checked sentences."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline simplifications for checked sentences.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../data/source_sentences_checked.xlsx"),
        help="Input Excel file with source sentences.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/sentences_with_negatives.xlsx"),
        help="Output Excel file with generated negatives.",
    )
    parser.add_argument(
        "--model",
        default="sbintuitions/sarashina2.2-3b-instruct-v0.1",
        help="Base model name.",
    )
    parser.add_argument(
        "--system-prompt",
        default="ユーザの発言を平易化・口語化してください。ですます口調で答えてください。",
        help="System prompt for generation.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def is_checked(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def make_negative_sentence(
    sentence: str,
    *,
    chat_pipeline,
    system_prompt: str,
    max_new_tokens: int,
) -> str:
    user_input = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sentence},
    ]

    responses = chat_pipeline(
        user_input,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_return_sequences=1,
    )
    return responses[0]["generated_text"][-1]["content"]


def main() -> None:
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    chat_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    set_seed(args.seed)

    dataframe = pd.read_excel(args.input)

    new_rows = []
    for _, row in dataframe.iterrows():
        original_sentence = row["sentence"]
        if is_checked(row.get("source_check")):
            negative_sentence = make_negative_sentence(
                original_sentence,
                chat_pipeline=chat_pipeline,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"Original: {original_sentence}")
            print(f"Negative: {negative_sentence}")
            print("-" * 50)
            new_row = row.to_dict()
            new_row["negative_sentence"] = negative_sentence
            new_row["positive_sentence"] = negative_sentence
        else:
            print("Skipping unchecked sentence.")
            new_row = row.to_dict()
            new_row["negative_sentence"] = ""
            new_row["positive_sentence"] = ""
        new_rows.append(new_row)

    new_dataframe = pd.DataFrame(new_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_dataframe.to_excel(args.output, index=False)
    print(f"New data saved to {args.output}")


if __name__ == "__main__":
    main()
