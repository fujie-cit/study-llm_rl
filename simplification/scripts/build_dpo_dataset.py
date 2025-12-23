"""Build JSONL preference data for DPO training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = "ユーザの発言を平易化し，さらに書き言葉から話し言葉に変換してください。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert spreadsheet preferences into DPO JSONL format.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../data/sentences_with_preferences.xlsx"),
        help="Input Excel file with positive/negative sentences.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/dpo_dataset.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument(
        "--system-prompt",
        default=SYSTEM_PROMPT,
        help="System prompt for DPO data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = pd.read_excel(args.input, sheet_name=args.sheet)

    dataframe = dataframe.dropna(subset=["positive_sentence"])

    data = []
    for _, row in dataframe.iterrows():
        prompt = (
            "<|system|>"
            f"{args.system_prompt}"
            "</s><|user|>"
            f"{row['sentence']}"
            "</s><|assistant|>"
        )
        chosen = f"{row['positive_sentence']}</s>"
        rejected = f"{row['negative_sentence']}</s>"
        data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in data:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
