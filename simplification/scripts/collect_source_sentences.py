"""Collect Japanese sentences from Wikipedia into a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import pandas as pd
import wikipedia


def extract_sentences(
    title: str,
    *,
    num_sentences: int = 5,
    min_sentence_length: int = 10,
    max_sentence_length: int = 50,
    wikipedia_lang: str = "ja",
) -> list[str]:
    """Extract sentences within the target length range from a Wikipedia page."""
    wikipedia.set_lang(wikipedia_lang)

    try:
        page = wikipedia.page(title)
    except wikipedia.DisambiguationError as exc:
        raise ValueError(f"Ambiguous title '{title}': {exc.options}") from exc
    except wikipedia.PageError as exc:
        raise ValueError(f"Page not found for title '{title}'") from exc

    lines = [line.strip() for line in page.content.split("\n") if line.strip()]
    lines = [line for line in lines if line.endswith("。") or line.endswith("．")]

    sentences: list[str] = []
    for line in lines:
        fragments = [
            fragment.strip() + "。"
            for fragment in line.replace("．", "。").split("。")
            if fragment.strip()
        ]
        sentences.extend(fragments)

    sentences = [
        sentence
        for sentence in sentences
        if min_sentence_length <= len(sentence) <= max_sentence_length
    ]

    if len(sentences) <= num_sentences:
        return sentences

    random.shuffle(sentences)
    return sentences[:num_sentences]


def get_page_names(keyword: str, wikipedia_lang: str = "ja") -> list[str]:
    """Search Wikipedia and return page titles for the keyword."""
    wikipedia.set_lang(wikipedia_lang)
    return wikipedia.search(keyword)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect sentences from Japanese Wikipedia into a CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/source_sentences.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--queries",
        nargs="*",
        default=["人工知能", "大学", "文学", "情報", "スポーツ"],
        help="Search keywords to query on Wikipedia.",
    )
    parser.add_argument("--lang", default="ja", help="Wikipedia language code.")
    parser.add_argument("--num-sentences", type=int, default=5)
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = None
    processed_title_set: set[str] = set()
    if output_path.exists():
        try:
            dataframe = pd.read_csv(output_path)
            processed_title_set = set(dataframe["title"].tolist())
        except Exception:
            dataframe = None

    if dataframe is None:
        dataframe = pd.DataFrame(columns=["query", "title", "sentence"])

    for query in args.queries:
        page_titles = get_page_names(query, wikipedia_lang=args.lang)
        for title in page_titles:
            if title in processed_title_set:
                continue
            print(f"Processing title: {title} (query: {query})")
            try:
                sentences = extract_sentences(
                    title,
                    num_sentences=args.num_sentences,
                    min_sentence_length=args.min_length,
                    max_sentence_length=args.max_length,
                    wikipedia_lang=args.lang,
                )
                sentences = [
                    sentence.replace("\n", " ").replace("\r", " ").replace(",", "，")
                    for sentence in sentences
                ]
                for sentence in sentences:
                    new_row = {"query": query, "title": title, "sentence": sentence}
                    dataframe = pd.concat(
                        [dataframe, pd.DataFrame([new_row])], ignore_index=True
                    )
            except ValueError as exc:
                print(exc)
            processed_title_set.add(title)
        dataframe.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
