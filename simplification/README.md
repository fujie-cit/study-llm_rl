# 強化学習（DPO）による日本語文平易化モデルの構築

## 概要

日本語 LLM の `sbintuitions/sarashina2.2-3b-instruct-v0.1` を基盤モデルとして、
DPO（Direct Preference Optimization）を用いて強化学習を行い、
日本語文の平易化モデルを構築する例です。

Unsloth, TRL を利用しています。

## ディレクトリ構成

- `scripts/`: 実行用のスクリプト
- `data/`: 生成・編集したデータ
- `outputs/`: 学習済み LoRA アダプタの出力先

## 作業手順

### ステップ1: 原文データの作成

> [!NOTE]
> このステップの実行結果はすでに `data/source_sentences.csv` にあります。

```bash
python scripts/collect_source_sentences.py \
  --output data/source_sentences.csv
```

Wikipedia から日本語の文書を取得し、文単位に分割して
`data/source_sentences.csv` に保存します。

この CSV には以下の列が含まれます:

| 列名 | 説明 |
| ---- | ---- |
| query | 記事の検索に使用したキーワード |
| title | 記事のタイトル |
| sentence | 記事から抽出された文 |

### ステップ2: 対象文の選定

> [!NOTE]
> このステップの実行結果はすでに `data/source_sentences_checked.xlsx` にあります。

このステップは手作業になります。

`data/source_sentences.csv` を Microsoft Excel などで開き、
新たに `source_check` 列を追加し、
平易化の対象にする文には `TRUE`、そうでないものには `FALSE` を入力します。

結果のファイルを `data/source_sentences_checked.xlsx` に保存し、
アップロードし直します。

### ステップ3: 負例の作成

> [!NOTE]
> このステップの実行結果はすでに `data/sentences_with_negatives.xlsx` にあります。

```bash
python scripts/generate_negatives.py \
  --input data/source_sentences_checked.xlsx \
  --output data/sentences_with_negatives.xlsx
```

`data/source_sentences_checked.xlsx` の平易化対象文に対して、
基盤モデル（`sbintuitions/sarashina2.2-3b-instruct-v0.1`）をそのまま
用いて平易化した結果を負例として生成し、
`data/sentences_with_negatives.xlsx` に保存します。

生成の際には、システムプロンプトに

```
ユーザの発言を平易化・口語化してください。ですます口調で答えてください。
```

を用いています。基盤モデルは平易化に特化していないため、
余計な出力を含むことが多々あります。

出力ファイルである `data/sentences_with_negatives.xlsx` には
以下の列が追加されます。

| 列名 | 説明 |
| ---- | ---- |
| negative_sentence | 基盤モデルによって生成された負例文 |
| positive_sentence | 同上 |

### ステップ4: 正例の作成

> [!NOTE]
> このステップの実行結果はすでに `data/sentences_with_preferences.xlsx` にあります。

このステップも手作業になります。

ステップ3で作成した `data/sentences_with_negatives.xlsx` を Microsoft Excel などで開き、
`negative_sentence` 列の内容を参考にして、
より平易な文（理想的な文）を `positive_sentence` 列に手作業で入力します。

ファイル名を `data/sentences_with_preferences.xlsx` として保存し、
アップロードし直します。

### ステップ5: 学習データの作成

> [!NOTE]
> このステップの実行結果はすでに `data/dpo_dataset.jsonl` にあります。

```bash
python scripts/build_dpo_dataset.py \
  --input data/sentences_with_preferences.xlsx \
  --output data/dpo_dataset.jsonl
```

`data/sentences_with_preferences.xlsx` から
学習に用いるデータを抽出し、`data/dpo_dataset.jsonl` に保存します。

### ステップ6: 学習

```bash
python scripts/train_dpo.py \
  --data data/dpo_dataset.jsonl \
  --output-dir outputs/dpo_lora
```

基盤モデル `sbintuitions/sarashina2.2-3b-instruct-v0.1` に対して
DPO（Direct Preference Optimization）を用いて強化学習を行い、
平易化モデルを構築します。

結果は `outputs/dpo_lora` ディレクトリに保存されます。

### ステップ7: デモ

```bash
python scripts/demo.py \
  --adapter-dir outputs/dpo_lora
```

基盤モデルに対して DPO で学習した平易化モデルを適用し、
いくつかのサンプル文を平易化します。
