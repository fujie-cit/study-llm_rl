# 強化学習（DPO）による日本語文平易化モデルの構築

## 概要

日本語LLMの `sbintuitions/sarashina2.2-3b-instruct-v0.1` を基盤モデルとして，
DPO（Direct Preference Optimization）を用いて強化学習を行い，
日本語文の平易化モデルを構築する例です．

Unsloth, TRL を利用しています．

詳しくはそれぞれのプログラムを参照してください．

## 作業

### ステップ1: 原文データの作成

> [!NOTE]
> このステップの実行結果はすでに `wikipedia_sentences.csv` にあります．

```
python make_source_sentence_data.py
```


Wikipediaから日本語の文書を取得し，文単位に分割して
`wikipedia_sentence.csv` に保存します．

このCSVには以下の列が含まれます:
| 列名 | 説明 |
| ---- | ---- |
| query | 記事の検索に使用したキーワード |
| title | 記事のタイトル |
| sentence | 記事から抽出された文 |

### ステップ2: 対象文の選定

> [!NOTE]
> このステップの実行結果はすでに `wikipedia_sentences_checked.xlsx` にあります．

このステップは手作業になります．

`wikipedia_sentences.csv` を Microsoft Excel などで開き，
新たに `source_checked` 列を追加し，
平易化の対象にする文には `TRUE`, そうでないものには `FALSE` を入力します．

結果のファイルを `wikipedia_sentences_checked.xlsx` に保存し，
アップロードし直します．


### ステップ　3: 負例の作成

> [!NOTE]
> このステップの実行結果はすでに `wikipedia_sentences_with_negatives.xlsx` にあります．

```
python make_negative_data.py
```

`wikipedia_sentences_checked.xlsx` の平易化対象文に対して，
基盤モデル（`sbintuitions/sarashina2.2-3b-instruct-v0.1`）をそのまま
用いて平易化した結果を負例として生成し，
`wikipedia_sentences_with_negatives.xlsx` に保存します．

生成の際には，システムプロンプトに
```
ユーザの発言を平易化・口語化してください。ですます口調で答えてください。
```
を用いています．
基盤モデルは平易化に特化していないため，余計な出力を含むことが多々あります．

出力ファイルである `wikipedia_sentences_with_negatives.xlsx` には
以下の列が追加されます．

| 列名 | 説明 |
| ---- | ---- |
| negative_sentence | 基盤モデルによって生成された負例文 |
| positive_sentence | 同上 |

### ステップ 4: 正例の作成

> [!NOTE]
> このステップの実行結果はすでに `wikipedia_sentences_with_positives_positives.xslx` にあります．

このステップも手作業になります．

ステップ 3 で作成した `wikipedia_sentences_with_negatives.xlsx` を Microsoft Excel などで開き，
`negative_sentence` 列の内容を参考にして，
より平易な文（理想的な文）を `positive_sentence` 列に手作業で入力します．

ファイル名を `wikipedia_sentences_with_positives_positives.xslx` として保存し，アップロードし直します．

### ステップ 5: 学習データの作成

> [!NOTE]
> このステップの実行結果はすでに `simplified_data.jsonl` にあります．

```
python make_data.py
```

`wikipedia_sentences_with_positives_positives.xslx` から
学習に用いるデータを抽出し，`simplified_data.jsonl` に保存します．

### ステップ 6: 学習

```
python train.py
```

基盤モデル `sbintuitions/sarashina2.2-3b-instruct-v0.1` に対して
DPO（Direct Preference Optimization）を用いて強化学習を行い，
平易化モデルを構築します．

結果は `dpo_lora_out` ディレクトリに保存されます．

### ステップ 7: デモ

```
python sarashina_rl_unsloth_demo.py
```

基盤モデルに対して DPO で学習した平易化モデルを適用し，
いくつかのサンプル文を平易化します．







