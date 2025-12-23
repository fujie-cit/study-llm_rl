# Wikipediaを検索して，タイトルや本文からランダムに文を抽出してデータ化するプログラム
import wikipedia
import pandas as pd
import os

def extract_sentences(title: str,
                      num_sentences: int = 5,
                      min_sentence_length: int = 10,
                      max_sentence_length: int = 50,
                      wikipedia_lang: str = "ja",
):
    """Wikipediaの指定ページから，本文のうち指定長さの文をランダムに抽出する関数

    Args:
        title (str): ページタイトル
        num_sentences_per_page (int, optional): 1ページあたり抽出する文の数. Defaults to 5.
        min_sentence_length (int, optional): 抽出する文の最小長さ. Defaults to 10.
        max_sentence_length (int, optional): 抽出する文の最大長さ. Defaults to 50.
        wikipedia_lang (str, optional): Wikipediaの言語. Defaults to "ja".

    Returns:
        List[str]: 抽出した文のリスト
    """
    wikipedia.set_lang(wikipedia_lang)

    try:
        page = wikipedia.page(title)
    except wikipedia.DisambiguationError as e:
        raise ValueError(f"Ambiguous title '{title}': {e.options}")
    except wikipedia.PageError:
        raise ValueError(f"Page not found for title '{title}'")
    
    text = page.content
    
    # textをstripし，改行で分割
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    # "。" または "．" で終わらないものは削除
    lines = [line for line in lines if line.endswith("。") or line.endswith("．")]
    
    # linesの中で，"。"または"．"で分割し，分割後の各文を再度linesに格納
    new_lines = []
    for line in lines:
        split_lines = [sline.strip()+"。" for sline in line.replace("．", "。").split("。") if sline.strip()]
        new_lines.extend(split_lines)
    lines = new_lines
    
    # 各行の長さを計算し，長い順にソート
    lines = sorted(lines, key=len, reverse=True)

    # 指定された長さの範囲内の行のみを抽出
    lines = [line for line in lines if len(line) >= min_sentence_length and len(line) <= max_sentence_length]

    # 指定数の行がない場合はそのまま返す
    if len(lines) < num_sentences:
        return lines    
    
    # # 中央のnum_sentences行を抽出
    # mid_index = len(lines) // 2
    # start_index = max(0, mid_index - num_sentences // 2)
    # lines = lines[start_index:start_index + num_sentences]
    # ランダムにnum_sentences行を抽出
    import random
    random.shuffle(lines)
    lines = lines[:num_sentences]
        
    return lines


def get_page_names(keyword: str, wikipedia_lang: str = "ja"):
    """Wikipediaでキーワード検索し，ページタイトルのリストを取得する関数

    Args:
        keyword (str): 検索キーワード
        wikipedia_lang (str, optional): Wikipediaの言語. Defaults to "ja".

    Returns:
        List[str]: ページタイトルのリスト
    """
    wikipedia.set_lang(wikipedia_lang)
    search_results = wikipedia.search(keyword)
    return search_results


def main(filename: str = "wikipedia_sentences.csv"):
    query_list = [
        "人工知能",
        "大学",
        "文学",
        "情報",
        "スポーツ",
    ]

    dataframe = None
    processed_title_set = set()
    if os.path.exists(filename):
        try:
            dataframe = pd.read_csv(filename)
            processed_title_set = set(dataframe["title"].tolist())
        except Exception as e:
            dataframe = None

    if dataframe is None:
        dataframe = pd.DataFrame(columns=["query", "title", "sentence"])
        
    for query in query_list:
        page_titles = get_page_names(query)
        for title in page_titles:
            if title in processed_title_set:
                continue
            print(f"Processing title: {title} (query: {query})")
            try:
                sentences = extract_sentences(title, num_sentences=5, min_sentence_length=10, max_sentence_length=50)
                # CSVセーフな文に変換
                sentences = [sentence.replace("\n", " ").replace("\r", " ").replace(",", "，") for sentence in sentences]
                for sentence in sentences:
                    new_row = {"query": query, "title": title, "sentence": sentence}
                    dataframe = pd.concat([dataframe, pd.DataFrame([new_row])], ignore_index=True)
                # print(dataframe.tail())
            except ValueError as e:
                print(e)
            processed_title_set.add(title)
        
        # CSVファイルの更新
        dataframe.to_csv(filename, index=False)

if __name__ == "__main__":
    main()
