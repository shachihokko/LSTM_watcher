
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
from keras.preprocessing import sequence
import os

### 作業場所ディレクトリの取得
dir_org = os.path.dirname(__file__) + "\\"
maxlen = 150 #paddingするときのリストの長さを指定
gensaki_list = ["gen", "saki"]


def token_text2sequence(gensaki):
  df_wakati = pd.read_csv(dir_org+'wakati\\'+gensaki+"_wakati.csv",
                          header=0, index_col=0, engine="python")
  ### 各行をリスト化して、それを要素としたリストを作成
  multi_list_df = df_wakati.values.tolist()
  ### 各要素に含まれているNANを削除してる
  multi_list_df_trimed = []
  for elem in multi_list_df:
    tmp = [x for x in elem if x is not np.nan]
    multi_list_df_trimed = multi_list_df_trimed + [tmp]
  ### 各単語を数値に変換
  tokenizer = Tokenizer(num_words=10000) #前準備
  tokenizer.fit_on_texts(multi_list_df_trimed) #文章を与える（内部で処理してる）
  tokens = tokenizer.texts_to_sequences(multi_list_df_trimed) #数値ベクトルとして返す
  ### padding(短い文の前半を0で埋めて、全ての文を同じ長さにする)
  padded = sequence.pad_sequences(tokens, maxlen=maxlen) #paddingの実行
  padded = pd.DataFrame(padded)
  return padded

for gensaki in gensaki_list:
  output = token_text2sequence(gensaki)
  output.to_csv()
