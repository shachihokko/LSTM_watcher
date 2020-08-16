
import pandas as pd
import os
import MeCab

### 作業場所ディレクトリの取得
dir_org = os.path.dirname(__file__) + "\\"

### YYMM表示の年月をループで作成
YYMM = []
edate = "2008"
for yy in range (18, 19):
  for mm in range(1,12):
    yymm = "{:02}".format(yy)+"{:02}".format(mm)
    if int(yymm)<=int(edate):
      YYMM.append(yymm)

### 文章を形態素解析して、単語ごとにリスト化している
def tokenize(text):
    wakati = MeCab.Tagger("-O wakati")
    wakati.parse("")
    return wakati.parse(text).strip().split()

### 各月のコメントを形態素解析してまとめている
tokens = []
for yymm in YYMM:
  dir_csv = dir_org + "preprocessed\\" + yymm + "gen_p.csv"
  data = pd.read_csv(dir_csv, header=0, engine="python")
  data= data["letter"]
  for line in data:
    tokens = tokens + [tokenize(line)]
df = pd.DataFrame(tokens)
df.to_csv(dir_org+'wakati\\'+'gen_wakati.csv', encoding="cp932")
