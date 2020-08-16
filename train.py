
from keras.models import Sequential
from keras.layers import Flatten, Dense, LSTM, Embedding
from keras.layers import LSTM
from keras import backend as K
import pandas as pd
import datetime
import random
import tensorflow as tf
import os

#-------------- 学習の再現性に関しての設定 -------------
ENABLE_REPRODUCIBLE_RANDOM = False
STANDARD_SEED = 77 #randomの乱数
NP_SEED       = 77 #numpyの乱数
TF_SEED       = 77 #tensorflowの乱数

if ENABLE_REPRODUCIBLE_RANDOM:
   os.environ['PYTHONHASHSEED'] = '0'
   random.seed(STANDARD_SEED)
   np.random.seed(NP_SEED)
   # threadの数を1にすると学習遅くなる??
   # ここは要検討
   session_conf = tf.ConfigProto(
       intra_op_parallelism_threads=1,
       inter_op_parallelism_threads=1
   )
   tf.set_random_seed(TF_SEED)
   sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
   K.set_session(sess)
#-------------------------------------- -------------


#-------------------- 各種設定 ------------------------
train_start = 2017
train_end   = 2017
test start = 2015
test_end = 2016

### Embeddingにおける設定
max_features = 10000 # 入力データの語彙数の上限
wordvec_dim = 32 # 単語ベクトルの次元数
### LSTMのモデルの設定
n_hidden_layer = 32 # LSTMの隠れ層の数
### LSTMの学習時のハイパーパラメータ
n_epoch = 10 #モデル訓練の試行回数（少ないとうまくいかないし、多いと過学習になる）
n_batch = 150 #学習時のバッチサイズ(入力1回分に対応するデータの長さ)
valid_rate = 0.2 #CrossValidationに使うデータの割合
#-------------------------------------- -------------



#-------------------- main process ------------------------

gensaki = pd.read_csv(csv_file, index_col=1)
train = gensaki.loc[train_start:train_end, :]
test = gensaki.loc[test_start:test_end, :]

input_train = train.iloc[:, 0:149]
input_test = test.iloc[:, 0:149]
y_train = train['genjou']
y_test = test['genjou']


### 機械学習モデルの設定
# NNを宣言（kerasのモデルは追加した順に積みあがっていく）
model = Sequential()
# 入力層の作成(ここで各単語をベクトル化している）
# ----- word2vec/fasttext等で改良すべきか?
# facebookが公開してるfasttextの性能が良さげ
# Embedding(語彙数, 分散ベクトルの次元数, options_args)
model.add(Embedding(max_features, wordvec_dim))
# 隠れ層（中間層）の作成
# LSTMを追加(バッチサイズ等は学習時に設定するので隠れそうのみ設定)
model.add(LSTM(n_hidden_layer))
# 出力層の作成
# Dense(出力データの次元（ノード数）, 活性化関数)
model.add(Dense(1, activation='softmax'))
# モデルのコンパイル
# compile(最適化関数、損失関数、評価指標)
model.compile(optimizer= 'rmsprop',
              loss= 'mean_squared_error',
              metrics=['acc'])
# モデルの学習
# model.fit(学習データ, 正解ラベル, options_args)
history = model.fit(input_train, y_train,
                    epochs=n_epoch,
                    batch_size=n_batch,
                    validation_split=valid_rate)
model.save(file_name)

# 予測フェイズ
# predict(入力データ, バッチサイズ, 進行状況のon/off, 評価の終了基準)
pred1 = model.predict(input_train, batch_size=n_batch, verbose=0, steps=None)
pred2 = model.predict(input_test, batch_size=n_batch, verbose=0, steps=None)

# 訓練後における誤差の評価
# evaluate(学習データ, 正解ラベル, バッチサイズ, 進行状況のon/off)
scorel = model.evaluate(input_train, batch_size=n_batch, verbose=0)
score2 = model.evaluate(input_test, batch_size=n_batch, verbose=0)

pred = pd.concat([pd.DataFrame(pred1), pd.DataFrame(pred2)], axis=1)
pred.columns = [ 'in_sample', 'out_of_sample']
file_name =
pred.to_csv(file_name)
