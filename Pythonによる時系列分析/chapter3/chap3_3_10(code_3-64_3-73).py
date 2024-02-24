#!/usr/bin/env python
# coding: utf-8

# __<span style="background-color:#ffffe0">第3章　時系列予測モデル構築・超入門</span>__

# # 3.3　時系列の予測モデルを構築してみよう

# ##  3.3.10　線形回帰で予測モデルを構築（REF法で変数選択）

# ### 準備（必要なモジュールとデータの読み込み）

# In[1]:


#
# 必要なモジュールの読み込み
#

import pandas as pd
import numpy as np
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from pmdarima.model_selection import train_test_split

import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
plt.style.use('ggplot') #グラフのスタイル
plt.rcParams['figure.figsize'] = [12, 9] # グラフサイズ設定
plt.rcParams['font.size'] = 14 #フォントサイズ


# In[2]:


#
# 必要なデータセット（時系列データ）の読み込み
#

dataset='df_tbl.csv'    #データセットのファイル名
df=pd.read_csv(
    dataset,
    index_col='Month',  #変数「Month」をインデックスに設定
    parse_dates=True)  #インデックスを日付型に設定

print(df) #確認


# In[3]:


# プロット
df.plot()
plt.title('Passengers')                               #グラフタイトル
plt.ylabel('Monthly Number of Airline Passengers')    #タテ軸のラベル
plt.xlabel('Month')                                   #ヨコ軸のラベル
plt.axvline(datetime.datetime(1960,1,1),color='blue')
plt.show()


# In[4]:


#
# データセットを学習データとテストデータ（直近12ヶ月間）に分割
#

# データ分割
train, test = train_test_split(
    df, test_size=12)

# 学習データ
y_train = train['y']              #目的変数y
X_train = train.drop('y', axis=1) #説明変数X

# テストデータ
y_test = test['y']              #目的変数y
X_test = test.drop('y', axis=1) #説明変数X


# ### 変数選択（特徴量選択）

# In[5]:


#
# 変数選択（特徴量選択）
#

# インスタンスの生成
rfecv = RFECV(
    estimator=LinearRegression(),
    n_jobs=-1,
    scoring='neg_mean_squared_error', #MSEで評価
    cv=10)

# 変数選択の実施
rfecv.fit(X_train, y_train)

# 結果
columns_selected = X_train.columns[rfecv.support_]
print('選択変数',columns_selected.values)


# In[6]:


#
# 新しい説明変数Xの生成
#

# 新しい学習データの説明変数X
X_train_selected = pd.DataFrame(
    rfecv.transform(X_train),
    columns=columns_selected,
    index=X_train.index)

# 新しいテストデータの説明変数X
X_test_selected = pd.DataFrame(
    rfecv.transform(X_test),
    columns=columns_selected,
    index=X_test.index)


# In[7]:


# テストデータ
print(X_test_selected)


# ### 予測モデルの学習（学習データ利用）

# In[8]:


#
# 予測モデルの学習（学習データ利用）
#

# 数理モデルのインスタンス生成
regressor = LinearRegression()

# 学習
regressor.fit(X_train_selected, y_train)

# 切片と回帰係数
print('切片:',regressor.intercept_)
print('回帰係数:',regressor.coef_)


# ### 予測モデルのテスト（テストデータ利用）

# In[9]:


#
# 予測の実施（学習期間）
#

train_pred = regressor.predict(X_train_selected)


# In[10]:


#
# 予測の実施（テストデータ期間）
#

# 学習データのコピー
y_train_new = y_train.copy()

# 説明変数Xを更新しながら予測を実施
for i in range(len(y_test)):
    
    #当期の予測の実施
    X_value =  X_test_selected.iloc[i:(i+1),:]
    y_value_pred = regressor.predict(X_value)
    y_value_pred = pd.Series(y_value_pred,index=[X_value.index[0]])
    y_train_new = pd.concat([y_train_new,y_value_pred])
    
    #次期の説明変数Xの計算
    lag12_new = y_train_new[-12] #lag12
    expanding_new = y_train_new.mean() #expanding
    
    #次期の説明変数Xの更新
    X_test_selected.iloc[(i+1):(i+2),0] = lag12_new
    X_test_selected.iloc[(i+1):(i+2),1] = expanding_new
    
# 予測値の代入
test_pred = y_train_new[-12:]
    
# 更新後の説明変数X
print(X_test_selected.head(12))


# In[11]:


#
# 予測モデルのテスト（テストデータ利用）
#

print('RMSE:\n',
      np.sqrt(mean_squared_error(
          y_test, test_pred)))
print('MAE:\n',
      mean_absolute_error(
          y_test, test_pred)) 
print('MAPE:\n',
      mean_absolute_percentage_error(
          y_test, test_pred))


# In[12]:


#
# グラフ（予測値と実測値）
#

fig, ax = plt.subplots()

# 実測値の描写
## 学習データ
ax.plot(
    train.index, 
    y_train, 
    linestyle='-',
    label='actual(train)')

## テストデータ
ax.plot(
    test.index, 
    y_test, 
    linestyle='--',
    label='actual(test)', 
    color='gray')

# 予測値の描写
## 学習データ
ax.plot(
    train.index, 
    train_pred, 
    linestyle=':',
    color='c')

## テストデータ
ax.plot(
    test.index, 
    test_pred, 
    linestyle=':',
    label="predicted", 
    color="c") 

# 学習データとテスデータの間の縦線の描写
ax.axvline(
    datetime.datetime(1960,1,1),
    color='blue')

# 凡例表示
ax.legend()

plt.show()

