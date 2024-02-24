#!/usr/bin/env python
# coding: utf-8

# __<span style="background-color:#ffffe0">第3章　時系列予測モデル構築・超入門</span>__

# # 3.4　季節成分が複数ある場合の予測モデル

# ##  3.4.5　線形回帰で構築する予測モデル

# ### 準備（必要なモジュールとデータの読み込み）

# In[1]:


#
# 必要なモジュールの読み込み
#

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
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

df=pd.read_csv('taylor_tbl_.csv') 

print(df) #確認


# In[3]:


#
# データセットを学習データとテストデータ（直近12ヶ月間）に分割
#

# データ分割
## 目的変数
y_train, y_test = train_test_split(
    df.y, test_size=336)

## 説明変数
X_train, X_test = train_test_split(
    df.drop('y', axis=1), test_size=336)


# ### 予測モデルの学習（学習データ利用）

# In[4]:


#
# 予測モデルの学習（学習データ利用）
#

# 数理モデルのインスタンス生成
regressor = LinearRegression()

# 学習
regressor.fit(X_train, y_train)


# ### 予測モデルのテスト（テストデータ利用）

# In[5]:


#
# 予測の実施（学習データ期間）
#

train_pred = regressor.predict(X_train)


# In[6]:


#
# 予測の実施（テストデータ期間）
#

# 学習データのコピー
y_train_new = y_train.copy()

# 説明変数Xを更新しながら予測を実施
for i in range(len(y_test)):
    
    #当期の予測の実施
    X_value =  X_test.iloc[i:(i+1),:]
    y_value_pred = regressor.predict(X_value)
    y_value_pred = pd.Series(y_value_pred,index=[X_value.index[0]])
    y_train_new = pd.concat([y_train_new,y_value_pred])
    
    #次期の説明変数Xの計算
    lag1_new = y_train_new.iloc[-1] #lag1
    window48_new = y_train_new[-48:].mean() #window48
    expanding_new = y_train_new.mean() #expanding
    
    #次期の説明変数Xの更新
    X_test.iloc[(i+1):(i+2),0] = lag1_new
    X_test.iloc[(i+1):(i+2),1] = window48_new
    X_test.iloc[(i+1):(i+2),2] = expanding_new
    
# 予測値の代入
test_pred = y_train_new[-336:]
    
# 更新後の説明変数X
print(X_test)


# In[7]:


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


# In[8]:


#
# グラフ（予測値と実測値）
#

fig, ax = plt.subplots()

# 実測値
ax.plot(
    y_test.index, 
    y_test.values, 
    linestyle='-',
    label='actual(test)')

# 予測値
ax.plot(
    y_test.index, 
    test_pred, 
    linestyle=':',
    label="predicted") 

# 凡例表示
ax.legend()

plt.show()


# In[9]:


#
# グラフ（予測値と実測値）
#

fig, ax = plt.subplots()

# 実測値の描写
## 学習データ
ax.plot(
    y_train.index,
    y_train.values, 
    linestyle='-',
    label='actual(train)')

## テストデータ
ax.plot(
    y_test.index, 
    y_test.values,
    linestyle='--',
    label='actual(test)',
    color='gray')

# 予測値の描写
## 学習データ
ax.plot(
    y_train.index,
    train_pred, 
    linestyle=':',
    color='c')

## テストデータ
ax.plot(
    y_test.index,
    test_pred,
    linestyle=':',
    label="predicted",
    color='c') 

# 学習データとテスデータの間の縦線の描写
ax.axvline(
    len(y_train),
    color='blue')

# 凡例表示
ax.legend()

plt.show()

