#!/usr/bin/env python
# coding: utf-8

# __<span style="background-color:#ffffe0">第3章　時系列予測モデル構築・超入門</span>__

# # 3.3　時系列の予測モデルを構築してみよう

# ## 3.3.7　時系列特徴量を生成しテーブルデータを作ろう！

# * ラグ特徴量（ラグ1）
# * ラグ特徴量（ラグ12）
# * ローリング特徴量（1期前までの12ヶ月平均）
# * エクスパンディング特徴量（1期前までの平均）
# * トレンド特徴量（線形）

# ### 準備（必要なモジュールとデータの読み込み）

# In[1]:


#
# 必要なモジュールの読み込み
#

import pandas as pd

import warnings
warnings.simplefilter('ignore')


# In[2]:


#
# 必要なデータセット（時系列データ）の読み込み
#

dataset='AirPassengers.csv'  #データセットのファイル名
df=pd.read_csv(
    dataset,
    index_col='Month',   #変数「Month」をインデックスに設定
    parse_dates=True)   #インデックスを日付型に設定


# ### ラグ特徴量の生成

# In[3]:


#
# ラグ特徴量（ラグ1）の生成
#

lag1 = df.shift(1)

print(lag1.head(15)) #確認


# In[4]:


#
# ラグ特徴量（ラグ12）の生成
#

lag12 = df.shift(12)

print(lag12.head(15)) #確認


# ### ローリング特徴量（1期前までの12ヶ月平均）

# In[5]:


#
# ローリング特徴量（1期前までの12ヶ月平均）
#

window12 = lag1.rolling(window=12).mean()

print(window12.head(15)) #確認


# ### エクスパンディング特徴量（1期前までの平均）

# In[6]:


#
# エクスパンディング特徴量（1期前までの平均）
#

expanding = lag1.expanding().mean()

print(expanding.head(15)) #確認


# ### 作成した時系列特徴量を結合しテーブルデータを生成

# In[7]:


#
# 作成した時系列特徴量を結合しテーブルデータを生成
#

## データを結合
df_tbl = pd.concat([df,
                    lag1,
                    lag12,
                    window12,
                    expanding],
                   axis=1)

## 変数名を設定
df_tbl.columns = ['y',
                  'lag1',
                  'lag12',
                  'window12',
                  'expanding']

print(df_tbl.head(15)) #確認


# In[8]:


# 欠測値削除
df_tbl = df_tbl.dropna()

print(df_tbl.head(15)) #確認


# ### トレンド特徴量（線形）

# In[9]:


#
# トレンド特徴量（線形）
#

df_tbl['t'] = pd.RangeIndex(start=0, 
                            stop=len(df_tbl))

print(df_tbl) #確認


# ### CSVファイルとして出力

# In[10]:


df_tbl.to_csv('df_tbl.csv' )

