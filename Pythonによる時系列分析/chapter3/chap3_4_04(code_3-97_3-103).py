#!/usr/bin/env python
# coding: utf-8

# __<span style="background-color:#ffffe0">第3章　時系列予測モデル構築・超入門</span>__

# # 3.4　季節成分が複数ある場合の予測モデル

# ## 3.4.4　時系列特徴量を生成しテーブルデータを作ろう！

# ### 生成する時系列特徴量
# 
# * ラグ特徴量（ラグ1）：lag1
# * ローリング特徴量（1期前までの1日間の平均の平均）：window48
# * エクスパンディング特徴量（1期前までの平均）：expanding
# * 三角関数特徴量（48周期）：sin48, cos48
# * 三角関数特徴量（336周期）：sin336, cos336

# ### 準備（必要なモジュールとデータの読み込み）

# In[1]:


#
# 必要なモジュールの読み込み
#

import numpy as np
import pandas as pd

import statsmodels.api as sm

import warnings
warnings.simplefilter('ignore')


# In[2]:


#
# 必要なデータセット（時系列データ）の読み込み
#

dataset = sm.datasets.get_rdataset(
    "taylor", "forecast")
y = dataset.data

print(y) #確認


# ### ラグ特徴量・ローリング特徴量・エクスパディング特徴量

# In[3]:


#
# ラグ特徴量・ローリング特徴量・エクスパディング特徴量
#

# ラグ特徴量（ラグ1）の生成
lag1 = y.shift(1)

# ローリング特徴量（1期前までの1日間の平均）の生成
window48 = lag1.rolling(window=48).mean()

# エクスパンディング特徴量（1期前までの平均）の生成
expanding = lag1.expanding().mean()


# In[4]:


#
# 作成した時系列特徴量を結合しテーブルデータを生成
#

## データを結合
df_tbl = pd.concat([y,
                    lag1,
                    window48,
                    expanding],
                   axis=1)

## 変数名を設定
df_tbl.columns = ['y',
                  'lag1',
                  'window48',
                  'expanding']

print(df_tbl) #確認


# ### 三角関数特徴量

# In[5]:


#
# 三角関数特徴量
#

# 空のデータフレーム生成
exog = pd.DataFrame()
exog.index = y.index

# 三角関数特徴量（Fourier terms）の生成関数
def fourier_terms_gen(seasonal,terms_num):
    
    #seasonal:周期
    #terms_num:Fourier termの数（sinとcosのセット数）
    
    for num in range(terms_num):
        num = num + 1
        sin_colname = 'sin'+str(seasonal)+'_'+ str(num)
        cos_colname = 'cos'+str(seasonal)+'_'+ str(num)
        exog[sin_colname] = np.sin(num * 2 * np.pi * exog.index / seasonal)
        exog[cos_colname] = np.cos(num * 2 * np.pi * exog.index / seasonal)
        
# 三角関数特徴量の生成
## 336周期
fourier_terms_gen(
    seasonal=336,
    terms_num=10)

## 48周期
fourier_terms_gen(
    seasonal=48,
    terms_num=10)

print(exog) #確認


# ### 2つのテーブルデータの結合と欠測値削除

# In[6]:


#
# 2つのテーブルデータの結合と欠測値削除
#

# データを結合
df_tbl = pd.concat([df_tbl,
                    exog],
                   axis=1)

# 欠測値削除
df_tbl = df_tbl.dropna()

print(df_tbl) #確認


# ### CSVファイルとして出力

# In[7]:


df_tbl.to_csv('taylor_tbl_.csv', index=False)

