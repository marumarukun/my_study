#!/usr/bin/env python
# coding: utf-8

# __<span style="background-color:#ffffe0">第3章　時系列予測モデル構築・超入門</span>__

# # 3.5　多変量時系列データの特徴把握と因果探索

# ## 3.5.2　相互相関係数によるアプローチ

# ### 準備（必要なモジュールとデータの読み込み）

# In[1]:


#
# 必要なモジュールの読み込み
#

import pandas as pd
import numpy as np
from scipy import stats

import statsmodels.api as sm

from graphviz import Digraph

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

dataset='MMM.csv' #データセットのファイル名
df = pd.read_csv(
    dataset,
    parse_dates=True,
    index_col='day')

print(df) #確認


# In[3]:


#
# 時系列データのグラフ化（折れ線グラフ）
#

df.plot()
plt.show()


# ### 相互相関コレログラム

# In[4]:


#
# 標準化（平均0、分散1）
#

df_std = stats.zscore(df)

print(df_std) #確認


# In[5]:


#
# 相互相関コレログラム
#

fig, ax = plt.subplots(3,1) #3行1列のグラフの枠を生成

# SaleaとOfflioneADの相互相関
ax[0].xcorr(df_std.iloc[:,0], df_std.iloc[:,1])
ax[0].title.set_text(str(df_std.columns[0]+' * '+df_std.columns[1]))

# SaleaとOnlineADの相互相関
ax[1].xcorr(df_std.iloc[:,0], df_std.iloc[:,2])
ax[1].title.set_text(str(df_std.columns[0]+' * '+df_std.columns[2]))

# OnlineADとOfflioneADの相互相関
ax[2].xcorr(df_std.iloc[:,2], df_std.iloc[:,1])
ax[2].title.set_text(str(df_std.columns[2]+' * '+df_std.columns[1]))

fig.tight_layout() #作成したグラフを整える
plt.show()


# ### 有向グラフ

# In[6]:


#
# 有向グラフ（先行系列＋一致系列）
#

# インスタンスの生成
graph = Digraph()

# ノードを追加
for i in range(len(df.columns)):
    graph.node(df.columns[i])

# 辺を追加
graph.edge(df.columns[1], df.columns[0]) #OfflineAD -> Sales
graph.edge(df.columns[1], df.columns[2]) #OfflineAD -> OnlineAD
graph.edge(df.columns[2], df.columns[0]) #OnlineAD -> Sales

# 有向グラフ表示
graph

