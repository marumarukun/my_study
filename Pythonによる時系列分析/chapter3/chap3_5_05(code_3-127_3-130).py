#!/usr/bin/env python
# coding: utf-8

# __<span style="background-color:#ffffe0">第3章　時系列予測モデル構築・超入門</span>__

# # 3.5　多変量時系列データの特徴把握と因果探索

# ## 3.5.5　非ガウスSVAR（VAR-LiNGAM）モデルで実施

# ### 準備（必要なモジュールとデータの読み込み）

# In[1]:


#
# 必要なモジュールの読み込み
#

import pandas as pd
import numpy as np

import lingam
from lingam.utils import make_dot

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


# ### VAR-LiNGAMで変数間の関係性を検討

# #### VAR-LiNGAMの構築

# In[4]:


#
# VAR-LiNGAMの構築
#

# モデルのインスタンス生成
model = lingam.VARLiNGAM(lags=2, prune=True)

# 学習
model.fit(df)

# 推定結果（有向グラフの隣接行列）
print(model.adjacency_matrices_)


# #### 有向グラフ（VAR-LiNGAM）

# In[5]:


#
# 有向グラフ（VAR-LiNGAM）
#

labels = ['Sales(t)', 'OfflineAD(t)', 'OnlineAD(t)', 
          'Sales(t-1)', 'OfflineAD(t-1)', 'OnlineAD(t-1)', 
          'Sales(t-2)', 'OfflineAD(t-2)', 'OnlineAD(t-2)']

make_dot(np.hstack(model.adjacency_matrices_),
         lower_limit=0.05,
         ignore_shape=True,
         labels=labels)

