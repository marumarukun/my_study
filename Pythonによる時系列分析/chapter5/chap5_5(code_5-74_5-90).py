#!/usr/bin/env python
# coding: utf-8

# __<span style="background-color:#ffffe0">第5章　時系列データを活用したビジネス事例</span>__

# # 5.5　チャーンマネジメントのための離反時期予測（携帯電話サービス）

# ## ステップ1：準備

# ### 必要なモジュールの読み込み

# In[1]:


#
# 必要なモジュールの読み込み
#

import pandas as pd 
import numpy as np

from lifelines import *
from lifelines.utils import median_survival_times
from lifelines.utils import concordance_index

from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
plt.style.use('ggplot') #グラフスタイル
plt.rcParams['figure.figsize'] = [12, 9] #グラフサイズ
plt.rcParams['font.size'] = 14 #フォントサイズ


# ### データセットの読み込み

# In[2]:


#
# 必要なデータセットの読み込み
#

dataset = 'chap5_7.csv'
df=pd.read_csv(dataset, index_col='id') 

print(df) #確認


# ## ステップ2：離反時期（ユーザであるまでの期間）分析

# ### カプランマイヤー推定（全体の生存時間の分析）

# In[3]:


#
# カプランマイヤー推定
#

# TとEの設定
T = df['tenure'] #生存期間（契約期間）
E = df['churn']  #イベント生起（離反）

# インスタンス生成
kmf = KaplanMeierFitter()

# 学習
kmf.fit(T, event_observed=E)


# In[4]:


#
# 生存関数（時刻tまでに離反が発生していない確率）
#

kmf.plot_survival_function()
plt.show()


# ### Ridge Cox回帰

# In[5]:


#
# Ridge Cox回帰
#

# インスタンス生成
cph = CoxPHFitter(
    penalizer=0.1, 
    l1_ratio=0)

# 学習
cph.fit(
    df, 
    duration_col='tenure', 
    event_col='churn') 

# 係数
print(cph.summary)


# In[6]:


#
# 係数のプロット
#

cph.plot()
plt.show()


# In[7]:


#
# family_discountによる生存関数の違い
#

cph.plot_partial_effects_on_outcome(
    'family_discount', values=[0, 1])
plt.show()


# In[8]:


#
# fonline_applicationによる生存関数の違い
#

cph.plot_partial_effects_on_outcome(
    'online_application', values=[0, 1])
plt.show()


# ## ステップ3：継続ユーザ（離反していないユーザ）の離反時期予測

# ### 離反時期予測モデルの検討

# In[9]:


#
# データセットの分割（学習データとテストデータ）
#

df_train, df_test = train_test_split(
    df,
    test_size=0.3,
    random_state=123)


# In[10]:


#
# 離反時期予測モデルの学習（学習データ）
#

# 学習
cph.fit(
    df_train, 
    duration_col='tenure', 
    event_col='churn') 


# In[11]:


#
# 精度評価（C-index）
#

# 学習データ
c_index_train = concordance_index(
    df_train['tenure'],               
    -cph.predict_partial_hazard(df_train),     
    df_train['churn'])

# テストデータ
c_index_test = concordance_index(
    df_test['tenure'],
    -cph.predict_partial_hazard(df_test),
    df_test['churn'])

# 確認
print('c-index (train data):\n', c_index_train)
print('c-index (test data):\n', c_index_test)


# ### 離反時期予測モデルの構築（全データで学習）

# In[12]:


#
# 離反時期予測モデルの学習（全データ）
#

# 学習
cph.fit(
    df, 
    duration_col='tenure', 
    event_col='churn') 

# 精度評価（C-index）
concordance_index(
    df['tenure'],
    -cph.predict_partial_hazard(df),
    df['churn'])


# ### 継続ユーザデータの抽出

# In[13]:


#
# 継続ユーザデータの抽出
#

customers_data = df[df['churn'] == 0]

print(customers_data) #確認


# ### 継続ユーザの生存関数（ユーザである確率）

# In[14]:


#
# 継続ユーザの生存関数
#

pred_survival_after = cph.predict_survival_function(
    customers_data,
    conditional_after=customers_data['tenure'])

print(pred_survival_after) #確認


# In[15]:


#
# 特定の継続ユーザの生存関数のグラフ化
#

customer1 = 3030576346
customer2 = 5249470446
customer3 = 4675530834

pred_survival_after[customer1].plot(label="id:"+str(customer1))
pred_survival_after[customer2].plot(label="id:"+str(customer2))
pred_survival_after[customer3].plot(label="id:"+str(customer3))

plt.legend()
plt.show()


# ### 継続ユーザの離反時期の予測（中央値）

# In[16]:


#
# 離反時期予測（中央値）
#

churn_pred = cph.predict_median(
    customers_data,
    conditional_after=customers_data['tenure'])

print(churn_pred) #確認


# In[17]:


#
# ヒストグラム（度数分布）※infは120と設定
#

plt.hist(
    churn_pred.replace([np.inf], 120), 
    bins=10, 
    rwidth=0.8)
plt.show()


# In[18]:


#
# 累積ヒストグラム（％）※infは120と設定
#

plt.hist(
    churn_pred.replace([np.inf], 120), 
    bins=10, 
    rwidth=0.8, 
    density=True, 
    cumulative=True)

plt.show()


# In[19]:


#
# 離反時期の早い順に並び替える
#

print(churn_pred.sort_values())


# In[20]:


#
# 3ヶ月以内に離反しそうなユーザ
#

print(churn_pred[churn_pred <= 3])


# In[21]:


#
# 離反時期予測（第1四分位数，中央値（第2四分位数），第3四分位数）
#

churn_pred = cph.predict_percentile(
    customers_data,
    p=[0.25,0.5,0.75],
    conditional_after=customers_data['tenure'])

print(churn_pred) #確認

