#!/usr/bin/env python
# coding: utf-8

# __<span style="background-color:#ffffe0">第5章　時系列データを活用したビジネス事例</span>__

# # 5.6　既存顧客のLTV予測による顧客選別（ECサイト）

# ## ステップ1：準備

# ### 必要なモジュールの読み込み

# In[1]:


#
# 必要なモジュールの読み込み
#

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes.utils import calibration_and_holdout_data
from lifetimes.plotting import plot_probability_alive_matrix

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

dataset = 'chap5_8.csv'
df=pd.read_csv(dataset)

print(df) #確認


# ### RFMデータセットの生成

# In[3]:


#
# トランザクションデータ（顧客×日付ごと購買金額）の作成
#

# 合計金額を計算
df['Amount'] = df['Quantity'] * df['Unit_Price']

# 顧客ID・日時・金額のトランザクションデータを生成
transaction_data = df.groupby(
    ['Customer_ID', 'Invoice_Date']
    )['Amount'].sum().reset_index()

print(transaction_data) #確認


# In[4]:


#
# 全データ期間のRFMデータセットの生成
#

# RFMデータセット生成
rfm = summary_data_from_transaction_data(
    transaction_data,
    'Customer_ID',
    'Invoice_Date',
    observation_period_end='2019-12-31',
    monetary_value_col = 'Amount')

# frequencyが1以上のデータに絞る
rfm = rfm.loc[rfm['frequency'] > 0]

# データセットの確認
print(rfm)


# In[5]:


# 基礎統計料

print(rfm.describe())


# In[6]:


#
# 期間分割されたRFMデータセットの生成
#（学習データ期間：2018年、テストデータ期間：2019年）
#

# 分割の実施
rfm_train_test = calibration_and_holdout_data(
    transaction_data,
    'Customer_ID',
    'Invoice_Date',
    calibration_period_end='2018-12-31',
    observation_period_end='2019-12-31',
    monetary_value_col = 'Amount')

# 学習データ期間のfrequency（frequency_cal）が1以上のデータに絞る
rfm_train_test = rfm_train_test.loc[
    rfm_train_test.frequency_cal > 0, :]

# データセットの確認
print(rfm_train_test)


# ## ステップ2：予測モデルの作り方の検討

# ### 購買回数の期待値を予測するためのBG/NBDモデルの学習と評価

# In[7]:


#
# BG/NBDモデルの学習（学習データ期間）
#

# インスタンス生成
bgf = BetaGeoFitter()

# 学習
bgf.fit(
    rfm_train_test['frequency_cal'],
    rfm_train_test['recency_cal'],
    rfm_train_test['T_cal'])

# パラメータ
print(bgf.summary)


# In[8]:


#
# テストデータ期間の予測
#

# 予測期間（日単位）
duration_holdout = 365

# 予測
predicted_bgf = bgf.predict(
    duration_holdout,
    rfm_train_test['frequency_cal'],
    rfm_train_test['recency_cal'],
    rfm_train_test['T_cal'])


# In[9]:


#
# 予測精度（テストデータ期間）
#

# 実測値と予測値の代入
actual = rfm_train_test['frequency_holdout']
pred = predicted_bgf

# 精度指標（テストデータ）
print('R2:\n', r2_score(actual, pred))


# ### 購買金額／回を予測するためのGGモデルの学習と評価

# In[10]:


#
# GGモデルの学習（学習期間）
#

# インスタンス生成
ggf = GammaGammaFitter()

# 学習
ggf.fit(
    rfm_train_test['frequency_cal'],
    rfm_train_test['monetary_value_cal'])

# パラメータ
print(ggf.summary)


# In[11]:


#
# テストデータ期間の予測
#

# 予測
predicted_ggf = ggf.conditional_expected_average_profit(
    rfm_train_test['frequency_cal'],
    rfm_train_test['monetary_value_cal'])


# In[12]:


#
# 予測精度（テストデータ期間）
#

# 実測値と予測値の代入
actual = rfm_train_test['monetary_value_holdout']
pred = predicted_ggf

# 精度指標（テストデータ）
print('R2:\n', r2_score(actual, pred))


# ### テストデータ期間のLTV（購買金額）予測の実施

# In[13]:


#
# LTV（購買金額）の予測
#

# 予測期間（月単位）
forecast_period = 12

# 予測の実施
ltv = ggf.customer_lifetime_value(
    bgf,
    rfm_train_test['frequency_cal'],
    rfm_train_test['recency_cal'],
    rfm_train_test['T_cal'],
    rfm_train_test['monetary_value_cal'],
    time=forecast_period,
    freq='D',
    discount_rate=0.01)

print(ltv) #確認


# In[14]:


#
# 予測精度（テストデータ期間）
#

# データセットに予測結果を追加
rfm_train_test['ltv'] = ltv

# 実測値と予測値の代入
actual = rfm_train_test['frequency_holdout']*         rfm_train_test['monetary_value_holdout']
pred = rfm_train_test['ltv'] 

# 予測精度
print('R2:\n', r2_score(actual, pred))


# In[15]:


#
# グラフ化
#

# 散布図
plt.scatter(pred, actual)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# ## ステップ3：LTV（購買金額）予測とリスト生成

# ### LTV（購買金額）予測モデルの構築（全データ期間のRFMデータセット）

# In[16]:


#
# LTV予測モデルの構築（全データ期間のRFMデータセット）
#

# BG/NBDモデルの学習
bgf = BetaGeoFitter().fit(
    rfm['frequency'],
    rfm['recency'],
    rfm['T'])

# GGモデルの学習
ggf = GammaGammaFitter().fit(
    rfm['frequency'],
    rfm['monetary_value'])


# ### 生存確率と離反確率

# In[17]:


#
# 生存確率と離反確率
#

# 生存確率
alive = bgf.conditional_probability_alive(
    rfm['frequency'],
    rfm['recency'],
    rfm['T'])

# 離反確率の確認
print(1-alive)


# In[18]:


#
# グラフ化
#

plot_probability_alive_matrix(bgf)
plt.show()


# ### LTV（購買金額）予測とリスト生成

# In[19]:


#
# 次年度のLTV（購買金額）の予測
#

# 予測期間（月単位）
forecast_period = 12

# 予測の実施
ltv = ggf.customer_lifetime_value(
    bgf, 
    rfm['frequency'],
    rfm['recency'],
    rfm['T'],
    rfm['monetary_value'],
    time=forecast_period,
    freq='D',
    discount_rate=0.01)

# データセットにLTV追加
rfm['ltv'] = ltv

print(rfm) #確認


# In[20]:


#
# LTV降順リスト
#

print(rfm[['ltv']].sort_values('ltv',ascending=False))


# In[21]:


#
# 多期間LTV（購買金額）予測とリスト生成
#

# LTV12予測（1年間）
ltv = ggf.customer_lifetime_value(
    bgf, 
    rfm['frequency'],
    rfm['recency'],
    rfm['T'],
    rfm['monetary_value'],
    time=12,
    freq='D',
    discount_rate=0.01)
rfm['ltv'] = ltv

# LTV36予測（3年間）
ltv = ggf.customer_lifetime_value(
    bgf, 
    rfm['frequency'],
    rfm['recency'],
    rfm['T'],
    rfm['monetary_value'],
    time=36,
    freq='D',
    discount_rate=0.01 )
rfm['ltv36'] = ltv

# LTV120予測（10年間）
ltv = ggf.customer_lifetime_value(
    bgf, 
    rfm['frequency'],
    rfm['recency'],
    rfm['T'],
    rfm['monetary_value'],
    time=120,
    freq='D',
    discount_rate=0.01)
rfm['ltv120'] = ltv

# LTV1200予測（100年間）
ltv = ggf.customer_lifetime_value(
    bgf, 
    rfm['frequency'],
    rfm['recency'],
    rfm['T'],
    rfm['monetary_value'],
    time=1200,
    freq='D',
    discount_rate=0.01)
rfm['ltv1200'] = ltv

# LTV降順リスト
print(
    rfm[['ltv','ltv36','ltv120','ltv1200']
       ].sort_values('ltv', ascending=False))


# In[22]:


# 基礎統計料

print(rfm.describe())

