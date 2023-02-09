#NGO - 계층적 군집분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage
pd.options.display.float_format = '{:.3f}'.format
df = pd.read_csv('21.csv', sep=',', encoding='UTF-8')
#df = pd.read_csv('1. NGO.csv',sep=',',encoding='CP949')
df1 = df[['PLED_RATE_FULFILLED','AGE','PAY_RATE_NOPAY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT','LONGEVITY_M','PLED_NUM','PLED_ACTIVE_NUM']]
df1 = df1.dropna()
df1 = df1[df1['PAY_NUM']>0]
df1 = df1[df1['PLED_NUM']>0]
df1['월후원율'] = df1['PAY_NUM'] / df1['LONGEVITY_M']
df1['평균후원금'] = df1['PAY_SUM_PAYMENTAMOUNT'] / df1['PAY_NUM']
df1['액티브비율'] = df1['PLED_ACTIVE_NUM'] / df1['PLED_NUM']

df2 = df1[['PLED_RATE_FULFILLED','월후원율']]
df2 = df2[df2['PLED_RATE_FULFILLED']<=1]
df2 = df2[df2['월후원율']<=1]
#df2 = df2[df2['액티브비율']<=1]
cluster = linkage(df2, method = 'average', metric='euclidean')
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

plt.title('Dendrogram')
plt.xlabel('index')
plt.xlabel('distance')
dendrogram(cluster)
plt.savefig('23.계층댄드로그램.png',dpi=200,edgecolor='blue', bbox_inches='tight',pad_inches=0.3)
plt.clf()