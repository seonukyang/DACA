#다항 로지스틱 회귀분석
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math


df = pd.read_csv('NGO연령대.csv', encoding='UTF-8')
del df['Unnamed: 0']

df['총납입금액구간'] = ''
for i in range(0,len(df),1) : 
    if df['PAY_SUM_PAYMENTAMOUNT'].iloc[i] < 100000 : 
        df['총납입금액구간'].iloc[i] = '0'
    elif df['PAY_SUM_PAYMENTAMOUNT'].iloc[i] < 300000 : 
        df['총납입금액구간'].iloc[i] = '1'
    elif df['PAY_SUM_PAYMENTAMOUNT'].iloc[i] < 500000 : 
        df['총납입금액구간'].iloc[i] = '2'
    else : df['총납입금액구간'].iloc[i] = '3'

a = len(df[df['CHURN']==0])
b = len(df[df['CHURN']==1])
c = len(df)
print(a,b,c)

print('명수파악',df.groupby(['총납입금액구간','CHURN']).size())
print('전체의 비율',df.groupby(['총납입금액구간']).size()/c)
print('비이탈자비율',df.groupby(['총납입금액구간','CHURN']).size()/a)
print('이탈자비율',df.groupby(['총납입금액구간','CHURN']).size()/b)
print('전체비율',df.groupby(['총납입금액구간','CHURN']).size()/c)

df.to_csv('NGO다항로지스틱회귀분석.csv', encoding='utf-8-sig')