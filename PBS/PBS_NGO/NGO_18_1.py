#NGO - 일원분산분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
import scikit_posthocs
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')
df.index= range(0,len(df),1)
df_X =  pd.read_csv('15.df_1.csv', sep=',', encoding='UTF-8')
df_X = df_X[df_X.columns.difference(['Unnamed: 0'])]

df_X['미납횟수'] = ''
for i in range(0,len(df_X),1) : 
    for j in range(0, len(df),1):
        if df_X['CONTACT_ID'].iloc[i] == df['CONTACT_ID'].iloc[j] : 
            if df['PAY_NUM_NOPAY'].iloc[j] >= -19 :
                df_X['미납횟수'].iloc[i] = df['PAY_NUM_NOPAY'].iloc[j]
            else : df_X['미납횟수'].iloc[i] = -20
df_X = df_X[df_X['미납횟수']>-20]
df_X.index = range(0,len(df_X),1)
print(df_X)

df_X.to_csv('18.다중회귀분석.csv', encoding='utf-8-sig')