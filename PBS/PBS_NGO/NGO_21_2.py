#NGO - 일원분산분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
import scikit_posthocs
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')
df_X =  pd.read_csv('15.df_1.csv', sep=',', encoding='UTF-8')
df_X = df_X[df_X.columns.difference(['Unnamed: 0'])]

df['고객등급'] = ''
for i in range(0,len(df),1) : 
    for j in range(0, len(df_X),1):
        if df['CONTACT_ID'].iloc[i] == df_X['CONTACT_ID'].iloc[j] : 
            if df['PAY_NUM_NOPAY'].iloc[i] >= -19 :
                df['고객등급'].iloc[i] = df_X['고객등급'].iloc[j]
            else : df_X['고객등급'].iloc[i] = -20
df_X = df_X[df_X['고객등급']>-20]
df.index = range(0,len(df),1)
print(df)

df.to_csv('21.csv', encoding='utf-8-sig')