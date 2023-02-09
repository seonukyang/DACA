#NGO - 일원분산분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
import scikit_posthocs
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')
df.index= range(0,len(df),1)
df_X =  pd.read_csv('고객등급표.csv', sep=',', encoding='UTF-8')
df_X = df_X[df_X.columns.difference(['Unnamed: 0'])]

df_X['주요채널'] = ''
for i in range(0,len(df_X),1) : 
    for j in range(0, len(df),1):
        if df_X['CONTACT_ID'].iloc[i] == df['CONTACT_ID'].iloc[j] : 
            if df['MOTI_CHANNEL'][j] == 'BROADCAST' :
                df_X['주요채널'].iloc[i] = 1
            elif df['MOTI_CHANNEL'][j] == 'DIGITAL':
                df_X['주요채널'].iloc[i] = 2  
            elif df['MOTI_CHANNEL'][j] == 'DM':
                df_X['주요채널'].iloc[i] = 3  
            elif df['MOTI_CHANNEL'][j] == 'EVENT':
                df_X['주요채널'].iloc[i] = 4  
            elif df['MOTI_CHANNEL'][j] == 'GENERAL AD':
                df_X['주요채널'].iloc[i] = 5
            elif df['MOTI_CHANNEL'][j] == 'RELATIONSH':
                df_X['주요채널'].iloc[i] = 6 
            elif df['MOTI_CHANNEL'][j] == 'TM':
                df_X['주요채널'].iloc[i] = 7
            elif df['MOTI_CHANNEL'][j] == 'UNKNOWN':
                df_X['주요채널'].iloc[i] = -2
            else : df_X['주요채널'].iloc[i] = -1
df_X = df_X[df_X['주요채널']>0]
df_X.index = range(0,len(df_X),1)

df_X['PAY_NUM'] = ''
for i in range(0,len(df_X),1) : 
    for j in range(0, len(df),1):
        if df_X['CONTACT_ID'].iloc[i] == df['CONTACT_ID'].iloc[j] :           
            if df['PAY_NUM'].iloc[j] >= 0 :
                df_X['PAY_NUM'].iloc[i] = df['PAY_NUM'].iloc[j]
            else : df_X['PAY_NUM'].iloc[i] = -1
df_X = df_X[df_X['PAY_NUM']>0]
df_X.index = range(0,len(df_X),1)

print(df_X)

df_X.to_csv('15.df_1.csv', encoding='utf-8-sig')