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
pd.options.display.float_format = '{:3f}'.format
#등분산 검정
df1 = df_X[['PAY_SUM_PAYMENTAMOUNT','고객등급','주요채널']]
고객등급 = []
for i in range(1,6,1):
    고객등급.append(df1[df1.고객등급==i].PAY_SUM_PAYMENTAMOUNT)
print('고객등급x총금액',sp.stats.levene(고객등급[0],고객등급[1],고객등급[2],고객등급[3],고객등급[4]))

주요채널 = []
for i in range(1,8,1):
    주요채널.append(df1[df1.주요채널==i].PAY_SUM_PAYMENTAMOUNT)
print('주요채널x총금액', sp.stats.levene(주요채널[0],주요채널[1],주요채널[2],주요채널[3],주요채널[4],주요채널[5],주요채널[6]))

#이원분산분석
print('이원분산분석\n', pg.anova(dv='PAY_SUM_PAYMENTAMOUNT', between=['고객등급','주요채널'], data=df_X))

#사후분석
df1['고객등급'] = df1['고객등급'].astype(str)
df1['주요채널'] = df1['주요채널'].astype(str)
print('고객등급x총매출액\n', scikit_posthocs.posthoc_scheffe(df1, val_col='PAY_SUM_PAYMENTAMOUNT', group_col='고객등급'))
print('주요채널x총매출액\n', scikit_posthocs.posthoc_scheffe(df1, val_col='PAY_SUM_PAYMENTAMOUNT', group_col='주요채널'))

#고객등급, 주요채널별 평균 총 매출액
print(pd.pivot_table(df1, index='고객등급', columns='주요채널', values='PAY_SUM_PAYMENTAMOUNT', aggfunc=np.mean))
