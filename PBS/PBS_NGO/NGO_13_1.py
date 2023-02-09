#NGO - 일원분산분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
import scikit_posthocs
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')
df_X =  pd.read_csv('고객등급점수표.csv', sep=',', encoding='UTF-8')

#print('상위 10퍼', np.percentile(df_X['총합'],90))
#print('상위 30퍼', np.percentile(df_X['총합'],70))
#print('상위 60퍼', np.percentile(df_X['총합'],40))
#print('상위 90퍼', np.percentile(df_X['총합'],10))
#print(df_X.groupby('총합').size())
df_X['고객등급']=''
for i in range(0,len(df_X),1) :
    if (df_X['총합'][i] >= 19) :
        df_X['고객등급'].iloc[i] = 1
    elif  (df_X['총합'][i] >= 17) :
       df_X['고객등급'].iloc[i] = 2
    elif  (df_X['총합'][i] >= 15) :
       df_X['고객등급'].iloc[i] = 3
    elif  (df_X['총합'][i] >= 11) :
       df_X['고객등급'].iloc[i] = 4  
    else  : df_X['고객등급'].iloc[i] = 5

pd.options.display.float_format = '{:3f}'.format
df1 = df_X[['고객등급','PAY_SUM_PAYMENTAMOUNT']]
df2 = df_X[['고객등급','LONGEVITY_D']]

#2. 등분산
고객등급1 = []
for i in range(1,6,1):
    고객등급1.append(df1[df1.고객등급==i].PAY_SUM_PAYMENTAMOUNT)
print('고객등급x총금액',sp.stats.levene(고객등급1[0],고객등급1[1],고객등급1[2],고객등급1[3],고객등급1[4]))
고객등급2 = []
for i in range(1,6,1):
    고객등급2.append(df2[df2.고객등급==i].LONGEVITY_D)
print('고객등급x가입일수',sp.stats.levene(고객등급2[0],고객등급2[1],고객등급2[2],고객등급2[3],고객등급2[4]))

#3. Welch 일원분산분석
print(pg.welch_anova(dv='PAY_SUM_PAYMENTAMOUNT', between='고객등급',data=df1))
df1['고객등급'] = df1['고객등급'].astype(str)
print(scikit_posthocs.posthoc_scheffe(df1, val_col='PAY_SUM_PAYMENTAMOUNT', group_col='고객등급'))
print(고객등급1[0].mean(),고객등급1[1].mean(),고객등급1[2].mean(),고객등급1[3].mean(),고객등급1[4].mean())

print(pg.welch_anova(dv='LONGEVITY_D', between='고객등급',data=df2))
df2['고객등급'] = df2['고객등급'].astype(str)
print(scikit_posthocs.posthoc_scheffe(df2, val_col='LONGEVITY_D', group_col='고객등급'))
print(고객등급2[0].mean(),고객등급2[1].mean(),고객등급2[2].mean(),고객등급2[3].mean(),고객등급2[4].mean())
#print(df_X[df_X.columns.difference(['Unnamed: 0'])])
df_X[df_X.columns.difference(['Unnamed: 0'])].to_csv('고객등급표.csv', encoding='utf-8-sig')