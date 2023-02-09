#NGO - 일표본 t-검정
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
from pingouin import partial_corr
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#2. 데이터 나누기
df_1 = df[['E_MAIL','LONGEVITY_M','PAY_NUM','SEX']]
df_1 = df_1.dropna()
df_1['월후원율'] = df_1['PAY_NUM']/df_1['LONGEVITY_M']
df_y = df_1[df_1.E_MAIL=='Yes']
df_n = df_1[df_1.E_MAIL=='No']

#3. 등분산검정
df_y2 = np.array(df_y.월후원율)
df_n2 = np.array(df_n.월후원율)
print('등분산검정 : ', stats.bartlett(df_y2, df_n2))


#4. 독립표본 t-검정 및 방문빈도 평균
print('이메일 수신여부에 따른 후원자 평균 월후원율의 t-검정 :',stats.ttest_ind(df_y2, df_n2, equal_var=False))
print('이메일 수신여부(Yes) 후원자 평균 월후원율 : ', df_y.월후원율.mean())
print('이메일 수신여부(No) 후원자  평균 월후원율 : ', df_n.월후원율.mean())

#2-1. 데이터 나누기
df_m = df_1[df_1.SEX==1]
df_w = df_1[df_1.SEX==2]

#3-1. 등분산검정
df_m2 = np.array(df_m.월후원율)
df_w2 = np.array(df_w.월후원율)
print('등분산검정 : ', stats.bartlett(df_m2, df_w2))


#4-1. 독립표본 t-검정 및 방문빈도 평균
print('성별에 따른 후원자 평균 월후원율의 t-검정 :',stats.ttest_ind(df_m2, df_w2, equal_var=False))
print('남성 후원자 평균 월후원율 : ', df_m.월후원율.mean())
print('여성 후원자 평균 월후원율 : ', df_w.월후원율.mean())