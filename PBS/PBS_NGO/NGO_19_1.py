#NGO - 더미변수를 이용한 회귀분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
import scikit_posthocs
import statsmodels.formula.api as smf
df =  pd.read_csv('15.df_1.csv', sep=',', encoding='UTF-8')
df = df[df.columns.difference(['Unnamed: 0'])]
df1 = df['PAY_SUM_PAYMENTAMOUNT']
#더미변수 생성
df2 = pd.get_dummies(df['고객등급'], prefix='고객등급',drop_first=True)
df3 = pd.concat([df1, df2], axis = 1)

Model1 = smf.ols(formula = 'PAY_SUM_PAYMENTAMOUNT ~  고객등급_2+고객등급_3+ 고객등급_4 + 고객등급_5', data=df3).fit()
print(Model1.summary())