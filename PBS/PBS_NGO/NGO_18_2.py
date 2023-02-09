#NGO - 다변량분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
from patsy import dmatrices
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

df =  pd.read_csv('18.다중회귀분석.csv', sep=',', encoding='UTF-8')
df = df[df.columns.difference(['Unnamed: 0'])]
df1 = df[['고객등급','미납횟수','PAY_SUM_PAYMENTAMOUNT','PAY_NUM']]
print(df1)

#2. 다중회귀분석 실행
model2 = smf.ols(formula = 'PAY_SUM_PAYMENTAMOUNT ~ 고객등급 + 미납횟수 + PAY_NUM', data=df1).fit()
print(model2.summary())

#3. 다중공선성 확인하기
y, X= dmatrices('PAY_SUM_PAYMENTAMOUNT ~ 고객등급 + 미납횟수 + PAY_NUM', data=df1, return_type='dataframe') #전처리 역할
print(np.round(variance_inflation_factor(X.values,1),3)) #서비스_만족도의 VIF 값
print(np.round(variance_inflation_factor(X.values,2),3))
print(np.round(variance_inflation_factor(X.values,3),3))