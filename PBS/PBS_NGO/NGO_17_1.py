#NGO - 단순회귀분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
import scikit_posthocs
import statsmodels.formula.api as smf
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')
df1 = df[['PAY_SUM_PAYMENTAMOUNT','PAY_NUM']]
df1 =df1.dropna()
df1.index = range(0,len(df1),1)

#2. 단순회귀분석 
model = smf.ols(formula = 'PAY_SUM_PAYMENTAMOUNT~ PAY_NUM', data=df1).fit()
print(model.summary())