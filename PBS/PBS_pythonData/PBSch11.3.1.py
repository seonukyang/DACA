#회귀분석 - 다중회귀분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')

#2. 다중회귀분석 실행
model2 = smf.ols(formula = '총_매출액 ~ 서비스_만족도 + 방문빈도 + 구매_카테고리_수', data=df).fit()
print(model2.summary())

#3. 다중공선성 확인하기
y, X= dmatrices('총_매출액 ~ 서비스_만족도 + 방문빈도 + 구매_카테고리_수', data=df, return_type='dataframe') #전처리 역할
print(np.round(variance_inflation_factor(X.values,1),3)) #서비스_만족도의 VIF 값
print(np.round(variance_inflation_factor(X.values,2),3))
print(np.round(variance_inflation_factor(X.values,3),3))