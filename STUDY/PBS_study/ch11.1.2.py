import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
df = pd.read_csv('Ashopping.csv', encoding='CP949')

#다중회귀분석
model2 = smf.ols(formula = '총_매출액 ~ 서비스_만족도 + 방문빈도 + 구매_카테고리_수', data=df).fit()
print(model2.summary())

#다중공선성 확인하기
y, X = dmatrices('총_매출액 ~ 서비스_만족도 + 방문빈도 + 구매_카테고리_수', data=df, return_type='dataframe')
print('서비스 만족도의 다중공선성 : ',np.round(variance_inflation_factor(X.values,1),3))
print('방문빈도의 다중공선성 : ',np.round(variance_inflation_factor(X.values,2),3))
print('구매 카테고리 수의 다중공선성 : ',np.round(variance_inflation_factor(X.values,3),3))