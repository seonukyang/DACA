from sys import platform
import pandas as pd
import statsmodels.formula.api as smf
df = pd.read_csv('Ashopping.csv', encoding='CP949')

#단순회귀분석 실행하기
model1 = smf.ols(formula = '총_매출액 ~ 방문빈도', data=df).fit()
print(model1.summary())

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Malgun Gothic'

fit = np.polyfit(df['방문빈도'], df['총_매출액'],1)
fit_fn = np.poly1d(fit)
print(np.round(fit,3))
print(fit_fn)

#산점도 그리기
plt.title('단순회귀분석')
plt.xlabel('방문빈도')
plt.ylabel('총매출액')
plt.plot(df['방문빈도'], df['총_매출액'], 'o')
plt.plot(df['방문빈도'], fit_fn(df['방문빈도']),'r')
plt.show()



#다중회귀분석
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
model2 = smf.ols(formula = '총_매출액 ~ 서비스_만족도 + 방문빈도 + 구매_카테고리_수', data=df).fit()
print(model2.summary())

#다중공선성 확인하기
y, X = dmatrices('총_매출액 ~ 서비스_만족도 + 방문빈도 + 구매_카테고리_수', data=df, return_type='dataframe')
print('서비스 만족도의 다중공선성 : ',np.round(variance_inflation_factor(X.values,1),3))
print('방문빈도의 다중공선성 : ',np.round(variance_inflation_factor(X.values,2),3))
print('구매 카테고리 수의 다중공선성 : ',np.round(variance_inflation_factor(X.values,3),3))



#더미변수 생성
df2 = pd.get_dummies(df['구매유형'], prefix='구매유형', drop_first=True)
df3 = pd.concat([df,df2],axis=1)

model3 = smf.ols(formula = '방문빈도 ~ 구매유형_2+ 구매유형_3 + 구매유형_4', data=df3).fit()
print(model3.summary())