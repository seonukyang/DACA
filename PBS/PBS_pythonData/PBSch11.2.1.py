#회귀분석 - 단순회귀분석 
#1. 모듈 및 데이터 탑재
import pandas as pd
import statsmodels.formula.api as smf
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')

#2. 단순회귀분석 실행하기
model1 = smf.ols(formula = '총_매출액 ~ 방문빈도', data= df).fit()
print(model1.summary())

#3. 시작적 표현하기
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#4. 한글깨짐현상 방지
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

#5. 선형회귀선 구하기
fit = np.polyfit(df['방문빈도'],df['총_매출액'],1)
print('fit : ',fit)
fit_fn = np.poly1d(fit)
print(np.round(fit,3))
print(fit_fn)

#6. 산점도와 선형회귀선 그리기
#%matplotlib inline
plt.title('단순회귀분석')
plt.xlabel('방문빈도')
plt.ylabel('chdaocnfdor')
plt.plot(df['방문빈도'],df['총_매출액'],'o')
plt.plot(df['방문빈도'],fit_fn(df['방문빈도']),'r')