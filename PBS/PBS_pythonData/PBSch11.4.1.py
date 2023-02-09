#.회귀분석 - 더미변수
#1. 모듈 및 데이터 탑재
import pandas as pd
import statsmodels.formula.api as smf
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')

#2. 더미변수 생성
df2 = pd.get_dummies(df['구매유형'],prefix='구매유형',drop_first=True) #(data, 더미변수명, 첫번째 범주 drop 여부-기준으로 사용)
df3 = pd.concat([df, df2], axis=1) #.더미변수로 만든 df2를 df에 이어붙임
print('df2 : ', df2)
print('df3 : ', df3)

#3. 더미변수를 이용한 회귀분석
Model3 = smf.ols(formula = '방문빈도 ~ 구매유형_2 + 구매유형_3 + 구매유형_4', data = df3).fit()
print(Model3.summary())