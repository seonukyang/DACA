# 분류예측분석 - 로지스틱회귀분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import statsmodels.api as sm
pd.options.display.float_format = '{:.3f}'.format
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')

#2. 더미변수 생성하기
df2 = pd.get_dummies(df['성별'], prefix='성별', drop_first=False) #범주형 데이터의 더미데이터화
df3 = pd.concat([df, df2], axis=1) #더미데이터를 df에 이러 붙임

#print('df3 : ', df3)

#3. 종속변수와 독립변수 구분하기
df3['intercept']=1.0
x = df3[['intercept','거래기간','Recency','성별_0']] #절편값을 넣어줘야 해서 임의로 지정
y = df3[['이탈여부']]

#4.로지스틱 회귀분석 실행하기
logit = sm.Logit(y,x).fit()

#5. 분석결과 출력하기
print(logit.summary2()) #오즈비는 별도로 구해야한다.
print(np.exp(logit.params)) #params는 로지스틱 계수이므로 이를 exp에 넣어서 오즈비를 구한다.
cf_m2 = pd.DataFrame(logit.pred_table())
cf_m2.columns=['예측 0', '예측 1']
cf_m2.index = ['실제 0', '실제 1']
print('\n분류행렬표 :\n', cf_m2)
total = cf_m2.iloc[1,1]+cf_m2.iloc[0,1]+cf_m2.iloc[1,0]+cf_m2.iloc[0,0]
true = cf_m2.iloc[1,1]+cf_m2.iloc[0,0]
act = true / total
print('정확도 : ', act*100, '%')