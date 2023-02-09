#표준 선형 회귀분석
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math


df = pd.read_csv('1. NGO.csv', encoding='cp949')
df['가입나이'] = round((df['AGE']*12 - df['LONGEVITY_M'])/12,0)
df = df[df['가입나이'] > 0]
df['PLED_FIRST_DAY'] = df['LONGEVITY_D'] - df['PLED_FIRST_LONGEVITY']

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
plt.rc('font', family=font_name)

# plt.title('가입나이 분포')
# sns.distplot(df['가입나이'])
# plt.show()
# plt.clf
# plt.title('고객나이 분포')
# sns.distplot(df['AGE'])
# plt.show()

#전처리 모듈 불러오기
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


#변수지정
df1 = df[['AGE','가입나이','LONGEVITY_D','SEX','PAY_RATE_NOPAY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT','CHURN']]
df1 = df1[df1['가입나이']>0]
df1 = df1[df1['SEX']!=0]
df1=df1.dropna()
#데이터 파악하기
user_mont = df1[df1['CHURN']==0]['PAY_SUM_PAYMENTAMOUNT'].sum()
user_mean = df1[df1['CHURN']==0]['PAY_SUM_PAYMENTAMOUNT'].mean()
user_count = df1[df1['CHURN']==0]['PAY_SUM_PAYMENTAMOUNT'].count()
nonuser_mont = df1[df1['CHURN']==1]['PAY_SUM_PAYMENTAMOUNT'].sum()
nonuser_mean = df1[df1['CHURN']==1]['PAY_SUM_PAYMENTAMOUNT'].mean()
nonuser_count = df1[df1['CHURN']==1]['PAY_SUM_PAYMENTAMOUNT'].count()
all_mont = user_mont + nonuser_mont
print('비이탈 고객의 총납입금액 총량 : ',user_mont)
print('비이탈 고객의 총납입금액 평균 : ',user_mean)
print('비이탈 고객의 수 : ',user_count)
print('이탈 고객의 총납입금액 총량 : ',nonuser_mont)
print('이탈 고객의 총납입금액 평균 : ',nonuser_mean)
print('이탈 고객의 수 : ',nonuser_count)
print('모든 고객의 총납입금액 총량 : ',all_mont)

#종속변수 로그화시켜서 약간 정규화
# df1['PAY_SUM_PAYMENTAMOUNT'] = np.log1p(df1['PAY_SUM_PAYMENTAMOUNT'])

num = ['AGE','가입나이','LONGEVITY_D','PAY_RATE_NOPAY','PAY_NUM']
cg = ['SEX']

X = df1[df1['CHURN']==0][num+cg]
Y = df1[df1['CHURN']==0]['PAY_SUM_PAYMENTAMOUNT']
# XY = df1[df1['CHURN']==0][['AGE','가입나이','LONGEVITY_D','PAY_RATE_NOPAY','PAY_NUM','SEX','REGIONID','PAY_SUM_PAYMENTAMOUNT']]

# print('X', X)
# print('Y', Y)
# print(XY.corr())
#3. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#4. 표준화 및 원핫인코딩
ct = ColumnTransformer([('scaling', StandardScaler(), num), ('onehot',OneHotEncoder(sparse = False, handle_unknown = 'ignore'), cg)])
ct.fit(X_train)
print(X_test[10:20])
print(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)



#모형 학습하기
#1. 모듈 불러오기
from sklearn.linear_model import LinearRegression

#2 모형 학습
lr = LinearRegression().fit(X_train, Y_train)

#3. 모형 예측
Y_pred_S = lr.predict(X_test)
print('표준 선형 평가용 데이터 세트에 대한 예측값 : ',Y_pred_S)

#4) 모형 평가
print('표준 선형 학습용 데이터 세트 결정계수 : ', round(lr.score(X_train, Y_train),2))
print('표준 선형 평가용 데이터 세트 결정계수 : ', round(lr.score(X_test, Y_test),2))

#RMSE평가
#1. 모듈 및 함수 불러오기
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_S = sqrt(mean_squared_error(Y_test, Y_pred_S))
print('선형 회귀 분석 RMSE : ', rmse_S)

#5) 절편 및 가중치출력
print('표준 선형 절편 : ',np.round(lr.intercept_,3))
print('표준 선형 가중치 : ', np.round(lr.coef_, 3))

# 모델에 이탈 고객 데이터 접목시키기
df_s_test = df1[df1['CHURN']==1][num+cg]
df_s_test = ct.transform(df_s_test)
nonuser_pred_S = lr.predict(df_s_test)

nonuser_pred_ad_S = nonuser_pred_S
# for i in range(0,len(nonuser_pred_S),1) : 
#  nonuser_pred_ad_S[i] = math.exp(nonuser_pred_S[i])
print(nonuser_pred_ad_S)
nonuser_pred_mont_S = nonuser_pred_ad_S.sum()
nonuser_pred_mean_S = nonuser_pred_ad_S.mean()
print('표준-이탈 고객이 비이탈일때 총납입금액 총량 : ', nonuser_pred_mont_S)
print('표준-이탈 고객의 비이탈일때 총납입금액 평균 : ', nonuser_pred_mean_S)
print('표준-이탈 고객의 잠재 총납입금액 총량 : ',nonuser_pred_mont_S- nonuser_mont)


#릿지 선형 회귀모형
#1. 모듈 및 함수 불러오기
from sklearn.linear_model import Ridge

#2. 모형 학습 및 예측
Rr = Ridge(random_state=0,alpha=1).fit(X_train, Y_train)
Y_pred_R = Rr.predict(X_test)
print('릿지 선형 학습용 데이터 세트 결정계수 : ', round(lr.score(X_train, Y_train),2))
print('릿지 선형 평가용 데이터 세트 결정계수 : ', round(lr.score(X_test, Y_test),2))
rmse_R = sqrt(mean_squared_error(Y_test, Y_pred_R))
print('릿지 회귀 분석 RMSE : ', rmse_R)
#3) 절편 및 가중치출력
print('릿지 선형 절편 : ',np.round(Rr.intercept_,3))
print('릿지 선형 가중치 : ', np.round(Rr.coef_, 3))

# 모델에 이탈 고객 데이터 접목시키기
df_r_test = df1[df1['CHURN']==1][num+cg]
df_r_test = ct.transform(df_r_test)
nonuser_pred_R = lr.predict(df_r_test)

nonuser_pred_ad_R = nonuser_pred_R
# for i in range(0,len(nonuser_pred_S),1) : 
#  nonuser_pred_ad_S[i] = math.exp(nonuser_pred_S[i])
nonuser_pred_mont_R = nonuser_pred_ad_R.sum()
nonuser_pred_mean_R = nonuser_pred_ad_R.mean()
print('랏지-이탈 고객이 비이탈일때 총납입금액 총량 : ', nonuser_pred_mont_R)
print('랏지-이탈 고객의 비이탈일때 총납입금액 평균 : ', nonuser_pred_mean_R)
print('랏지-이탈 고객의 잠재 총납입금액 총량 : ',nonuser_pred_mont_R- nonuser_mont)

#라소 선형 회귀모형
#1) 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
from sklearn.linear_model import Lasso

#2) 모형 학습 및 예측
Lr = Lasso(random_state=0, alpha=0.0001, max_iter=10000).fit(X_train,Y_train)
Y_pred_L = Lr.predict(X_test)

#2) 모형 평가
print('라쏘 선형 학습용 데이터 세트 결정계수 : ', round(Lr.score(X_train, Y_train),2))
print('라쏘 선형 평가용 데이터 세트 결정계수 : ', round(Lr.score(X_test, Y_test),2))
rmse_L = sqrt(mean_squared_error(Y_test, Y_pred_R))
print('라쏘 회귀 분석 RMSE : ', rmse_L)
#3) 절편 및 가중치출력
print('라쏘 선형 절편 : ',np.round(Lr.intercept_,3))
print('라쏘 선형 가중치 : ', np.round(Lr.coef_, 3))

# 모델에 이탈 고객 데이터 접목시키기
df_l_test = df1[df1['CHURN']==1][num+cg]
df_l_test = ct.transform(df_l_test)
nonuser_pred_L = lr.predict(df_l_test)

nonuser_pred_ad_L = nonuser_pred_L
# for i in range(0,len(nonuser_pred_S),1) : 
#  nonuser_pred_ad_S[i] = math.exp(nonuser_pred_S[i])
nonuser_pred_mont_L = nonuser_pred_ad_L.sum()
nonuser_pred_mean_L = nonuser_pred_ad_L.mean()
print('라쏘-이탈 고객이 비이탈일때 총납입금액 총량 : ', nonuser_pred_mont_L)
print('라쏘-이탈 고객의 비이탈일때 총납입금액 평균 : ', nonuser_pred_mean_L)
print('라쏘-이탈 고객의 잠재 총납입금액 총량 : ',nonuser_pred_mont_L- nonuser_mont)