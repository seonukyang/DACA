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

#종속변수 로그화시켜서 약간 정규화
df1['PAY_SUM_PAYMENTAMOUNT'] = np.log1p(df1['PAY_SUM_PAYMENTAMOUNT'])

num = ['AGE','가입나이','LONGEVITY_D','PAY_RATE_NOPAY','PAY_NUM']
cg = ['SEX']

X = df1[df1['CHURN']==0][num+cg]
Y = df1[df1['CHURN']==0]['PAY_SUM_PAYMENTAMOUNT']

#3. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#4. 표준화 및 원핫인코딩
ct = ColumnTransformer([('scaling', StandardScaler(), num), ('onehot',OneHotEncoder(sparse = False, handle_unknown = 'ignore'), cg)])
ct.fit(X_train)
print(X_test[10:20])
print(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

scaler=StandardScaler().fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

#2) 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
from sklearn.neighbors import KNeighborsRegressor

#2. 모형 생성
for i in range(1,15,1) : 
    model = KNeighborsRegressor(n_neighbors=i, p=2)

    #3). 모형 학습 및 예측
    model.fit(X_train, Y_train)
    Y_pred_K = model.predict(X_test)

    #모형 평가
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    print('k-nn 학습용 데이터 세트 결정계수 k = ',i,' : ', round(model.score(X_train, Y_train),2))
    print('k-nn 선형 평가용 데이터 세트 결정계수 k = ',i,' : ', round(model.score(X_test, Y_test),2))
    rmse_R = sqrt(mean_squared_error(Y_test, Y_pred_K))
    print('k-nn 분석 RMSE k = ',i,' : ', rmse_R)

    # #3) 절편 및 가중치출력
    # print('k-nn 선형 절편 : ',np.round(model.intercept_,3))
    # print('k-nn 선형 가중치 : ', np.round(model.coef_, 3))

    # 모델에 이탈 고객 데이터 접목시키기
    df_k_test = df1[df1['CHURN']==1][num+cg]
    df_k_test = ct.transform(df_k_test)
    nonuser_pred_K = model.predict(df_k_test)

    nonuser_pred_ad_K = nonuser_pred_K
    nonuser_pred_mont_K = nonuser_pred_ad_K.sum()
    nonuser_pred_mean_K = nonuser_pred_ad_K.mean()
    print('k-nn-이탈 고객이 비이탈일때 총납입금액 총량 k = ',i,' : ', nonuser_pred_mont_K)
    print('k-nn-이탈 고객의 비이탈일때 총납입금액 평균 k = ',i,' : ', nonuser_pred_mean_K)
    print('k-nn-이탈 고객의 잠재 총납입금액 총량 k = ',i,' : ',nonuser_pred_mont_K- nonuser_mont)
