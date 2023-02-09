#K_NN 수치예측
#1) 변수 지정 및 전처리
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv('Ashopping.csv', encoding='cp949')
# print(df['총매출액'].head())
# df['총매출액str'] = df['총매출액']
# print(df['총매출액str'][0])
# print(type(df['총매출액str'][0]))
# df['총매출액str'][0] = str(df['총매출액str'][0])
# print(type(df['총매출액str'][0]))
# a = 'abc'
# print(a[0:1])


#1. 모듈 및 함수 불러오기
import numpy as np

#2 . 변수 지정 및 로그변환
X = df[df.이탈여부 ==0][['총매출액', '1회 평균매출액','총 할인 금액','구매카테고리수','Frequency']]
Y_o = df[df.이탈여부 == 0]['평균 구매주기']
Y = np.log1p(df[df.이탈여부 == 0]['평균 구매주기'])
# print(Y_o)
# print(Y)

#3. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#4. 표준화
scaler = StandardScaler().fit(X_train)
# print(X_test)
X_test = scaler.transform(X_test)
# print(X_test)
X_train = scaler.transform(X_train)

#2) 모형 학습 및 에측
#1. 모듈 및 학습 불러오기
from sklearn.neighbors import KNeighborsRegressor

#2. 모형 생성
model = KNeighborsRegressor(n_neighbors=5, p=2)

#3. 모형 학습 및 예측
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print('평가용 데이터 세트에 대한 예측값 :',Y_pred)

#3)모형 평가
# 결정계수 평가
print('학습용 데이터 세트 정확도 : ',model.score(X_train, Y_train))
print('평가용 데이터 세트 정확도 : ',model.score(X_test, Y_test))
#수치 예측 모형 일때는 결정 계수를 반환한다.

#RMSE 평가
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print('RMSE : ', rmse)
