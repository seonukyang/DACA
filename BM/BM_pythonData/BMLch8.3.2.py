#SVM 수치 예측 모형
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
df = pd.read_csv('Ashopping.csv', encoding='cp949')

#1) 변수 지정 및 전처리
#1. 모듈 및 함수 불러오기
import numpy as np

#2. 변수 지정 및 종속변수 로그변환
X = df[['방문빈도','총 할인 금액','고객등급','거래기간','할인민감여부','평균 구매주기']]
Y = np.log1p(df['1회 평균매출액'])

#3. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#4. 표준화 및 원핫인코딩
ct = ColumnTransformer([('scaling', StandardScaler(), ['방문빈도','총 할인 금액','거래기간',
'거래기간','평균 구매주기']), ('onehot',OneHotEncoder(sparse = False),['고객등급','할인민감여부'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

#2) 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
from sklearn.svm import SVR

#2. 모형 생성
model = SVR(C=10, epsilon = 0.1, gamma = 0.01)

#3. 모형 학습 및 예측
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print('평가용 데이터 세트에 대한 예측값 : ',Y_pred)

#3) 모형 평가
#결정계수 평가
print('학습용 데이터 세트 결정계수 : ',model.score(X_train, Y_train))
print('평가용 데이터 세트 결정계수 : ',model.score(X_test, Y_test))

#RMSE 평가
#1. 모듈 및 함수 불러오기
from sklearn.metrics import mean_squared_error
from math import sqrt

#2. RMSE 계산
rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print('RMSE : ', rmse)