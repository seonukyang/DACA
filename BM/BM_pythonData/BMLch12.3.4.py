#랜덤포레스트 - 수치 예측
#1) 변수 지정 및 전처리
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
df = pd.read_csv('Ashopping.csv', encoding = 'cp949')

#1. 변수 지정
X = df[['Recency','Frequency','Monetary','총매출액','방문빈도']]
Y = df['구매카테고리수']

#2. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#2) 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
from sklearn.ensemble import RandomForestRegressor

#2. 모형 생성
model = RandomForestRegressor(random_state=0, n_estimators = 100, max_depth = 4)

#3. 모형 학습 및 예측
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print('평가용 데이터 세트에 대한 예측값 : ', Y_pred)

#3) 모형 평가
#결정계수 평가
print('학습용 데이터 세트 결정계수 : ', round(model.score(X_train, Y_train),3))
print('평가용 데이터 세트 결정계수 : ', round(model.score(X_test, Y_test),3))

#RMSE 평가
#1. 모듈 및 함수 불러오기
from sklearn.metrics import mean_squared_error
from math import sqrt

#2. RMSE 계산
rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print('RMSE : ', round(rmse,3))