#인공신경망
#분류 예측 실습
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE


df= pd.read_csv('Ashopping.csv', encoding='CP949')

import numpy as np
X = df[df.이탈여부==0][['방문빈도','총 할인 금액','고객등급','구매유형','거래기간','할인민감여부','평균 구매주기']]
Y = np.log1p(df[df.이탈여부==0]['1회 평균매출액'])


X_train, Y_train = SMOTE(random_state=0).fit_sample(X_train, Y_train)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)
ct = ColumnTransformer([('scaling', StandardScaler(),['방문빈도','총 할인 금액','거래기간','평균 구매주기']), 
('onehot', OneHotEncoder(sparse = False), ['고객등급','할인민감여부','구매유형'])])
ct.fit(X_train, Y_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

from sklearn.neural_network import MLPRegressor
model = MLPRegressor(random_state=0, alpha=1, max_iter=1000, hidden_layer_sizes=[50,50])
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#결정계수
print('학습용 데이터 세트 결정 계수 : ', model.score(X_train, Y_train))
print('평가용 데이터 세트 결정 계수 : ', model.score(X_test, Y_test))

#RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print('RMSE : ', rmse)