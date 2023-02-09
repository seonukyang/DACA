#SVM 분류예측
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

df = pd.read_csv('Ashopping.csv', encoding='CP949')

X = df[['총매출액','구매금액대','거래기간']]
Y = df['할인민감여부']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

ct = ColumnTransformer([('scaling', StandardScaler(),['총매출액','거래기간']), 
('onehot', OneHotEncoder(sparse = False), ['구매금액대'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

X_train, Y_train = SMOTE(random_state=0).fit_sample(X_train, Y_train)

from sklearn.svm import SVC
model = SVC(C=1000, gamma=10, random_state=0)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)


#정확도 평가
print('3학습용 데이터 세트 정확도 : ', model.score(X_train, Y_train))
print('3평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

#수치 예측 
import numpy as np
X = df[['방문빈도','총 할인 금액','고객등급','거래기간','할인민감여부','평균 구매주기']]
Y = np.log1p(df['1회 평균매출액'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

ct = ColumnTransformer([('scaling', StandardScaler(),['방문빈도','총 할인 금액','거래기간','평균 구매주기']), 
('onehot', OneHotEncoder(sparse = False), ['고객등급','할인민감여부'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

from sklearn.svm import SVR

model = SVR(C = 10, epsilon = 0.1, gamma = 0.01)

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