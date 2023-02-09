#분류예측
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df= pd.read_csv('Ashopping.csv', encoding='CP949')


X = df[['총매출액','방문빈도','1회 평균매출액','거래기간','평균 구매주기']]
Y = df['이탈여부']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

X_train, Y_train = SMOTE(random_state = 0).fit_sample(X_train, Y_train)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
mylist = list(range(2,50))
parameter_grid = {'n_neighbors':mylist}

grid_search = GridSearchCV(KNeighborsClassifier(), parameter_grid, cv=10)
grid_search.fit(X_train, Y_train)
print('최적의 인자 : ', grid_search.best_params_)

model_7 = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
model_3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
model_7.fit(X_train, Y_train)
model_3.fit(X_train, Y_train)
Y_pred_7 = model_7.predict(X_test)
Y_pred_3 = model_3.predict(X_test)

#정확도 평가
print('3학습용 데이터 세트 정확도 : ', model_3.score(X_train, Y_train))
print('3평가용 데이터 세트 정확도 : ', model_3.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_3))

#정확도 평가
print('7학습용 데이터 세트 정확도 : ', model_7.score(X_train, Y_train))
print('7평가용 데이터 세트 정확도 : ', model_7.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_7))



#수치예측
import numpy as np

X = df[df.이탈여부==0][['총매출액','1회 평균매출액','총 할인 금액','구매카테고리수','Frequency']]
Y = np.log1p(df[df.이탈여부==0]['평균 구매주기'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors = 5, p=2)
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

