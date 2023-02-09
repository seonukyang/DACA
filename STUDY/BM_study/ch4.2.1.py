import pandas as pd
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df= pd.read_csv('Ashopping.csv', encoding='CP949')

font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

X = df[df.이탈여부==0][['고객ID','할인권 사용 횟수', '1회 평균매출액', '총매출액', '평균 구매주기', 'Recency', 'Frequency', 'Monetary']]
Y = df[df.이탈여부==0]['구매카테고리수']

X_train1, X_test1, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)
X_train = X_train1[['할인권 사용 횟수', '1회 평균매출액', '총매출액', '평균 구매주기', 'Recency', 'Frequency', 'Monetary']]
X_test = X_test1[['할인권 사용 횟수', '1회 평균매출액', '총매출액', '평균 구매주기', 'Recency', 'Frequency', 'Monetary']]
ct = ColumnTransformer([('scaling', StandardScaler(),
['할인권 사용 횟수', '1회 평균매출액', '총매출액', '평균 구매주기', 'Recency', 'Frequency', 'Monetary'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)


#릿지 선형 회귀모형
from sklearn.linear_model import Ridge
result = X_test1['고객ID']
Rr = Ridge(random_state=0).fit(X_train, Y_train)
Y_pred = Rr.predict(X_test)
for i in range(0,len(Y_pred), 1):
    if Y_pred[i] >=5 :
        print(result.iloc[i])


#결정계수
print('학습용 데이터 세트 결정 계수 : ', Rr.score(X_train, Y_train))
print('평가용 데이터 세트 결정 계수 : ', Rr.score(X_test, Y_test))

#RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print('RMSE : ', rmse)

print('절편 : ',Rr.intercept_)
print('가중지 : ', Rr.coef_)