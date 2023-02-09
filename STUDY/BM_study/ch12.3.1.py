#앙상블
#그래디언트 부스팅 분류 예측
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

df = pd.read_csv('Ashopping.csv', encoding = 'CP949')

X = df[['고객ID','총매출액', '1회 평균매출액', '할인권 사용 횟수', '총 할인 금액']]
Y = df['평균 구매주기']
X_train1, X_test1, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=13)
X_train = X_train1[['총매출액', '1회 평균매출액', '할인권 사용 횟수', '총 할인 금액']]
X_test = X_test1[['총매출액', '1회 평균매출액', '할인권 사용 횟수', '총 할인 금액']]
scaler = StandardScaler()
scaler.fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=13, max_depth=3, n_estimators=30, learning_rate=0.1)

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

n=0
for i in range(0,len(Y_pred),1):
    if Y_pred[i] < 65 :
        n = n+1
print(n)

result = X_test1['고객ID']

for j in range(0,len(Y_pred),1):
    print(result.iloc[j], '번째',Y_pred[j])

