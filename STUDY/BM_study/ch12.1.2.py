#앙상블
#랜덤 포레스트 분류 예측
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

df = pd.read_csv('Ashopping.csv', encoding = 'CP949')

X = df[['방문빈도','1회 평균매출액','할인권 사용 횟수','총 할인 금액','거래기간','평균 구매주기','구매유형']]
Y = df['고객등급']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

smote = SMOTE(random_state=0)
X_train, Y_train = smote.fit_sample(X_train, Y_train)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state = 0, n_estimators=300, max_depth=5)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

#정확도 평가
print('학습용 데이터 세트 정확도 : ', model.score(X_train, Y_train))
print('평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


#랜덤포레스트 수치 예측
X = df[['Recency','Frequency','Monetary','총매출액','방문빈도']]
Y = df['구매카테고리수']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=0, n_estimators=100, max_depth=4)

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

