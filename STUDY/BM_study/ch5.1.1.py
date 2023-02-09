#이항 분류 예측
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df= pd.read_csv('Ashopping.csv', encoding='CP949')

X = df[['총매출액','거래기간','방문빈도']]
Y = df['이탈여부']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

from imblearn.over_sampling import SMOTE
from collections import Counter

smote = SMOTE(random_state = 0)
X_train_over, Y_train_over = smote.fit_sample(X_train, Y_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C = 1, random_state=0)

model.fit(X_train_over, Y_train_over)
Y_pred =model.predict(X_test)

#정확도 평가
print('학습용 데이터 세트 정확도 : ', model.score(X_train_over, Y_train_over))
print('평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

#이항 로지스틱 회귀계수 출력
print('총매출액 회귀계수 : ', model.coef_[0,0])
print('거래기간 회귀계수 : ', model.coef_[0,1])
print('방문빈도 회귀계수 : ', model.coef_[0,2])


#다항 분류 예측 실슥
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


X = df[['방문빈도','총 할인 금액','거래기간','할인민감여부']]
Y = df['구매금액대']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

ct = ColumnTransformer([('scaling',StandardScaler(),['방문빈도','총 할인 금액','거래기간']), 
('onehot', OneHotEncoder(sparse = False),['할인민감여부'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

model = LogisticRegression(random_state=0, C=0.1, solver='newton-cg', multi_class='multinomial')
model.fit(X_train, Y_train)

#정확도 평가
print('학습용 데이터 세트 정확도 : ', model.score(X_train_over, Y_train_over))
print('평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

print('구매금액대 0의 회귀계수')
print('방문빈도 회귀계수 : ', model.coef_[0,0])
print('총 할인 금액 회귀계수 : ', model.coef_[0,1])
print('거래기간 회귀계수 : ', model.coef_[0,2])
print('할인 민감여부_1 : ', model.coef_[0,4])

print('구매금액대 1의 회귀계수')
print('방문빈도 회귀계수 : ', model.coef_[1,0])
print('총 할인 금액 회귀계수 : ', model.coef_[1,1])
print('거래기간 회귀계수 : ', model.coef_[1,2])
print('할인 민감여부_1 : ', model.coef_[1,4])

print('구매금액대 2의 회귀계수')
print('방문빈도 회귀계수 : ', model.coef_[2,0])
print('총 할인 금액 회귀계수 : ', model.coef_[2,1])
print('거래기간 회귀계수 : ', model.coef_[2,2])
print('할인 민감여부_1 : ', model.coef_[2,4])