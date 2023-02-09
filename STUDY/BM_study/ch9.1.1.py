#가우시안 나이브 베이즈 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

df= pd.read_csv('Ashopping.csv', encoding='CP949')

X = df[['구매유형','거래기간','구매카테고리수']]
Y = df['할인민감여부']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

ct = ColumnTransformer([('scaling', StandardScaler(),['거래기간','구매카테고리수']), 
('onehot', OneHotEncoder(sparse = False), ['구매유형'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

X_train, Y_train = SMOTE(random_state=0).fit_sample(X_train, Y_train)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#정확도 평가
print('3학습용 데이터 세트 정확도 : ', model.score(X_train, Y_train))
print('3평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

#분류 확률의 추정치 확인
yproba = model.predict_proba(X_test)
print(yproba.head())


