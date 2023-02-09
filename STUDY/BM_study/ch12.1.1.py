#앙상블
#보팅 앙상블 분류 예측
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

ct = ColumnTransformer([('scaling', StandardScaler(), ['1회 평균매출액','방문빈도','총 할인 금액',
'할인권 사용 횟수','거래기간','평균 구매주기']),('onehot',OneHotEncoder(sparse = False, handle_unknown='ignore'),['구매유형'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

smote = SMOTE(random_state=0)
X_train, Y_train = smote.fit_sample(X_train, Y_train)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

dtree = DecisionTreeClassifier(random_state=0)
knn = KNeighborsClassifier()

model = VotingClassifier(estimators=[('K-NN',knn),('Dtree',dtree)], voting ='soft')

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#정확도 평가
print('Voting 분류기 정확도 : ',model.score(X_test, Y_test))

classifiers = [dtree, knn]
for classifier in classifiers:
    classifier.fit(X_train, Y_train)
    class_name = classifier.__class__.__name__
    print(class_name,'정확도 : ',classifier.score(X_test, Y_test))

from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))



#수치 예측 실습
X=df[['Recency','Frequency','Monetary','총매출액','방문빈도']]
Y=df['구매카테고리수']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

#표준화
scaler = StandardScaler()
scaler.fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

svr=SVR()
mlp = MLPRegressor(random_state=0)

model = VotingRegressor(estimators=[('SVR',svr),('MLP',mlp)])

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#결정계수
model.fit(X_train, Y_train)
print('Voting 앙상블 결정계수 : ',model.score(X_train, Y_train))

#개별 모형의 결정계수
Regressors = [svr, mlp]
for Regressor in Regressors:
    Regressor.fit(X_train, Y_train)
    class_name = Regressor.__class__.__name__
    print(class_name,'결정계수 : ',Regressor.score(X_train, Y_train))

#RMSE평가
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print('RMSE : ', rmse)