#분류 예측 실습
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


df= pd.read_csv('Ashopping.csv', encoding='CP949')


X = df[['총매출액','구매금액대','할인권 사용 횟수','총 할인 금액','구매유형','구매카테고리수','성별','거래기간','방문빈도','할인민감여부']]
Y = df['이탈여부']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

smote = SMOTE(random_state=0)
X_train, Y_train = smote.fit_sample(X_train, Y_train)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0, max_depth=3)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#정확도 평가
print('3학습용 데이터 세트 정확도 : ', model.score(X_train, Y_train))
print('3평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

#변수 중요도 출력
feature_name = X.columns
feature_importances = model.feature_importances_
sorted(zip(feature_importances, feature_name), reverse=True)

#모형의 시각화
import graphviz
from sklearn.tree import export_graphviz
export_graphviz(model, out_file='tree.dot', class_names=['비이탈','이탈'], 
feature_names = feature_name, impurity=True, filled=True)
with open('tree.dot', encoding = 'utf-8') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


#수치예측 실습
X = df[df.이탈여부==0][['총매출액','구매금액대','1회 평균매출액','할인권 사용 횟수',
'총 할인 금액','고객등급','구매유형','구매카테고리수','할인민감여부','성별']]
Y = df[df.이탈여부==0]['평균 구매주기']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=0, max_depth=6)
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

#변수 중요도 출력
feature_name = X.columns
feature_importances = model.feature_importances_
sorted(zip(feature_importances, feature_name), reverse=True)

#모형의 시각화
export_graphviz(model, out_file='tree.dot', class_names=['비이탈','이탈'], 
feature_names = feature_name, impurity=True, filled=True)
with open('tree.dot', encoding = 'utf-8') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)