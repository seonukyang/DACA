#의사결정나무 - 수치 예측
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv('Ashopping.csv', encoding='cp949')

#1) 변수 지정 및 전처리
#1. 변수 지정
X = df[df.이탈여부==0][['총매출액','구매금액대','1회 평균매출액','할인권 사용 횟수','총 할인 금액',
'고객등급','구매유형','구매카테고리수','할인민감여부','성별']]
Y = df[df.이탈여부==0]['평균 구매주기']

#2. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#2) 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
from sklearn.tree import DecisionTreeRegressor

#2. 모형 생성
model = DecisionTreeRegressor(random_state=0, max_depth=6)

#3. 모형 학습 및 예측
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print('평가용 데이터 세트에 대한 예측값 : ',Y_pred)

#3) 모형 평가
#결정계수 
print('학습용 데이터 세트 정확도 : ',model.score(X_train, Y_train))
print('평가용 데이터 세트 정확도 : ',model.score(X_test, Y_test))

#RMSE 평가
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print('RMSE : ', rmse)

#4) 변수 중요도 출력
feature_name = X.columns
feature_importances = model.feature_importances_
print(sorted(zip(feature_importances, feature_name), reverse=True))

#5) 모형의 시각화
#1. 모듈 및 함수 불러오기
import graphviz
from sklearn.tree import export_graphviz

#2. tree.dot 파일 생성
export_graphviz(model, out_file='tree.dot', class_names=['비이탈','이탈'],
feature_names = feature_name, impurity=True, filled=True)

#3. tree.dot 파일 읽기
with open("tree.dot", encoding = 'utf-8') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)