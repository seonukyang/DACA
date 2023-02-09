#분류 예측 실습
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


df= pd.read_csv('Ashopping.csv', encoding='CP949')

X = df[['고객ID','총매출액', '구매금액대', '할인권 사용 횟수', '총 할인 금액', 
'구매유형', '구매카테고리수', '성별', '거래기간', '방문빈도', '이탈여부']]
Y = df['할인민감여부']

X_train1, X_test1, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state=1)
X_train = X_train1[['총매출액', '구매금액대', '할인권 사용 횟수', '총 할인 금액', 
'구매유형', '구매카테고리수', '성별', '거래기간', '방문빈도', '이탈여부']]
X_test = X_test1[['총매출액', '구매금액대', '할인권 사용 횟수', '총 할인 금액', 
'구매유형', '구매카테고리수', '성별', '거래기간', '방문빈도', '이탈여부']]
smote = SMOTE(random_state=1)
X_train, Y_train = smote.fit_sample(X_train, Y_train)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=1, max_depth=5)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# result = X_test1['고객ID']
# for i in range(0,len(Y_pred), 1):
#     if Y_pred[i] == 1 :
#         print(result.iloc[i])


#정확도 평가
print('학습용 데이터 세트 정확도 : ', model.score(X_train, Y_train))
print('평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

#변수 중요도 출력
feature_name = X.columns
feature_importances = model.feature_importances_
print(sorted(zip(feature_importances, feature_name), reverse=True))

#모형의 시각화
# import graphviz
# from sklearn.tree import export_graphviz
# export_graphviz(model, out_file='tree.dot', class_names=['비이탈','이탈'], 
# feature_names = feature_name, impurity=True, filled=True)
# with open('tree.dot', encoding = 'utf-8') as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph)

