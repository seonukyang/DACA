#보팅 앙상블 - 분류 예측
#1) 변수 지정 및 전처리
#1. 모듈 및 함수 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

#2. 데이터 불러오기
df = pd.read_csv('Ashopping.csv', encoding = 'cp949')

#3. 변수 지정 및 데이터 세트 분할
X = df[['방문빈도','1회 평균매출액','할인권 사용 횟수','총 할인 금액','거래기간','평균 구매주기','구매유형']]
Y = df['고객등급']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#4. 표준화 및 원핫인코딩
ct = ColumnTransformer([('scaling',StandardScaler(),['1회 평균매출액','방문빈도','총 할인 금액','할인권 사용 횟수',
'거래기간','평균 구매주기']),('onehot',OneHotEncoder(sparse= False, handle_unknown = 'ignore'),['구매유형'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

#5. 오버샘플링
smote = SMOTE(random_state=0)
X_train, Y_train = smote.fit_sample(X_train, Y_train)

#2) 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

#2. 단일 모형 객체 생성 (의사결정나무 모형, K-NN 모형)
dtree = DecisionTreeClassifier(random_state=0)
knn = KNeighborsClassifier()

#3. 소프트 보팅 기반의 앙상블 모형 생성
model = VotingClassifier(estimators=[('K-NN',knn),('Dtree',dtree)], voting = 'soft')

#4. 모형 학습 및 예측
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print('평가용 데이터 세트에 대한 예측값 : ', Y_pred)

#3) 모형 평가
#정확도 평가
#1. 보팅 모형의 정확도
print('Voting 분류기 정확도 : ', model.score(X_test, Y_test))

#2. 개별 모형의 정확도
classifiers = [dtree, knn]
for classifier in classifiers : 
    classifier.fit(X_train, Y_train)
    class_name = classifier.__class__.__name__
    print(class_name,' 정확도 : ', classifier.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어 평가
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))