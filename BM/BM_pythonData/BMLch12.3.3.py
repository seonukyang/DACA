#랜덤 포레스트 - 분류 예측
#1) 변수 지정 및 전처리
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
df = pd.read_csv('Ashopping.csv', encoding = 'cp949')

#1. 변수 지정 및 데이터 세트 분할
X = df[['방문빈도','1회 평균매출액','할인권 사용 횟수','총 할인 금액','거래기간','평균 구매주기','구매유형']]
Y = df['고객등급']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#2. 오버 샘플링
smote = SMOTE(random_state=0)
X_train, Y_train = smote.fit_sample(X_train, Y_train)

#2) 모형 학습 및 예측
#1. 모형 및 함수 불러오기
from sklearn.ensemble import RandomForestClassifier

#2. 모형 생성
model = RandomForestClassifier(random_state = 0, n_estimators = 300, max_depth = 5)

#3. 모형 학습 및 예측
model.fit(X_train, Y_train)

#4. 평가용 데이터 세트에 대한 예측값 출력
Y_pred = model.predict(X_test)
print('평가용 데이터 세트에 대한 예측값 : ', Y_pred)

#3) 모형 평가
#정확도 평가
print('학습용 데이터 세트 정확도 : ', round(model.score(X_train, Y_train),3))
print('평가용 데이터 세트 정확도 : ', round(model.score(X_test, Y_test),3))

#정밀도, 재현율, F1 스코어 평가
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

