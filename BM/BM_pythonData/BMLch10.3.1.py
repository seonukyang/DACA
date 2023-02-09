#인공신경망 - 분류 예측
#1) 변수 지정 및 전처리
#1. 모듈 및 함수 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

#2. 변수 지정
df = pd.read_csv('Ashopping.csv', encoding='cp949')
X = df[['총매출액','구매금액대','1회 평균매출액','거래기간','평균 구매주기']]
Y = df['할인민감여부']

#3. 데이터 분할 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#4. 표준화 및 원핫인코딩
ct = ColumnTransformer([('scaling', StandardScaler(), ['총매출액','1회 평균매출액','거래기간','평균 구매주기']),
('onehot', OneHotEncoder(sparse = False),['구매금액대'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

#5. 오버 샘플링
smote = SMOTE(random_state = 0)
X_train, Y_train = smote.fit_sample(X_train, Y_train)

#2) 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
from sklearn.neural_network import MLPClassifier

#2. 모형 생성
model = MLPClassifier(random_state=0, alpha = 0.001, hidden_layer_sizes = [50])

#3) 모형 학습 및 예측
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print('평가용 데이터 세트에 대한 예측값 : ',Y_pred)

#3) 모형 평가
#정확도 평가
print('학습용 데이터 세트 정확도 : ',model.score(X_train, Y_train))
print('평가용 데이터 세트 정확도 : ',model.score(X_test, Y_test))

#-정밀도, 재현율, F1 스코어 평가
#1. 모듈 및 함수 불러오기
from sklearn.metrics import classification_report

#2. 재현율, 정밀도, F1 스코어 평가
print(classification_report(Y_test, Y_pred))

