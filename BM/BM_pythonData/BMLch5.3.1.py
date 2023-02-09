#이항 분류 예측 실습
#1) 변수 지정 및 전처리
#1. 모듈 및 함수 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#2. 데이터 불러오기
df = pd.read_csv('Ashopping.csv', encoding='cp949')

#3. 변수 지정(독립변수, 종속변수)
X = df[['총매출액','거래기간','방문빈도']]
Y = df['이탈여부']

#4. 데이터 분할(학습용/평가용 데이터 세트)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#5. 표준화
scaler = StandardScaler()
scaler.fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

#데이터 균형화
#1. 모듈 및 함수 불러오기
from imblearn.over_sampling import SMOTE
from collections import Counter

#2. 오버 샘플링
smote = SMOTE(random_state=0)
X_train_over, Y_train_over = smote.fit_sample(X_train, Y_train)

#3. 결과 출력
#print(Counter(Y))
# print(Counter(Y_train))
# print(Counter(Y_train_over))

#2). 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
from sklearn.linear_model import LogisticRegression

#2. 모형 생성 
model = LogisticRegression(C=1, random_state=0)

#3. 모형 학습 및 예측
model.fit(X_train_over, Y_train_over)
Y_pred = model.predict(X_test)
# print('예측값 : ', Y_pred)

#3) 모형 평가
#정확도 평가
print('학습용 데이터 세트 정확도 : ',model.score(X_train_over, Y_train_over))
print('평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어 평가
#1. 모듈 및 함수 불러오기
from sklearn.metrics import classification_report

#2. 정밀도, 재현율, F1 스코어 출력
print(classification_report(Y_test, Y_pred))

#4) 이항 로지스틱 회귀계수 출력
print('총매출액 회귀계수 : {0:.3f}, 거래기간 회귀계수 : {1:.3f}, 방문빈도 회귀계수 : {2:.3f}'.
format(model.coef_[0,0],model.coef_[0,1], model.coef_[0,2]))
