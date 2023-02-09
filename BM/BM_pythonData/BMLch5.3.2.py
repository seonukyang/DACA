#다항 분류 예측 실습
#1) 변수 지정 및 전처리
#1. 모듈 및 함수 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 

df = pd.read_csv('Ashopping.csv', encoding='cp949')

#2. 변수 지정(독립변수, 종속변수), 데이터 분할
X = df[['방문빈도','총 할인 금액','거래기간','할인민감여부']]
Y = df['구매금액대']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#3. 표준화 및 원핫인코딩
ct = ColumnTransformer([('scaling', StandardScaler(), ['방문빈도','총 할인 금액','거래기간']),
('onehot', OneHotEncoder(sparse = False),['할인민감여부'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

#2) 모형 학습 및 예측
#1. 모형 생성
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0, C=0.1, solver='newton-cg', multi_class='multinomial')

#2. 모형 학습 및 예측
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print('평가용 데이터 세트에 대한 예측값 : ', Y_pred)

#3) 모형 평가
#- 정확도 평가
print('학습용 데이터 세트 정확도 : ',model.score(X_train, Y_train))
print('평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))

#정밀도, 재현율, F1 스코어 평가
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

#4) 다항 로지스틱 회귀계수 출력
print('---구매금액대 0의 회귀계수---')
print('방문빈도 회귀계수:{0:.3f}, 총 할인 금액 회귀계수:{1:.3f}, 거래기간:{2:.3f}, 할인 민감여부_1:{3:.3f}'.
format(model.coef_[0,0],model.coef_[0,1],model.coef_[0,2],model.coef_[0,4]))

print('---구매금액대 1의 회귀계수---')
print('방문빈도 회귀계수:{0:.3f}, 총 할인 금액 회귀계수:{1:.3f}, 거래기간:{2:.3f}, 할인 민감여부_1:{3:.3f}'.
format(model.coef_[1,0],model.coef_[1,1],model.coef_[1,2],model.coef_[1,4]))

print('---구매금액대 2의 회귀계수---')
print('방문빈도 회귀계수:{0:.3f}, 총 할인 금액 회귀계수:{1:.3f}, 거래기간:{2:.3f}, 할인 민감여부_1:{3:.3f}'.
format(model.coef_[2,0],model.coef_[2,1],model.coef_[2,2],model.coef_[2,4]))