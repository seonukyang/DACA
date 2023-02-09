#표준 선형 회귀모형
#1. 모듈 및 함수 불러오기
import pandas as pd
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#2. 데이터 불러오기
df = pd.read_csv('Ashopping.csv', encoding ='cp949')

#3. 폰트 설정
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
plt.rc('font', family=font_name)

#4. 히스토그램 생성
plt.title('평균 구매주기 분포')
sns.histplot(df['평균 구매주기'])
plt.hist(df['평균 구매주기'])
#plt.show()
plt.clf()
plt.title('로그 변환 후 평균 구매주기')
df['평균 구매주기'] = np.log1p(df['평균 구매주기']) #로그화 시켜서 변수를 정규분포화 시킨다.
plt.hist(df['평균 구매주기'])
#plt.show()

#2) 변수 지정 및 전처리
#1. 모듈 및 함수 불렁괴
import sklearn.model_selection as sel
import sklearn.compose as com
import sklearn.preprocessing as pre

#2. 변수 지정(독립변수, 종속변수)
num=['총매출액','1회 평균매출액','할인권 사용 횟수','총 할인 금액','구매카테고리수','Recency','Frequency','Monetary']
cg = ['구매금액대','고객등급','구매유형','클레임접수여부','거주지역','성별','고객 나이대','할인민감여부']
X = df[df.이탈여부==0][num+cg]
Y = df[df.이탈여부==0]['평균 구매주기']
XY = df[df.이탈여부==0][num+cg,'평균 구매주기']
print(XY.corr())

#3. 데이터 분할 (학습용, 평가용 데이터 세트)
X_train, X_test, Y_train, Y_test = sel.train_test_split(X, Y, test_size=0.3, random_state=0)

#4. 표준화 및 원핫인코딩
ct = com.ColumnTransformer([("scaling", pre.StandardScaler(),num),("onehot", 
   pre.OneHotEncoder(sparse = False), cg)])
ct.fit(X_train)
X_train=ct.transform(X_train)
X_test=ct.transform(X_test)
# print(X_train)

#3) 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
import sklearn.linear_model as lin
#2. 모형 학습
lr = lin.LinearRegression().fit(X_train, Y_train)

#3. 모형 예측
Y_pred = lr.predict(X_test)
print('평가용 데이터 세트에 대한 예측값\n', Y_pred)

#4) 모형 평가
#결정계수 평가
print("학습용 데이터 세트 결정계수: {:.3f}".format(lr.score(X_train, Y_train)))
print("평가용 데이터 세트 결정계수: {:.3f}".format(lr.score(X_test, Y_test)))

#RMSE 평가
#1. 모듈 및 함수 불러오기
from sklearn.metrics import mean_squared_error
from math import sqrt

#2. RMSE 계산
rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print("RMSE: {:.3f}".format(rmse))

#5. 절편 및 가중치 출력
print("절편:", np.round(lr.intercept_,3))
print("가중치", np.round(lr.coef_,3))
