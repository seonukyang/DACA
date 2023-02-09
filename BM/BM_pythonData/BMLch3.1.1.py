#k-겹 교차검증
#1. 모듈과 함수 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

#2 데이터 불러오기
df = pd.read_csv('Ashopping.csv', encoding='cp949')

#3. 변수 지정
X = df[['총매출액','1회 평균매출액','할인권 사용 횟수']]
Y = df['평균 구매주기']

#4. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state=0)
#testdata의 비율을 정해줌
#5. 모형 생성
model = KNeighborsRegressor() #K-NN 수치 예측 모형을 생성한다.

#6. 5겹 교차검증 수행
scores = cross_val_score(model, X_train, Y_train, cv=5)
print('교차검증 점수:',scores)