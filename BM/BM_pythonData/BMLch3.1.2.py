#분류 문제에 대한 층화 k-겹 교차검증
#1 모듈과 함수 불러오기
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd

df = pd.read_csv('Ashopping.csv', encoding='cp949')

#2. 변수 지정
X = df[['총매출액','거래기간','방문빈도']]
Y = df['이탈여부']

#3. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#4. 모델 생성
model = KNeighborsClassifier()

#5. 층화 5-겹 교차검증 수행
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
#n_splits = 5 5개의 폴드로 데이터 분리, 
# shuffle로 각 범주 클래스를 섞어 전체 데이터 상의 각 클래스 비율이 폴드마다 동일하게 분리 될 수 있게 한다.
#즉 각 폴드마다 클래스의 비율이 동일하다. 이렇게 섞어준 폴드를 cross_val의 cv에 넣어준다.
print(skf)
score = cross_val_score(model, X_train, Y_train, cv=skf)
print('층화 교차검증 점수 :', score)