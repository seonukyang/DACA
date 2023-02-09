#그리드 서치
#1 모듈과 함수 불러오기
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

df = pd.read_csv('Ashopping.csv', encoding='cp949')




#1. 모듈 및 함수 불러오기
from sklearn.model_selection import GridSearchCV

#2. 변수 지정
X = df[['총매출액','거래기간','방문빈도']]
Y = df['이탈여부']

#3. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#4. 검색 대상 인자 생성
mylist = list(range(1,50))
k_list = [x for x in mylist if x %2 != 0]
parameter_grid = {'n_neighbors':k_list}

#5. 그리드 서치 수행
grid_search = GridSearchCV(KNeighborsClassifier(), parameter_grid, cv=10)
grid_search.fit(X_train, Y_train)
print('최적의 인자 : ', grid_search.best_params_)