#K-겹 교차검증
from numpy.core.numeric import cross
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

df= pd.read_csv('Ashopping.csv', encoding='CP949')

X = df[['총매출액','1회 평균매출액','할인권 사용 횟수']]
Y = df['평균 구매주기']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

model = KNeighborsRegressor()

scores = cross_val_score(model, X_train, Y_train, cv=5)
print('교차검수 점수 : ', scores)

#층화 K-겹 교차검증
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

X = df[['총매출액','거래기간','방문빈도']]
Y = df['이탈여부']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

model = KNeighborsClassifier()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_train, Y_train, cv=skf)
print('층화 교차검증 점수 : ',scores)
