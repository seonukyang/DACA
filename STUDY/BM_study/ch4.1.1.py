#표준 선형 회귀모형
import pandas as pd
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df= pd.read_csv('Ashopping.csv', encoding='CP949')

font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
plt.rc('font', family=font_name)

plt.title('평균 구매주기 분포')
sns.distplot(df['평균 구매주기'])
plt.show()
plt.clf()

plt.title('로그 변환 후 평균 구매주기')
df['평균 구매주기'] = np.log1p(df['평균 구매주기'])
sns.distplot(df['평균 구매주기'])
plt.show()

#변수 지정 및 전처리
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

num = ['총매출액','1회 평균매출액','할인권 사용 횟수','총 할인 금액','구매카테고리수','Recency','Frequency','Monetary']
cg = ['구매금액대','고객등급','구매유형','클레임접수여부','거주지역','성별','고객 나이대','할인민감여부']
X = df[df.이탈여부==0][num+cg]
Y = df[df.이탈여부==0]['평균 구매주기']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

#표준화 및 원핫인코딩
ct = ColumnTransformer([('scaling', StandardScaler(),num), ('onehot', OneHotEncoder(sparse = False), cg)])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

#도형 학습 및 예측
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, Y_train)

Y_pred = lr.predict(X_test)

#결정계수
print('학습용 데이터 세트 결정 계수 : ', lr.score(X_train, Y_train))
print('평가용 데이터 세트 결정 계수 : ', lr.score(X_test, Y_test))

#RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print('RMSE : ', rmse)

print('절편 : ',lr.intercept_)
print('가중지 : ', lr.coef_)


#릿지 선형 회귀모형
from sklearn.linear_model import Ridge

Rr = Ridge(random_state=0).fit(X_train, Y_train)
Y_pred = Rr.predict(X_test)

#결정계수
print('학습용 데이터 세트 결정 계수 : ', Rr.score(X_train, Y_train))
print('평가용 데이터 세트 결정 계수 : ', Rr.score(X_test, Y_test))

#RMSE
rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print('RMSE : ', rmse)

print('절편 : ',Rr.intercept_)
print('가중지 : ', Rr.coef_)

#라쏘 선형 회귀모형
from sklearn.linear_model import Lasso
mylist = list(range(1,1000))
mylist[0] = mylist[0]/100
k_list = mylist
parameter_grid = {'alpha':k_list}
from sklearn.model_selection import GridSearchCV
# grid_search = GridSearchCV(Lasso(), parameter_grid, cv=10)
# grid_search.fit(X_train, Y_train)
# print('최적의 인자 : ', grid_search.best_params_)

Lr = Lasso(random_state=0, alpha=0.001, max_iter=10000).fit(X_train, Y_train)
Y_pred = Lr.predict(X_test)

#결정계수
print('학습용 데이터 세트 결정 계수 : ', Lr.score(X_train, Y_train))
print('평가용 데이터 세트 결정 계수 : ', Lr.score(X_test, Y_test))

#RMSE
rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print('RMSE : ', rmse)

print('절편 : ',Lr.intercept_)
print('가중지 : ', Lr.coef_)