from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
df= pd.read_csv('Ashopping.csv', encoding='CP949')

X=df[['총매출액','거래기간','방문빈도']]
Y=df['이탈여부']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

mylist = list(range(1,50))
k_list = [x for x in mylist if x % 2 !=0]
parameter_grid = {'n_neighbors':k_list}

from sklearn.neighbors import KNeighborsClassifier
grid_search = GridSearchCV(KNeighborsClassifier(), parameter_grid, cv=10)
grid_search.fit(X_train, Y_train)
print('최적의 인자 : ', grid_search.best_params_)