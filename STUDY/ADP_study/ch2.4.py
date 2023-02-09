#교차검증
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

iris = load_iris()
features = iris.data
label = iris.target
iris_df = pd.DataFrame(data=features, columns=iris.feature_names)
iris_df['label'] = label
print(iris_df.head())

X_train, X_test, y_train, y_test  = train_test_split(iris.data, label, test_size=0.2, random_state=11)

dt_clf = DecisionTreeClassifier(random_state=156)

kfold = KFold(n_splits=5)
cv_accuracy = []
n_iter = 0
for train_index, test_index in kfold.split(iris_df):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test, pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(n_iter,'번째')
    print('교차검증 정확도 : ', accuracy)
    print('학습 데이터 크기 : ',train_size)
    print('검증 데이터 크기 : ',test_size)
    cv_accuracy.append(accuracy)

print('평균 검증 정확도 : ',np.mean(cv_accuracy))

#GridSearchCV
max_depth = range(1,4,1)
min_sample_split = range(2,4,1)
grid_parameters = {'max_depth':max_depth, 'min_samples_split' : min_sample_split}

X_train, X_test, y_train, y_test  = train_test_split(iris.data, label, test_size=0.2, random_state=121)
dtree = DecisionTreeClassifier()
parameters = grid_parameters = {'max_depth':max_depth, 'min_samples_split' : min_sample_split}

grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)
grid_dtree.fit(X_train,y_train)

scores_df = pd.DataFrame(grid_dtree.cv_results_)
print(scores_df[['params','split0_test_score']].sort_values('split0_test_score',ascending=False))
print('GridSearchCV 최적 파라미더', grid_dtree.best_params_)
print('GridSdarchCV 최고 정확도 : ',grid_dtree.best_score_)

best_grid_dtree = grid_dtree.best_estimator_
pred = best_grid_dtree.predict(X_test)
print('세트 정확도 : ',accuracy_score(y_test, pred))