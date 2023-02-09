from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv('./titanic_train.csv')
print(titanic_df.info())
print(titanic_df.describe())
print(titanic_df.isna().sum())

titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)
print('데이터 세트 null 값 개수 : \n', titanic_df.isna().sum())

colnames = titanic_df.columns
for colname in colnames : 
    print(colname,'의 값 분포 : \n', titanic_df[colname].value_counts())

titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]

print(titanic_df.groupby(['Sex','Survived'])['Survived'].count())
sns.barplot(x='Sex', y='Survived', data=titanic_df)
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)

titanic_df['Age_cat'] = 0
for age in titanic_df['Age'].index :
    if titanic_df['Age'][age] <= -1 : titanic_df.loc[age,'Age_cat']='Unknown'
    elif titanic_df['Age'][age] <=5 : titanic_df.loc[age,'Age_cat']='Baby'
    elif titanic_df['Age'][age] <=12 : titanic_df.loc[age,'Age_cat']='Child'
    elif titanic_df['Age'][age] <=18 : titanic_df.loc[age,'Age_cat']='Teenager'
    elif titanic_df['Age'][age] <=25 : titanic_df.loc[age,'Age_cat']='Student'
    elif titanic_df['Age'][age] <=30 : titanic_df.loc[age,'Age_cat']='Yong Adult'
    elif titanic_df['Age'][age] <=60 : titanic_df.loc[age,'Age_cat']='Adult'
    else : titanic_df.loc[age,'Age_cat'] = 'Elderly'

plt.figure(figsize=(10,6))
group_names = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Elderly']
sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
print(titanic_df['Age_cat'].head())

def encode_features(dataDF):
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])
    return dataDF

titanic_df = encode_features(titanic_df)
print(titanic_df.info())

#null 처리 함수
def fillna(df) : 
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df
drop_columns = ['PassengerId','Name','Ticket']
def drop_features(df, drop_columns) : 
    df.drop(drop_columns, axis=1, inplace=True)
    return df
def format_features(df) : 
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df
def transform_features(df) : 
    df = fillna(df)
    df = drop_features(df, drop_columns)
    df = format_features(df)
    return df


titanic_df = pd.read_csv('titanic_train.csv')

titanic_df=transform_features(titanic_df)
Y = titanic_df['Survived']
X = titanic_df.drop('Survived', axis=1)
X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size=0.2, random_state=11)

print(X)
dt_clf = DecisionTreeClassifier(random_state=11)
rt_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

models = [dt_clf, rt_clf, lr_clf]
for model in models : 
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(model,'의 정확도',accuracy_score(y_test, pred))

max_depth = range(1,11,1)
min_samples_split = range(2,6,1)
min_samples_leaf = range(1,9,1)
cv = 5
parameters = {'max_depth':max_depth,'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}
grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=cv)
grid_dclf.fit(X_train, y_train)
print('최적의 파마미터 : ', grid_dclf.best_params_)
print('학습 데이터 최고 정확도 : ', grid_dclf.best_score_)
best_dclf = grid_dclf.best_estimator_

best_pred = best_dclf.predict(X_test)
best_accuracy = accuracy_score(y_test, best_pred)
print('테스트 데이터의 정확도 : ', best_accuracy)
