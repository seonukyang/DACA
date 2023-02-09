#가우시안 이항 분석
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import random

#데이터 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import *

#데이터 균형화
from imblearn.over_sampling import SMOTE
from collections import Counter

#로지스틱
from sklearn.linear_model import LogisticRegression
#의사결정나무
from sklearn.tree import DecisionTreeClassifier
#가우시안 나이브 베이즈
from sklearn.naive_bayes import GaussianNB
#보팅 앙상블
from sklearn.ensemble import VotingClassifier
#랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
#그래디언트 부스팅
from sklearn.ensemble import GradientBoostingClassifier
#나이브베이즈
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

#정밀도, 재현율, F1 스코어 평가, 교차검증
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from collections import Counter


df = pd.read_csv('winemag-data-130k-v2_2.csv', encoding='latin_1')

X = df.description
X = X.dropna()
Y = df.province
Y = Y.dropna()
print(Counter(Y))

#언더샘플링 하고 7:3 나누기
tv = TfidfVectorizer()
X = tv.fit_transform(X)
X,Y = RandomUnderSampler(random_state=0).fit_resample(X,Y)
X_under, X_test, Y_under, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#7:3나누고 언더샘플링
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# tv = TfidfVectorizer()
# X_train = tv.fit_transform(X_train)
# X_test = tv.fit_transform(X_test)
# X_under,Y_under = RandomUnderSampler(random_state=0).fit_resample(X_train,Y_train)

print(Counter(Y_under))
print(X_under.shape)
print(X_test.shape)




model = MultinomialNB()
model.fit(X_under, Y_under)
Y_pred = model.predict(X_test)

print('학습용 데이터 세트 정확도 : ',model.score(X_under, Y_under))
print('평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))


#평가 리포트
from sklearn import metrics
print(classification_report(Y_test, Y_pred))

print(metrics.precision_score(Y_test, Y_pred, average='macro'))

print(metrics.recall_score(Y_test, Y_pred, average='macro'))

print(metrics.f1_score(Y_test, Y_pred, average='macro'))