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

#정밀도, 재현율, F1 스코어 평가, 교차검증
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

df = pd.read_csv('tripadvisor_hotel_reviews.csv', encoding='UTF-8')

X = df['Review']
Y=df['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

cv = CountVectorizer(binary=True, stop_words='english', min_df=5)
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

model = BernoulliNB()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print('학습용 데이터 세트 정확도 : ',model.score(X_train, Y_train))
print('평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))


#평가 리포트
print(classification_report(Y_test, Y_pred))
