#가우시안 다항 분석
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


#정밀도, 재현율, F1 스코어 평가, 교차검증
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

df = pd.read_csv('NGO다항로지스틱회귀분석.csv', encoding='UTF-8')

del df['Unnamed: 0']
#3. 변수 지정(독립변수, 종속변수)
df_train = df[df['CHURN']==0]
X = df_train[['AGE','SEX','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','가입나이']]
Y = df_train['총납입금액구간']
target = df[df['CHURN']==1]

target_X = target[['AGE','SEX','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','가입나이']]


#4. 데이터 분할(학습용/평가용 데이터 세트)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#교차검증용 다회차 
#2. 오버 샘플링
smote = SMOTE(random_state=0)
X_train, Y_train = smote.fit_sample(X_train, Y_train)


#2. 모형 생성 
model = RandomForestClassifier(random_state=0,n_estimators=324, max_depth=3)

#3. 모형 학습 및 예측
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
target_pred = model.predict(target_X)
#3) 모형 평가
#정확도 평가
print('학습용 데이터 세트 정확도 : ',model.score(X_train, Y_train))
print('평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))
#교차검증 
v_score = cross_val_score(model, X, Y, cv=10)
print('교차검증 평균 : ',v_score.mean())


print(classification_report(Y_test, Y_pred))
results = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
print(results)
target['예측납입금액대'] = ''
target['예측납입금액대'] = target_pred
print('타겟 고객층 인구수 : \n',target.groupby('예측납입금액대').size())
print('타겟 고객층 비율 : \n',target.groupby('예측납입금액대').size()/len(target))
target.to_csv('랜덤포레스트다항분석결과.csv', encoding='utf-8-sig')