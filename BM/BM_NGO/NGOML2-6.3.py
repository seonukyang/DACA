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
df_model = {'n_estimators':[],'max_depth':[],'학습용정확도':[],'평가용정확도':[]}
df_model = pd.DataFrame(df_model)

del df['Unnamed: 0']
#3. 변수 지정(독립변수, 종속변수)
df_train = df[df['CHURN']==0]
X = df_train[['AGE','SEX','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','가입나이']]
Y = df_train['총납입금액구간']
target = df[df['CHURN']==1]

#4. 데이터 분할(학습용/평가용 데이터 세트)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#교차검증용 다회차 
#2. 오버 샘플링
smote = SMOTE(random_state=0)
X_train, Y_train = smote.fit_sample(X_train, Y_train)

#모델 생성

#2. 모형 생성 
for i in range(100,501,1) : 
    for j in range(3,10,1) : 
        model = RandomForestClassifier(n_estimators=i, max_depth=j)
        #3. 모형 학습 및 예측
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        #3) 모형 평가
        #정확도 평가
        train_accuracy = model.score(X_train, Y_train)
        test_accuracy = model.score(X_test, Y_test)

        newdata = {'n_estimators':i,'max_depth':j,'학습용정확도': train_accuracy, '평가용정확도':test_accuracy}
        df_model = df_model.append(newdata, ignore_index=True)
        print('진행상황 for문 : ',i,'-',j,'번째')


#모델결과 내보내기
df_model.to_csv('랜덤포레스트다항모델결과.csv', encoding='utf-8-sig')