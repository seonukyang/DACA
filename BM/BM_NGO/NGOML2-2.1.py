#다항 로지스틱 회귀분석
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#데이터 균형화
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.linear_model import LogisticRegression

#정밀도, 재현율, F1 스코어 평가
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('NGO다항로지스틱회귀분석.csv', encoding='UTF-8')
df_model = {'C':[],'solver':[],'학습용정확도':[],'평가용정확도':[]}
df_model = pd.DataFrame(df_model)

del df['Unnamed: 0']
#3. 변수 지정(독립변수, 종속변수)
df_train = df[df['CHURN']==0]
X = df_train[['AGE','SEX','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','가입나이']]
Y = df_train['총납입금액구간']
target = df[df['CHURN']==1]

target_test = target[['AGE','SEX','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','가입나이']]



k = 0
while k<10 : 
    #4. 데이터 분할(학습용/평가용 데이터 세트)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

    #5. 표준화
    ct = ColumnTransformer([('scaling',StandardScaler(),['AGE','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','가입나이']),
    ('onehot',OneHotEncoder(sparse = False),['SEX'])])
    ct.fit(X_train)
    X_train = ct.transform(X_train)
    X_test = ct.transform(X_test)

    #2. 모형 생성 newton-cg
    for i1 in range(1,100,1) : 
        i1 = i1/10
        model = LogisticRegression(C=i1, solver='newton-cg', multi_class='multinomial')

    #3. 모형 학습 및 예측
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
    #3) 모형 평가
    #정확도 평가
        train_accuracy = model.score(X_train, Y_train)
        test_accuracy = model.score(X_test, Y_test)

        newdata = {'C':i1,'solver':'newton-cg','학습용정확도': train_accuracy, '평가용정확도':test_accuracy}
        df_model = df_model.append(newdata, ignore_index=True)
        print('진행상황 while문 : ',k,'번째   for문 : ',i1,'번째','newton')

#2. 모형 생성 sag
    for i2 in range(1,100,1) : 
        i2 = i2/10
        model = LogisticRegression(C=i2, solver='sag', multi_class='multinomial')

    #3. 모형 학습 및 예측
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
    #3) 모형 평가
    #정확도 평가
        train_accuracy = model.score(X_train, Y_train)
        test_accuracy = model.score(X_test, Y_test)

        newdata = {'C':i2,'solver':'sag','학습용정확도': train_accuracy, '평가용정확도':test_accuracy}
        df_model = df_model.append(newdata, ignore_index=True)
        print('진행상황 while문 : ',k,'번째   for문 : ',i2,'번째','sag')

#2. 모형 생성 saga
    for i3 in range(1,100,1) : 
        i3 = i3/10
        model = LogisticRegression(C=i3, solver='saga', multi_class='multinomial')

    #3. 모형 학습 및 예측
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
    #3) 모형 평가
    #정확도 평가
        train_accuracy = model.score(X_train, Y_train)
        test_accuracy = model.score(X_test, Y_test)

        newdata = {'C':i3,'solver':'saga','학습용정확도': train_accuracy, '평가용정확도':test_accuracy}
        df_model = df_model.append(newdata, ignore_index=True)
        print('진행상황 while문 : ',k,'번째   for문 : ',i3,'번째','saga')

    k = k+1

#모델결과 내보내기
df_model.to_csv('다항로지스틱모델결과.csv', encoding='utf-8-sig')