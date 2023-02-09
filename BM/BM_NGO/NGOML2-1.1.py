#이항 로지스틱 회귀분석
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

#정밀도, 재현율, F1 스코어 평가, 교차검증
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score




df = pd.read_csv('NGO연령대.csv', encoding='UTF-8')
df_model = {'C':[],'학습용정확도':[],'평가용정확도':[]}
df_model = pd.DataFrame(df_model)

del df['Unnamed: 0']
#3. 변수 지정(독립변수, 종속변수)
X = df[['AGE','SEX','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT','가입나이']]
Y = df['CHURN']
target = df[df['CHURN']==0]
target = target[target['가입나이연령대']=='10대']
target = target[target['연령대']=='20대']
print(target.index)
target_index = [ 43, 133, 175,271, 273, 357, 560, 617, 638, 674]
target = target.loc[target_index]
target_test = target[['AGE','SEX','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT','가입나이']]


df = df.drop(target_index)

# k = 0
# while k<10 : 
#     #4. 데이터 분할(학습용/평가용 데이터 세트)
#     X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#     #5. 표준화
#     ct = ColumnTransformer([('scaling',StandardScaler(),['AGE','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT','가입나이']),
#     ('onehot',OneHotEncoder(sparse = False),['SEX'])])
#     ct.fit(X_train)
#     X_train = ct.transform(X_train)
#     X_test = ct.transform(X_test)
#     scaler = StandardScaler()
#     scaler.fit(X_train)
#     X_test = scaler.transform(X_test)
#     X_train = scaler.transform(X_train)

#     #교차검증용 다회차 
#     #2. 오버 샘플링
#     smote = SMOTE(random_state=0)
#     X_train_over, Y_train_over = smote.fit_sample(X_train, Y_train)

#     #2. 모형 생성 
#     for i in range(1,100,1) : 
#         i = i/10
#         model = LogisticRegression(C=i)

#     #3. 모형 학습 및 예측
#         model.fit(X_train_over, Y_train_over)
#         Y_pred = model.predict(X_test)
#     #3) 모형 평가
#     #정확도 평가
#         train_accuracy = model.score(X_train_over, Y_train_over)
#         test_accuracy = model.score(X_test, Y_test)

#         newdata = {'C':i,'학습용정확도': train_accuracy, '평가용정확도':test_accuracy}
#         df_model = df_model.append(newdata, ignore_index=True)
#         print('진행상황 while문 : ',k,'번째   for문 : ',i,'번째')
#     k = k+1

# #모델결과 내보내기
# df_model.to_csv('이항로지스틱모델결과.csv', encoding='utf-8-sig')

#타겟데이터 넣기
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#5. 표준화
ct = ColumnTransformer([('scaling',StandardScaler(),['AGE','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT','가입나이']),
('onehot',OneHotEncoder(sparse = False),['SEX'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)
target_test = ct.transform(target_test)
scaler = StandardScaler()
scaler.fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)
target_test = scaler.transform(target_test)

#2. 오버 샘플링
smote = SMOTE(random_state=0)
X_train_over, Y_train_over = smote.fit_sample(X_train, Y_train)

#2. 모형 생성 
model = LogisticRegression(C=1.7)

#3. 모형 학습 및 예측
model.fit(X_train_over, Y_train_over)
Y_pred = model.predict(X_test)
target_pred = model.predict(target_test)
#3) 모형 평가
print('학습용 데이터 세트 정확도 : ',model.score(X_train_over, Y_train_over))
print('평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))


#교차검증 
v_score = cross_val_score(model, X, Y, cv=10)
print(v_score.mean())




print(classification_report(Y_test, Y_pred))
target['예측이탈'] = ''
target['예측이탈'] = target_pred
# target.to_csv('이항로지스틱분석결과.csv', encoding='utf-8-sig')
results = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
print(results)