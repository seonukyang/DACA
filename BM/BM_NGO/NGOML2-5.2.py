#가우시안 이항분석
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

#정밀도, 재현율, F1 스코어 평가, 교차검증
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


df = pd.read_csv('NGO연령대.csv', encoding='UTF-8')
df_model = {'max_depth':[],'criterion':[],'학습용정확도':[],'평가용정확도':[]}
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

#타겟데이터 넣기
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#5. 표준화
ct = ColumnTransformer([('scaling',StandardScaler(),['AGE','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT','가입나이']),
('onehot',OneHotEncoder(sparse = False),['SEX'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)
target_test = ct.transform(target_test)


#2. 오버 샘플링
smote = SMOTE(random_state=0)
X_train, Y_train = smote.fit_sample(X_train, Y_train)

#2. 모형 생성 
model = model1 = GaussianNB(priors=[0.9,0.1])
logi = LogisticRegression(C=1.7)
dtree = DecisionTreeClassifier(random_state=0, max_depth = 7, criterion='entropy')
gaus = GaussianNB()
model = VotingClassifier(estimators=[('Logi',logi),('Dtree',dtree),('Gaus',gaus)], voting='soft')


#3. 모형 학습 및 예측
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
target_pred = model.predict(target_test)
#3) 모형 평가
print('학습용 데이터 세트 정확도 : ',model.score(X_train, Y_train))
print('평가용 데이터 세트 정확도 : ', model.score(X_test, Y_test))

#교차검증 
v_score = cross_val_score(model, X, Y, cv=10)
print('교차검증 평균 : ',v_score.mean())

#평가 리포트
print(classification_report(Y_test, Y_pred))
target['예측이탈'] = ''
target['예측이탈'] = target_pred

results = confusion_matrix(y_true=Y_test, y_pred=Y_pred)
print(results)

#파일 저장
target.to_csv('보팅앙상블이항분석결과.csv', encoding='utf-8-sig')