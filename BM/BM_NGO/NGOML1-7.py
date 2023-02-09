#심층 신경망
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

df = pd.read_csv('1. NGO.csv', encoding='cp949')
df['가입나이'] = round((df['AGE']*12 - df['LONGEVITY_M'])/12,0)
df = df[df['가입나이'] > 0]
df['PLED_FIRST_DAY'] = df['LONGEVITY_D'] - df['PLED_FIRST_LONGEVITY']

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
plt.rc('font', family=font_name)

#전처리 모듈 불러오기
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


#변수지정
df1 = df[['AGE','가입나이','LONGEVITY_D','SEX','PAY_RATE_NOPAY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT','CHURN']]
df1 = df1[df1['가입나이']>0]
df1 = df1[df1['SEX']!=0]
df1=df1.dropna()
#데이터 파악하기
user_mont = df1[df1['CHURN']==0]['PAY_SUM_PAYMENTAMOUNT'].sum()
user_mean = df1[df1['CHURN']==0]['PAY_SUM_PAYMENTAMOUNT'].mean()
user_count = df1[df1['CHURN']==0]['PAY_SUM_PAYMENTAMOUNT'].count()
nonuser_mont = df1[df1['CHURN']==1]['PAY_SUM_PAYMENTAMOUNT'].sum()
nonuser_mean = df1[df1['CHURN']==1]['PAY_SUM_PAYMENTAMOUNT'].mean()
nonuser_count = df1[df1['CHURN']==1]['PAY_SUM_PAYMENTAMOUNT'].count()
all_mont = user_mont + nonuser_mont

#종속변수 로그화시켜서 약간 정규화
# df1['PAY_SUM_PAYMENTAMOUNT'] = np.log1p(df1['PAY_SUM_PAYMENTAMOUNT'])

num = ['AGE','가입나이','LONGEVITY_D','PAY_RATE_NOPAY','PAY_NUM']
cg = ['SEX']

X = df1[df1['CHURN']==0][num+cg]
Y = df1[df1['CHURN']==0]['PAY_SUM_PAYMENTAMOUNT']

#3. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#4. 표준화 및 원핫인코딩
ct = ColumnTransformer([('scaling', StandardScaler(), num), ('onehot',OneHotEncoder(sparse = False, handle_unknown = 'ignore'), cg)])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

scaler=StandardScaler().fit(X_train)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

#모형 학습 및 예측
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import Accuracy
model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_dim=7, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1))

model.compile(loss='mse', optimizer='SGD')
svdf = pd.read_csv('DNN평가.csv', encoding='cp949')
for i in range(5,100,1) : 
    model.fit(X_train, Y_train, epochs=i, batch_size=64, verbose=0)
    train_score = model.evaluate(X_train, Y_train, verbose=0)
    test_score = model.evaluate(X_test, Y_test, verbose=0)
    print('학습용 데이터 세트 MSE epochs=',i,': ', train_score)
    print('평가용 데이터 세트 MSE epochs=',i,': ', test_score)
    newdata = {'epochs':i,'trainMSE':train_score,'testMSE':  test_score}
    svdf = svdf.append(newdata, ignore_index=True)
svdf.to_csv('DNN평가2.csv', encoding='utf-8-sig')