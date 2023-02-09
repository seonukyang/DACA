#DNN 실습
#분류 예측
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE


df= pd.read_csv('Ashopping.csv', encoding='CP949')


X = df[['총매출액','구매금액대','1회 평균매출액','거래기간','평균 구매주기']]
Y = df['할인민감여부']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

ct = ColumnTransformer([('scaling', StandardScaler(),['총매출액','1회 평균매출액','거래기간','평균 구매주기']), 
('onehot', OneHotEncoder(sparse = False), ['구매금액대'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import Accuracy

np.random.seed(0)
tf.random.set_seed(0)

model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_dim=7, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

#모형학습
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=64, verbose=2)

#시각화
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='lower right')

acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper right')
plt.show()

#모형 재학습
model.fit(X_train, Y_train, epochs=20, batch_size=64, verbose=0)
Y_pred = np.round(model.predict(X_test, verbose=0),3)
Y_classes = model.predict_classes(X_test, verbose=0)


#정확도 평가
train_score = model.evaluate(X_train, Y_train, verbose=0)
test_score = model.evaluate(X_test, Y_test, verbose=0)
print('학습용 데이터 세트 정확도 : ', train_score[0], train_score[1])
print('평가용 데이터 세트 정확도 : ', test_score[0], test_score[1])

#DNN 수치예측
X = df[df.이탈여부==0][['방문빈도','총 할인 금액','구매카테고리수','거래기간']]
Y = np.log1p(df[df.이탈여부==0]['1회 평균매출액'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

np.random.seed(0)
tf.random.set_seed(0)

model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_dim=4, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1))

model.compile(loss='mse', optimizer='SGD')

model.fit(X_train, Y_train, epochs=20, batch_size=64, verbose=0)
Y_pred = np.round(model.predict(X_test[:5], verbose=0),3)


train_score = model.evaluate(X_train, Y_train, verbose=0)
test_score = model.evaluate(X_test, Y_test, verbose=0)
print('학습용 데이터 세트 MSE : ', train_score)
print('평가용 데이터 세트 MSE : ', test_score)




