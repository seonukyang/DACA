#DNN 실습
#분류 예측
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE


df= pd.read_csv('Ashopping.csv', encoding='CP949')

X = df[df.이탈여부==0][['고객ID','Recency', 'Frequency', 'Monetary', '총매출액', '방문빈도']]
Y = df[df.이탈여부==0]['구매카테고리수']

X_train1, X_test1, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=111)
X_train = X_train1[['Recency', 'Frequency', 'Monetary', '총매출액', '방문빈도']]
X_test = X_test1[['Recency', 'Frequency', 'Monetary', '총매출액', '방문빈도']]
ct = ColumnTransformer([('scaling', StandardScaler(),['Recency', 'Frequency', 'Monetary', '총매출액', '방문빈도'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import Accuracy

np.random.seed(111)
tf.random.set_seed(111)

model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_dim=5, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1))

model.compile(loss='mse', optimizer='adam')

#모형학습
model.fit(X_train, Y_train, validation_split=0.2, epochs=20, batch_size=64, verbose=0)

Y_pred = np.round(model.predict(X_test, verbose=0),3)
Y_classes = model.predict_classes(X_test, verbose=0)



train_score = model.evaluate(X_train, Y_train, verbose=0)
test_score = model.evaluate(X_test, Y_test, verbose=0)
print('학습용 데이터 세트 MSE : ', train_score)
print('평가용 데이터 세트 MSE : ', test_score)



result = X_test1['고객ID']

for i in range(0,len(Y_pred),1) : 
    print(result.iloc[i],'번째',Y_pred[i])