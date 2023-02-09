#딥러닝 CNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import Accuracy

from keras.datasets import mnist
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
print('학습용 데이터 이미지 수 :', X_train.shape[0])
print('평가용 데이터 이미지 수 :', X_test.shape[0])

from keras.utils import np_utils

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float64') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float64') / 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

from keras.layers import Conv2D, MaxPooling2D, Flatten
np.random.seed(0)
tf.random.set_seed(0)

model = Sequential()
model.add(Conv2D(4, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X_train, Y_train, epochs=10, batch_size=128, verbose=0)
Y_pred = np.round(model.predict(X_test, verbose=0),3)
Y_classes = model.predict_classes(X_test, verbose=0)

print('평가용 데이터 세트에 대한 예측 확률\n',Y_pred[:5])
print('평가용 데이터 세트에 대한 예측 클래스\n',Y_classes[:5])

#정확도 평가
train_score = model.evaluate(X_train, Y_train, verbose=0)
test_score = model.evaluate(X_test, Y_test, verbose=0)
print('학습용 데이터 세트 정확도 : ', train_score[0], train_score[1])
print('평가용 데이터 세트 정확도 : ', test_score[0], test_score[1])
