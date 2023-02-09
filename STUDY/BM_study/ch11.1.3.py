#딥러닝 RNN
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

from keras.datasets import imdb
from collections import Counter

np.random.seed(0)
tf.random.set_seed(0)

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()

plt.hist([len(s) for s in X_train], bins=50)

from keras.preprocessing import sequence

X_train = sequence.pad_sequences(X_train, maxlen=300)
X_test = sequence.pad_sequences(X_test, maxlen=300)

from keras.layers import Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(10000, 128))
model.add(SimpleRNN(128, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=5, batch_size=300)
Y_classes = model.predict_classes(X_test, verbose=0)


#정확도 평가
train_score = model.evaluate(X_train, Y_train, verbose=0)
test_score = model.evaluate(X_test, Y_test, verbose=0)
print('학습용 데이터 세트 정확도 : ', train_score[0], train_score[1])
print('평가용 데이터 세트 정확도 : ', test_score[0], test_score[1])
