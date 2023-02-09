import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import Accuracy
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.datasets import imdb
from collections import Counter

df = pd.read_csv('tripadvisor_hotel_reviews.csv', encoding='UTF-8')
X = df['Review']
Y=df['target']
#시드값 설정
np.random.seed(0)
tf.random.set_seed(0)
X2 = X
for k in range(0,len(X),1) : 
    vocab = sorted(set(X[k]))
    char2idx = {u: i for i, u in enumerate(X[k])}
    idx2char = np.array(vocab)
    X2[k] = np.array([char2idx[c] for c in X[k]])
    # print(X2[k])
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y, test_size=0.3, random_state=0)

print('학습용 데이터 개수',X_train.shape)
print('평가용 데이터 개수',X_test.shape)
print('클래스 빈도수',Counter(Y_train))

plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
# plt.show()


from keras.preprocessing import sequence

X_train = sequence.pad_sequences(X_train, maxlen = 1500)
X_test = sequence.pad_sequences(X_test, maxlen = 1500)
print(X_train.shape)
print(X_test.shape)

from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(15000, 128))
model.add(SimpleRNN(128, activation='tanh'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

history = model.fit(X_train, Y_train, epochs=5, batch_size=300)
Y_classes = model.predict_classes(X_test, verbose=0)
print('평가용 데이터 세트에 대힌 예측 클래스', Y_classes[:5])

train_score_cnn = model.evaluate(X_train, Y_train, verbose=0)
test_score_cnn = model.evaluate(X_test, Y_test, verbose=0)
print('rnn학습용 데이터 세트 오차와 정확도',train_score_cnn[0], train_score_cnn[1])
print('rnn평가용 데이터 세트 오차와 정확도',test_score_cnn[0], test_score_cnn[1])