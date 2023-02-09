#가우시안 이항 분석
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
#덴드로그램
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

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
#그래디언트 부스팅
from sklearn.ensemble import GradientBoostingClassifier
#나이브베이즈
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
#군집분석
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

#정밀도, 재현율, F1 스코어 평가, 교차검증
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

from PIL import Image
Y_train = pd.read_csv('archive\csvTrainLabel 13440x1.csv', encoding='UTF-8')
Y_train = Y_train[:4000]
Y_test = pd.read_csv('archive\csvTestLabel 3360x1.csv', encoding='UTF-8')
Y_test = Y_train[:1000]

import cv2
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import Accuracy

from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, Flatten
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)


train_files = listdir('archive\Train Images 13440x32x32')
test_files = listdir('archive\Test Images 3360x32x32')
train_path = 'archive\Train Images 13440x32x32/'
test_path = 'archive\Test Images 3360x32x32/'
X_train = []
X_test = []
for i in range(0,4000,1) : 
    path = train_path+train_files[i]
    im = Image.open(path)
    X_train.insert(i,im)
    print('X_train',i)


for j in range(0,1000,1) : 
    path = test_path+test_files[j]
    im = Image.open(path)
    X_train.insert(j,im)
    print('X_test',j)

# X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float64')/255
# X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float64')/255



np.random.seed(0)
tf.random.set_seed(0)

model_cnn = Sequential()
model_cnn.add(Conv2D(4,(3,3), input_shape=(28,28,1), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2,2)))
model_cnn.add(Flatten())
model_cnn.add(Dense(10,activation='softmax'))



model_cnn.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model_cnn.fit(X_train, Y_train, epochs=10, batch_size=128, verbose=0)
Y_pred_cnn = np.round(model_cnn.predict(X_test, verbose=0),3)
Y_classes_cnn = model_cnn.predict_classes(X_test, verbose=0)

train_score_cnn = model_cnn.evaluate(X_train, Y_train, verbose=0)
test_score_cnn = model_cnn.evaluate(X_test, Y_test, verbose=0)
print('cnn평가용데이터에대한 예측 클래스',Y_classes_cnn)
print('cnn학습용 데이터 세트 오차와 정확도',train_score_cnn[0], train_score_cnn[1])
print('cnn평가용 데이터 세트 오차와 정확도',test_score_cnn[0], test_score_cnn[1])


model_rnn = Sequential()
model_rnn.add(Enbedding(10000,128))
model_rnn.add(SimpoleRNN(128,activation='tanh'))
model.add(Dense(1,activation='sigmoid'))
model_rnn.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model_rnn.fit(X_train, Y_train, epochs=5, batch_size=300)

Y_classes_rnn = model_rnn.predict_classes(X_test, verbose=0)
print('rnn평가용데이터에대한 예측 클래스',Y_classes_rnn)
train_score_rnn = model_rnn.evaluate(X_train, Y_train, verbose=0)
test_score_rnn = model_rnn.evaluate(X_test, Y_test, verbose=0)
print('rnn학습용 데이터 세트 오차와 정확도',train_score_rnn[0], train_score_rnn[1])
print('rnn평가용 데이터 세트 오차와 정확도',test_score_rnn[0], test_score_rnn[1])