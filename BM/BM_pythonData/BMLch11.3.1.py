#1. 모듈 및 함수 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
#from imblearn.over_sampling import SMOTE


#2. 변수 지정(독립변수/종속변수)
df = pd.read_csv('Ashopping.csv', encoding = 'cp949')
X = df[['총매출액', '구매금액대', '1회 평균매출액','거래기간', '평균 구매주기']]
Y = df['할인민감여부']

#3. 데이터 분할(학습용/평가용 데이터 세트)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#4. 표준화 및 원핫인코딩
ct = ColumnTransformer([('scaling', StandardScaler(), ['총매출액', '1회 평균매출액', '거래기간', 
'평균 구매주기']), ('onehot', OneHotEncoder(sparse = False),['구매금액대'])])
ct.fit(X_train)
X_train=ct.transform(X_train)
X_test=ct.transform(X_test)

#1. 모듈 및 함수 불러오기
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import Accuracy

#2. 시드 값 설정
np.random.seed(0)
tf.random.set_seed(0)

#3. 모형 생성
model = keras.models.Sequential()
model.add(keras.layers.Dense(64, input_dim=7, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

#4. 학습 과정 설정
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#5. 모형 학습
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=64, verbose=2)

#1. 모듈 및 함수 불러오기
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='Malgun Gothic' 
matplotlib.rcParams['axes.unicode_minus']=False
#2. Figure와 Axes 객체 생성
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

#3. 오차 (loss) 그래프 그리기
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='lower right')

#4. 정확도 (accuracy) 그래프 그리기
acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper right')

plt.show()

#1. 모형 재학습
model.fit(X_train, Y_train, epochs=20, batch_size=64, verbose=0)

#2. 모형 예측
Y_pred = np.round(model.predict(X_test, verbose=0),3)
Y_classes = model.predict_classes(X_test, verbose=0)

print('평가용 데이터 세트에 대한 예측 확률\n', Y_pred[:5])
print(' ')
print('평가용 데이터 세트에 대한 예측 클래스\n',Y_classes[:5])

train_score = model.evaluate(X_train, Y_train, verbose=0)
test_score = model.evaluate(X_test, Y_test, verbose=0)
print('학습용 데이터 세트 오차와 정확도: {:.3f}, {:.3f}'.format(train_score[0], train_score[1]))
print('평가용 데이터 세트 오차와 정확도: {:.3f}, {:.3f}'.format(test_score[0], test_score[1]))