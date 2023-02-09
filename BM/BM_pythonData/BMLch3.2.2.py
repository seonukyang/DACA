#데이터 밸런싱
#1 모듈과 함수 불러오기
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

df = pd.read_csv('Ashopping.csv', encoding='cp949')

#2. 변수 지정
X = df.drop(['고객ID','이탈여부'], axis=1) #인덱스와 종속변수를 제거한 모든 변수, axis=1은 열을 삭제
Y = df['이탈여부']

#3. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#1) 언더 샘플링
#1. 모듈 및 함수 불러오기
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

#2 랜덤 언더 샘플링 실행
X_train_under, Y_train_under = \
    RandomUnderSampler(random_state=0).fit_sample(X_train, Y_train)

#3. 결과 출력
print('Oroginal dataset shape %s' % Counter(Y)) #본래 유지7, 이탈3의 비율이었다.
print(Y)
print('sampled dataset shape %s' % Counter(Y_train)) #전체의 70% 샘플이며 유지:이탈 7:3비율이 어느정도 유지된다.
print('Resampled dataset shape %s' % Counter(Y_train_under)) #완전이 유지:이탈을 1:1로 하였다.

#2) 오버 샘플링
#1. 모듈 및 함수 불러오기
from imblearn.over_sampling import SMOTE

#2. SMOTE 샘플링 실행
X_train_over, Y_train_over = SMOTE(random_state=0).fit_sample(X_train, Y_train)

#3. 결과출력
print('Original dataset shape %s' % Counter(Y))
print('sampled dataset shape %s' % Counter(Y_train))
print('Resampled dataset shape %s' % Counter(Y_train_over))
print(Y_train_over)