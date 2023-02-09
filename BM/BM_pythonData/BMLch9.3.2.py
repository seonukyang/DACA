#나이브 베이즈 - 베르누이 나이브 베이즈

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE


#1) 데이터 살펴보기
#1. 데이터 불러오기
import pandas as pd
data = pd.read_csv('spam.csv', encoding='latin1')
#2. 데이터 확인하기
print(data.head())

data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])

print('메일의 개수 : ', len(data))
print('레이블 값 분포 :', pd.Series(data['v1'].value_counts()))

#2) 데이터 전처리
#1. 모듈 및 함수 불러오기
from sklearn.feature_extraction.text import CountVectorizer

#2. 변수 지정
X = data['v2']
Y = data['v1']

#3. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#4. 문서 단어 행렬 DTM로 변환
cv = CountVectorizer(binary=True, stop_words='english', min_df=3)
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

#5. 변환 결과 출력
print('단어별 인덱스 부여 결과 : \n', cv.vocabulary_)
print('')
print('문서 단어 행렬 변환 결과 : \n', X_train.toarray())

#3) 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
from sklearn.naive_bayes import BernoulliNB

#2. 모형 학습 및 예측
model = BernoulliNB()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print('평가용 데이터 세트에 대한 예측값 : ',Y_pred)

#4) 모형 평가
#정확도 평가
print('학습용 데이터 세트 정확도 : ',model.score(X_train, Y_train))
print('평가용 데이터 세트 정확도 : ',model.score(X_test, Y_test))

#-정밀도, 재현율, F1 스코어 평가
#1. 모듈 및 함수 불러오기
from sklearn.metrics import classification_report

#2. 재현율, 정밀도, F1 스코어 평가
print(classification_report(Y_test, Y_pred))
