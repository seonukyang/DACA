#분류예측분석 - 선형판별분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')

#2. 종속변수와 독립변수 구분하기
X = df[['방문빈도','1회_평균매출액','거래기간']]
Y = df[['이탈여부']]

#3. 선형판별분석
lda = LDA().fit(X,Y) #.독립,종속

#4. 예측결과
print('판별식 선형계수 : ', np.round(lda.coef_,3))
print('\n판별식 절편 :',np.round(lda.intercept_,3))
print('\n예측 결과 :',pd.DataFrame(lda.predict(X)))
print('\n예측 스코어 : ', pd.DataFrame(lda.predict_proba(X)))
print('\n예측 정확도 :',lda.score(X,Y))

#5. 분류행렬표 출력
cf_m = pd.DataFrame(confusion_matrix(Y, lda.predict(X)))
cf_m.columns = ['예측 0','예측 1']
cf_m.index = ['실제 0','실제 1']
print('\n분류행렬표 : \n',cf_m)