#밀도 군집분석 - DBSCAN 
#사전처리
import pandas as pd
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import seaborn as sns

#2.변수 지정
df = pd.read_csv('Ashopping.csv', encoding='cp949')
X = df[['1회 평균매출액','평균 구매주기','구매카테고리수','총매출액','방문빈도']]

#3. 한글 폰트 설정
font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

#2. 변수 리스트 생성
list=['1회 평균매출액','평균 구매주기','구매카테고리수','총매출액','방문빈도']

#3. IQR 범위 산정
for a in list : 
    Q1 = X[a].quantile(0.25)
    Q3 = X[a].quantile(0.75)
    IQR = Q3 - Q1
#4. 이상치 제거
    outlier_index = X[(X[a] < Q1 - 1.5*IQR)|(X[a] > Q3 + 1.5*IQR)].index
    X.drop(outlier_index, inplace=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_stand = scaler.transform(X)

#1) DBSCAN 군집분석
#1. 모듈 및 함수 불러오기
from sklearn.cluster import DBSCAN
import numpy as np
#2. 모형 생성
dbscan = DBSCAN(eps = 0.7, min_samples=10)

#3. 모형 학습 및 예측
Y_dbscan = dbscan.fit_predict(X_stand)
print(Y_dbscan)
Y_dbscanDF = pd.Series(Y_dbscan)
print(Y_dbscanDF.value_counts() )

#2) 군집 품질 평가
from sklearn.metrics import silhouette_score, calinski_harabasz_score
S_score = silhouette_score(X_stand, Y_dbscan)
C_score = calinski_harabasz_score(X_stand, Y_dbscan)
print('실루엣 계수 : ', round(S_score,3))
print('CH 점수 : ', round(C_score, 3))