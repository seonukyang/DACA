#군집분석 기법 간의 품질 비교와 군집 프로파일링

#데이터 전처리
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

#1. 모듈 및 함수 불러오기
from sklearn.preprocessing import StandardScaler

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

#5. 표준화
scaler = StandardScaler()
scaler.fit(X)
X_stand = scaler.transform(X)



# 계층적 군집분석 수행
#1. 모듈 및 함수 불러오기
from sklearn.cluster import AgglomerativeClustering
#2. 군집분석 모형 생성
agg = AgglomerativeClustering(linkage='ward')
#3. 군집분석 수행
Y_agg = agg.fit_predict(X_stand)

#1. 모듈 및 함수 불러오기
from sklearn.metrics import silhouette_score, calinski_harabasz_score
#2. 실루엣 계수 및 CH점수 출력
S_score_agg = silhouette_score(X_stand, Y_agg)
C_score_agg = calinski_harabasz_score(X_stand, Y_agg)

# K-평균 군집분석 수행
#1. 모형 생성
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)

#2. 군집분석 수행
Y_kmeans = kmeans.fit_predict(X_stand)

#3) 군집 품질 평가
S_score_kmeans = silhouette_score(X_stand, Y_kmeans)
C_score_kmeans = calinski_harabasz_score(X_stand, Y_kmeans)

#밀도 군집 분석
from sklearn.cluster import DBSCAN
#2. 모형 생성
dbscan = DBSCAN(eps = 0.7, min_samples=10)

#3. 모형 학습 및 예측
Y_dbscan = dbscan.fit_predict(X_stand)

#2) 군집 품질 평가
S_score_dbscan = silhouette_score(X_stand, Y_dbscan)
C_score_dbscan = calinski_harabasz_score(X_stand, Y_dbscan)
state = {'계층적 군집분석' : [S_score_agg, C_score_agg], 'K-평균 군집분석' : [S_score_kmeans, C_score_kmeans],
'DBSCAN':[S_score_dbscan, C_score_dbscan]}
scoreDF = pd.DataFrame(state)
scoreDF.index_set = ['실루엣 계수', 'CH 점수']
print(scoreDF)

#2)  군집별 고객 프로파일링
#1. 군집별 각 칼럼의 평균값을 시리즈 형태로 저장
X['cluster'] = Y_kmeans

a = pd.Series(X.groupby('cluster')['1회 평균매출액'].mean())
b = pd.Series(X.groupby('cluster')['평균 구매주기'].mean())
c = pd.Series(X.groupby('cluster')['구매카테고리수'].mean())
d = pd.Series(X.groupby('cluster')['총매출액'].mean())
e = pd.Series(X.groupby('cluster')['방문빈도'].mean())

#2. 군집별 요약 정보를 갖는 데이터 프레임 생성
df2 = pd.concat([pd.Series([0,1]),a,b,c,d,e,], axis=1)
df2.columns = ["ClusterID", "1회 평균매출액", "평균 구매주기", "구매카테고리수",'총매출액', '방문빈도']
print(df2.head())

#막대그래프
fig, axs = plt.subplots(1,5, figsize = (50,20))
sns.barplot(x=df2.ClusterID, y=df2['1회 평균매출액'], ax = axs[0])
sns.barplot(x=df2.ClusterID, y=df2['평균 구매주기'], ax = axs[1])
sns.barplot(x=df2.ClusterID, y=df2['구매카테고리수'], ax = axs[2])
sns.barplot(x=df2.ClusterID, y=df2['총매출액'], ax = axs[3])
sns.barplot(x=df2.ClusterID, y=df2['방문빈도'], ax = axs[4])
