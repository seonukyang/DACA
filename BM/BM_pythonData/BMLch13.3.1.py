#군집분석 - 계층적 군집분석
#1) 변수 지정 및 전처리
#상자 그림 시각화
#1. 모듈 및 함수 불러오기
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

#4. 5개의 서브 플롯 생성
fig, axs = plt.subplots(2,5,figsize=(20,5))

#5. 상자 그림 생성
sns.boxplot(X['1회 평균매출액'], ax = axs[0,0])
sns.boxplot(X['평균 구매주기'], ax = axs[0,1])
sns.boxplot(X['구매카테고리수'], ax = axs[0,2])
sns.boxplot(X['총매출액'], ax = axs[0,3])
sns.boxplot(X['방문빈도'], ax = axs[0,4])
#plt.show()

print(X.describe())

#데이터 전처리
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

#6. 결과 출력
print(X_stand.shape)

#5. 상자 그림 재확인
sns.boxplot(X['1회 평균매출액'], ax = axs[1,0])
sns.boxplot(X['평균 구매주기'], ax = axs[1,1])
sns.boxplot(X['구매카테고리수'], ax = axs[1,2])
sns.boxplot(X['총매출액'], ax = axs[1,3])
sns.boxplot(X['방문빈도'], ax = axs[1,4])
# plt.show()

#2) 계층적 군집분석 수행
#1. 모듈 및 함수 불러오기
from sklearn.cluster import AgglomerativeClustering

#2. 군집분석 모형 생성
agg = AgglomerativeClustering(linkage='ward')

#3. 군집분석 수행
Y_agg = agg.fit_predict(X_stand)
# print(Y_agg)

#3) 군집 품질 평가
#1. 모듈 및 함수 불러오기
from sklearn.metrics import silhouette_score, calinski_harabasz_score

#2. 실루엣 계수 및 CH점수 출력
S_score = silhouette_score(X_stand, Y_agg)
C_score = calinski_harabasz_score(X_stand, Y_agg)

print('실루엣 계수 : ', round(S_score,3))
print('CH 점수 : ', round(C_score, 3))

#4) 덴드로그램 생성
#1. 모듈 및 함수 불러오기
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

#2. 군집분석 수행
h_cluster = linkage(X_stand, method = 'ward')

#3. 덴드로그램 출력
plt.clf()
plt.figure(figsize=(10,10))
plt.title('Dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
dendrogram(h_cluster)
plt.show()