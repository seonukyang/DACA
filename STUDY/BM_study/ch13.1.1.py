#군집분석
#계층적 군집분석
import pandas as pd
from matplotlib import font_manager,rc
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Ashopping.csv',encoding='CP949')
X = df[['1회 평균매출액','평균 구매주기','구매카테고리수','총매출액','방문빈도']]

font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
rc('font',family=font_name)

fig,axs = plt.subplots(1,5,figsize=(20,5))

sns.boxplot(X['1회 평균매출액'], ax=axs[0])
sns.boxplot(X['평균 구매주기'], ax=axs[1])
sns.boxplot(X['구매카테고리수'], ax=axs[2])
sns.boxplot(X['총매출액'], ax=axs[3])
sns.boxplot(X['방문빈도'], ax=axs[4])
plt.show()
plt.clf()

from sklearn.preprocessing import StandardScaler

list =['1회 평균매출액','평균 구매주기','구매카테고리수','총매출액','방문빈도']

for a in list : 
    Q1 = X[a].quantile(0.25)
    Q3 = X[a].quantile(0.75)
    IQR = Q3 = Q1

    outlier_index = X[(X[a]<Q1-1.5*IQR) | (X[a]>Q3+1.5*IQR)].index
    X.drop(outlier_index, inplace=True)

scaler = StandardScaler()
scaler.fit(X)
X_stand = scaler.transform(X)

print(X_stand.shape)

#계층적 군집분석 수행
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(linkage='ward')

Y_agg = agg.fit_predict(X_stand)
print(Y_agg)

#군집 품질 평가
from sklearn.metrics import silhouette_score, calinski_harabasz_score

S_score = silhouette_score(X_stand, Y_agg)
C_score = calinski_harabasz_score(X_stand, Y_agg)

print('실루엣 계수 : ',S_score)
print('CH 점수 : ', C_score)

#덴드로그램 생성
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

h_cluster = linkage(X_stand, method='ward')

plt.figure(figsize=(10,10))
plt.title('Dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
dendrogram(h_cluster)
plt.show()
plt.clf()

