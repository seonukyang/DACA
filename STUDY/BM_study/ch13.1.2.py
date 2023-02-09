#k-mean 군집분석
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
# plt.show()
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

#최적 군집 수 판단
from sklearn.cluster import KMeans
inertia = []

for k in range(1,11) : 
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit_predict(X_stand)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,11), inertia)
plt.title('The Elbow Method', fontsize=15)
plt.xlabel('K', fontsize=13)
plt.ylabel('inertia', fontsize=13)
# plt.show()
plt.clf()

kmeans = KMeans(n_clusters=2, random_state=0)

Y_kmeans = kmeans.fit_predict(X_stand)

#군집 품질 평가
from sklearn.metrics import silhouette_score, calinski_harabasz_score

S_score = silhouette_score(X_stand, Y_kmeans)
C_score = calinski_harabasz_score(X_stand, Y_kmeans)

print('실루엣 계수 : ',S_score)
print('CH 점수 : ', C_score)
X['cluster'] = Y_kmeans


a = pd.Series(X.groupby('cluster')['1회 평균매출액'].mean())
b = pd.Series(X.groupby('cluster')['평균 구매주기'].mean())
c = pd.Series(X.groupby('cluster')['구매카테고리수'].mean())
d = pd.Series(X.groupby('cluster')['총매출액'].mean())
e = pd.Series(X.groupby('cluster')['방문빈도'].mean())

df2 = pd.concat([pd.Series([0,1]),a,b,c,d,e], axis=1)
df2.columns = ['ClusterID', '1회 평균매출액', '평균 구매주기','구매카테고리수','총매출액','방문빈도']

fig, axs = plt.subplots(1,5,figsize=(20,5))
sns.barplot(x=df2.ClusterID, y=df2['1회 평균매출액'], ax=axs[0])
sns.barplot(x=df2.ClusterID, y=df2['평균 구매주기'], ax=axs[1])
sns.barplot(x=df2.ClusterID, y=df2['구매카테고리수'], ax=axs[2])
sns.barplot(x=df2.ClusterID, y=df2['총매출액'], ax=axs[3])
sns.barplot(x=df2.ClusterID, y=df2['방문빈도'], ax=axs[4])
