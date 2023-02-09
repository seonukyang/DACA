#군집분석
#계층적 군집분석
import pandas as pd
from matplotlib import font_manager,rc
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Ashopping.csv',encoding='CP949')
X = df[['평균 구매주기', '구매카테고리수', '총매출액', '방문빈도', 'Recency', 'Frequency']]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_stand = scaler.transform(X)

print(X_stand.shape)

#계층적 군집분석 수행
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')

Y_agg = agg.fit_predict(X_stand)
print(Y_agg)

#군집 품질 평가
from sklearn.metrics import silhouette_score, calinski_harabasz_score

S_score = silhouette_score(X_stand, Y_agg)
C_score = calinski_harabasz_score(X_stand, Y_agg)

print('실루엣 계수 : ',S_score)
print('CH 점수 : ', C_score)
import collections

print(collections.Counter(Y_agg)[1])
