#군집분석 - k-means
#1. 모듈 및 데이터 탑재
import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')
X = df[['Recency','Frequency','Monetary']]

#2. 비계층적 군집분석
model = KMeans(n_clusters=3, max_iter=20, random_state=19).fit(X)
X['cluster_id'] = model.labels_ #라벨링하기 위해 labels_로 군집번호를 불러와 X객체의 열 이름을 cluster_id로 정한다.

#3. 군집별 고객 수 확인
clu1 = X[X.cluster_id==0]
clu2 = X[X.cluster_id==1]
clu3 = X[X.cluster_id==2]
print('군집1의 고객수 : ',clu1.cluster_id.count())
print('\n군집2의 고객수 : ',clu2.cluster_id.count())
print('\n군집3의 고객수 : ',clu3.cluster_id.count())

#4. 군집별 평균 RFM 확인
print('군집1의 RFM평균\n',clu1.Recency.mean(), clu1.Frequency.mean(), clu1.Monetary.mean())
print('군집2의 RFM평균\n',clu2.Recency.mean(), clu1.Frequency.mean(), clu1.Monetary.mean())
print('군집3의 RFM평균\n',clu3.Recency.mean(), clu1.Frequency.mean(), clu1.Monetary.mean())