#비계층적 군집분석 분석
import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv('Ashopping.csv', encoding='CP949')
X = df[['Recency','Frequency','Monetary']]

model = KMeans(n_clusters = 3, max_iter=20, random_state=19).fit(X)
X['cluster_id'] = model.labels_

clu1 = X[X.cluster_id == 0]
clu2 = X[X.cluster_id == 1]
clu3 = X[X.cluster_id == 2]
print('군집 1의 고객수 : ', clu1.cluster_id.count())
print('군집 2의 고객수 : ', clu2.cluster_id.count())
print('군집 3의 고객수 : ', clu3.cluster_id.count())

#군집별 평균 RFM확인
print('군집1의 RFM평균 \n', clu1.Recency.mean(), clu1.Frequency.mean(), clu1.Monetary.mean())
print('군집2의 RFM평균 \n', clu2.Recency.mean(), clu2.Frequency.mean(), clu2.Monetary.mean())
print('군집3의 RFM평균 \n', clu3.Recency.mean(), clu3.Frequency.mean(), clu3.Monetary.mean())

