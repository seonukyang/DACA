#계층적 군집분석
import pandas as pd
from scipy.cluster.hierarchy import linkage
df = pd.read_csv('Ashopping.csv', encoding='CP949')
data_temp = df.sample(n=500, random_state=111)
X = data_temp[['Recency','Frequency','Monetary']]

cluster = linkage(X, method = 'average', metric='euclidean')

from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
plt.title('Dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
dendrogram(cluster)
plt.show()
plt.clf()

