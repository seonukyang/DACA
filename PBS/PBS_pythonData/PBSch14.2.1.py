#군집분석 - 계층적 군집분석
#1. 모듈 및 데이터 탑재
import pandas as pd
from scipy.cluster.hierarchy import linkage
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')
data_temp = df.sample(n=500, random_state=111)
X = data_temp[['Recency','Frequency','Monetary']]

#2. 계층적 군집분석
cluster=linkage(X, method = 'average', metric='euclidean') #data, 군집화 방법, 유사성 측정 방법
print(cluster[490:]) #1~500까지 데이터들을 군집으로 분류하는 과정을 보여준다. 군집의 개수는 결국 사용자의 판단

#3. 덴드로그램 그리기
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

plt.title('Dendrogram')
plt.xlabel('index')
plt.xlabel('distance')
dendrogram(cluster)
plt.show()