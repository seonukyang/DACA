#단순 경쟁구조 분석
import pandas as pd
from sklearn.manifold import MDS
df = pd.read_csv('MDS1.csv', sep=',', encoding='CP949')

clf = MDS(n_components=2, random_state=123).fit(df.loc[:, '이미지':'가격만족도'])
X_mds = clf.fit_transform(df.loc[:,'이미지':'가격만족도'])
print(X_mds)

import matplotlib
import matplotlib.pylab as plt

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

labels = df.shop
for label, x_count, y_count in zip(labels, X_mds[:,0], X_mds[:,1]):
    plt.annotate(label,
            xycoords='data',
            textcoords='offset points',
            xy=(x_count, y_count),
            xytext=(5,-5))
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.scatter(X_mds[:,0], X_mds[:,1])
plt.show()
