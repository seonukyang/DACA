#이상점-경쟁구조 분석 과정
import pandas as pd
from sklearn.manifold import MDS
df1 = pd.read_csv('MDS1.csv', sep=',', encoding='CP949')
df3 = pd.read_csv('MDS3.csv', sep=',', encoding='CP949')

#다차원척도법 분석
clf = MDS(n_components=2, random_state=123)
X_mds1 = clf.fit_transform(df1.loc[:, '이미지':'가격만족도'])
X_mds3 = clf.fit_transform(df3.loc[:,'A쇼핑':'G쇼핑'])

import matplotlib
import matplotlib.pylab as plt

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

labels = df1.shop
for label, x_count, y_count in zip(labels, X_mds1[:,0], X_mds1[:,1]) : 
    plt.annotate(label,
                    xycoords='data',
                    textcoords='offset points',
                    xy=(x_count, y_count),
                    xytext=(5,-5))
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.scatter(X_mds1[:,0], X_mds1[:,1])
plt.scatter(X_mds3[:,0], X_mds3[:,1], label='이상점')
plt.scatter(X_mds3[:,0].mean(), X_mds3[:,1].mean(), label='평균 이상점')
plt.legend(loc='upper right')
plt.show()