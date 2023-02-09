#단순 경쟁구조 분석
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import statsmodels.formula.api as smf
df = pd.read_csv('MDS1_1.csv', sep=',', encoding='CP949')
print(df)
df1 = df[['이미지','접근성','서비스','친절성','신속성','정확성','상품다양성']]
df1 = (df1 - df1.min())/(df1.max() - df1.min())
clf = MDS(n_components=2, random_state=123).fit(df1.loc[:, '이미지':'상품다양성'])
X_mds = clf.fit_transform(df1.loc[:, '이미지':'상품다양성'])
print(X_mds)

import matplotlib
import matplotlib.pylab as plt



#속성 차원 좌표 값 계산
df1['차원1'] = X_mds[:,0]
df1['차원2'] = X_mds[:,1]
model = []
model.append(smf.ols(formula = '이미지 ~ 차원1+차원2', data=df1).fit())
model.append(smf.ols(formula = '접근성 ~ 차원1+차원2', data=df1).fit())
model.append(smf.ols(formula = '서비스 ~ 차원1+차원2', data=df1).fit())
model.append(smf.ols(formula = '친절성 ~ 차원1+차원2', data=df1).fit())
model.append(smf.ols(formula = '신속성 ~ 차원1+차원2', data=df1).fit())
model.append(smf.ols(formula = '정확성 ~ 차원1+차원2', data=df1).fit())
model.append(smf.ols(formula = '상품다양성 ~ 차원1+차원2', data=df1).fit())

속성 = []
for i in range(0,7,1):
    속성.append([model[i].params[1], model[i].params[2]])
속성 = np. array(속성)
자극점및속성 = np.hstack([X_mds, 속성])
print(자극점및속성)

import matplotlib
import matplotlib.pylab as plt

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

#속성-경쟁구조 시각화
labels = df.shop
for label, x_count, y_count in zip(labels, 자극점및속성[:,0], 자극점및속성[:,1]) : 
    plt.annotate(label,
                xycoords = 'data',
                textcoords = 'offset points',
                xy = (x_count, y_count),
                xytext=(5,-5))
labels2 = df1.columns

for label, x_count, y_count in zip(labels2, 자극점및속성[:,2], 자극점및속성[:,3]):
    plt.annotate(label,
            xycoords = 'data',
            textcoords = 'offset points',
            xy=(x_count, y_count),
            xytext = (5,-5))
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.scatter(자극점및속성[:,0], 자극점및속성[:,1])
plt.scatter(자극점및속성[:,2], 자극점및속성[:,3])
plt.show()