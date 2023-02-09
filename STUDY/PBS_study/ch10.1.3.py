import pandas as pd
import scikit_posthocs
import numpy as np
from statsmodels.multivariate.manova import MANOVA

df = pd.read_csv('Ashopping.csv', encoding='CP949')
df1 = df[['총_매출액','방문빈도','구매유형','거주지역']]

#다변량분산분석
print(MANOVA.from_formula('방문빈도 + 총_매출액 ~ 구매유형 + 거주지역', data=df1).mv_test())

#사후분석
df1['구매유형'] = df1['구매유형'].astype(str)
df1['거주지역'] = df1['거주지역'].astype(str)
print('총_매출액x구매유형\n',scikit_posthocs.posthoc_scheffe(df1, val_col='총_매출액', group_col='구매유형'),'\n')
print('총_매출액x거주지역\n',scikit_posthocs.posthoc_scheffe(df1, val_col='총_매출액', group_col='거주지역'),'\n')
print('방문빈도x구매유형\n',scikit_posthocs.posthoc_scheffe(df1, val_col='방문빈도', group_col='구매유형'),'\n')
print('방문빈도x구매유형\n',scikit_posthocs.posthoc_scheffe(df1, val_col='방문빈도', group_col='거주지역'),'\n')

#총매출액, 방문빈도에 대한 피벗테이블
평균총매출액 = pd.pivot_table(df1, index='구매유형', columns='거주지역', values = '총_매출액', aggfunc=np.mean)
평균방문빈도 = pd.pivot_table(df1, index='구매유형', columns='거주지역', values = '총_매출액', aggfunc=np.mean)
print('구매유형, 거주지역별 평균총 매출액\n',평균총매출액)
print('구매유형, 거주지역별 평균방문빈도 \n', 평균방문빈도)

import matplotlib
import matplotlib.pyplot as plt

X = np.hstack(평균방문빈도.values[0:4])
Y = np.hstack(평균총매출액.values[0:4])

labels = []
for i in range(1,5,1) : 
    for j in range(1,8,1):
        labels.append(str(i)+str(j))

for label, x_count, y_count in zip(labels, X, Y):
    plt.annotate(label,
            xycoords='data',
            textcoords='offset points',
            xy = (x_count, y_count),
            xytext=(5,5))

plt.title('평균점 산점도')
plt.xlabel('평균 방문빈도')
plt.ylabel('평균 총매출액')
plt.plot(X, Y, 'o')