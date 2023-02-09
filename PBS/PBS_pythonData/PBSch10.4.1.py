#분산분석 - 다변량분석 MANOVA
#1. 모듈 및 데이터 탑재
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')
df1 = df[['총_매출액','방문빈도','구매유형','거주지역']]
pd.options.display.float_format = '{:.3f}'.format

#2. 다변량분산분석
print(MANOVA.from_formula('방문빈도 + 총_매출액 ~ 구매유형 + 거주지역', data=df1).mv_test()) #.종속 +종속 ~ 독립+독립
# mv_test()는 앞서 설정한 다변량분산분석 가설 검정을 수행해 통계량을 산출하는 역할

#3. 사후분석
import scikit_posthocs
import numpy as np
df1['구매유형']=df1['구매유형'].astype(str)
df1['거주지역']=df1['거주지역'].astype(str)
print('구매유형 총매출액 사후분석 \n', scikit_posthocs.posthoc_scheffe(df1, val_col = '총_매출액', group_col='구매유형'))
print('구매유형 방문빈도 사후분석 \n', scikit_posthocs.posthoc_scheffe(df1, val_col = '방문빈도', group_col='구매유형'))
print('거주지역 총매출액 사후분석 \n', scikit_posthocs.posthoc_scheffe(df1, val_col = '총_매출액', group_col='거주지역'))
print('거주지역 방문빈도 사후분석 \n', scikit_posthocs.posthoc_scheffe(df1, val_col = '방문빈도', group_col='거주지역'))

#4. 구매유형, 거주지역별 평균 총매출액, 구매유형, 거주지역별 평균 방문빈도
평균총매출액 = pd.pivot_table(df1, index='구매유형', columns='거주지역', values='총_매출액', aggfunc=np.mean)
print('구매유형, 거주지역 별 평균총매출액', 평균총매출액)
평균방문빈도 = pd.pivot_table(df1, index='구매유형', columns='거주지역', values='방문빈도', aggfunc=np.mean)
print('구매유형, 거주지역 별 평균방문빈도', 평균방문빈도)

#5 산점도 그리기
import matplotlib
import matplotlib.pyplot as plt

X = np.hstack(평균방문빈도.values[0:4])
Y = np.hstack(평균총매출액.values[0:4])

#6. 한글 깨짐 방지
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

#7. 평균점 산점도 그리기
labels= []
for i in range(1,5,1) :
    for j in range(1,8,1):
        labels.append(str(i)+str(j)) #라벨에 11부터 47까지 번호 매겨주기

for label, x_count, y_count in zip(labels, X, Y) :
    plt.annotate(label, 
                xycoords='data',
                textcoords='offset points',
                xy=(x_count, y_count),
                xytext=(5,-5))
# annotate 좌표값에 텍스트를 찍는다. (텍스트, xycoords : x,y축 좌표체계, textcoords 주석이 찍히는 좌표체계, xy 좌표값, xytext 텍스트 찍히는 위치)

plt.title('평균점 산점도')
plt.xlabel('평균 방문빈도')
plt.ylabel('평균 총매출액')
plt.plot(X,Y,'o')