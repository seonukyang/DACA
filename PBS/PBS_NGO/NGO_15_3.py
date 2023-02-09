#NGO - 다변량분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
import scikit_posthocs
from statsmodels.multivariate.manova import MANOVA
df =  pd.read_csv('15.df_1.csv', sep=',', encoding='UTF-8')
print(df)
df = df[df.columns.difference(['Unnamed: 0'])]

df1 = df[['고객등급','주요채널','PAY_SUM_PAYMENTAMOUNT','PAY_NUM']]
#등분산 검정
고객등급1 = []
for i in range(1,6,1):
    고객등급1.append(df1[df1.고객등급==i].PAY_SUM_PAYMENTAMOUNT)
print('고객등급x총금액',sp.stats.levene(고객등급1[0],고객등급1[1],고객등급1[2],고객등급1[3],고객등급1[4]))

고객등급2 = []
for i in range(1,6,1):
    고객등급2.append(df1[df1.고객등급==i].PAY_NUM)
print('고객등급x납입횟수',sp.stats.levene(고객등급2[0],고객등급2[1],고객등급2[2],고객등급2[3],고객등급2[4]))

주요채널1 = []
for i in range(1,8,1):
    주요채널1.append(df1[df1.주요채널==i].PAY_SUM_PAYMENTAMOUNT)
print('주요채널x총금액', sp.stats.levene(주요채널1[0],주요채널1[1],주요채널1[2],주요채널1[3],주요채널1[4],주요채널1[5],주요채널1[6]))

주요채널2 = []
for i in range(1,8,1):
    주요채널2.append(df1[df1.주요채널==i].PAY_SUM_PAYMENTAMOUNT)
print('주요채널x납입횟수', sp.stats.levene(주요채널2[0],주요채널2[1],주요채널2[2],주요채널2[3],주요채널2[4],주요채널2[5],주요채널2[6]))

pd.options.display.float_format = '{:3f}'.format
#2. 다변량 분석
print(MANOVA.from_formula('PAY_SUM_PAYMENTAMOUNT + PAY_NUM ~ 고객등급 + 주요채널', data=df1).mv_test())

#사후분석
import scikit_posthocs

df1['고객등급']=df1['고객등급'].astype(str)
df1['주요채널']=df1['주요채널'].astype(str)
print('고객등급 총납입금액 사후분석 \n', scikit_posthocs.posthoc_scheffe(df1, val_col = 'PAY_SUM_PAYMENTAMOUNT', group_col='고객등급'))
print('고객등급 납입횟수 사후분석 \n', scikit_posthocs.posthoc_scheffe(df1, val_col = 'PAY_NUM', group_col='고객등급'))
print('주요채널 총납입금액 사후분석 \n', scikit_posthocs.posthoc_scheffe(df1, val_col = 'PAY_SUM_PAYMENTAMOUNT', group_col='주요채널'))
print('주요채널 납입횟수 사후분석 \n', scikit_posthocs.posthoc_scheffe(df1, val_col = 'PAY_NUM', group_col='주요채널'))

#4. 고객등급, 주요채널별 평균 총납입금액, 고객등급, 주요채널별 평균 납입횟수
평균총납입금액 = pd.pivot_table(df1, index='고객등급', columns='주요채널', values='PAY_SUM_PAYMENTAMOUNT', aggfunc=np.mean)
print('구매유형, 거주지역 별 평균총납입금액', 평균총납입금액)
평균납입횟수 = pd.pivot_table(df1, index='고객등급', columns='주요채널', values='PAY_NUM', aggfunc=np.mean)
print('구매유형, 거주지역 별 평균납입횟수', 평균납입횟수)

#5 산점도 그리기
import matplotlib
import matplotlib.pyplot as plt

X = np.hstack(평균납입횟수.values[0:5])
Y = np.hstack(평균총납입금액.values[0:5])

#6. 한글 깨짐 방지
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

#7. 평균점 산점도 그리기
labels= []
for i in range(1,6,1) :
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
plt.xlabel('평균 납입횟수')
plt.ylabel('평균 총납입금액')
plt.plot(X,Y,'o')
plt.savefig('15.다변량분석.png',dpi=200,edgecolor='blue', bbox_inches='tight',pad_inches=0.3)