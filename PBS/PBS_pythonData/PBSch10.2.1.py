#분산분석 - 일원분산분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import scipy as sp
import numpy as np
import pingouin as pg
import scikit_posthocs
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')
df1=df[['구매유형','총_매출액']]
pd.options.display.float_format = '{:.3f}'.format #수치형 데이터를 소수점 3자리 까지 출력하겠다.

#2. 등분산 검정
구매유형 = []
for i in range(1,5,1):
    print(i)
    구매유형.append(df1[df1.구매유형==i].총_매출액)
    print(i,'번째 구매유형의 총 매출액', 구매유형)

lev = sp.stats.levene(구매유형[0], 구매유형[1], 구매유형[2], 구매유형[3])
print('lev',lev)

#3. Welch 일원분산분석 등분산이 아닐 때 사용한다.
print(pg.welch_anova(dv='총_매출액', between='구매유형',data=df1)) #dv=종속변수, between=독립변수

#4. 사후분석
df1['구매유형'] = df1['구매유형'].astype(str) #범주형 자료이므로 문자열로 행태 변경
print(scikit_posthocs.posthoc_scheffe(df1, val_col='총_매출액', group_col='구매유형'))
#행렬로 독립변수의 성분 끼리의 사후분석을 한다. 이렇게 나온 값들은 유의확률이라서 값이 작다면 
# 1과2그룹간에는 유의한 차이가 있다 정도로 해석될 수 있다.
#각 집단의 크기가 달라서 scheffe를 이용. data, val_col=종속변수, group_cel=구매유형

#5. 구매유형별 평균 총매출액
print('구매유형[0]',구매유형[0].mean())
print('구매유형[1]',구매유형[0].mean())
print('구매유형[2]',구매유형[0].mean())
print('구매유형[3]',구매유형[0].mean())
