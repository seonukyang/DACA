#분산분석 - 이원분산분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import scipy as sp
import numpy as np
import pingouin as pg
import scikit_posthocs
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')
df1 = df[['총_매출액','구매유형','거주지역']]
pd.options.display.float_format = '{:.3f}'.format

#2. 이원분산분석
print(pg.anova(dv='총_매출액',between=['구매유형','거주지역'],data=df1))

#3. 사후분석
df1['구매유형']=df1['구매유형'].astype(str)
df1['거주지역']=df1['거주지역'].astype(str)
print(scikit_posthocs.posthoc_scheffe(df1, val_col='총_매출액',group_col='구매유형'))
print(scikit_posthocs.posthoc_scheffe(df1, val_col='총_매출액',group_col='거주지역'))
#각 독립변수 그룹 마다 사후분석을 해준다.
#사후분석은 유의 확률이며 0.05보다 작다면 각 그룹간의 종속변수 차이가 유의하게 나타난 것이다.

#4. 구매유형, 거주지역별 평균 총 매출액
print(pd.pivot_table(df, index='구매유형', columns='거주지역', values='총_매출액', aggfunc=np.mean))