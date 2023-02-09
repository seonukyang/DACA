import pandas as pd
import scipy as sp
import numpy as np
import pingouin as pg
import scikit_posthocs
df = pd.read_csv('Ashopping.csv', encoding='CP949')
df1 = df[['구매유형','총_매출액']]

#등분산 검정
구매유형 = []
for i in range(1,5,1):
    구매유형.append(df1[df.구매유형==i].총_매출액)
print(sp.stats.levene(구매유형[0],구매유형[1],구매유형[2],구매유형[3]))

#Welch 일원분삭분석
print(pg.welch_anova(dv='총_매출액', between='구매유형',data=df1))

#사후분석
df1['구매유형'] = df1['구매유형'].astype(str)
print(scikit_posthocs.posthoc_scheffe(df1, val_col='총_매출액', group_col='구매유형'))
print('구매유형1 : ',구매유형[0].mean())
print('구매유형2 : ',구매유형[1].mean())
print('구매유형3 : ',구매유형[2].mean())
print('구매유형4 : ',구매유형[3].mean())
