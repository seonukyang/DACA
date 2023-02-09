import pandas as pd
import scipy as sp
import numpy as np
import pingouin as pg
import scikit_posthocs
df = pd.read_csv('Ashopping.csv', encoding='CP949')
df1 = df[['구매유형','총_매출액','거주지역']]

#이원분산분석
print(pg.anova(dv='총_매출액', between=['구매유형','거주지역'], data=df1))

#사후분석
df1['구매유형'] = df1['구매유형'].astype(str)
df1['거주지역'] = df1['거주지역'].astype(str)
print(scikit_posthocs.posthoc_scheffe(df1, val_col='총_매출액', group_col='구매유형'))
print(scikit_posthocs.posthoc_scheffe(df1, val_col='총_매출액', group_col='거주지역'))

#구매유형, 거주지역별 평균 총 매출액
pd.pivot_table(df1, index='구매유형', columns='거주지역', values='총_매출액', aggfunc=np.mean)
