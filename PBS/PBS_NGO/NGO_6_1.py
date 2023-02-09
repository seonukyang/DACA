#NGO - 일표본 t-검정
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
from pingouin import partial_corr
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#2. 데이터 나누기
df_t = df[['LONGEVITY_D','PAY_NUM','PAY_SUM_PAYMENTAMOUNT']]
df_t = df_t.dropna()

df_t['평균후원금액'] = df_t['PAY_SUM_PAYMENTAMOUNT']/df_t['PAY_NUM']
print(df_t)
df_1y = df_t[df_t['LONGEVITY_D']<=365]
df_2y = df_t[df_t['LONGEVITY_D']<=365*2]
df_2y = df_2y[df_2y['LONGEVITY_D']>365]
print('가입후 1년 이내의 평균후원금액의 평균 : ',df_1y.평균후원금액.mean())
print('가입후 1년 이상 2년 이내의 평균후원금액의 평균 : ',df_2y.평균후원금액.mean())

#3. 평균후원금액의 평균 및 일표본 t-검정
print(stats.ttest_1samp(df_2y['평균후원금액'], df_1y['평균후원금액'].mean()*1.1))