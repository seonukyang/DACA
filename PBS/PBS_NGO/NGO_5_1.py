#NGO - 순서형 변수의 상관관계
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
from pingouin import partial_corr
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#
df1 = df[['AGE','LONGEVITY_D','PLED_FIRST_LONGEVITY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT'
,'PLED_NUM','PLED_RATE_FULFILLED','MOTI_NUM_CHANNEL']]
df1 = df1.dropna()
df1['평균후원금액'] = round(df1['PAY_SUM_PAYMENTAMOUNT']/df1['PAY_NUM'],0)
df1['최초후원까지DAY'] = round(df1['LONGEVITY_D']-df1['PLED_FIRST_LONGEVITY'],0)
print(df1)
#2. 스피어만 상관계수 출력
spear = stats.spearmanr(df1['PLED_FIRST_LONGEVITY'], df1['AGE'])
print(spear)