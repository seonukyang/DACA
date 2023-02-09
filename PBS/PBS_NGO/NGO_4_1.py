#NGO - 편상관 상관관계 분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
from pingouin import partial_corr
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#2. 수치형 데이터들만 뽑아낸다.
df1 = df[['AGE','LONGEVITY_D','PLED_FIRST_LONGEVITY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT'
,'PLED_NUM','PLED_RATE_FULFILLED','MOTI_NUM_CHANNEL']]
df1 = df1.dropna()
df1['평균후원금액'] = round(df1['PAY_SUM_PAYMENTAMOUNT']/df1['PAY_NUM'],2)
df1['최초후원까지DAY'] = round(df1['LONGEVITY_D']-df1['PLED_FIRST_LONGEVITY'],0)
print('1)가입일수x첫플릿지 기준 고객 기간, 최초후원까지DAY : ')
print(partial_corr(data=df1, x='LONGEVITY_D', y='PLED_FIRST_LONGEVITY', covar='최초후원까지DAY'),'\n')
print('2)가입일수x최초후원까지DAY, 첫플릿지 기준 고객 기간 : ')
print(partial_corr(data=df1, x='LONGEVITY_D', y='최초후원까지DAY', covar='PLED_FIRST_LONGEVITY'),'\n')
print('3)납입횟수x납입총후원금액, 평균후원금액 : ')
print(partial_corr(data=df1, x='PAY_NUM', y='PAY_SUM_PAYMENTAMOUNT', covar='평균후원금액'),'\n')
print('4)플릿지횟수x개발채널수, 납입횟수 : ')
print(partial_corr(data=df1, x='PLED_NUM', y='MOTI_NUM_CHANNEL', covar='PAY_NUM'),'\n')