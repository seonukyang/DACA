#NGO - 공분산분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
import scikit_posthocs
from statsmodels.multivariate.manova import MANOVA

df =  pd.read_csv('15.df_1.csv', sep=',', encoding='UTF-8')
df = df[df.columns.difference(['Unnamed: 0'])]

df1 = df[['고객등급','PAY_SUM_PAYMENTAMOUNT','PAY_NUM']]
pd.options.display.float_format = '{:3f}'.format

#2. 공분산분석
print('공분산분석 결과\n',pg.ancova(dv='PAY_SUM_PAYMENTAMOUNT', between='고객등급', covar='PAY_NUM', data=df1))

#3. 일원분산분석
print('\n일원분산분석 결과\n', pg.anova(dv='PAY_SUM_PAYMENTAMOUNT', between='고객등급', data = df1))