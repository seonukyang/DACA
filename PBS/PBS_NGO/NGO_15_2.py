#NGO - 다변량분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
import scikit_posthocs
from statsmodels.multivariate.manova import MANOVA

df =  pd.read_csv('15.df_1.csv', sep=',', encoding='UTF-8')
df = df[df.columns.difference(['Unnamed: 0'])]

df1 = df[['고객등급','주요채널','PAY_SUM_PAYMENTAMOUNT','PAY_NUM']]
#등분산 검정 15_3으로
pd.options.display.float_format = '{:3f}'.format
#다변량분산분석 15_3으로

#사후분석 15_3으로