#NGO - 수치형 변수의 상관관계 분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#2. 수치형 데이터들만 뽑아낸다.
df1 = df[['LONGEVITY_M','PAY_SUM_PAYMENTAMOUNT']]
df1 = df1.dropna()

correlationMatrix = df1.corr()
print(correlationMatrix)
