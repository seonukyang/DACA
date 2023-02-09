#NGO - 수치형 변수의 상관관계 분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#2. 수치형 데이터들만 뽑아낸다.
df1 = df[['AGE','LONGEVITY_D','PLED_FIRST_LONGEVITY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT'
,'PLED_NUM','PLED_RATE_FULFILLED','MOTI_NUM_CHANNEL']]
df1 = df1.dropna()
df1['평균후원금액'] = round(df1['PAY_SUM_PAYMENTAMOUNT']/df1['PAY_NUM'],2)
df1['최초후원까지DAY'] = round(df1['LONGEVITY_D']-df1['PLED_FIRST_LONGEVITY'],0)

correlationMatrix = df1.corr()
correlationMatrix.to_csv('3.수치형상관관계.csv', encoding='utf-8-sig')

def high_correlated_p(x, cutoff):
    index_list = []
    for i in range(0, len(x)):
        for j in range(i+1, len(x)):
            if x.iloc[i][j] >= cutoff:
                index_list.append([x.columns[i], x.columns[j]])
    print(index_list)
def high_correlated_n(x, cutoff):
    index_list = []
    for i in range(0, len(x)):
        for j in range(i+1, len(x)):
            if x.iloc[i][j] <= cutoff:
                index_list.append([x.columns[i], x.columns[j]])
    print(index_list)


# 함수 실행
print(high_correlated_p(correlationMatrix, 0.5))
print(high_correlated_n(correlationMatrix, -0.5))

