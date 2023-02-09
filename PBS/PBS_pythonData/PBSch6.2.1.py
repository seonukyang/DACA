#1. 모듈 및 데이터 탑재
import pandas as pd
from scipy import stats
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')

#2. 총매출액 평균 및 일표본 t-검정
print('총매출액 평균 : ',df.총_매출액.mean()) #아래거랑 동일
print('총매출액 평균 : ',df['총_매출액'].mean())
print(stats.ttest_1samp(df['총_매출액'], 7700000))