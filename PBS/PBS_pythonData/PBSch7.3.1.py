#1. 모듈 및 데이터 탑재
import pandas as pd
from scipy import stats
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')
df1 = df[['1회_평균매출액','방문빈도']]

#2. 스피어만 상관계수 출력
spear = stats.spearmanr(df1['1회_평균매출액'], df1['방문빈도'])
print(spear)