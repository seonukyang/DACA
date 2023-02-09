#1. 모듈 및 데이터 탑재
import pandas as pd
from pingouin import partial_corr
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')
df1 = df[['총_매출액','방문빈도','구매_카테고리_수']]

#2. 편 상관계수 출력
partial_corr(data=df1, x='총_매출액', y='방문빈도', covar='구매_카테고리_수')
print(partial_corr(data=df1, x='총_매출액', y='방문빈도', covar='구매_카테고리_수'))