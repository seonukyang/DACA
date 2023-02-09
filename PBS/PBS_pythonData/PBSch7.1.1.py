#1. 모듈 및 데이터 탑재
import pandas as pd
from scipy import stats
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')
df1 = df[['총_매출액','방문빈도','구매_카테고리_수']]

#2-1. 피어슨 상관계수 출력 - 수치형 변수의 상관관계 분석 과정
print('총_매출액x방문빈도 : ',stats.pearsonr(df1.총_매출액, df1.방문빈도))
print('총_매출액x구매_카테고리_수 : ',stats.pearsonr(df1.총_매출액, df1.구매_카테고리_수))
print('방문빈도x구매_카테고리_수 : ',stats.pearsonr(df1.방문빈도, df1.구매_카테고리_수))
#결과값 총_매출액x방문빈도 :  (0.6311706453193395, 3.051960449687332e-112) (상관관계, p-value)


#2-2. 피어슨 상관계수 출력 - 상관관계 테이블 출력
df1.corr(method = 'pearson')
print(df1.corr(method = 'pearson'))