#1 모듈 및 데이터 탑재
import pandas as pd
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')
df_1 = df[['할인권_사용 횟수','성별']]

#2. 성별 평균, 분산, 표준편차 구하기
print(df_1.groupby('성별').mean())
print(df_1.groupby('성별').var())
print(df_1.groupby('성별').std())
