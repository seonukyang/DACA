#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')

#2. 4분위수 구하기
print(df_1[colname].mean())
print(np.percentile(df['할인권_사용 횟수'],50))
print(np.percentile(df['할인권_사용 횟수'],75))
print(np.percentile(df['할인권_사용 횟수'],100))

#3. 최빈값 출력
print('최빈값:', df['할인권_사용 횟수'].value_counts().idxmax())

#4. 기술통계량 출력
print('기술통계량:\n', df['할인권_사용 횟수'].describe())
#print('기술통계량에서 평균 뽑아내기:\n', df['할인권_사용 횟수'].describe().mean()) 기술통계량 8개 값들에 대한 평균을 구해버린다.