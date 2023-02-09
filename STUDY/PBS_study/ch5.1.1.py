#평균, 분산, 표준편차 분석
import pandas as pd
import numpy as np
df = pd.read_csv('Ashopping.csv', encoding='CP949')
df_1 = df[['할인권_사용 횟수','성별']]

print(df_1.groupby('성별').mean())
print(df_1.groupby('성별').var())
print(df_1.groupby('성별').std())

#첨도, 왜도
print(df.서비스_만족도.skew())
print(df.서비스_만족도.kurt())

#히스토그램 그리기
df.서비스_만족도.hist(bins=7)

#5.3 기타 기술통계량

#4분위수 구하기
print(np.percentile(df['할인권_사용 횟수'],25))
print(np.percentile(df['할인권_사용 횟수'],50))
print(np.percentile(df['할인권_사용 횟수'],75))
print(np.percentile(df['할인권_사용 횟수'],100))

#최빈값 출력
print('최반값 :',df['할인권_사용 횟수'].value_counts().idxmax())

#기술통계량 출력
print('기술통계량:\n', df['할인권_사용 횟수'].describe())