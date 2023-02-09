#1 모듈 및 데이터 탑재
import pandas as pd
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')

#2. 왜도와 첨도 출력하기
print(df.서비스_만족도.skew()) #왜도
print(df.서비스_만족도.kurt()) #첨도

#3. 히스토그램 그리기
print(df.서비스_만족도.hist(bins=7))