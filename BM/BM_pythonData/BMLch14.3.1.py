#1. 장바구니 분석을 통한 교차판매 규칙 생성
#2) 데이터 살펴보기
#1. 모듈 및 함수 불러오기
import pandas as pd

#2. 데이터 불러오기
df = pd.read_csv('Retail_data.csv', engine='python',encoding='CP949')
print(df.head())
print(df.isnull().sum())

print(df.size)
print(df.values)
print(df['Product1'].value_counts())
print(df['Product1'].value_counts()[0])

df1 = df['Product1']
df1 = df1.replace(['Bread','Juice','Nachos','Fruits'],[0,1,2,3])
print(df1)
print(df1.value_counts())

#3) 데이터 전처리
#1. 변수 선택
df = df.iloc[:, 1:]

#2. 데이터 변환
df_dummy = pd.get_dummies(df)

#4) Apriori 모형기반 연관규칙 생성
#1. 모듈 및 함수 불러오기
from mlxtend.frequent_patterns import apriori, association_rules

#2. Apriori를 이용한 빈발항목세트 생성
frequent_items = apriori(df_dummy, min_support=0.1, use_colnames=True)
print(frequent_items.head())

#1. 연관 규칙 생성
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.6)

#결과 출력
print(rules[['antecedents','consequents','support','confidence','lift']].sort_values(by='lift', ascending=False).head(10))