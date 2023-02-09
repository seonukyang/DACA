#연관규칙
#장바구니 분석을 통한 교차판매 규칙 생성
import pandas as pd
df = pd.read_csv('Retail_data.csv', engine='python', encoding='CP949')
df = df.iloc[:, 1:]
df_dummy = pd.get_dummies(df)

from mlxtend.frequent_patterns import apriori, association_rules

#Apriori알고리즘을 이용한 빈발항목세트 생성
frequent_items = apriori(df_dummy, min_support=0.2, use_colnames=True)

#연관규칙 생성
rules = association_rules(frequent_items, metric='support', min_threshold=0.1)
print(frequent_items.sort_values(by='support', ascending=False))


#결과출력
print(rules[['antecedents','consequents','support','confidence','lift']].sort_values(by='lift', ascending=False))