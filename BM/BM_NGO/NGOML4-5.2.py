import pandas as pd
df= pd.read_csv('웹스크래핑_월드비전_트위터2.csv', encoding='cp949')
print(df['month'].value_counts())
print(df['month'].sum())
print(df['month'].count())