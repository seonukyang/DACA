import pandas as pd
df= pd.read_csv('BML16웹스크래핑.csv', encoding='utf-8')
print(df['score'].value_counts())