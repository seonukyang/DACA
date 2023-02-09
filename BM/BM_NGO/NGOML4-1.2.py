#연관규칙 분석
import pandas as pd

#연관규칙 Apriori 모형
from mlxtend.frequent_patterns import apriori, association_rules



df = pd.read_csv('1. NGO_연관규칙.csv', encoding='CP949')

df = df[['CHURN','PLED_NUM_R_CS_INT_C','PLED_NUM_R_CS_DOM_C','PLED_NUM_R_PN_INT_C',
'PLED_NUM_R_PN_FST_C','PLED_NUM_R_PN_DOM_C','PLED_NUM_R_PN_NKOR_C','PLED_NUM_R_PN_ALL_C']]

df = df.dropna()
print(df.shape)

df = df[df['PLED_NUM_R_CS_INT_C']+df['PLED_NUM_R_CS_DOM_C']+df['PLED_NUM_R_PN_INT_C']+
df['PLED_NUM_R_PN_FST_C']+df['PLED_NUM_R_PN_DOM_C']+df['PLED_NUM_R_PN_NKOR_C']+df['PLED_NUM_R_PN_ALL_C']==2]
print(df.shape)
list_name = ['PLED_NUM_R_CS_INT_C','PLED_NUM_R_CS_DOM_C','PLED_NUM_R_PN_INT_C',
'PLED_NUM_R_PN_FST_C','PLED_NUM_R_PN_DOM_C','PLED_NUM_R_PN_NKOR_C','PLED_NUM_R_PN_ALL_C']

df['후원종류1'] = 0
df['후원종류2'] = 0
print(df['후원종류1'])
df.index = range(0,len(df),1)

for i in range(0,len(df),1) : 
    for j in range(0,len(list_name),1) : 
        if df['후원종류1'].iloc[i] == 0 : 
            if df[list_name[j]].iloc[i] == 1:
                df['후원종류1'].iloc[i] = list_name[j]
                for k in range(j+1,len(list_name),1) : 
                    if df['후원종류2'].iloc[i] == 0 : 
                        if df[list_name[k]].iloc[i] == 1:
                            df['후원종류2'].iloc[i] = list_name[k]

print(df[['후원종류1','후원종류2']])

# df2 = df[['CHURN','SEX','연령대','PLED_NUM_R_CS_INT_C','PLED_NUM_R_CS_DOM_C','PLED_NUM_R_PN_INT_C', 
# 'PLED_NUM_R_PN_FST_C','PLED_NUM_R_PN_DOM_C','PLED_NUM_R_PN_NKOR_C']]


df_dummy = pd.get_dummies(df[['후원종류1','후원종류2']])
frequent_items = apriori(df_dummy, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.3)
print(rules[['antecedents','consequents','support','confidence','lift']].sort_values(by='lift', ascending=False))

print(df.groupby(['후원종류1','후원종류2']).count())
#결과 내보내기
# rules.to_csv('연관규칙분석결과4.csv', encoding='utf-8-sig')