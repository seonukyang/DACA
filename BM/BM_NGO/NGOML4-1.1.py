#연관규칙 분석
import pandas as pd

#연관규칙 Apriori 모형
from mlxtend.frequent_patterns import apriori, association_rules



df = pd.read_csv('1. NGO_연관규칙.csv', encoding='CP949')

df = df[['SEX','AGE','CHURN','PLED_NUM_R_CS_INT_C','PLED_NUM_R_CS_DOM_C','PLED_NUM_R_PN_INT_C',
'PLED_NUM_R_PN_FST_C','PLED_NUM_R_PN_DOM_C','PLED_NUM_R_PN_NKOR_C','PLED_NUM_R_PN_ALL_C']]
# print(df.head())
df = df.dropna()
df = df[df['SEX']!=0]
df = df[df['AGE']>0]
df['SEX'] = df['SEX'].replace([1,2],['남성','여성'])
df['CHURN'] = df['CHURN'].replace([0,1],['비이탈','이탈'])


df['PLED_NUM_R_CS_INT_C'] = df['PLED_NUM_R_CS_INT_C'].replace([0,1],['비후원','후원'])
df['PLED_NUM_R_CS_DOM_C'] = df['PLED_NUM_R_CS_DOM_C'].replace([0,1],['비후원','후원'])
df['PLED_NUM_R_PN_INT_C'] = df['PLED_NUM_R_PN_INT_C'].replace([0,1],['비후원','후원'])
df['PLED_NUM_R_PN_FST_C'] = df['PLED_NUM_R_PN_FST_C'].replace([0,1],['비후원','후원'])
df['PLED_NUM_R_PN_DOM_C'] = df['PLED_NUM_R_PN_DOM_C'].replace([0,1],['비후원','후원'])
df['PLED_NUM_R_PN_NKOR_C'] = df['PLED_NUM_R_PN_NKOR_C'].replace([0,1],['비후원','후원'])
df['PLED_NUM_R_PN_ALL_C'] = df['PLED_NUM_R_PN_ALL_C'].replace([0,1],['비후원','후원'])

df['연령대'] = ''
for i in range(0, len(df), 1):
    if df['AGE'].iloc[i] < 10 : 
        df['연령대'].iloc[i] = '유아'
    elif df['AGE'].iloc[i] <20 : 
        df['연령대'].iloc[i] = '10대'
    elif df['AGE'].iloc[i] <30 : 
        df['연령대'].iloc[i] = '20대'
    elif df['AGE'].iloc[i] <40 : 
        df['연령대'].iloc[i] = '30대'
    elif df['AGE'].iloc[i] <50 : 
        df['연령대'].iloc[i] = '40대'
    elif df['AGE'].iloc[i] <60 : 
        df['연령대'].iloc[i] = '50대'
    elif df['AGE'].iloc[i] <70 : 
        df['연령대'].iloc[i] = '60대'
    else : df['연령대'].iloc[i] = '70대'
# print(df.head())
print(df.shape)

df1 = df[['CHURN','SEX','연령대','PLED_NUM_R_CS_INT_C','PLED_NUM_R_CS_DOM_C','PLED_NUM_R_PN_INT_C', 
'PLED_NUM_R_PN_FST_C','PLED_NUM_R_PN_DOM_C','PLED_NUM_R_PN_NKOR_C','PLED_NUM_R_PN_ALL_C']]
df2 = df[['CHURN','SEX','연령대','PLED_NUM_R_CS_INT_C','PLED_NUM_R_CS_DOM_C','PLED_NUM_R_PN_INT_C', 
'PLED_NUM_R_PN_FST_C','PLED_NUM_R_PN_DOM_C','PLED_NUM_R_PN_NKOR_C']]


df_dummy = pd.get_dummies(df2)
frequent_items = apriori(df_dummy, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_items, metric='confidence', min_threshold=0.3)
print(rules[['antecedents','consequents','support','confidence','lift']].sort_values(by='lift', ascending=False))

#결과 내보내기
rules.to_csv('연관규칙분석결과3.csv', encoding='utf-8-sig')