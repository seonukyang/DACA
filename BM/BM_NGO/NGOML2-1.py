#이항 로지스틱 회귀분석
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

df = pd.read_csv('1. NGO.csv', encoding='cp949')
df = df[df['AGE'] > 0]
df['가입나이'] = round((df['AGE']*12 - df['LONGEVITY_M'])/12,0)
df = df[df['가입나이'] > 0]
df = df[df['SEX']!=0]
df['가입나이연령대'] = ''
df['연령대'] = ''
# if df['가입나이'][1] < 10 :
#          df['가입나이연령대'].iloc[1] = '유아'
for i in range(0,len(df), 1):
    if df['가입나이'].iloc[i] < 10 :
        df['가입나이연령대'].iloc[i] = '유아'
    elif df['가입나이'].iloc[i] < 20 :
        df['가입나이연령대'].iloc[i] = '10대'
    elif df['가입나이'].iloc[i] < 30 :
        df['가입나이연령대'].iloc[i] = '20대'
    elif df['가입나이'].iloc[i] < 40 :
        df['가입나이연령대'].iloc[i] = '30대'
    elif df['가입나이'].iloc[i] < 50 :
        df['가입나이연령대'].iloc[i] = '40대'
    elif df['가입나이'].iloc[i] < 60 :
        df['가입나이연령대'].iloc[i] = '50대'
    elif df['가입나이'].iloc[i] < 70 :
        df['가입나이연령대'].iloc[i] = '60대'
    else : df['가입나이연령대'].iloc[i] = '70대 이상'

for i in range(0,len(df), 1):
    if df['AGE'].iloc[i] < 10 :
        df['연령대'].iloc[i] = '유아'
    elif df['AGE'].iloc[i] < 20 :
        df['연령대'].iloc[i] = '10대'
    elif df['AGE'].iloc[i] < 30 :
        df['연령대'].iloc[i] = '20대'
    elif df['AGE'].iloc[i] < 40 :
        df['연령대'].iloc[i] = '30대'
    elif df['AGE'].iloc[i] < 50 :
        df['연령대'].iloc[i] = '40대'
    elif df['AGE'].iloc[i] < 60 :
        df['연령대'].iloc[i] = '50대'
    elif df['AGE'].iloc[i] < 70 :
        df['연령대'].iloc[i] = '60대'
    else : df['연령대'].iloc[i] = '70대 이상'


df1 = df[['CONTACT_ID','AGE','SEX','LONGEVITY_M','PAY_RATE_NOPAY','PAY_NUM','PAY_SUM_PAYMENTAMOUNT','CHURN','가입나이','가입나이연령대','연령대']]
df1 = df1.dropna()

print(df1['가입나이연령대'].head())
df1.to_csv('NGO연령대.csv', encoding='utf-8-sig')
