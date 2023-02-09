import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
df = pd.read_csv('Ashopping.csv', encoding='CP949')
print(df.columns)
#더미변수 생성
df1 = df['총_매출액']
df2 = df['고객_나이대']
# print(df2[2])
for i in range(0,len(df2),1):
    if df2[i] > 6 :
        df2[i] = 3
    elif df2[i] > 3 :
        df2[i] = 2
    else : df2[i]=1
df2 = pd.get_dummies(df['고객_나이대'], prefix='고객_나이대', drop_first=True)
df3 = pd.concat([df,df2],axis=1)
print(df3)

model3 = smf.ols(formula = '총_매출액 ~ 고객_나이대_2 + 고객_나이대_3', data=df3).fit()
print(model3.summary())
