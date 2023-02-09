import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
df = pd.read_csv('Ashopping.csv', encoding='CP949')

#더미변수 생성
df2 = pd.get_dummies(df['구매유형'], prefix='구매유형', drop_first=True)
df3 = pd.concat([df,df2],axis=1)

model3 = smf.ols(formula = '방문빈도 ~ 구매유형_2+ 구매유형_3 + 구매유형_4', data=df3).fit()
print(model3.summary())
