#NGO - 독립성 검정
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
from pingouin import partial_corr
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#2. 데이터 나누기
df_1 = df[['PLED_NUM_R_CS_INT','PLED_NUM_R_CS_DOM','PLED_NUM_R_PN_INT','PLED_NUM_R_PN_FST','PLED_NUM_R_PN_DOM','PLED_NUM_R_PN_NKOR']]
df_1 = df_1.dropna()

df_1['해외아동여부']=""
df_1['국내아동여부']=""
df_1['해외사업여부']=""
df_1['긴급사업여부']=""
df_1['국내사업여부']=""
df_1['북한사업여부']=""
columns_bs = ['PLED_NUM_R_CS_INT','PLED_NUM_R_CS_DOM','PLED_NUM_R_PN_INT','PLED_NUM_R_PN_FST','PLED_NUM_R_PN_DOM','PLED_NUM_R_PN_NKOR']
columns_add = ['해외아동여부','국내아동여부','해외사업여부','긴급사업여부','국내사업여부','북한사업여부']

for i in range(0,len(columns_bs),1) : 
    colname_add = columns_add[i]
    colname_bs = columns_bs[i]
    for j in range(0,len(df_1),1) : 
        if (df_1[colname_bs][j] > 0) :
            df_1[colname_add][j] = 1
        else : 
            df_1[colname_add][j] = 0
df_2 = df_1[['해외아동여부','국내아동여부','해외사업여부','긴급사업여부','국내사업여부','북한사업여부']]

#3. 빈도교차표 생성, 카이제곱 독립성 검정하기 해외아동여부, 국내아동여부
X_1 = pd.crosstab(df_2.해외아동여부, df_2.국내아동여부, margins=True)
X_2 = pd.crosstab(df_2.해외아동여부, df_2.국내아동여부, margins=False)
print('해외아동여부x국내아동여부의 빈도표 : ')
print(X_1)

#4. 빈도교차표 생성, 카이제곱 독립성 검정하기, 각 사업마다
chi_X=stats.chi2_contingency(X_2)
print('해외아동여부x국내아동여부의 독립성 검정 : ',chi_X,'\n')

for i in range(2,len(columns_add),1) :    
    for j in range(i+1, len(columns_add),1) :
        colname_i = columns_add[i]
        colname_j = columns_add[j]
        Y_1 = pd.crosstab(df_2[colname_i], df_2[colname_j], margins=True)
        Y_2 = pd.crosstab(df_2[colname_i], df_2[colname_j], margins=False)
        chi_Y=stats.chi2_contingency(Y_2)
        print(i-1,'-',j-1,'. ',colname_i,'와',colname_j,'의 빈도표 : ')
        print(Y_1)
        print(i-1,'-',j-1,'. ',colname_i,'와',colname_j,'의 독립성 검정 : ')
        print(chi_Y,'\n')
