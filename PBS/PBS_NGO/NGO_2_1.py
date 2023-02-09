#NGO - 기술통계분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#2. 데이터 나누기
df_1 = df[['PAY_SUM_PAYMENT_R_CS_INT','PAY_SUM_PAYMENT_R_CS_DOM','PAY_SUM_PAYMENT_R_PN_INT','PAY_SUM_PAYMENT_R_PN_FST','PAY_SUM_PAYMENT_R_PN_DOM',
'PAY_SUM_PAYMENT_R_PN_NKOR','PAY_SUM_PAYMENT_R_PN_ALL']]
#결측값 처리
df_1 = df_1.dropna()
col = list(df_1.columns)
rows_list = ['평균','분산','표준편차','왜도','첨도','사분위수1','사분위수2','사분위수3','사분위수4','최빈값'] #10개
columns_list = ['기술통계','정기해외아동','정기국내아동','정기해외사업','정기긴급사업',
'정기국내사업','정기북한사업','정기전체사업'] #8개
size_col = len(columns_list)
size_row = len(rows_list)


value_list = [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]
,[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
resultDF = pd.DataFrame(data = value_list, columns = columns_list)

#print(col[0])
#print(df_1[col[1-1]].mean())

for i in range (0,size_row,1) : 
    resultDF.iloc[[i],[0]] = rows_list[i]
for i in range(1,size_col,1) : 
    colname = col[i-1]
    df_2 = df_1[df_1[colname]!=0]
    resultDF.iloc[[0],[i]] = round(df_2[colname].mean(),2)
    resultDF.iloc[[1],[i]] = round(df_2[colname].var(),2)
    resultDF.iloc[[2],[i]] = round(df_2[colname].std(),2)
    resultDF.iloc[[3],[i]] = round(df_2[colname].skew(),2)
    resultDF.iloc[[4],[i]] = round(df_2[colname].kurt(),2)
    resultDF.iloc[[5],[i]] = np.percentile(df_2[colname],25)
    resultDF.iloc[[6],[i]] = np.percentile(df_2[colname],50)
    resultDF.iloc[[7],[i]] = np.percentile(df_2[colname],75)
    resultDF.iloc[[8],[i]] = np.percentile(df_2[colname],100)
    resultDF.iloc[[8],[i]] = df_2[colname].value_counts().idxmax()
print(resultDF)
resultDF.to_csv('2.통계분석.csv', encoding='utf-8-sig')