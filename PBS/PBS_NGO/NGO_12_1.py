#NGO - 내적일관성분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
from pingouin import partial_corr
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#데이터 나누기
df_1 = df[['PAY_NUM','PAY_SUM_PAYMENTAMOUNT','PAY_RATE_NOPAY','PLED_FIRST_LONGEVITY','LONGEVITY_D','CONTACT_ID']]
df_1 = df_1.dropna()
df_1[['ratepay','책임감','지속성','후원력','납부력']] = ""
df_1['ratepay'] = df_1['PAY_NUM']/df_1['PLED_FIRST_LONGEVITY']*30

#측정항목 생성
df_1.index = range(0,len(df_1),1)
#책임감
for i in range(0,len(df_1),1) :
    if (df_1['PAY_RATE_NOPAY'][i] >= 0.9) :
        df_1['책임감'].iloc[i] = 1
    elif  (df_1['PAY_RATE_NOPAY'][i] >= 0.8) :
        df_1['책임감'].iloc[i] = 2
    elif  (df_1['PAY_RATE_NOPAY'][i] >= 0.6) :
        df_1['책임감'].iloc[i] = 3
    elif  (df_1['PAY_RATE_NOPAY'][i] >= 0.4) :
        df_1['책임감'].iloc[i] = 4
    elif  (df_1['PAY_RATE_NOPAY'][i] >= 0.2) :
        df_1['책임감'].iloc[i] = 5
    elif  (df_1['PAY_RATE_NOPAY'][i] >= 0.1) :
        df_1['책임감'].iloc[i] = 6
    else  : df_1['책임감'].iloc[i] = 7
#지속성
for i in range(0,len(df_1),1) :
    if (df_1['ratepay'][i] >= 0.9) :
        df_1['지속성'].iloc[i] = 7
    elif  (df_1['ratepay'][i] >= 0.8) :
        df_1['지속성'].iloc[i] = 6
    elif  (df_1['ratepay'][i] >= 0.6) :
        df_1['지속성'].iloc[i] = 5
    elif  (df_1['ratepay'][i] >= 0.4) :
        df_1['지속성'].iloc[i] = 4
    elif  (df_1['ratepay'][i] >= 0.2) :
        df_1['지속성'].iloc[i] = 3
    elif  (df_1['ratepay'][i] >= 0.1) :
        df_1['지속성'].iloc[i] = 2
    else  : df_1['지속성'].iloc[i] = 1

#print(df_1.groupby('지속성').size())

#후원력
for i in range(0,len(df_1),1) :
    if (df_1['PAY_NUM'][i] >= 18) :
        df_1['후원력'].iloc[i] = 7
    elif  (df_1['PAY_NUM'][i] >= 15) :
        df_1['후원력'].iloc[i] = 6
    elif  (df_1['PAY_NUM'][i] >= 12) :
        df_1['후원력'].iloc[i] = 5
    elif  (df_1['PAY_NUM'][i] >= 9) :
        df_1['후원력'].iloc[i] = 4
    elif  (df_1['PAY_NUM'][i] >= 6) :
        df_1['후원력'].iloc[i] = 3
    elif  (df_1['PAY_NUM'][i] >= 3) :
        df_1['후원력'].iloc[i] = 2
    else  : df_1['후원력'].iloc[i] = 1
#print(df_1.groupby('후원력').size())
#납부력
for i in range(0,len(df_1),1) :
    if (df_1['PAY_SUM_PAYMENTAMOUNT'][i]*0.01 >= 5000) :
        df_1['납부력'].iloc[i] = 7
    elif  (df_1['PAY_SUM_PAYMENTAMOUNT'][i]*0.01 >= 4000) :
        df_1['납부력'].iloc[i] = 6
    elif  (df_1['PAY_SUM_PAYMENTAMOUNT'][i]*0.01 >= 3500) :
        df_1['납부력'].iloc[i] = 5
    elif  (df_1['PAY_SUM_PAYMENTAMOUNT'][i]*0.01 >= 3000) :
        df_1['납부력'].iloc[i] = 4
    elif  (df_1['PAY_SUM_PAYMENTAMOUNT'][i]*0.01 >= 1000) :
        df_1['납부력'].iloc[i] = 3
    elif  (df_1['PAY_SUM_PAYMENTAMOUNT'][i]*0.01 >= 500) :
        df_1['납부력'].iloc[i] = 2
    else  : df_1['납부력'].iloc[i] = 1
#print(df_1.groupby('납부력').size())
X = df_1[['책임감','지속성','후원력','납부력','PAY_SUM_PAYMENTAMOUNT','LONGEVITY_D','CONTACT_ID']]
X.to_csv('12.df_1.csv', encoding='utf-8-sig')

#전체의 크론바흐 알파 계수 출력
df_X = pd.read_csv('12.df_1.csv', sep=',', encoding='utf-8-sig')
X0 = df_X[['책임감','지속성','후원력','납부력']]
X1 = df_X[['지속성','후원력','납부력']]
X2 = df_X[['책임감','후원력','납부력']]
X3 = df_X[['책임감','지속성','납부력']]
X4 = df_X[['책임감','지속성','후원력']]
#print(X)

CA_all = pg.cronbach_alpha(data=X0)
CA_X1 = pg.cronbach_alpha(data=X1)
CA_X2 = pg.cronbach_alpha(data=X2)
CA_X3 = pg.cronbach_alpha(data=X3)
CA_X4 = pg.cronbach_alpha(data=X4)

print('책임감, 지속성, 후원력, 납부력',CA_all)
print('지속성, 후원력, 납부력',CA_X1)
print('책임감, 후원력, 납부력',CA_X2)
print('책임감, 지속성, 납부력',CA_X3)
print('책임감, 지속성, 후원력',CA_X4)

df_X['총합'] = df_X['책임감']+df_X['후원력']+df_X['납부력']
print('총합의 평균',df_X['총합'].mean())
print('상위 10퍼', np.percentile(df_X['총합'],90))
print('중위값',np.percentile(df_X['총합'],50))


df_X[['책임감','후원력','납부력','총합','PAY_SUM_PAYMENTAMOUNT','LONGEVITY_D','CONTACT_ID']].to_csv('고객등급점수표.csv', encoding='utf-8-sig')