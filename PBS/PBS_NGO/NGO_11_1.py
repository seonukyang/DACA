#NGO - 동질성 검정
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
from pingouin import partial_corr
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#2. 데이터 나누기
df_1 = df[['AGE','MOTI_CHANNEL']]
df_1 = df_1[df_1['AGE']!=0]
df.dropna(inplace=True)
#df_1 = df_1.dropna()
df_1['고객연령대'] = ""
df_1['주요채널'] = ""
df_1.index = range(0,len(df_1),1)

for i in range(0,len(df_1),1) : 
    if (df_1['AGE'][i] < 40) : 
        df_1['고객연령대'][i] = '청년층'
    else :
        df_1['고객연령대'][i] = '장년층'

for i in range(0,len(df_1),1) : 
    if (df_1['MOTI_CHANNEL'][i] == 'BROADCAST' ) : 
        df_1['주요채널'][i] = '1'
    elif df_1['MOTI_CHANNEL'][i] == 'DIGITAL' : df_1['주요채널'][i] = '2'
    elif df_1['MOTI_CHANNEL'][i] == 'DM' : df_1['주요채널'][i] = '3'
    elif df_1['MOTI_CHANNEL'][i] == 'EVENT' : df_1['주요채널'][i] = '4'
    elif df_1['MOTI_CHANNEL'][i] == 'GENERAL AD' : df_1['주요채널'][i] = '5'
    elif df_1['MOTI_CHANNEL'][i] == 'RELATIONSH' : df_1['주요채널'][i] = '6'
    elif df_1['MOTI_CHANNEL'][i] == 'TM' : df_1['주요채널'][i] = '7'
    elif df_1['MOTI_CHANNEL'][i] == 'UNKNOWN' : df_1['주요채널'][i] = '0'

#3. 두 모집단 랜덤표본추출
df1 = df_1.loc[df_1.고객연령대=='청년층'] 
df2 = df_1.loc[df_1.고객연령대=='장년층']
df1_sample = df1.sample(200, random_state = 29)
df2_sample = df2.sample(200, random_state = 29)
df3 = df1_sample.append(df2_sample)

#4. 빈도교차표 생성하기
X_f = pd.crosstab(df3.고객연령대, df3.주요채널, margins=False)
X_t = pd.crosstab(df3.고객연령대, df3.주요채널, margins=True)
print('연령대별 주요채널의 빈도교차표')
print(X_t,'\n')
#5. 카이제곱 동질성 검정하기
chi=stats.chi2_contingency(X_f)
print('연령대별 주요채널의 동질성 검정 :\n', chi)



