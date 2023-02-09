#NGO - 적합성 검정
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
from pingouin import partial_corr
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#2. 빈도표 생성하기
df_1 = df[['PLED_NUM_R_PN_INT','PLED_NUM_R_PN_FST','PLED_NUM_R_PN_DOM','PLED_NUM_R_PN_NKOR']]
df_1 = df_1.dropna()
columns_k = ['정기해외사업 플릿지 수','정기긴급사업 플릿지 수','정기국내사업 플릿지 수','정기북한사업 플릿지 수']
columns_e = ['PLED_NUM_R_PN_INT','PLED_NUM_R_PN_FST','PLED_NUM_R_PN_DOM','PLED_NUM_R_PN_NKOR']
index = [0]
c = [[0,0,0,0]]
pd_f = pd.DataFrame(data = c, columns = columns_k, index=index )

for i in range(0,len(columns_k),1) :
    colname = columns_e[i]
    pd_f.iloc[[0],[i]] = sum(df_1[colname])



#3. 관측도수, 기대도수 추출하기
Ob = pd_f.values[0,:4]
Pr = np.array([0.25,0.25,0.4,0.1])
n=0
for i in range(0,len(columns_k),1) :
    colname = columns_e[i]
    n = n + sum(df_1[colname])
E=n*Pr

#4. 카이제곱 적합도 검정하기
ch = stats.chisquare(Ob, E)
print(pd_f)
print('예상 비율 : ', Pr)
print('전체 표본 수 : ', n)
print('후원 종류별 카이제곱 적합도 ',ch)









