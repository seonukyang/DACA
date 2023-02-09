#NGO - 계층적 군집분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
pd.options.display.float_format = '{:.3f}'.format
df = pd.read_csv('21.csv', sep=',', encoding='UTF-8')
#df = pd.read_csv('1. NGO.csv',sep=',',encoding='CP949')
df1 = df[['PAY_SUM_PAYMENT_R_CS_INT','PAY_SUM_PAYMENT_R_CS_DOM','PAY_SUM_PAYMENT_R_PN_INT','PAY_SUM_PAYMENT_R_PN_FST','PAY_SUM_PAYMENT_R_PN_DOM',
'PAY_SUM_PAYMENT_R_PN_NKOR','PAY_SUM_PAYMENT_O_GN_INT','PAY_SUM_PAYMENT_O_GN_DOM','PAY_SUM_PAYMENT_O_PN_INT','PAY_SUM_PAYMENT_O_PN_FST',
'PAY_SUM_PAYMENT_O_PN_DOM','PAY_SUM_PAYMENT_O_PN_NKOR','CONTACT_ID']]
df1 = df1.dropna()

df1['해외아동'] = df1['PAY_SUM_PAYMENT_R_CS_INT'] + df1['PAY_SUM_PAYMENT_O_GN_INT']
df1['국내아동'] = df1['PAY_SUM_PAYMENT_R_CS_DOM'] + df1['PAY_SUM_PAYMENT_O_GN_DOM']
df1['해외사업'] = df1['PAY_SUM_PAYMENT_R_PN_INT'] + df1['PAY_SUM_PAYMENT_O_PN_INT']
df1['긴급사업'] = df1['PAY_SUM_PAYMENT_R_PN_FST'] + df1['PAY_SUM_PAYMENT_O_PN_FST']
df1['국내사업'] = df1['PAY_SUM_PAYMENT_R_PN_DOM'] + df1['PAY_SUM_PAYMENT_O_PN_DOM']
df1['북한사업'] = df1['PAY_SUM_PAYMENT_R_PN_NKOR'] + df1['PAY_SUM_PAYMENT_O_PN_NKOR']
#df1 = df1[df1['해외아동']!=360000.0&&df1['국내아동']!=360000.0&&df1['']]
df2 = df1[['해외아동','국내아동','해외사업','긴급사업','국내사업','북한사업']]
#2. 비계층적 군집분석
df2 = df2.drop(415)
model = KMeans(n_clusters=6, max_iter=20, random_state=19).fit(df2)

df2['cluster_id'] = model.labels_ #라벨링하기 위해 labels_로 군집번호를 불러와 X객체의 열 이름을 cluster_id로 정한다.

#3. 군집별 고객 수 확인
clu1 = df2[df2.cluster_id==0]
clu2 = df2[df2.cluster_id==1]
clu3 = df2[df2.cluster_id==2]
clu4 = df2[df2.cluster_id==3]
clu5 = df2[df2.cluster_id==4]
clu6 = df2[df2.cluster_id==5]
print('군집1의 고객수 : ',clu1.cluster_id.count())
print('\n군집2의 고객수 : ',clu2.cluster_id.count())
print('\n군집3의 고객수 : ',clu3.cluster_id.count())
print('\n군집4의 고객수 : ',clu4.cluster_id.count())
print('\n군집5의 고객수 : ',clu5.cluster_id.count())
print('\n군집6의 고객수 : ',clu6.cluster_id.count())

#4. 군집별 평균 RFM 확인
print('군집1의 RFM평균\n',clu1.해외아동.mean(), clu1.국내아동.mean(), clu1.해외사업.mean(), 
clu1.긴급사업.mean(),clu1.국내사업.mean(),clu1.북한사업.mean())
print('군집2의 RFM평균\n',clu2.해외아동.mean(), clu2.국내아동.mean(), clu2.해외사업.mean(), 
clu2.긴급사업.mean(),clu2.국내사업.mean(),clu2.북한사업.mean())
print('군집3의 RFM평균\n',clu3.해외아동.mean(), clu3.국내아동.mean(), clu3.해외사업.mean(), 
clu3.긴급사업.mean(),clu3.국내사업.mean(),clu3.북한사업.mean())
print('군집4의 RFM평균\n',clu4.해외아동.mean(), clu4.국내아동.mean(), clu4.해외사업.mean(), 
clu4.긴급사업.mean(),clu4.국내사업.mean(),clu4.북한사업.mean())
print('군집5의 RFM평균\n',clu5.해외아동.mean(), clu5.국내아동.mean(), clu5.해외사업.mean(), 
clu5.긴급사업.mean(),clu5.국내사업.mean(),clu5.북한사업.mean())
print('군집6의 RFM평균\n',clu6.해외아동.mean(), clu6.국내아동.mean(), clu6.해외사업.mean(), 
clu6.긴급사업.mean(),clu6.국내사업.mean(),clu6.북한사업.mean())