import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy import stats
df = pd.read_csv('CCA.csv', sep=',',encoding='CP949')
U=df[['품질','가격','디자인']]
V=df[['직원 서비스','매장 시설','고객관리']]
print(df.head())

#2. 정준변수 구하기
cca = CCA(n_components=1).fit(U,V) #n_components 각 변수 그룹을 몇 개의 요인으로 할당하여 구할 것인다. 
U_c, V_c = cca.transform(U,V) #. 정준변수를 산출한다. 제1 정준상관계수를 보여줄 그룹을 구한 것이다.
print('U_c : ',U_c, '      V_c : ', V_c)
U_c1 = pd.DataFrame(U_c)[0]
V_c1 = pd.DataFrame(V_c)[0]

#3. 정준상관계수 구하기
CC1 = stats.pearsonr(U_c1, V_c1) #정준상관관계수가 나오는 그룹을 구했으니 이 그룹으로 상관분석을 해준다.
print('제1정준상관계수 : ', CC1)

#4. 정준적재량, 교차적재량 구하기
print('제품 만족도 정준변수와 해당 변수들간 정준적재량: ',np.corrcoef(U_c1.T, U.T)[0,1:4])
print('제품 만족도 정준변수와 매장 만족도 변수들간 교차적재량:',np.corrcoef(U_c1.T, V.T)[0,1:])
print('매장 만족도 정준변수와 해당 변수들간 정준적재량: ', np.corrcoef(V_c1.T, V.T)[0,1:])
print('매장 만족도 정준변수와 제품 만족도 변수들간 교차적재량: ',np.corrcoef(V_c1.T, U.T)[0,1:4])