#1. 모듈 및 데이터 탑재
import pandas as pd
from scipy import stats
import numpy as np
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')

#2. 등분산검정
no_claim = df[df.클레임접수여부==0]
df2 = np.array(no_claim.방문빈도) #행렬식으로 뽑다? no_claim의 방문빈도만 1xn 으로 뽑음
print(df2)
claim = df[df.클레임접수여부==1]
df3 = np.array(claim.방문빈도)
stats.bartlett(df2, df3)
print(stats.bartlett(df2, df3))

#3. 독립표본 t-검정 및 방문빈도 평균
print(stats.ttest_ind(df2, df3, equal_var=False))
print('클레임 접수여부(0) 고객 평균방문빈도 : ', no_claim.방문빈도.mean())
print('클레임 접수여부(1) 고객 평균방문빈도 : ', claim.방문빈도.mean())