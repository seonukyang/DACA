import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('Ashopping.csv',encoding='CP949')

#일표본 t-검정
print('총매출액의 평균 : ',df.총_매출액.mean())
print(stats.ttest_1samp(df['총_매출액'], 7700000))

#독립표본 t-검정
#등분산검정
no_claim = df[df.클레임접수여부==0]
df2 = np.array(no_claim.방문빈도)
claim = df[df.클레임접수여부==1]
df3 = np.array(claim.방문빈도)
print(stats.bartlett(df2, df3))

#독립표본 t-검정 및 방문빈도 평균
print(stats.ttest_ind(df2, df3, equal_var=False))
print('클레임 접수여부(0) 고객 평균방문빈도 : ',no_claim.방문빈도.mean())
print('클레임 접수여부(1) 고객 평균방문빈도 : ',claim.방문빈도.mean())

#쌍체표본 t-검정
print(stats.ttest_rel(df['멤버쉽_프로그램_가입후_만족도'], df['멤버쉽_프로그램_가입전_만족도']))

