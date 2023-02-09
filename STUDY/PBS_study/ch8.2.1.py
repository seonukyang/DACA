#적합도 검정
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('Ashopping.csv', encoding='CP949')
print(df.columns)
#빈도 교차표 작성
X = pd.crosstab(df.성별, df.고객등급, margins=True)
print(X)

#관측도수, 기대도수 추출하기
Ob = X.values[1,:2]
Pr = np.array([0.8,0.2])
n = X.values[1,2]
E = n*Pr
print(E)
#카이제곱 적합도 검정하기
print(stats.chisquare(Ob, E))