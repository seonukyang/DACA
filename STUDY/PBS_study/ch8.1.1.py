#적합도 검정
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('Ashopping.csv', encoding='CP949')

#빈도 교차표 작성
X = pd.crosstab(df.클레임접수여부, df.구매유형, margins=True)
print(X)

#관측도수, 기대도수 추출하기
Ob = X.values[1,:4]
Pr = np.array([0.1,0.3,0.2,0.4])
n = X.values[1,4]
E = n*Pr

#카이제곱 적합도 검정하기
print(stats.chisquare(Ob, E))

#독립성 검정
Y=pd.crosstab(df.성별, df.클레임접수여부, margins=False)
print(Y)
#카이제곱 독립성 검정하기
print(stats.chi2_contingency(Y))

#동질성 검정
df['고객연령대'] = ''
for i in range(0,len(df['고객_나이대']),1) : 
    if df['고객_나이대'][i] >5 :
        df['고객연령대'][i] = '1'
    else : df['고객연령대'][i] = '2'

#두 모집단 랜덤표본추출
df1 = df.loc[df.고객연령대=='1']
df2 = df.loc[df.고객연령대=='2']
df1_sample = df1.sample(200, random_state = 29)
df2_sample = df2.sample(200, random_state=29)
df3 = df1_sample.append(df2_sample)

#빈도교차표 생성하기
Z = pd.crosstab(df3.고객연령대, df3.구매유형, margins=False)
print(Z)

#카이제곱 동질성 검정하기
print(stats.chi2_contingency(Z))
