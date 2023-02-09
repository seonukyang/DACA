#카이제곱 적합도 검정
#1 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')

#2. 빈도교차표 생성하기
X=pd.crosstab(df.클레임접수여부, df.구매유형, margins=True)
print(X)

#3. 관측도수, 기대도수 추출하기
Ob = X.values[1,:4]
Pr = np.array([0.1,0.3,0.2,0.4])
n=X.values[1,4]
E=n*Pr

#4. 카이제곱 적합도 검정하기
ch = stats.chisquare(Ob, E) #관측도수, 기대도수 순으로 집어넣는다.
print(ch)