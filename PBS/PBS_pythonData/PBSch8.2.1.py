#카이제곱 독립성 검정
#1 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')

#2. 빈도교차표 생성하기
X = pd.crosstab(df.성별, df.클레임접수여부, margins=False)

#3. 카이제곱 독립성 검정하기
chi=stats.chi2_contingency(X)
print(chi)
