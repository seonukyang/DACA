#카이제곱 동질성 검정
#1 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')

df["고객연령대"] = ""
df.고객연령대[df.고객_나이대==1]="1"
df.고객연령대[df.고객_나이대==2]="1"
df.고객연령대[df.고객_나이대==3]="1"
df.고객연령대[df.고객_나이대==4]="1"
df.고객연령대[df.고객_나이대==5]="1"
df.고객연령대[df.고객_나이대==6]="2"
df.고객연령대[df.고객_나이대==7]="2"
df.고객연령대[df.고객_나이대==8]="2"
df.고객연령대[df.고객_나이대==9]="2"

#3. 두 모집단 랜덤표본추출
df1 = df.loc[df.고객연령대=='1'] #df에서 고객연령대==1인 애들만 모은다. 앞에 나온 where와 비슷하네
print(df.head())
print(df1)
df2 = df.loc[df.고객연령대=='2']
df1_sample = df1.sample(200, random_state = 29)
df2_sample = df2.sample(200, random_state = 29)
df3 = df1_sample.append(df2_sample)
print('df3',df3)

#4. 빈도교차표 생성하기
X = pd.crosstab(df3.고객연령대, df3.구매유형, margins=False)

#5. 카이제곱 동질성 검정하기
chi=stats.chi2_contingency(X)
print(chi)