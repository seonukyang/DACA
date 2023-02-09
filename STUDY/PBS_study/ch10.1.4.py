import pandas as pd
import pingouin as pg
df = pd.read_csv('Ashopping.csv', encoding='CP949')
df1 = df[['총_매출액','방문빈도','거주지역']]

#공분산 분석
print('공분산분석 결과\n', pg.ancova(dv='총_매출액', between='거주지역', covar='방문빈도', data=df1))

#일원분산분석
print('일원분산분석 결과\n', pg.anova(dv='총_매출액', between='거주지역', data=df1))
