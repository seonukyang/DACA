#분산분석 - 공분산분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import pingouin as pg
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')
df1 = df[['총_매출액','방문빈도','거주지역']]
pd.options.display.float_format = '{:.3f}'.format

#2. 공분산분석
print('공분산분석 결과\n', pg.ancova(dv='총_매출액',between='거주지역', covar='방문빈도', data=df1))
#dv=종속변수, between=독립변수, covar=공변량(제거할거), data

#3. 일원분산분석
print('\n일원분산분석 결과\n', pg.anova(dv='총_매출액',between='거주지역', data=df1))
