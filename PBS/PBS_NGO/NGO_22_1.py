#NGO - 로지스틱회귀분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import statsmodels.api as sm
pd.options.display.float_format = '{:.3f}'.format
df = pd.read_csv('21.csv', sep=',', encoding='UTF-8')
#df = pd.read_csv('1. NGO.csv',sep=',',encoding='CP949')
df = df[df.columns.difference(['Unnamed: 0'])]
#df1 = df[['AGE','SEX','CHURN','MOTI_CHANNEL']]
df1 = df[['AGE','SEX','CHURN','PAY_NUM','PAY_SUM_PAYMENTAMOUNT','고객등급']]
df1 = df1.dropna()
df1 = df1[df1['AGE']>0]
#df1 = df1[df1['SEX']>0]
df1.index = range(0,len(df1),1)
df2 = pd.get_dummies(df1['고객등급'],prefix='고객등급', drop_first=False)
df4 = pd.concat([df1,df2], axis=1)
df4['intercept'] = 1.0
df4.index = range(0,len(df4),1)
print(df4)
#df4.to_csv('22.로지스틱.csv', encoding='utf-8-sig')
#df4 = pd.read_csv('22.로지스틱.csv', sep=',', encoding='UTF-8')
x = df4[['intercept','AGE','고객등급_5.0']]
y = df4['CHURN']
dfo = df[['CHURN','고객등급']]
print(pd.crosstab(dfo.CHURN, dfo.고객등급, margins=True))


#4로지스틱 회귀분석 실행하기
logit = sm.Logit(y,x).fit()

#분석결과 출력하기
print(logit.summary2())
print(np.exp(logit.params)) #params는 로지스틱 계수이므로 이를 exp에 넣어서 오즈비를 구한다.
cf_m2 = pd.DataFrame(logit.pred_table())
cf_m2.columns=['예측 0', '예측 1']
cf_m2.index = ['실제 0', '실제 1']
print('\n분류행렬표 :\n', cf_m2)
total = cf_m2.iloc[1,1]+cf_m2.iloc[0,1]+cf_m2.iloc[1,0]+cf_m2.iloc[0,0]
true = cf_m2.iloc[1,1]+cf_m2.iloc[0,0]
act = true / total
print('정확도 : ', act*100, '%')