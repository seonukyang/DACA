#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
import scikit_posthocs
import statsmodels.formula.api as smf
from scipy import stats
from factor_analyzer import FactorAnalyzer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
#df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')
df = pd.read_csv('21.csv', sep=',', encoding='UTF-8')
#2. 변수추출
df_1 = df[['PAY_NUM','PAY_SUM_PAYMENTAMOUNT','PLED_NUM','PLED_ACTIVE_NUM','PAY_RATE_NOPAY','CHURN']]
df_1 = df_1.dropna()
df_1['액티브비율'] = df_1['PLED_ACTIVE_NUM']/df_1['PLED_NUM']
#df_1['월후원율'] = df_1['PAY_NUM']/df_1['LONGEVITY_M']
#df2 = pd.get_dummies(df['고객등급'], prefix='고객등급',drop_first=False)
#df_1 = pd.concat([df_1, df2], axis = 1)
X = df_1[['액티브비율','PAY_RATE_NOPAY']]
Y = df_1['CHURN']
XY = pd.concat([X,Y], axis = 1)

#3. 조건 검사
#3-1 독립변수는 정규분포를 따라야한다.
print('독립변수의 정규성 검사')
for i in range(0,len(X.columns),1):
   print(X.columns[i],'의 정규성 검사 : ',stats.shapiro(X[X.columns[i]]))
#3-2. 종속변수와 독립변수는 상관관계를 가져야 한다.
#3-3. 독립변수들 사이에 상관관계가 없거나 작아야 한다.
corr = XY.corr(method='pearson')
print(corr)
#corr.to_csv('21.상관관계.csv', encoding='utf-8-sig')

#3-4. 종속변수로 구분되는 각 집단 별 공분산 행렬리 유사해야 한다.

#4. 선형판별분석
lda = LDA().fit(X,Y)

#5. 예측결과
print('판별식 선형계수 : ', np.round(lda.coef_,3))
print('\n판별식 절편 :',np.round(lda.intercept_,3))
print('\n예측 결과 :',pd.DataFrame(lda.predict(X)))
print('\n예측 스코어 : ', pd.DataFrame(lda.predict_proba(X)))
print('\n예측 정확도 :',lda.score(X,Y))

#6. 분류행렬표 출력
cf_m = pd.DataFrame(confusion_matrix(Y, lda.predict(X)))
cf_m.columns = ['예측 0','예측 1']
cf_m.index = ['실제 0','실제 1']
print('\n분류행렬표 : \n',cf_m)


