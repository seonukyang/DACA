#1. 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import scipy as sp
import pingouin as pg
import scikit_posthocs
import statsmodels.formula.api as smf
from factor_analyzer import FactorAnalyzer
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')
#2. 변수추출
X = df[['PLED_NUM','PLED_NUM_FULFILLED','CHILD_NUM_COUNTRYCODE','PAY_NUM','PAY_SUM_PAYMENTAMOUNT',
'PAY_NUM_NOPAY','INTR_NUM_REQUEST','INTR_NUM_COMM']]

X = X.dropna()
#3. 탐색적요인분석
fa = FactorAnalyzer(method='principal',n_factors=3, rotation='varimax').fit(X)
print('fa : ',fa)
#4. 결과 출력
print('요인적재량 :\n', pd.DataFrame(fa.loadings_, index=X.columns)) 
print('\n공통성 :\n', pd.DataFrame(fa.get_communalities(), index=X.columns))
ev, v=fa.get_eigenvalues() 
print('\n고유값 :\n', pd.DataFrame(ev))
print('\n요인점수 :\n', fa.transform(X.dropna()))