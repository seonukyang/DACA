import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
df = pd.read_csv('Ashopping.csv', encoding='CP949')

#회귀분석 분류예측
X = df[['방문빈도','1회_평균매출액','거래기간']]
Y = df[['이탈여부']]

lda = LDA().fit(X,Y)

print('판별식 선형계수 : ', lda.coef_)
print('판별식 절편 : ', lda.intercept_)
print('예측결과 : ',pd.DataFrame(lda.predict(X)))
print('예측스코어 : ', pd.DataFrame(lda.predict_proba(X)))
print('예측 정확도 : ', lda.score(X,Y))

cf_m = pd.DataFrame(confusion_matrix(Y, lda.predict(X)))
cf_m.columns = ['예측 0', '예측 1']
cf_m.index = ['실제 0', '실제 1']
print(cf_m)

#로지스틱 회귀분석
import statsmodels.api as sm
df2 = pd.get_dummies(df['성별'], prefix='성별', drop_first=False)
df3 = pd.concat([df,df2], axis=1)

df3['intercept'] = 1.0
x = df3[['intercept','거래기간','Recency','성별_0']]
y = df3[['이탈여부']]

logit = sm.Logit(y,x).fit()
print(logit.summary2())
print('오즈비 : ',print(np.exp(logit.params)))
cf_m2=pd.DataFrame(logit.pred_table())
cf_m2.columns = ['예측0','예측1']
cf_m2.index = ['실제0','실제1']
print('분류행렬표\n',cf_m2)
