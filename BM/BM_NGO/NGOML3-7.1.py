#가우시안 이항 분석
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import random

#데이터 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


#데이터 균형화
from imblearn.over_sampling import SMOTE
from collections import Counter
#덴드로그램
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

#로지스틱
from sklearn.linear_model import LogisticRegression
#의사결정나무
from sklearn.tree import DecisionTreeClassifier
#가우시안 나이브 베이즈
from sklearn.naive_bayes import GaussianNB
#보팅 앙상블
from sklearn.ensemble import VotingClassifier
#랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
#그래디언트 부스팅
from sklearn.ensemble import GradientBoostingClassifier
#나이브베이즈
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
#군집분석
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

#정밀도, 재현율, F1 스코어 평가, 교차검증
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB



df = pd.read_csv('NGO연령대.csv', encoding='UTF-8')
df_model = {'eps':[],'min_samples':[],'실루엣':[],'CP':[]}
df_model = pd.DataFrame(df_model)


del df['Unnamed: 0']
#3. 변수 지정(독립변수, 종속변수)
target = df[df['CHURN']==0]
df = df[df['LONGEVITY_M']>12]
df = df[df['CHURN']==0]
X = df[['AGE','LONGEVITY_M','PAY_SUM_PAYMENTAMOUNT','가입나이']]


target = target[target['LONGEVITY_M']==12]
print(target.index)
target_index = [61, 74, 167, 199, 253, 261, 280, 342, 343, 423, 460, 499, 502, 543, 570, 690]
target = target.loc[target_index]
target_test = target[['AGE','LONGEVITY_M','PAY_SUM_PAYMENTAMOUNT','가입나이']]


font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font_name)

# fig, axs = plt.subplots(2,3,figsize=(20,5))

# sns.boxplot(X['AGE'], ax = axs[0,0])
# sns.boxplot(X['LONGEVITY_M'], ax = axs[0,1])
# sns.boxplot(X['PAY_RATE_NOPAY'], ax = axs[0,2])
# sns.boxplot(X['PAY_NUM'], ax = axs[1,0])
# sns.boxplot(X['PAY_SUM_PAYMENTAMOUNT'], ax = axs[1,1])
# sns.boxplot(X['가입나이'], ax = axs[1,2])
# plt.show()

#전처리과정
list = ['AGE','LONGEVITY_M','PAY_SUM_PAYMENTAMOUNT','가입나이']

for a in list : 
    Q1 = X[a].quantile(0.25)
    Q3 = X[a].quantile(0.75)
    IQR = Q3 - Q1
    outlier_index = X[(X[a]<Q1 - 1.5*IQR)|(X[a] > Q3 + 1.5*IQR)].index
    X.drop(outlier_index, inplace=True)

#표준화
scaler = StandardScaler()
scaler.fit(X)
X_stand = scaler.transform(X)
target_stand = scaler.transform(target_test)


# for i in range(5,11,1): 
#     i = i/10
#     for j in range(5,21,1):
#         dbscan_test = DBSCAN(eps=i, min_samples=j)
#         Y_dbscan = dbscan_test.fit_predict(X_stand)
#         db_S_score = silhouette_score(X_stand, Y_dbscan)
#         db_C_score = calinski_harabasz_score(X_stand, Y_dbscan)
#         newdata = {'eps':i,'min_samples':j,'실루엣':db_S_score,'CP':db_C_score}
#         df_model = df_model.append(newdata, ignore_index=True)
#         print('진행상황 for문 : ',i,'-',j,'번째')

# df_model.to_csv('dbscan모델찾기결과.csv', encoding='utf-8-sig')

dbscan = DBSCAN(eps=0.7, min_samples=10)
Y_dbscan = dbscan.fit_predict(X_stand)
db_S_score = silhouette_score(X_stand, Y_dbscan)
db_C_score = calinski_harabasz_score(X_stand, Y_dbscan)


print('dbscan 군집분석 실루엣 계수 : ',db_S_score)
print('dbscan 군집분석 CH 점수 : ',db_C_score)



X['result'] = ''
X['result'] = Y_dbscan

target_pred_d = dbscan.fit_predict(target_stand)


target['DBSCAN군집예측'] = ''
target['DBSCAN군집예측'] = target_pred_d
# X.to_csv('dbscan군집모델결과.csv', encoding='utf-8-sig')
# target.to_csv('dbscan군집분석결과.csv', encoding='utf-8-sig')

a = pd.Series(X.groupby('result')['AGE'].mean())
b = pd.Series(X.groupby('result')['LONGEVITY_M'].mean())
c = pd.Series(X.groupby('result')['PAY_SUM_PAYMENTAMOUNT'].mean())
d = pd.Series(X.groupby('result')['가입나이'].mean())

df2 = pd.concat([pd.Series([0,1]),a,b,c,d,], axis=1)
df2.columns = ["ClusterID", 'AGE','LONGEVITY_M','PAY_SUM_PAYMENTAMOUNT','가입나이']

#막대그래프
fig, axs = plt.subplots(1,4, figsize = (50,20))
sns.barplot(x=df2.ClusterID, y=df2['AGE'], ax = axs[0])
sns.barplot(x=df2.ClusterID, y=df2['LONGEVITY_M'], ax = axs[1])
sns.barplot(x=df2.ClusterID, y=df2['PAY_SUM_PAYMENTAMOUNT'], ax = axs[2])
sns.barplot(x=df2.ClusterID, y=df2['가입나이'], ax = axs[3])
plt.show()