#1) 일변량 통계기반 선택
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

df= pd.read_csv('Ashopping.csv', encoding='CP949')
X = df.drop(['고객ID','이탈여부'], axis=1)
Y = df['이탈여부']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

feat_selector = SelectKBest(chi2)
feat_selector.fit(X_train, Y_train)

feat_scores = pd.DataFrame()
feat_scores['Chi-squared stats'] = feat_selector.scores_
feat_scores['P Value'] = feat_selector.pvalues_
feat_scores['Support'] = feat_selector.get_support()
feat_scores['Attribute'] = X_train.columns
print(feat_scores[feat_scores['Support']==True])

#모형기반 선택
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

feat_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0), threshold='median')
feat_selector.fit(X_train, Y_train)

feat_scores = pd.DataFrame()
feat_scores['Attribute'] = X_train.columns
feat_scores['Support'] = feat_selector.get_support()
print(feat_scores[feat_scores['Support']==True])

#반복적 변수 선택
from sklearn.feature_selection import RFE
import numpy as np
select = RFE(RandomForestClassifier(n_estimators = 100, random_state = 0))
select.fit(X_train, Y_train)

features_bool = np.array(select.get_support())
features = np.array(X.columns)
result = features[features_bool]
print(result)


#언더 샘플링
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

X_train_under, Y_train_under = RandomUnderSampler(random_state=0).fit_sample(X_train, Y_train)
print('Original dataset shape :',Counter(Y))
print('sampled dataset shape : ',Counter(Y_train))
print('Resampled dataset shape : ', Counter(Y_train_under))

#오버 샘플링
from imblearn.over_sampling import SMOTE

X_train_over, Y_train_over = SMOTE(random_state=0).fit_sample(X_train, Y_train)
print('Original dataset shape :',Counter(Y))
print('sampled dataset shape : ',Counter(Y_train))
print('Resampled dataset shape : ', Counter(Y_train_under))


