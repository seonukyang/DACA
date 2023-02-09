#일변량 통계기반 선택을 이용한 성능향상
#1 모듈과 함수 불러오기
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

df = pd.read_csv('Ashopping.csv', encoding='cp949')

#2. 변수 지정
X = df.drop(['고객ID','이탈여부'], axis=1) #인덱스와 종속변수를 제거한 모든 변수, axis=1은 열을 삭제
Y = df['이탈여부']

#3. 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#4. 일변량 통계기반 변수 선택
feat_selector = SelectKBest(chi2, k=3) #k로 선택할 변수의 수 결정
feat_selector.fit(X_train, Y_train)
print('feat_selector : ',feat_selector)
#5. 선택된 변수 출력
feat_scores = pd.DataFrame()
feat_scores['Chi-squared stats'] = feat_selector.scores_
feat_scores['P Value'] = feat_selector.pvalues_
feat_scores['Support'] = feat_selector.get_support()
feat_scores['Attribute'] = X_train.columns
print(feat_scores[feat_scores['Support']==True])

#모형기반 선택
#1. 모듈 및 함수 불러오기
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

#2. 모형기반 변수 선택
feat_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, #나무의 개수, 클수록 좋으나 오래걸림
random_state=0), threshold='median')
#랜덤 포레스트 모형을 이용한 유의미한 독립변수의 구분이다.
feat_selector.fit(X_train, Y_train)

#3. 선택된 변수  출력
feat_scores = pd.DataFrame()
feat_scores['Attribute'] = X_train.columns
feat_scores['Support'] = feat_selector.get_support()
print(feat_scores[feat_scores['Support']==True])


#반복적 변수 선택
#1. 모듈 및 함수 불러오기
from sklearn.feature_selection import RFE
import numpy as np

#2 반복적 변수 선택
select = RFE(RandomForestClassifier(n_estimators = 100, random_state=0))
select.fit(X_train, Y_train)

#3. 선택된 변수 출력
features_bool = np.array(select.get_support()) #선택된 변수의 True, False 여부
features = np.array(X.columns)
print(select.get_support())
result = features[features_bool] #true, false의 array를 array에 집어넣으면 true 순번의 값만 나오나 보다.
print(result)
