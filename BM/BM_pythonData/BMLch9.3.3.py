#나이브 베이즈 - 다항분포 나이브 베이즈

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

#1) 데이터 살펴보기
#1. 모듈 및 함수 불러오기
from sklearn.datasets import fetch_20newsgroups

#2. 데이터 불러오기
newsgroups = fetch_20newsgroups(subset='all')

#3.데이터 속성 확인하기
# print(dir(newsgroups))
# print(newsgroups.columns)
