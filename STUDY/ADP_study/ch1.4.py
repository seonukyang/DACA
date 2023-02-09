#판다스 다루기
import pandas as pd
from pandas.core.algorithms import value_counts
titanic_df = pd.read_csv('titanic_train.csv')
print(titanic_df.head(3))
print('titanic 변수 type:',type(titanic_df))
print(titanic_df.describe())
print('titanic의 크기 : ',titanic_df.shape)
print(titanic_df.info())
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts[1])
print(value_counts[3])
print(type(titanic_df['Pclass']))
print(titanic_df['Pclass'][2])

value_counts = titanic_df['Pclass'].value_counts()
print(type(value_counts))
print(value_counts)

import numpy as np
col_name1=['col1']
list1 = [1,2,3]
array1 = np.array(list1)
print('array1 shape:', array1.shape)
#리스트를 이용해 DataFrame 생성
df_list1 = pd.DataFrame(list1, columns=col_name1)
print('1차원 리스트로 만든 DataFrame :\n',df_list1)
#넘파이 ndarray를 이용해 DataFrame 생성
df_array1 = pd.DataFrame(array1, columns=col_name1)
print('1차원 ndarray로 만든 DataFrame:\n',df_array1)

col_name2=['col1','col2','col3']
list2 = [[1,2,3],[4,5,6]]
array2 = np.array(list2)
print('array2 shape :',array2.shape)
df_list2 = pd.DataFrame(list2, columns=col_name2)
print('2차원 리스트로 만든 Dataframe : \n',df_list2)
df_array2 = pd.DataFrame(array2, columns=col_name2)
print('2차원 ndarray로 만든 DataFrame:\n',df_array2)
df_array2['col4'] = 0
print('2차원 ndarray로 만든 DataFrame:\n',df_array2)

#Key는 문자열 컬럼명으로 매핑, Value는 리스트 형(또는 ndarray)칼럼 데이터로 매핑
dict={'col1':[1,11], 'col2':[2,22],'col3':[3,33]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame : \n',df_dict)

#DataFrame을 넘파이 ndarray, 리스트, 딕셔너리로 변환하기
#DataFrame을 ndarray로 변환
array3 = df_dict['col1'].values
print('df_dict.values 타입:',type(array3))
print('df_dict.values shape : ',array3.shape)
print(array3)

list3 = df_dict.values.tolist()
print('df_dict.values.tolist() 타입',type(list3))
print(list3)

dict3 = df_dict.to_dict('list')
print('df_dict.to_dict() 타입 : ',type(dict3))
print(dict3)

titanic_df['Age_0'] = 0
print(titanic_df.head(3))
titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch']+1
print(titanic_df.head())
titanic_df['Age_by_10'] = titanic_df['Age_by_10'] + 100
print(titanic_df.head())

#DataFrame 데이터 삭제
titanic_drop_df = titanic_df.drop('Age_0',axis=1)
print(titanic_drop_df.head())

drop_result = titanic_df.drop(['Age_0','Age_by_10','Family_No'], axis=1, inplace=True)
print('inplace=True로 drop한 후 반환된 값 : ', drop_result)

pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth', 15)
print('### before axis 0 drop ###')
print(titanic_df.head())
titanic_df.drop([0,1,2], axis=0, inplace=True)

print('### after axis 0 drop ###')
print(titanic_df.head())

#index
titanic_df=pd.read_csv('titanic_train.csv')
#index 객체 추출
indexes = titanic_df.index
print(indexes)
#index 객체를 실제 값 array로 변환
print('Index 객체 array값',indexes.values)

series_fair = titanic_df['Fare']
list_fair = series_fair.tolist()
print(type(series_fair))
print('Fair Series max 값 : ',series_fair.max())
print(type(list_fair))
print(series_fair)
#print('List_fair max 값 : ',list_fair.max())
print('Fair Series + 3\n',(series_fair + 3).head())

titanic_reset_df = titanic_df.reset_index(inplace=False)
print(titanic_reset_df)

value_counts = titanic_df['Pclass'].value_counts()
new_value_counts = value_counts.reset_index(inplace=False)
print(new_value_counts)

#데이터 셀렉션 및 필터링
print('단일 칼럼 데이터 추출:\n', titanic_df['Pclass'].head())
print('\n여러 칼럼의 데이터 추출:\n', titanic_df[['Survived','Pclass']].head())
print('titanic_df[[\'Survived\',\'Pclass\']][1:5] :\n',titanic_df[['Survived','Pclass']][1:5])

print(titanic_df[titanic_df['Pclass']==3].head())

data = {'Name':['Chulmin','Eunkyung','Jinwoong','Soobeom'],
        'Year':[2011,2016,2015,2015],
        'Gender':['Male','Female','Male','Male']
        }
data_df = pd.DataFrame(data)

#iloc
data_df_reset = data_df.reset_index()
data_df_reset = data_df_reset.rename(columns={'index':'old_index'})
data_df_reset.index = data_df_reset.index+1
print(data_df_reset)
print(data_df_reset.iloc[0,0])

#loc
print(data_df_reset.loc[data_df_reset.index,'Name'])
print(data_df_reset.loc[1:2,'Name'])

#불린 인덱싱
titanic_df = pd.read_csv('titanic_train.csv')
titanic_boolean = titanic_df[titanic_df['Age']>60]
print(len(titanic_boolean))
print(titanic_df[titanic_df['Age']>60][['Name','Age']].head())
print(titanic_df['Age']>60)
titanic_df.loc[titanic_df['Age']>60 , ['Name','Age']]

cond1 = titanic_df['Age']>60
cond2 = titanic_df['Pclass']==1
cond3 = titanic_df['Sex']=='female'
print(titanic_df[cond1&cond2&cond3])

#정렬, Aggregation 함수, GroupBy 적용
titanic_sorted = titanic_df.sort_values(by=['Name'])
print(titanic_sorted.head())
titanic_sorted = titanic_df.sort_values(by=['Pclass','Name'], ascending=False)
print(titanic_sorted)
print(titanic_df.count())

print(titanic_df[['Age','Fare']].mean())

titanic_groupby = titanic_df.groupby(by='Pclass').count()
print(titanic_groupby)
print(titanic_df.groupby('Pclass')[['Age','Survived']].agg([max,min]))

print(titanic_df.isnull().sum())