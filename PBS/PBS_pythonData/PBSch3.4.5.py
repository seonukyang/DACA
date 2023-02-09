import pandas as pd

sales_data = {'화장품':[300,274,150,524,211],
                '의류':[773,657,699,324,487],
                '식음료':[362,131,593,348,98],
                '전자제품':[458,667,123,524,662]}
columns_list = ['화장품','의류','식음료','전자제품']                
index_list = ['2014','2015','2016','2017','2018']

df_store = pd.DataFrame(data=sales_data, columns=columns_list, index=index_list)
print(df_store)

#평균출력
print(df_store.mean()) #dataframe을 그대로 집어넣으면 각 컬럼들의 평균을 보여준다.

#표준편차 출력
print(df_store.std())

#인덱스별 평균 구하기
print((df_store.mean(axis=1)))  #axis=1은 행을 기준으로 평균을 구한다. default는 axis=0로 열을 기준으로 평균구함

#dataframe의 기초통계량을 구한다.
df_store.describe() 
print(df_store.describe())