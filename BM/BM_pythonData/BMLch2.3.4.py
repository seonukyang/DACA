import pandas as pd
sales_data = {'화장품': [300,274,150,524,211],
                '의류': [773,657,699,324,487],
                '식음료': [362,131,593,348,98],
                '전자제품':[458,667,123,521,662]}
columns_list = ['화장품','의류','식음료','전자제품']
index_list = ['2014','2015','2016','2017','2018']
df_store = pd.DataFrame(data = sales_data, columns = columns_list, index = index_list)
print(df_store)

#평균 출력
print(df_store.mean())
print(df_store.mean()[3]) #해당 결과의 index 3번의 결과를 가져온다.
#표준편차 출력
print(df_store.std())

#인덱스 별로 평균 계산하기
print((df_store.mean(axis=1))) #1은 행별, 0은 열별 계산이다.

#통계량
print(df_store.describe())

#인덱스 명을 가지고 해당 행을 가져온다.
print(df_store.loc['2016'])
print(df_store.loc['2016'][3],type(df_store.loc['2016'][3])) #이러면 일단은 int값으로 가져온다.

#컬럼명과 인덱스로 가져오기
print(df_store['화장품']['2016':'2018'])
print(df_store['화장품']['2016':'2018'][2]) #이렇게 추출한 쪼개진 데이터 프레임에서 몇 번째 값을 지정할 수 있다.
