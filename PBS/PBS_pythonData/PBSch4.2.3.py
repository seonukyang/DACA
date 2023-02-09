#1. 모듈 및 데이터 탑재
import pandas as pd
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')

#2. 데이터 분할하기
Under_500 = df[df.고객ID<=500]
Upper_500 = df[df.고객ID>500]

#3. 불러온 데이터 확인하기
print(Under_500.tail())
print(Upper_500.tail())

#데이터 추가하기  행 끼리 합치기
df_join=Under_500.append(Upper_500, ignore_index=True) #ignore_index가 True면 합쳐지는 데이터에 새로운 index를 부여한다. false면 기존의 index유지
print(df_join)

#데이터 병합하기  열 끼리 합치기

#필드 추출하기
df_1 = df[['고객ID','방문빈도']]
df_2 = df[['고객ID','총_매출액']]
df_3 = df[['고객ID','고객등급']]

#데이터 병합하기
df_merge = df_1.merge(df_2)  #공통된 컬럼인 고객ID는 겹쳐지고 총 3개의 컬럼을 가지게 된다. 
#merge안에 있는 컬럼들이 오른쪽으로 붙게 된다. groupby 할지 안할지, 어느쪽으로 추가 할 지도 설정할 수 있나보다.
# df_merge2 = df_1.merge(df_2, df_3) 이건 안되나 보다
print(df_merge.head())
#print(df_merge2)
