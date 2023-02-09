#1 모듈 및 데이터 탑재
import pandas as pd
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')

#2. 레코드 추출하기
churn_customer = df[df.이탈여부==1]

#3. 불러온 데이터 확인하기
print(churn_customer.head())

#필드 추출하기
df_1 = df[['고객ID','방문빈도']]
print(df_1.head())

#2. 무작위 표본추출하기
data_temp = df.sample(n=10, replace=False, random_state=123)
#smaple(n=추출할 데이터의 크기, replace= 복원추출 유무,True면 복원추출 False면 비복원추출, random_state는 무작위 표본추출 시 사용할 난수 값)
print(data_temp)