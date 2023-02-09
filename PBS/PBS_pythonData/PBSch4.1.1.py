#모듈 및 데이터 탑제
import pandas as pd
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')
#print(df)

#2. 무작위 표본추출하기
data_temp = df.sample(n=10, replace=False, random_state=123)
#smaple(n=추출할 데이터의 크기, replace= 복원추출 유무,True면 복원추출 False면 비복원추출, random_state는 무작위 표본추출 시 사용할 난수 값)
print(data_temp)
