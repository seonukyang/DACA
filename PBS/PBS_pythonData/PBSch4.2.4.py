import pandas as pd
import numpy as np
df = pd.read_csv('Ashopping.csv', sep=',', encoding ='CP949')

#2 표현형태 변환하기
df['남녀']=np.where(df.성별 ==0, '남자', '여자') #where(조건, true, false)

#3. 데이터 확인하기
print(df[['고객ID','성별','남녀']])

#척도 변환하기
df['New고객등급']=np.where(df.총_매출액 >= 50000000, '상', np.where(df.총_매출액>=3000000,'중','하'))

print(df[['고객ID','총_매출액','New고객등급']])

#모델링을 활옹해 파생변수 생성하기
df['New_1회_평균매출액'] = df['총_매출액']/df['방문빈도']

print(df[['고객ID','총_매출액','New_1회_평균매출액']])

#거래이력 요약을 통한 파생변수 생성하기
print("df.이탈여부==1 : ",df.이탈여부==1) #df에서 이탈여부가 1인 애들은 true, 아닌애들은 false을 output한다.
print("df[df.이탈여부==1] : ",df[df.이탈여부==1]) #df.이탈여부==1에서 true가 된 애들을 가지고 df를 재정의한다.
churn_customer = df[df.이탈여부==1]
Non_churn_customer = df[df.이탈여부==0]

print("churn_customer.총_매출액 : ",churn_customer.총_매출액) #기존의 df에서 조건에 맞는애들만 필터링 했기때문에 기존 df의 속성인 총_매출액이 존재한다.
print("Non_churn_customer.총_매출액 : ",Non_churn_customer.총_매출액)
print(sum(churn_customer.총_매출액))
print(sum(Non_churn_customer.총_매출액))