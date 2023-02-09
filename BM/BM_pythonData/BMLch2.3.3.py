#[3]pandas
import pandas as pd
series = pd.Series([1,2,3,4,5])
print(series, type(series))

#리스트를 활용한 데이터프로엠 생성
columns_list = ['고객이름','나이','등급']
value_list = [['원종호',26,'Silver'],['김수현',23,'Gold']]
pd1 = pd.DataFrame(data = value_list, columns = columns_list)
print(pd1)
print(pd1['나이'],type(pd1['나이']))
print(pd1['나이'][0],type(pd1['나이'][0]))
print(pd1['나이'][0] + pd1['나이'][1])
print(pd1[columns_list[1]][0])
#딕셔너리를 활용한 데이터프레임 생성


