import pandas as pd

series = pd.Series([1,2,3,4,5]) #pandas의 1차원 배열이다. 자동으로 인덱스가 부여된다.
print(series)

columns_list=['name','age','grade']
value_list = [['원종호','26','Silver'],['김수현','23','Gold']]
DataFrame1 = pd.DataFrame(data = value_list, columns = columns_list)
print(DataFrame1)
dict = {'고객이름':['원종호','김수현'],'age':['26','23'],'grade':['Selver','Gold']}
DataFrame2 = pd.DataFrame(dict)
print(DataFrame2)

