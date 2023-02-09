import pandas as pd
import numpy as np

dep_data = {'명동점':[59060,49296,62015,48621,46712,31216,38467],
            '분당점':[9312,1267,6893,7226,8878,13622,18228],
            '광주점':[2627,4145,4088,4321,4679,4994,5544],
            '부산점':[14211,11071,11234,15424,12146,39415,57866],
            '송도점':[np.nan,np.nan,9912,9224,8395,9786,9667]}
col_list = ['명동점','분당점','송도점','부산점','광주점']
index_list = ['2011','2012','2013','2014','2015','2016','2017']

df_store = pd.DataFrame(data=dep_data, columns=col_list, index=index_list)
print(df_store)
print(df_store[0:2])
print(df_store.loc['2016'])

print(df_store['명동점'])

print(df_store['명동점']['2013':'2015'])

#print(df_store['명동점':'부산점']['2012':'2016'])