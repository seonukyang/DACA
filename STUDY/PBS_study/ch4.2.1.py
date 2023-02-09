import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Ashopping.csv',encoding='CP949')
print(df.head())

#2. 한글깨짐현상 방지
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

#파이차트
#구매유형별 고객 그룹화하기
groupby_거주지역 = df.groupby('거주지역')
print(groupby_거주지역.describe())
print(df['거주지역'].value_counts()/len(df['거주지역']))

#파이차트 속정 지정
labels = ['1','2','3','4']
sizes = [43,317,144,496]
colors = ['yellowgreen','gold','lightskyblue','lightcoral']
explode = (0,0.1,0,0)

#파이차트 작성하기
plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct='%1.1f%%', shadow = True, startangle = 90)
plt.axis('equal')
plt.title('구매유형에 따른 고객 분포')
plt.show()
plt.clf()