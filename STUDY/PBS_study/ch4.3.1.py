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

#3. 히스토그램 작성하기
plt.hist(df['서비스_만족도'], alpha=0.4, bins=7, rwidth=1, color='red', label='서비스만족도')

#4. 각종 옵션설정
plt.legend()
plt.grid()
plt.xlabel('서비스 만족도')
plt.ylabel('빈도')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()
plt.clf()

#산점도 작성하기
df.plot.scatter(x='방문빈도',y='총_매출액',grid=True, title='방문 빈도와 총 매출액간 관계')
plt.clf()

#파이차트
#구매유형별 고객 그룹화하기
groupby_구매유형 = df.groupby('구매유형')
print(groupby_구매유형.describe())

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

#상자 그림
sns.boxplot(x='성별',y='총_할인_금액', hue='성별', data=df)
