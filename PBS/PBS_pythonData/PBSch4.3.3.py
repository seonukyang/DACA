# 원그래프
#1. 모듈 및 데이터 탑재
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')

#2. 한글 깨짐현상 방지
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

#3. 구매유형별 고객 그룹화 하기
groupby_구매유형 = df.groupby('구매유형') #구매유형이 같은 데이터끼리 그룹으로 묶임
groupby_구매유형.describe() #기술 통계량을 확인할 수 있다.
print(groupby_구매유형.describe())

#4. 파이차트 속성 지정
#파이차트 함수에 넣기 위해 사전에 지정해줌
labels = ['1','2','3','4'] #라벨 지정, 그룹의 이름
sizes = [43,317,144,496] #크기 지정, 각 그룹별 크기, 고객수 count참조
color = ['yellowgreen','gold','lightskyblue','lightcoral'] #색지정
explode = (0,0.1,0,0) #특정 그룹에 대한 시각적인 분리. 0이면 그룹끼리 붙고 같이 커질 수록 분리된 피자조각 같다.

#5. 파이차트 작성하기
plt.pie(sizes, explode = explode, labels = labels, colors = color, autopct='%1.1f%%', shadow = True, startangle=90)
#autopct는 그래프에 비율을 표기하는 부분이다. 정규표현식으로 %1.1f%% 즉 00.0%로 표현하겠다는 것이다.
plt.axis('equal')
plt.title('구매유형에 따른 고객 분포')
print(plt.show())