#NGO_1 파이차트 
#1 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#2 한글깨짐 현상 방지
matplotlib.rcParams['font.family']='Malgun Gothic' 
matplotlib.rcParams['axes.unicode_minus']=False 
#2-1 이상값 처리 - AGE
dfin = df[df['AGE']!=0]

#3. 파생변수 만들기
df_1 = dfin[['AGE','SEX']]
#3-1. 연령대, 성별 구분
df_1['연령대']=pd.cut(df_1.AGE,[0,9,19,29,39,49,59,69,79,89],
labels=['유아','10대','20대','30대','40대','50대','60대','70대','80대'])
man = df_1[df_1.SEX==1]
woman = df_1[df_1.SEX==2]

#4-1-0. 파이차트 만들기 - 연령대
groupby_연령대 = df_1.groupby('연령대').count() #구매유형이 같은 데이터끼리 그룹으로 묶임
groupby_연령대 = groupby_연령대['AGE']
#4-1-1. 파이차트 속성 지정
labels = ['유아','10대','20대','30대','40대','50대','60대','70대','80대']
color = ['yellowgreen','gold','lightskyblue','lightcoral','blue','red','white','black','pink']
explode = (0,0,0,0,0.1,0,0,0,0) 
size = groupby_연령대.values
#4-1-2. 파이차트 작성하기
plt.pie(size, explode = explode, labels = labels, colors = color, autopct='%1.1f%%', shadow = False, startangle=90)
plt.axis('equal')
plt.title('연령대별 후원자 분포',position=(0.1, 3))
#4-1-3 이미지 저장
plt.savefig('1.pie연령대별후원자.png',dpi=200,edgecolor='black', bbox_inches='tight',pad_inches=0.3)
#4-1-4 plt 초기화
plt.clf()

#4-2-0. 파이차트 만들기 - 남성연령대
mangroupby_연령대 = man.groupby('연령대').count() #구매유형이 같은 데이터끼리 그룹으로 묶임
mangroupby_연령대 = mangroupby_연령대['AGE']
#4-2-1. 파이차트 속성 지정
labels = ['유아','10대','20대','30대','40대','50대','60대','70대','80대']
color = ['yellowgreen','gold','lightskyblue','lightcoral','blue','red','white','black','pink']
explode = (0,0,0,0,0.1,0,0,0,0) 
size = mangroupby_연령대.values
#4-2-2. 파이차트 작성하기
plt.pie(size, explode = explode, labels = labels, colors = color, autopct='%1.1f%%', shadow = False, startangle=90)
plt.axis('equal')
plt.title('남성 연령대별 후원자 분포',position=(0.1, 3))
#4-2-3 이미지 저장
plt.savefig('1.pie남성연령대별후원자.png',dpi=200,edgecolor='blue', bbox_inches='tight',pad_inches=0.3)
#4-1-4 plt 초기화
plt.clf()

#4-3-0. 파이차트 만들기 - 여성연령대
womangroupby_연령대 = woman.groupby('연령대').count() #구매유형이 같은 데이터끼리 그룹으로 묶임
womangroupby_연령대 = womangroupby_연령대['AGE']
#4-3-1. 파이차트 속성 지정
labels = ['유아','10대','20대','30대','40대','50대','60대','70대','80대']
color = ['yellowgreen','gold','lightskyblue','lightcoral','blue','red','white','black','pink']
explode = (0,0,0,0,0.1,0,0,0,0) 
size = womangroupby_연령대.values
#4-3-2. 파이차트 작성하기
plt.pie(size, explode = explode, labels = labels, colors = color, autopct='%1.1f%%', shadow = False, startangle=90)
plt.axis('equal')
plt.title('여성 연령대별 후원자 분포',position=(0.1, 3))
#4-3-3 이미지 저장
plt.savefig('1.pie여성연령대별후원자.png',dpi=200,edgecolor='blue', bbox_inches='tight',pad_inches=0.3)
#4-3-4 plt 초기화
plt.clf()