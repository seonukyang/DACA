#NGO_1 히스토그램 
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
dropin = df[df['AGE']==0].index
dfin = df.drop(dropin)
#3. 파생변수 만들기
df_1 = dfin[['AGE','SEX']]
man = df_1[df_1.SEX==1]
woman = df_1[df_1.SEX==2]

#4. 히스토그램 작성-남성
plt.hist(man['AGE'], alpha=0.4,bins=np.arange(0,90,10), rwidth=1, color='blue', label='남성 연령대')

#4-1 각종 옵션설정
plt.title("남성 연령대별 후원자 수")
plt.legend() #범례 표시
plt.grid() #격자선 추가
plt.xlabel('연령대') #x축 이름
plt.ylabel('회원수') #y축 이름
plt.xticks(fontsize = 14) #글씨 크기 설정
plt.yticks(fontsize = 14)

#4-2 이미지 저장
from pylab import figure, axes, pie, title, savefig
plt.savefig('1.hist남성연령대후원자.png',dpi=200,edgecolor='blue', bbox_inches='tight',pad_inches=0.3)

#4-3 plt 초기화
plt.clf()

#5. 히스토그램 작성-여성
plt.hist(woman['AGE'], alpha=0.4, bins=np.arange(0,90,10), rwidth=1, color='red', label='여성 연령대')
#5-1 각종 옵션설정
plt.title("여성 연령대별 후원자 수")
plt.legend() #범례 표시
plt.grid() #격자선 추가
plt.xlabel('연령대') #x축 이름
plt.ylabel('회원수') #y축 이름
plt.xticks(fontsize = 14) #글씨 크기 설정
plt.yticks(fontsize = 14)
#4-2 이미지 저장
from pylab import figure, axes, pie, title, savefig
plt.savefig('1.hist여성연령대후원자.png',dpi=200,edgecolor='red', bbox_inches='tight',pad_inches=0.3)