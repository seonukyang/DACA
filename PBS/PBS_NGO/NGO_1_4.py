#NGO_1 상자그림 
#1 모듈 및 데이터 탑재
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#2 한글깨짐 현상 방지
matplotlib.rcParams['font.family']='Malgun Gothic' 
matplotlib.rcParams['axes.unicode_minus']=False
#2-1 파생변수 나누기
df_1=df[['SEX','PAY_SUM_PAYMENTAMOUNT']].dropna()
man = df_1[df_1.SEX==1]
woman = df_1[df_1.SEX==2]
man_pay = man['PAY_SUM_PAYMENTAMOUNT'].values
woman_pay = woman['PAY_SUM_PAYMENTAMOUNT'].values
both_pay = df_1['PAY_SUM_PAYMENTAMOUNT'].values
#3-1. 상자그림 작성하기
plt.boxplot([man_pay,woman_pay,both_pay], labels=['남성', '여성','전체'])
plt.title("성별x납입총후원금액 상자그림")
plt.legend() #범례 표시
plt.ylabel('총후원금액(단위 : 백만)') #y축 이름
plt.xticks(fontsize = 12) #글씨 크기 설정
plt.yticks(fontsize = 12)
#3-2 이미지 저장
plt.savefig('1.box성별x납입총후원금액.png',dpi=200,edgecolor='blue', bbox_inches='tight',pad_inches=0.3)

#plt.show()