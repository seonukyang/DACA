# 히스토그램
#1 모듈 및 데이터 탑재
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt #히스토 그램 작성하기 위해
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')

#2 한글깨짐 현상 방지
matplotlib.rcParams['font.family']='Malgun Gothic' #글꼴을 맑음고딕으로 설정
matplotlib.rcParams['axes.unicode_minus']=False # -기호가 깨지지 않도록 해줌
#print(df['서비스_만족도'].head())
#3 히스토그램 작성하기
#%matplotlib inline #주비터 노트북에서 사용하고 있는 웹브라우저에 직접 시각화 출력물 표현
plt.hist(df['서비스_만족도'], alpha=0.4, bins=7, rwidth=1, color='red', label='서비스만족도')
#데이터 값, alpha=그래프 투명도, bins=계급의 계수, rwidth=bar사이의 간격, color=bar의 색상, label=범례표시 텍스트)

#4 각종 옵션설정
plt.legend() #범례 표시
plt.grid() #격자선 추가
plt.xlabel('서비스만족도') #x축 이름
plt.ylabel('빈도') #y축 이름
plt.xticks(fontsize = 14) #글씨 크기 설정
plt.yticks(fontsize = 14)
print(plt.show())
