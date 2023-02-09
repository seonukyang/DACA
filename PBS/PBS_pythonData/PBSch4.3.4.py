#[4] 상자그림
#1. 모듈 및 데이터 탑재
import pandas as pd
import matplotlib
import seaborn as sns
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')

#2. 한글 깨짐현상 방지
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

#3. 상자그림 작성하기
sns.boxplot(x='성별',y='총_할인_금액',hue='성별',data=df)