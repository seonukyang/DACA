# 산점도
#1. 모듈 및 데이터 탑재
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')

#2. 한글 깨짐현상 방지
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

#3. 산점도 작성하기
df.plot.scatter(x='방문빈도', y='총_매출액', grid=True, title='방문 빈도와 총 매출액간 관계')
