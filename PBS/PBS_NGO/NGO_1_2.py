#NGO - 통계그래프 - 산점도
#1. 모듈 및 데이터 탑재
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
df = pd.read_csv('1. NGO.csv', sep=',', encoding='CP949')

#print(df.loc[287:289,'LONGEVITY_D'])
#2. 한글 깨짐현상 방지
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False

#3. 산점도 작성하기
#3-1-0 이상값 처리 - AGE
dropin = df[df['AGE']==0].index
dfin = df.drop(dropin)
#3-1-1. 연령x후원횟수 산점도
plt.scatter(dfin['AGE'], dfin['PAY_NUM'])
#3-1-2 각종 옵션설정
plt.title("나이x정기후원총횟수 파이차트")
plt.legend() #범례 표시
plt.grid() #격자선 추가
plt.xlabel('나이') #x축 이름
plt.ylabel('정기후원총횟수') #y축 이름
plt.xticks(fontsize = 14) #글씨 크기 설정
plt.yticks(fontsize = 14)
#3-1-3 이미지 저장
from pylab import figure, axes, pie, title, savefig
plt.savefig('1.sca연령대x정기후원총횟수.png',dpi=200,edgecolor='blue', bbox_inches='tight',pad_inches=0.3)
#3-1-3 plt 초기화
plt.clf()

#3-2-0 특이값 제거 - 가입일수
df_1=df.drop(288)
#3-2-1. 가입일수(일)x후원횟수 산점도
plt.scatter(df_1['LONGEVITY_D'], df_1['PAY_NUM'])
#3-2-2 각종 옵션설정
plt.title("가입일수x정기후원총횟수 파이차트")
plt.legend() #범례 표시
plt.grid() #격자선 추가
plt.xlabel('가입일수') #x축 이름
plt.ylabel('정기후원총횟수') #y축 이름
plt.xticks(fontsize = 14) #글씨 크기 설정
plt.yticks(fontsize = 14)
#3-2-3 이미지 저장
plt.savefig('1.sca가입일수x정기후원총횟수.png',dpi=200,edgecolor='blue', bbox_inches='tight',pad_inches=0.3)
#3-2-4 plt 초기화
plt.clf()

#3-3. 정기후원횟수x일시후원횟수 산점도
plt.scatter(df['PAY_NUM_REGULAR'], df['PAY_NUM_ONETIME'])
#3-3-1 각종 옵션설정
plt.title("정기후원횟수x일시후원횟수 파이차트")
plt.legend() #범례 표시
plt.grid() #격자선 추가
plt.xlabel('정기후원횟수') #x축 이름
plt.ylabel('일시후원횟수') #y축 이름
plt.xticks(fontsize = 14) #글씨 크기 설정
plt.yticks(fontsize = 14)
#3-3-2 이미지 저장
plt.savefig('1.sca정기후원횟수x일시후원횟수.png',dpi=200,edgecolor='blue', bbox_inches='tight',pad_inches=0.3)
#3-3-3 plt 초기화


