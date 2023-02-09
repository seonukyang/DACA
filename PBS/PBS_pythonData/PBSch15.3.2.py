#포지셔닝 분석 - 상응분석 - 다중 상응분석
#1. 모듈 및 데이터 탑재
import pandas as pd
import prince
cor = pd.read_csv('Correspondence.csv', sep=',',encoding='CP949')
X = cor[['resort','slope','traffic','lodging','etc']]
#빈도교차표를 만들 수 없으니 필요가 없나보다
#2. 다중상응분석 좌표값
mca = prince.MCA(n_components=2).fit(X)
print('\n각 변수별 차원좌표\n', mca.column_coordinates(X))

#그래프로 표현하기
#3. 모듈 및 패키지 불러오기
import matplotlib

#4. 한글깨짐현상 방지
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

#5. 다중 상응분석 시각화
mca.plot_coordinates(X=X, show_column_labels=True)