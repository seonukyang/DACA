#포지셔닝 분석 - 상응분석 - 단순 상응분석 
#1. 모듈 및 데이터 탑재
import pandas as pd
import prince
cor = pd.read_csv('Correspondence.csv', sep=',',encoding='CP949')
X=pd.crosstab(cor.resort, cor.slope, margins=False)

#2. 단순 상응분석(차원 좌표 값 계산)
ca = prince.CA(n_components=2).fit(X)
#print(ca) ca자체로는 볼 수가 없다. 행 변수들의 좌표. 같은 식으로 별도로 뽑아내야 한다.
print('리조트 기준 차원좌표\n', ca.row_coordinates(X))
print('\n슬로프 기준 차원좌표\n',ca.column_coordinates(X))

#상응분석을 그래프로 표현하기
#3. 패키지 불러오기
import matplotlib
import matplotlib.pylab as plt

#4. 한글깨짐현상 방지
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

#5. 단순 상응분석 시각화
#caplot = ca.plot_coordinates(X=X) #X=data 그냥 빈도교차표를 집어넣어도 알아서 그래프를 구해주나 보다. 
#caplot.show()


#ㅈ망. 범례만 모아놓은 라벨이 필요하다. 흉내만 낸거지 상응분석이 아니다.
resortC = ca.row_coordinates(X)
slopeC = ca.column_coordinates(X)
labelsR=['대명','리솜','무주','용평','한화']
labelsS=['slope-H','Slope-L','Slope-M']
for label, x_count, y_count in zip(labelsR, resortC.loc[:,0], resortC.loc[:,1]):
    plt.annotate(label,
                xycoords='data',
                textcoords='offset points',
                xy=(x_count, y_count),
                xytext = (-5,5))
for label, x_count, y_count in zip(labelsS, slopeC.loc[:,0], slopeC.loc[:,1]):
    plt.annotate(label,
                xycoords='data',
                textcoords='offset points',
                xy=(x_count, y_count),
                xytext = (-5,5))

plt.xlabel('Component 0')
plt.ylabel('Component 1')
plt.scatter(resortC.loc[:,0],resortC.loc[:,1], label='resort')
plt.scatter(slopeC.loc[:,0],slopeC.loc[:,1], label='slope') #좌표점들에 대한 범례용 라벨도 가능
plt.legend(loc='upper right') #점들의 라벨을 upper right로 지정해준다.
plt.show()
#하지만 이것이 단순 상응분석을 의미하지는 않는다.