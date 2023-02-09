#포지셔닝 분석 - 다차원 척도법 - 이상점-경쟁구조 분석 과정
#1. 모듈 및 데이터 탑재
import pandas as pd
from sklearn.manifold import MDS
df1 = pd.read_csv('MDS1.csv', sep=',', encoding='CP949')
df3 = pd.read_csv('MDS3.csv', sep=',', encoding='CP949')

#2. 다차원척도법 분석(좌표 값 계산)
clf = MDS(n_components=2, random_state=123) #.분석을 위한 모듈을 사전 설정이라 생각하자. 뒤에 fit을 해주면 바로 분석하는 식
X_mds1 = clf.fit_transform(df1.loc[:,'이미지':'가격만족도']) 
X_mds3 = clf.fit_transform(df3.loc[:,'A쇼핑':'G쇼핑']) 
print('자극점 차원좌표\n',X_mds1)
print('이상점 차원좌표\n',X_mds3)

#이상점-경쟁구조 그래프 그리기
#3. 모듈 및 패키지 불러오기
import matplotlib
import matplotlib.pylab as plt

#4. 한글깨짐현상 방지
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

#5. 이상점-경쟁구조 시각화
labels=df1.shop
for label, x_count, y_count in zip(labels, X_mds1[:,0], X_mds1[:,1]):
    plt.annotate(label,
                xycoords='data',
                textcoords='offset points',
                xy=(x_count, y_count),
                xytext = (-5,5))
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.scatter(X_mds1[:,0],X_mds1[:,1])
plt.scatter(X_mds3[:,0],X_mds3[:,1], label='이상점') #좌표점들에 대한 범례용 라벨도 가능
plt.scatter(X_mds3[:,0].mean(),X_mds3[:,1].mean(), label='평균 이상점')
plt.legend(loc='upper right') #점들의 라벨을 upper right로 지정해준다.
plt.show()