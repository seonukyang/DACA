#포지셔닝 - 다차원척도법 - 단순 경쟁구조 분석 과정
#1. 모듈 및 데이터 탑재
import pandas as pd
from sklearn.manifold import MDS
df = pd.read_csv('MDS1.csv', sep=',',encoding='CP949')

#2. 다차원척도법 분석(차원 좌표 값 계산)
clf = MDS(n_components=2, random_state=123).fit(df.loc[:,'이미지':'가격만족도']) 
#(n_components=차원수, 난수) 에다가 df의 독립변수들을 넣어줌. 알아서 종속변수의 변수 개수만큼 좌표값을 구해준다.
X_mds = clf.fit_transform(df.loc[:,'이미지':'가격만족도'])
print(X_mds)

#단순 경쟁구조를 그래프로 표현하기
#3. 모듈 및 패키지 불러오기
import matplotlib
import matplotlib.pylab as plt

#4. 한글깨짐현상 방지
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

#5. 단순 경쟁구조 시각화
labels = df.shop
for label, x_count, y_count in zip(labels, X_mds[:,0], X_mds[:,1]):
    plt.annotate(label,
                xycoords='data',
                textcoords='offset points',
                xy=(x_count, y_count),
                xytext=(5,-5))
plt.xlabel('Dimension1')
plt.ylabel('Dimension2')
plt.scatter(X_mds[:,0],X_mds[:,1])
plt.show()