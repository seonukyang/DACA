#포지셔닝 분석 - 다차원 척도법 - 속성-경쟁구조 분석 과정
#1. 모듈 및 데이터 탑재
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.manifold import MDS
import numpy as np
df = pd.read_csv('MDS1.csv', sep=',', encoding='CP949')
df1 = df.loc[:, '이미지':'가격만족도']
df1 = (df1 - df1.min())/(df1.max()-df1.min()) #속성들을 정규화 한다. 시작점을 0으로, 최대값을 1로 하고 df1.max 비율로 조정

#2. 다차원척도법 분석(차원 좌표 값 계산)
clf = MDS(n_components=2, random_state=123).fit(df1)
X_mds = clf.fit_transform(df1) #fit_transform으로 () 안의 값을 X_mds에 저장한다는데?

#3. 속성 차원 좌표 값 계산
df1['차원1'] = X_mds[:,0]
df1['차원2'] = X_mds[:,1]
model = []
model.append(smf.ols(formula = '이미지 ~ 차원1 + 차원2', data=df1).fit())
model.append(smf.ols(formula = '접근성 ~ 차원1 + 차원2', data=df1).fit())
model.append(smf.ols(formula = '서비스 ~ 차원1 + 차원2', data=df1).fit())
model.append(smf.ols(formula = '친절성 ~ 차원1 + 차원2', data=df1).fit())
model.append(smf.ols(formula = '편의시설 ~ 차원1 + 차원2', data=df1).fit())
model.append(smf.ols(formula = '인지도 ~ 차원1 + 차원2', data=df1).fit())
model.append(smf.ols(formula = '가격만족도 ~ 차원1 + 차원2', data=df1).fit())
print('model : ', model)
속성 = []
for i in range(0,7,1):
    속성.append([model[i].params[1], model[i].params[2]])
속성 = np.array(속성)
print('속성', 속성)
자극점및속성 = np.hstack([X_mds,속성]) #열로 붙인다
print('자극점및속성',자극점및속성)

# 그래프로 포현하기
#4. 모듈 및 패키지 불러오기
import matplotlib
import matplotlib.pylab as plt

#5. 한글깨짐현상 방지
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

#6. 속성-경쟁구조 시각화
label = df.shop
for label, x_count, y_count in zip(label, 자극점및속성[:,0], 자극점및속성[:,1]): #단순히 라벨이름을 붙이는 과정이다.
    plt.annotate(label,
                xycoords='data',
                textcoords='offset points',
                xy=(x_count, y_count),
                xytext = (-5,5))
                
label2 = df1.columns
for label2, x_count, y_count in zip(label2, 자극점및속성[:,2], 자극점및속성[:,3]):
    plt.annotate(label2,
                xycoords='data',
                textcoords='offset points',
                xy=(x_count, y_count),
                xytext = (-5,5))
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.scatter(자극점및속성[:,0],자극점및속성[:,1]) #자극점 좌표 그래프
plt.scatter(자극점및속성[:,2],자극점및속성[:,3]) #좌표에 점을 찍는건 이쪽
plt.show()