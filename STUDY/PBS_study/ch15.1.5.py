#다중 상응분석
import pandas as pd
import prince
cor = pd.read_csv('Correspondence.csv', sep=',',encoding='CP949')
X = cor[['resort','slope','traffic','lodging','etc']]

mca = prince.MCA(n_components=2).fit(X)
print('각 변수별 차원좌표\n', mca.column_coordinates(X))

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
mca.plot_coordinates(X=X, show_column_labels=True)
