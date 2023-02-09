#단순 상응분석
import pandas as pd
import prince
cor = pd.read_csv('Correspondence.csv', sep=',', encoding='CP949')
X  = pd.crosstab(cor.resort, cor.slope, margins=False)

#단순 상응분석(차원 좌표 값 계산)
ca = prince.CA(n_components=2).fit(X)
print('리조트 기준 차원좌표\n', ca.row_coordinates(X))
print('슬로프 기준 차원좌표\n', ca.column_coordinates(X))

import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'

ca.plot_coordinates(X=X)

