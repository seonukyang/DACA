import pandas as pd
from scipy import stats
from pingouin import partial_corr
df = pd.read_csv('Ashopping.csv',encoding='CP949')
df1 = df[['총_매출액','방문빈도','구매_카테고리_수']]

#피어슨 상관계수 출력
print(stats.pearsonr(df1.총_매출액, df1.방문빈도))
print(stats.pearsonr(df1.총_매출액, df1.구매_카테고리_수))
print(stats.pearsonr(df1.방문빈도, df1.구매_카테고리_수))

print(df1.corr(method='pearson'))

#편 상관계수 출력
print(partial_corr(data=df1, x='총_매출액',y='방문빈도',covar='구매_카테고리_수'))

#스피어만 상관관계 분석
print(stats.spearmanr(df['1회_평균매출액'], df1['방문빈도']))

from sklearn.cross_decomposition import CCA
data = pd.read_csv('CCA.csv', encoding='CP949')
U = data[['품질','가격','디자인']]
V = data[['직원 서비스','매장 시설','고객관리']]

#정준변수 구하기
cca = CCA(n_components=1).fit(U,V)
U_c, V_c = cca.transform(U, V)
U_c1 = pd.DataFrame(U_c)[0]
V_c1 = pd.DataFrame(V_c)[0]
print(U_c)
print(V_c)

#정준 상관계수 구하기
CC1 = stats.pearsonr(U_c1, V_c1)
print('제1정준상관계수 : ',CC1)



