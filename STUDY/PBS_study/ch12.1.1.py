import pandas as pd
from factor_analyzer import FactorAnalyzer
df = pd.read_csv('Ashopping.csv', encoding='CP949')

#변수추출
X = df[['상품_품질','상품_다양성','가격_적절성','상품_진열_위치','상품_설명_표시','매장_청결성','공간_편의성',
'시야_확보성','음향_적절성','안내_표지판_설명']]

#탐색적 요인분석
fa = FactorAnalyzer(method='principal', n_factors=2, rotation='varimax').fit(X)

#결과출력
print('요인적재량\n', pd.DataFrame(fa.loadings_, index=X.columns))
print('공통성\n', pd.DataFrame(fa.get_communalities(), index=X.columns))
ev, v = fa.get_eigenvalues()
print('고유값 \n', pd.DataFrame(ev))
print('요인점수 \n', fa.transform(X.dropna()))

test = fa.transform(X.dropna())

