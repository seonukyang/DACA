import pandas as pd
from factor_analyzer import FactorAnalyzer
df = pd.read_csv('Ashopping (1).csv', encoding='CP949')
print(df.columns)
#변수추출
X = df[['프로모션_다양성', '상품_다양성', '혜택_다양성', '콘텐츠_다양성', '상품_품질']]

#탐색적 요인분석
fa = FactorAnalyzer(method='principal', n_factors=2, rotation='varimax').fit(X)

#결과출력
print('요인적재량\n', pd.DataFrame(fa.loadings_, index=X.columns))
print('\n공통성\n', pd.DataFrame(fa.get_communalities(), index=X.columns))
ev, v = fa.get_eigenvalues()
print('\n고유값 \n', pd.DataFrame(ev))
print('\n요인점수 \n', fa.transform(X.dropna()))

test = fa.transform(X.dropna())

