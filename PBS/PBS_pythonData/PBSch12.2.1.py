#요인분석 - 탐색적 요인분석
#1. 모듈 및 데이터 탑재
import pandas as pd
from factor_analyzer import FactorAnalyzer
df = pd.read_csv('Ashopping.csv',sep=',',encoding='CP949')

#2. 변수 추출
X = df[['상품_품질','상품_다양성','가격_적절성','상품_진열_위치','상품_설명_표시','매장_청결성','공간_편의성','시야_확보성',
'음향_적절성','안내_표지판_설명']]

#3. 탐색적 요인 분석
fa = FactorAnalyzer(method='principal', n_factors=2, rotation='varimax').fit(X)
print('fa : ' , fa)
#methos 요인추출방법 principal 주성분분석법 ml 최대우도 요인추출법, f_factors 축약하고자 하는 요인의 수, rotation 요인의 회전 방식
#fit로 X 객체에 적합시킨다.

#4. 결과 출력
print('요인적재량 :\n', pd.DataFrame(fa.loadings_, index=X.columns)) #보기 편하게 imdex로 x의 컬럼명을 넣어줌
print('\n공통성 :\n', pd.DataFrame(fa.get_communalities(), index=X.columns))
ev, v=fa.get_eigenvalues() #고유값 계산, 고유값, 공통요인 고유값 두 개의 값을 ev, v에 저장한다.
print('\n고유값 :\n', pd.DataFrame(ev))
print('\n요인점수 :\n', fa.transform(X.dropna())) #.transform으로 요인점수 산출. 결측치 오류 대비 droppna()사용
