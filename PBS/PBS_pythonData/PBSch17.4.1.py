#비모수 통계분석 - 상관성 - Kendall 서열상관 분석
#1. 모듈 및 함수 불러오기
from scipy.stats import kendalltau
#아래의 변수들은 변수에 따른 집단들의 순위이다.
x = [3,4,5,2,1]
y = [3,2,1,4,5]
z = [5,3,1,2,4]

#2. Kendall 검정 분석 결과 출력
print(kendalltau(x,y,z))
#결과 상관계수, p값
