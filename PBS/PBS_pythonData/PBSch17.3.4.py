#비모수 통계분석 - 동질성 - Friedman
#1. 모듈 및 데이터 탑재
from scipy.stats import friedmanchisquare
#아래의 변수들은 열의 기준이다. 실험마다 집단이 매긴 순위이다.
a = [1,2,1,1,2]
b = [3,3,3,2,1]
c = [2,4,4,4,3]
d = [4,1,2,3,4]

#2. Friedman 검정 분석
print(friedmanchisquare(a,b,c,d))
#결과 검정통계량, p값.  유의수준은 0.05