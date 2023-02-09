#비모수 통계분석 - Wilcoxon 부호-순위
#1. 모듈 및 데이터 탑재
from scipy.stats import wilcoxon
x = [10,30,9,21,35,12,17]
y = [8,27,16,25,30,13,11]

#2. 부호 순위 검정 분석
print(wilcoxon(x,y))
#결과 검정통계량= wilcoxon 부호-서열 통계량, p값

