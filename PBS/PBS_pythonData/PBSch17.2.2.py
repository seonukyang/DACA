#비모수 통계분석 - Kolmogorov-Smirnov 검정 분석
#1. 모듈 및 데이터 탑재
from statsmodels.stats.diagnostic import kstest_normal
import numpy as np
x = [88,75,79,84,68,51,70,75,88,90,92,88,63,72,94,80,78,98,81,67,85,87,79,81,
    85,48,79,86,53,100,87,80,80,32,60,75,62,82,40,57]
print(x)
x = np.array(x)
print(x)
#2. Kolmogorov-Smirnov 검정 분석
#kstest_normal(x, dist='norm')
print(kstest_normal(x, dist='norm')) #norm 정규분포, exp 지수분포 두가지만 설정
