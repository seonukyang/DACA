#비모수 통계분석 - RUN검정
#1.모듈 및 데이터 탑재
from statsmodels.sandbox.stats.runs import Runs
import numpy as np
x = [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1]
x = np.array(x)

#2. RUN 검정 분석
print(Runs(x).runs_test())
#결과 (Z값, P값) 유의수준 0.1


#비모수 통계분석 - Kolmogorov-Smirnov 검정 분석
#1. 모듈 및 데이터 탑재
from statsmodels.stats.diagnostic import kstest_normal
import numpy as np
x = [20,50,32,20,55,30,20,20,20,45,50,20,41,20,45,20,20,35,25,45,40,60,45,45]
x = np.array(x)

#2. Kolmogorov-Smirnov 검정 분석
#kstest_normal(x, dist='norm')
print(kstest_normal(x, dist='norm')) #norm 정규분포, exp 지수분포 두가지만 설정
