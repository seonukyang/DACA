#비모수 통계분석 - RUN검정
#1.모듈 및 데이터 탑재
from statsmodels.sandbox.stats.runs import Runs
import numpy as np
x = [1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,0,0]
x = np.array(x)

#2. RUN 검정 분석
print(Runs(x).runs_test())
#결과 (Z값, P값) 유의수준 0.1
