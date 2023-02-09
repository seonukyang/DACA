#run 검정 분석 과정
from statsmodels.sandbox.stats.runs import Runs
import numpy as np
x = [1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,0,0]
x = np.array(x)

Runs(x).runs_test()

#kolmogorov-Smirnov 검정 (단일표본)
