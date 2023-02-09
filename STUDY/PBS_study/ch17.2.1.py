from statsmodels.stats.diagnostic import kstest_normal
import numpy as np
x = [70, 75, 80, 85, 50, 55, 65, 57, 56 ,97, 90, 100, 79, 
48, 32, 67, 85, 78 ,81, 80, 70, 51, 68, 84, 88, 78, 81, 57, 40, 62, 80, 32, 60, 75, 48, 79, 100, 87, 80, 98]
x = np.array(x)
print(kstest_normal(x,dist='norm'))
print(kstest_normal(x,dist='exp'))