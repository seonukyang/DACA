#비모수 통계분석 - 동질성 - Mann-Whitney 검정
#1. 모듈 및 데이터 탑재
import pandas as pd
from scipy.stats import mannwhitneyu
x = [12,11,13,14,15]
y = [16,15,17,19,20]

#2. Mann-Whitney 검정 분석
print(mannwhitneyu(x,y))

#3. 생산량 평균 순위 출력
xy = pd.DataFrame(x+y) #두 집단이 단일 컬럼으로 묶인다. 순서는 x - y순이다.
xy['생산량 순위'] = xy.rank(ascending=False) #rank로 index를 대신한다.
xy['공장이름']=['A','A','A','A','A','B','B','B','B','B']
print(xy.groupby('공장이름').mean())
print(xy.groupby('공장이름').sum())
#결과 (U통계량, p값) p값으로 다르다는 것은 알 수 있는데 서열을 가르기 위해 순위를 매겼다.
