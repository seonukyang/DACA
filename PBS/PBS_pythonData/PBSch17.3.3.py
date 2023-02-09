#비모수 통계분석 - 동질성 - Kruskal-Wallis H 검정
#1. 모듈 및 함수 불러오기
import pandas as pd
from scipy.stats import kruskal
a = [35,41,45,42,33,36,47,45,31,32,40,44]
b = [40,38,44,48,45,46,42,39,40,41,38,47]
c = [30,34,38,39,40,41,38,37,40,41,39,38]

#2. Kruskal-Wallis H 검정 분석
print(kruskal(a,b,c))

#3. 생산량 평균 순위 출력
abc = pd.DataFrame(a+b+c)
abc['생산량순위'] = abc.rank(ascending=False)
abc['공장이름'] = ''
abc['공장이름'][0:12] = 1
abc['공장이름'][13:24] = 2
abc['공장이름'][25:36] = 3
print(abc.groupby('공장이름').mean())