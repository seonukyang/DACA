#신뢰도 내적일관성 분석
#1 모듈 및 데이터 탑재
import pandas as pd
import pingouin as pg
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')
X = df[['친절성','신속성','책임성','정확성','전문성']]

#2. 크론바흐 알파 계수 출력
cron = pg.cronbach_alpha(data=X)
print(cron)
#결과 (0.8117458883194828, array([0.793, 0.83 ])) 크론바흐 알파 계수, 신뢰구간

#3. 각 변수를 제외한 크론바흐 알파 계수 출력
X1 = df[['신속성','책임성','정확성','전문성']]
X2 = df[['친절성','책임성','정확성','전문성']]
X3 = df[['친절성','신속성','정확성','전문성']]
X4 = df[['친절성','신속성','책임성','전문성']]
X5 = df[['친절성','신속성','책임성','정확성']]
print('X1 : ', pg.cronbach_alpha(data=X1))
print('X2 : ', pg.cronbach_alpha(data=X2))
print('X3 : ', pg.cronbach_alpha(data=X3))
print('X4 : ',pg.cronbach_alpha(data=X4))
print('X5 : ', pg.cronbach_alpha(data=X5))