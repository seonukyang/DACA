#1. 모듈 및 데이터 탑재
import pandas as pd
from scipy import stats
df = pd.read_csv('Ashopping.csv', sep=',', encoding='CP949')

#2. 쌍체표본 t-검정
rel = stats.ttest_rel(df['멤버쉽_프로그램_가입후_만족도'],df['멤버쉽_프로그램_가입전_만족도'])
print(rel)