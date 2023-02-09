#(1)속성수준별 효용가치 계산
import pandas as pd
df = pd.read_csv('conjoint.csv', sep=',', encoding='CP949')

import statsmodels.api as sm
from sympy import *

#속성수준 더미변수생성 및 독립변수 객체 생성
con_df = pd.get_dummies(df, columns=['데이터양','데이터속도','통화량'], drop_first=False)
con_df['intercept'] = 1.0
X = con_df[['intercept','데이터양_1GB','데이터양_2GB','데이터속도_중','데이터속도_상','통화량_150분','통화량_180분']]
print(X)
#개별 응답자의 속성수준 효용가치 계산하기
res = []
intercept = []
for i in range(50):
    Y = con_df[con_df.columns[2:52][i]]
    LR1 = sm.OLS(Y, X).fit()
#연립방정식을 이용한 응답자의 속성수준별 효용가치 계산
    x,y,z = symbols('a_1 a_2 a_3')
    a1 = solve([Eq(x-z, LR1.params[1]), Eq(y-z, LR1.params[2]), Eq(x+y+z,0)], [x,y,z])
    x,y,z = symbols('b_1 b_2 b_3')
    b1 = solve([Eq(x-z, LR1.params[3]), Eq(y-z, LR1.params[4]), Eq(x+y+z,0)], [x,y,z])
    x,y,z = symbols('c_1 c_2 c_3')
    c1 = solve([Eq(x-z, LR1.params[5]), Eq(y-z, LR1.params[6]), Eq(x+y+z,0)], [x,y,z])

#속성수준 효용가치 병합
    a_1 = list(a1.values())
    b_1 = list(b1.values())
    c_1 = list(c1.values())
    a_1.extend(b_1)
    a_1.extend(c_1)
    res.append(a_1)
    d_1=LR1.params[0]
    intercept.append(d_1)

#데이터프레임으로 전환 후 출력
result = pd.DataFrame(res)
result.columns=['데이터양_1GB','데이터양_2GB','데이터양_500MB','데이터속도_중','데이터속도_상','데이터속도_하',
'통화량_150분','통화량_180분','통화량_120분']
print(result.shape)
print(result.head())


#(2)대안별 효용가치 계산
import numpy as np
con_df2 = con_df[['데이터양_1GB','데이터양_2GB','데이터양_500MB','데이터속도_중','데이터속도_상','데이터속도_하',
'통화량_150분','통화량_180분','통화량_120분']]

#대안별 응답자들의 효용가치 계산
alt1 = []
for i in range(0,9,1):
    alt1.append(con_df2.loc[i])
alt2 = []
for i in range(0,9,1):
    for k in range(0,50,1):
        alt2.append(np.dot(alt1[i], result.loc[k]))

#개별응답자의 대안별 효용가치 데이터프레임 생성
cust_Utility = []
for i in range(0,401,50):
    cust_Utility.append(alt2[i:i+50])
cust_Utility = pd.DataFrame(cust_Utility).T

cust_Utility.columns = ['대안1','대안2','대안3','대안4','대안5','대안6','대안7','대안8','대안9']
cust_Utility = cust_Utility+np.array(intercept).mean()
print(cust_Utility.shape)
print('개별응답자의 대안별 효용가치\n', cust_Utility.head())


#(3)최적 대안 선택 및 속성 중요도 평가
#대안별 평균 효용가치 계산 및 최적 대안 찾기
alt_mean = pd.DataFrame(cust_Utility.mean(), columns=['효용가치'])
alt_mean['Rank'] = alt_mean['효용가치'].rank(ascending=False)
print('대안별 효용가치 순위\n', alt_mean.sort_values(by=['Rank'], ascending=True))

#속성별 중요도
a_dif = max(result.mean()[0:3])-min(result.mean()[0:3])
b_dif = max(result.mean()[3:6])-min(result.mean()[3:6])
c_dif = max(result.mean()[6:9])-min(result.mean()[6:9])
print('데이터양 중요도 :', round(a_dif/(a_dif+b_dif+c_dif)*100,2))
print('데이터속도 중요도 :', round(b_dif/(a_dif+b_dif+c_dif)*100,2))
print('통화량 중요도 :', round(c_dif/(a_dif+b_dif+c_dif)*100,2))
print('데이터양 효용가치(1GB, 2GB, 500MB : \n', result.mean()[0:3])
print('데이터속도 효용가치(중, 상, 하 : \n', result.mean()[3:6])
print('통화량 효용가치(150분, 180분, 120분 : \n', result.mean()[6:9])

#(4) 최대선호 모형 기반 시장점유율 예측
#9가지 대안에 대한 개별응답자의 계산
pre1 = []
for i in range(0,50,1):
    market_sum1 = np.where(cust_Utility.loc[i].values == cust_Utility.loc[i].max(),1,0)
    pre1.append(market_sum1)
MAX = pd.DataFrame(pre1)
MAX.columns = cust_Utility.columns
print('9가지 선택 대안에 대한 최대선호 현황 : \n',MAX)

#최대선호 모형을 이용한 시장점유율 예측
pre2=[]
for i in range(0,9,1):
    MAX_sum1 = MAX.iloc[:,i].sum()
    pre2.append(MAX_sum1)
prefer = pd.DataFrame(pre2).T
prefer.coluumns = cust_Utility.columns
print('최대선호 모형을 이용한 예측 시장점유율\n', prefer/50*100)


#(5)선택확률 모형 기반 시장점유율 예측
#개별응답자의 9가지 선택 대안에 대한 선택확률
pro1=[]
for i in range(0,50,1):
    market_sum2 = cust_Utility.loc[i]/cust_Utility.loc[i].sum()
    pro1.append(market_sum2)
select = round(pd.DataFrame(pro1).astype(float),3)
select.columns = cust_Utility.columns
print('9가지 선택 대안에 대한 선택확률 현황 :\n',select)

#9가지 선택 대안에 대한 개별 고객들의 시장점유율
pro2 = []
for i in range(0,9,1):
    select_sum3 = select.iloc[:,i].sum()
    pro2.append(select_sum3)

select2 = round(pd.DataFrame(pro2).astype(float).T,3)
select2.columns=cust_Utility.columns
print('선택확률 모형을 이용한 시장점유율 예측\,',select2/50*100)



