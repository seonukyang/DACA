#컨조인트 분석 - 라팅 방격법 이용
#1. 모듈 및 데이터 탑재
import pandas as pd
df = pd.read_csv('conjoint.csv', sep=',',encoding='CP949')
#print(df)

#(1)속성수준별 효용가치 계산
#1. 패키지 불러오기
import statsmodels.api as sm
from sympy import *

#2. 속성수준 더미변수생성 및 독립변수 객체 생성
con_df = pd.get_dummies(df, columns=['데이터양','데이터속도','통화량'], drop_first=False)
#print(con_df)
con_df['intercept']=1.0
X = con_df[['intercept','데이터양_1GB','데이터양_2GB','데이터속도_중','데이터속도_상','통화량_150분','통화량_180분']]
#더미데이터 특성상 속성 기준이 다 들어가면 안된다. -1개만 들어간다.
#3. 개별 응답자의 속성수준 효용가치 계산하기
res = [] #개별응답자들의 속성수준별 효용가치 저장 리스트
intercept = [] #절편 저장
for i in range(50):
    Y = con_df[con_df.columns[2:52][i]] 
    #no, 전체평균을 제외하고 2부터 시작. 속성은 9개지만 추가한 더미 컬럼은 3개라서 52까지인듯 하다. 이들중에서 [i]번째 애들을 고름.
    LR1 = sm.OLS(Y, X).fit() #OLS(종속, 독립) 즉 소비자 답변 하나 마다 회귀계수를 구해준다.
#회귀계수라 하면 z-y 처럼 변수들의 차이를 의미한다.

#3-1 연립방정식을 이용한 응답자의 속성수준별 효용가치 계산
    x,y,z = symbols('a_1 a_2 a_3') #방정식에 사용될 변수를 선언
    a1 = solve([Eq(x-z, LR1.params[1]),Eq(y-z, LR1.params[2]), Eq(x+y+z,0)],[x,y,z]) 
    #solve는 연립 방법식 풀이 함수(식,변수). Eq(변수식, =등수) 여기서 z은 X에서 속성의 하나 빠진 속성 기준이다.
    x,y,z = symbols('b_1 b_2 b_3') 
    b1 = solve([Eq(x-z, LR1.params[3]),Eq(y-z, LR1.params[4]), Eq(x+y+z,0)],[x,y,z])
    x,y,z = symbols('c_1 c_2 c_3') 
    c1 = solve([Eq(x-z, LR1.params[5]),Eq(y-z, LR1.params[6]), Eq(x+y+z,0)],[x,y,z]) 
#그래서 풀어진 변수 c_1, c_2, c_3이 c1에 저장된다.

#3-2 속성수준 효용가치 병합
    a_1=list(a1.values())
    b_1=list(b1.values())
    c_1=list(c1.values())
    a_1.extend(b_1)
    a_1.extend(c_1)
    res.append(a_1)
    d_1=LR1.params[0]
    intercept.append(d_1)
#print(res)

#4. 데이터프레임으로 전환 후 출력
result = pd.DataFrame(res)
result.columns=['데이터양_1GB','데이터양_2GB','데이터양_500MB','데이터속도_중','데이터속도_상','데이터속도_하',
'통화량_150분','통화량_180분','통화량_120분']
print(result.shape)
print(result.head())


#(2) 대안별 효용가치 계산
#1 패키지 불러오기
import numpy as np

#2 대안별 더미변수 상태 행렬 생성
con_df2 = con_df[['데이터양_1GB','데이터양_2GB','데이터양_500MB','데이터속도_중','데이터속도_상','데이터속도_하',
'통화량_150분','통화량_180분','통화량_120분']] #더미쪽만 때옴

#3. 대안별 응답자들의 효용가치 계산
alt1 = []
for i in range(0,9,1):
    alt1.append(con_df2.loc[i])
#더미변수들과 속성 효용가치를 합성곱 해야한다.
#9x9의 더미변수행렬을 [1x9,1x9,...]로 쪼개어 놓는 과정이다. 쉬운 계산을 위해

alt2 = []
for i in range(0,9,1) :
    for k in range(0,50,1):
        alt2.append(np.dot(alt1[i], result.loc[k])) #더미변수 x 모든 속성효용가치의 합성곱. 스칼라 값이다. 리스트 타입이다.
        #즉 한 더미변수마다 50개의 스칼라 값이 생성된다. alt2에 총 450개의 데이터를 나란히 넣었다는 것인가? Yes
#print(alt2.shape) 리스트 타입은 shpae가 안 먹힌다.
 
#4. 개별응답자의 대안별 효용가치 데이터프레임 생성
cust_Utility = [] #빈 리스트
for i in range(0,401,50): #0 50 100 150 200 250 300 350 400 총 9개 속성의 시작점
    cust_Utility.append(alt2[i:i+50]) #나름 똑똑한 정렬법이다. 리스트를 append하면 아래로 붙여서 행렬을 만드나 보다.
#print(alt2) [데이터들]
#print(cust_Utility) [[속성1 효용가치 50개],[속성2 효용가치 50개],...] 50개씩 나눠서 행렬식으로 넣어줘다. 9x50으로 만듬.
cust_Utility = pd.DataFrame(cust_Utility).T #T는 행과 열을 전치시킨다. 즉 50x9형태이다.
cust_Utility.columns = ['대안1','대안2','대안3','대안4','대안5','대안6','대안7','대안8','대안9']
cust_Utility = cust_Utility + np.array(intercept).mean() #모든 절편값의 평균을 더해준다. 모두 동일한 값을 더하여 순위 변화는 없음
print(cust_Utility.shape)
print('개별응답자의 대안별 효용가치\n', cust_Utility.head())
#cust_Utility는 각 소비자가 각 대안에 매긴 효용가치이다. 


#(3) 최적 대안 선택 및 속성 중요도 평가
#1. 대안별 평균 효용가치 계산 및 최적 대안 찾기
alt_mean = pd.DataFrame(cust_Utility.mean(), columns=['효용가치']) #데이터프레임을 평균내면 컬럼별로 평균을 구한다.
alt_mean['Rank'] = alt_mean['효용가치'].rank(ascending=False) #alt_mean에 rank컬럼 추가. ascending=False면 클 수부터 1로 랭크를 매긴다.
print('\n대안별 효용가치 순위\n', alt_mean.sort_values(by=['Rank'], ascending=True)) #sort로 정렬 rank를 기준으로 ascending=True 작은수부터 정렬

#2. 속성별 중요도 및 효용가치계산
a_dif = max(result.mean()[0:3])-min(result.mean()[0:3]) #첫번째 대안 속성의 각 속성 기준들의 효용가치를 평균하여 차이를 구한다. 
b_dif = max(result.mean()[3:6])-min(result.mean()[3:6]) #result.mean()[3:6]은 3,1인데 이중에서 max를 고르니 스칼라 값이 결과로 나온다.
c_dif = max(result.mean()[6:9])-min(result.mean()[6:9])
a_dif_1 = max(result.mean()[0:1])-min(result.mean()[0:1]) #이렇게 1,1로 구하면 max, min값이 같으니 당연히 0이 나온다.
a_dif_2 = max(result.mean()[1:2])-min(result.mean()[1:2])
a_dif_3 = max(result.mean()[2:3])-min(result.mean()[2:3])
print('max(result.mean()[0:3])',max(result.mean()[0:3]))
print('min(result.mean()[0:3])',min(result.mean()[0:3]))
print('a_dif_1 = ', a_dif_1)
print('a_dif_2 = ', a_dif_2)
print('a_dif_3 = ', a_dif_3)
print('a_dif = ', a_dif)
print('\n데이터양 중요도 :', round(a_dif/(a_dif+b_dif+c_dif)*100,2)) #전체 속성의 효용가치 중에 첫 번째 대안 속성이 차지하는 비율
print('데이터속도 중요도 :', round(b_dif/(a_dif+b_dif+c_dif)*100,2)) #아무튼 소비자가 점수를 매긴 것이니 중요도라는 표현이 좋다.
print('통화량 중요도 :', round(c_dif/(a_dif+b_dif+c_dif)*100,2)) 
print('\n데이터양 효용가치(1GB, 2GB, 500MB :\n',result.mean()[0:3])
print('\n데이터속도 효용가치(중, 상, 하) :\n',result.mean()[3:6])
print('\n통화량 효용가치(150분, 180분, 120분) :\n',result.mean()[6:9])


#(4) 최대선호 모형 기반 시장점유율 예측
#1. 9가지 대안에 대한 개별응답자의 최대선호 계산
pre1 = []
for i in range(0,50,1) :
    market_sum1 = np.where(cust_Utility.loc[i].values == cust_Utility.loc[i].max(),1,0) #where(이것이=이값이면, true값 , false값)
    #cust_Utility의 i번째 행에 대해 where문을 실행하는 것이다. 즉 1,9는 유지된다. 최대 점수를 받은 선택 대안만 1이 된다.
    pre1.append(market_sum1) #50,9 의 0과1로 이루어진 선택 대안의 최대선호 테이블이 완성된다.
MAX = pd.DataFrame(pre1)
MAX.columns = cust_Utility.columns
print('9가지 선택 대안에 대한 최대선호 현황:\n', MAX)

#2. 최대선호 모형을 이용한 시장점유율 예측
pre2 = []
for i in range(0,9,1):
    MAX_sum1 = MAX.iloc[:,i].sum() #각 대안별 최대로 선택된 횟수를 구하는 것이다. 9,1
    pre2.append(MAX_sum1)
prefer = pd.DataFrame(pre2).T # 1,9로 만들어 보기 편하게 한다.
prefer.columns = cust_Utility.columns
print('\n최대선호 모형을 이용한 예측 시장점유율\n', prefer/50*100) #50은 소비자수


#(5) 선택확률 모형 기반 시장점유율 예측
#1. 개별응답자의 9가지 선택 대안에 대한 선택확률
pro1=[]
for i in range(0,50,1):
    market_sum2 = cust_Utility.loc[i]/cust_Utility.loc[i].sum() 
    #cust_utility의 i번째 행 각각에 i번째 행 전체의 합을 나눈 것이다. 그래서 1,9는 유지되고 
    #i번째 소비자가 1~9까지의 대안을 선택할 확률이 구해진다. 완벽하네
    pro1.append(market_sum2)
select = round(pd.DataFrame(pro1).astype(float),3) #실수형 float으로 수정해줌. 50,9
select.columns = cust_Utility.columns
print('9가지 선택 대안에 대한 선택확률 현황\n', select)

#2. 9가지 선택 대안에 대한 개별 고객들의 시장점유율
pro2 = []
for i in range(0,9,1):
    select_sum3 = select.iloc[:,i].sum() #각 대안에 대한 개별 고객들의 선택 확률을 모두 더한다. 스칼라
    pro2.append(select_sum3) #9개의 스칼라 9,1
select2 = round(pd.DataFrame(pro2).astype(float).T,3) #1,9
select2.columns = cust_Utility.columns
print('\n선택확률 모형을 이용한 시장점유율 예측\n', select2/50*100) #이 확률에 소비자 수만큼 나눠준다.

