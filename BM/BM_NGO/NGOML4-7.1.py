
#버즈분석
import json
import urllib.request

#2. 요청 정보 입력
request = urllib.request.Request('https://openapi.naver.com/v1/datalab/search')
request.add_header('X-Naver-Client-Id','oprd6_ik_1u81jSCM9HV')
request.add_header('X-Naver-Client-Secret','SEHecvi92Q')
request.add_header('Content-Type','application/json')

#3. 요청 본문 생성
# body_dict1 = {'startDate' : '2017-01-01',
#             'endDate' : '2020-12-31',
#             'timeUnit' : 'month',
#             'keywordGroups': [{'groupName':'월드비전','keywords':['월드비전']}]}
# body1 = json.dumps(body_dict1)

body_dict = {'startDate' : '2017-01-01',
            'endDate' : '2020-12-31',
            'timeUnit' : 'month',
            'keywordGroups': [{'groupName':'월드비전 연말정산','keywords':['월드비전 연말정산']}]}
body = json.dumps(body_dict)

#1. 서버에 정보 요청
# response1 = urllib.request.urlopen(request, data=body1.encode('utf-8'))
response = urllib.request.urlopen(request, data=body.encode('utf-8'))

#2. 응답 상태 코드 가져오기
# rescode1 = response1.getcode()
rescode = response.getcode()

#3. 값이 200인 경우에만 데이터 추출
# if(rescode1==200):
#     scraped1 = response1.read()
# else : 
#     print('Error Code:'+rescode1)

if(rescode==200):
    scraped = response.read()
else : 
    print('Error Code:'+rescode)

#4. json 데이터 타입 변환
# result1 = json.loads(scraped1)
result = json.loads(scraped)


#3) 버즈 그래프 출력
#1. 모듈 및 함수 임포트
import pandas as pd
import matplotlib.pyplot as plt

#2. 데이터 프레임 변환
# data1 = pd.DataFrame(result1['results'][0]['data']).set_index('period')
data2 = pd.DataFrame(result['results'][0]['data']).set_index('period')
# print(data1.shape)
print(data2.shape)

#3. 그래프 생성
plt.plot(data2, color = 'g', figsize = (20,10), marker='o', markerfacecolor='red')

plt.legend(fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()