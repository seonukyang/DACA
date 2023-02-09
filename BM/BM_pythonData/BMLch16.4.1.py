
#버즈분석
import json
import urllib.request

#2. 요청 정보 입력
request = urllib.request.Request('https://openapi.naver.com/v1/datalab/search')
request.add_header('X-Naver-Client-Id','oprd6_ik_1u81jSCM9HV')
request.add_header('X-Naver-Client-Secret','SEHecvi92Q')
request.add_header('Content-Type','application/json')

#3. 요청 본문 생성
body_dict = {'startDate' : '2019-01-01',
            'endDate' : '2019-12-20',
            'timeUnit' : 'month',
            'keywordGroups': [{'groupName':'기생충','keywords':['기생충']}]}
body = json.dumps(body_dict)
print(body)

#1. 서버에 정보 요청
response = urllib.request.urlopen(request, data=body.encode('utf-8'))

#2. 응답 상태 코드 가져오기
rescode = response.getcode()

#3. 값이 200인 경우에만 데이터 추출
if(rescode==200):
    scraped = response.read()
else : 
    print('Error Code:'+rescode)

#4. json 데이터 타입 변환
result = json.loads(scraped)
print(result)

#3) 버즈 그래프 출력
#1. 모듈 및 함수 임포트
import pandas as pd
import matplotlib.pyplot as plt

#2. 데이터 프레임 변환
data = pd.DataFrame(result['results'][0]['data']).set_index('period')

#3. 그래프 생성
data.plot(color = 'g', figsize = (20,10))
plt.legend(fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()