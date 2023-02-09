#리뷰 스크래핑
#1. 모듈 및 함수 불러오기
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import re

#2. 리뷰와 평점을 담을 빈 리스트 생성
score = []
review = []

#3. 리뷰 스크래핑
for i in range(1,1001) : 
    
    #3-1. HTML 소스코드 가져오기
    base_url='https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=161967&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page='+str(i)
    html = requests.get(base_url)

    #3-2. HTML 파싱
    soup = bs(html.text, 'html.parser')

    for j in range(10):
        score.append(soup.select('div.star_score > em')[j].text)
        review.append(soup.find_all('span', {'id':re.compile('_filtered_ment_\d')})[j].text.strip())
        print('페이지수 : ',i,'  리뷰 수 : ',j)

df = pd.DataFrame(list(zip(review,score)), columns=['review','score'])
df.to_csv('BML16웹스크래핑.csv', encoding='utf-8-sig')

