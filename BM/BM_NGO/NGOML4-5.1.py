#리뷰 스크래핑
#1. 모듈 및 함수 불러오기
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import re

#2. 리뷰와 평점을 담을 빈 리스트 생성
title = []
question = []
date = []

#3. 리뷰 스크래핑

#3-1. HTML 소스코드 가져오기
base_url='https://kin.naver.com/qna/detail.nhn?d1id=6&dirId=61002&docId=375024444&qb=7JuU65Oc67mE7KCE&enc=utf8&section=kin&rank=1&search_sort=0&spq=0'
html = requests.get(base_url)

#3-2. HTML 파싱
soup = bs(html.text, 'html.parser')



print(soup.select('div.title')[0].text)
print(soup.select('div.c-heading__content')[0].text)
print(soup.select('span.c-userinfo__info')[0].text)

title.append(soup.select('div.title').text)
question.append(soup.select('div.c-heading__content').text)
date.append(soup.select('span.c-userinfo__info').text)

df = pd.DataFrame(list(zip(date, title, question)), columns=['date','title','question'])
print(df)
# df.to_csv('BML16웹스크래핑.csv', encoding='utf-8-sig')

