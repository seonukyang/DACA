#텍스트 빈도 분석
#2) 데이터 전처리
#1. 모듈 및 함수 불러오기
import pandas as pd
from konlpy.tag import Okt

df= pd.read_csv('BML16웹스크래핑.csv', encoding='utf-8')
df= df.dropna()
review = df['review']
score = df['score']


#2. 문자열 변환
string = ','.join(review)

#3. 명사 추출
okt = Okt()
nouns=okt.nouns(string)

#4. 두 글자 이상 단어 추출
word_list = [x for x in nouns if len(x) >= 2]

#5. 불용어 제거
stopwords = ['영화','그냥','정말','진짜']
word_list=[i for i in word_list if i not in stopwords]
# print(word_list)

#3) 빈도 분석 수행
from collections import Counter
count = Counter(word_list)
print(count)

#4) 워드 클라우드 생성
#1. 라이브러리 설치 및 모듈 / 함수 불러오기
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#2. 워드클라우드생성
font_path = '/usr/share/fonts/truetrype/nanum/malgun.ttf'
wc = WordCloud(width = 800, height = 800, font_path = font_path, background_color='white')

#3. 시각화
plt.figure(figsize=(10,10))
plt.imshow(wc.generate_from_frequencies(count))
plt.axis("off")
plt.show()
