#텍스트 빈도 분석
import pandas as pd
from konlpy.tag import Okt

from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df= pd.read_csv('웹스크래핑_월드비전_트위터2.csv', encoding='cp949')

content_all = df['content']
content_12 = df[df['month']==12].content
content_11 = df[df['month']==11].content
content_5 = df[df['month']==5].content
content_list = [content_all, content_12, content_11, content_5]
font_path = '/usr/share/fonts/truetrype/nanum/malgun.ttf'
#2. 문자열 변환
for i in range(0,4,1) : 
    content = content_list[i]
    string = ','.join(content)

    #3. 명사 추출
    okt = Okt()
    nouns=okt.nouns(string)

    #4. 두 글자 이상 단어 추출
    word_list = [x for x in nouns if len(x) >= 2]

    #5. 불용어 제거
    stopwords = ['그냥','정말','진짜']
    word_list=[i for i in word_list if i not in stopwords]
    # print(word_list)

    #3) 빈도 분석 수행
    
    count = Counter(word_list)
    print(count)

    #4) 워드 클라우드 생성
    #1. 라이브러리 설치 및 모듈 / 함수 불러오기
    
    #2. 워드클라우드생성
    
    wc = WordCloud(width = 800, height = 800, font_path = font_path, background_color='white')

    #3. 시각화
    plt.figure(figsize=(10,10))
    plt.imshow(wc.generate_from_frequencies(count))
    plt.axis("off")
    plt.show()
    plt.clf()
