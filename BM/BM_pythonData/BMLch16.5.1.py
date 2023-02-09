#토픽 모델링
#1) 데이터 전처리
#1. 모듈 및 함수 불러오기
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df= pd.read_csv('BML16웹스크래핑.csv', encoding='utf-8')
df= df.dropna()
review = df['review']
score = df['score']
#2. 문서 단어 행렬 변환
stopword = ['영화','하지만','근데','대한','이게','없는','영화가','영화는','영화를','ㅋㅋ','내내','봤습니다','보고',
'그냥','많이','내가','그리고','진짜','정말','너무','나는','있는','가장','ㅎㅎ']

tv = TfidfVectorizer(max_df=.15, ngram_range = (1,4), min_df=2, stop_words=stopword)
vect = tv.fit_transform(review)

print('문서 단어 행렬 변환 결과 : ',vect.toarray())

#2) 토픽 모델링 수행
#1. 모듈 및 함수 불러오기
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

#2. LDA 모형 생성 및 변환
model = LatentDirichletAllocation(n_components=5, learning_method='batch', max_iter=25, random_state=0)
model.fit_transform(vect)

#3. 토픽별 주요 단어 출력

for topic_index, topic in enumerate(model.components_):

    #3.1 중요도 내림차순 정렬 및 인덱스 추출
    print('Topic ',topic_index+1)
    topic_index = topic.argsort()[::-1]
    print(topic_index)
    #3-2 단어 추출
    feature_names = ['1','2','3','4','5','6','7','8','9','0']
    for i in range(0,10,1) : 
        j = topic_index[i] 
        feature_names[i] = tv.get_feature_names()[j]
    print(feature_names)
