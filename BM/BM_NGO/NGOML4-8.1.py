#토픽 모델링
#1) 데이터 전처리
#1. 모듈 및 함수 불러오기
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
df= pd.read_csv('웹스크래핑_월드비전_트위터2.csv', encoding='cp949')
df= df.dropna()
content_all = df['content']
content_12 = df[df['month']==12].content
content_11 = df[df['month']==11].content
content_5 = df[df['month']==5].content
content_list = [content_all, content_12, content_11, content_5]

#2. 문서 단어 행렬 변환
stopword = ['아니','하지만','근데','대한','이게','없는','ㅋㅋ','내내','그냥','내가','그리고','진짜','정말','너무','나는','있는','가장','ㅎㅎ','#']
for i in range(0,4,1) : 
    content = content_list[i]
    tv = TfidfVectorizer(max_df=.15, ngram_range = (1,4), min_df=2, stop_words=stopword)
    vect = tv.fit_transform(content)

    print('문서 단어 행렬 변환 결과 : ',vect.toarray())

    #2. LDA 모형 생성 및 변환
    model = LatentDirichletAllocation(n_components=2, learning_method='batch', max_iter=25, random_state=0)
    model.fit_transform(vect)

    #3. 토픽별 주요 단어 출력

    for topic_index, topic in enumerate(model.components_):

        #3.1 중요도 내림차순 정렬 및 인덱스 추출
        print('Topic ',topic_index+1)
        topic_index = topic.argsort()[::-1]
        #3-2 단어 추출
        feature_names = ['1','2','3','4','5','6','7','8','9','0']
        for i in range(0,10,1) : 
            j = topic_index[i] 
            feature_names[i] = tv.get_feature_names()[j]
        print(feature_names)
