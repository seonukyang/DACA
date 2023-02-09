#사용자 기반 협업 필터링
#1) 데이터 살펴보기
#1. 모듈 및 함수 불러오기

from surprise import Dataset
import pandas as pd
#2. 데이터 로딩
data = Dataset.load_builtin('ml-100k')


ratings = pd.DataFrame(data.raw_ratings, columns=['user','item','rate','timestamp'])
print(ratings.head())
print(ratings.shape)

u_cols = ['user_id','age','sex','occupation','zip_code']

users = pd.read_csv('u.user',sep='|', names=u_cols)
print(users.shape)
print(users.head())

m_cols = ['movie_id','title','release_date','video_releas_data','imdb_url']
movies = pd.read_csv('u.item',sep='|', names=m_cols, usecols=range(5), encoding = 'latin1')
print(movies.shape)
print(movies.head())

#2) 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
from surprise.model_selection import train_test_split
from surprise import KNNBasic

#2. 데이터 분할
trainset, testset = train_test_split(data, test_size=0.3, random_state=0)


#3. 모형 학습 및 예측
model = KNNBasic(name = 'cosine', user_base=True)
predictions = model.fit(trainset).test(testset)
print(predictions[:3])

#3) 모형 평가
from surprise import accuracy
rmse = accuracy.rmse(predictions)
print('rmse',rmse)

#4) 고객별 추천 영화 리스트 출력
#1. 함수 정의
def recommend(predictions, n, k) : 
    print('-----고객별 추천 영화 리스트----')
#2. n명의 고객 ID 추출
    uids = [pred.uid for pred in predictions][:n]

#3. 고객별 추천 리스트 출력
    for uid in uids : 

        #3-1 고객이 관람하지 않은 영화 추출
        seen_movies = ratings[ratings.user == uid]['item'].tolist()
        total_movies = movies['movie_id'].tolist()
        unseen_movies=[movie for movie in total_movies if movie not in seen_movies]

        #3-2 k개의 미관람 영화에 대한 평점 예측
        pred = [model.predict(str(uid), str(item)) for item in unseen_movies]
        pred.sort(key=lambda pred: pred[3], reverse=True)
        top_pred = pred[:k]

        #3-3. 예측 결과로부터 영화 ID, 제목, 예측 평점 추출
        top_ids = [int(pred.iid) for pred in top_pred]
        top_titles = movies[movies.movie_id.isin(top_ids)]['title']
        top_rating = [pred.est for pred in top_pred]
        top_preds = [(id, title, rating) for id, title, rating in zip(top_ids, top_titles, top_rating)]

        #3-4. 추천 리스트 출력
        print('#고객 ID:', uid)
        for top_movie in top_preds : 
            print(top_movie[1], ':', top_movie[2])
#4. 함수 호출
recommend(predictions, 10,3)