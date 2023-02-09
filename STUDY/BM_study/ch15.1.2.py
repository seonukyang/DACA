#협업 필터링
#아이템 기반 협업 필터링
from surprise import Dataset, trainset
data = Dataset.load_builtin('ml-100k')
import pandas as pd
ratings = pd.DataFrame(data.raw_ratings, columns=['user','item','rate','timestamp'])
print(ratings.head())

#유저정보
u_cols = ['user_id', 'age','sex','occupation','xip_code']
users = pd.read_csv('u.user', sep='|', names=u_cols)

#영화 정보
m_cols = ['movie_id','title','release_date','video_release_date','imdb_url']
movies = pd.read_csv('u.item', sep='|', names=m_cols, usecols=range(5), encoding='latin1')

#모형
from surprise.model_selection import train_test_split
from surprise import KNNBasic
trainset, testset = train_test_split(data, test_size = 0.3, random_state=0)
model = KNNBasic(name = 'cosine', user_base=False)
predictions = model.fit(trainset).test(testset)


from surprise import accuracy
rmse = accuracy.rmse(predictions)
print('RMSE : ', rmse)


#고객별 추천 영화 리스트 출력
def recommend(predictions, n, k):
    uids = [pred.uid for pred in predictions][:n]
    for uid in uids : 
        seen_movies = ratings[ratings.user==uid]['item'].tolist()
        total_movies = movies['movie_id'].tolist()
        unseen_movies = [movie for movie in total_movies if movie not in seen_movies]

        pred = [model.predict(str(uid), str(item)) for item in unseen_movies]
        pred.sort(key=lambda pred: pred[3], reverse=True)
        top_pred = pred[:k]

        top_ids = [int(pred.iid) for pred in top_pred]
        top_titles = movies[movies.movie_id.isin(top_ids)]['title']
        top_rating = [pred.est for pred in top_pred]
        top_preds = [(id, title, rating) for id, title, rating in zip(top_ids, top_titles, top_rating)]

        print('#고객 ID : ',uid)
        for top_movie in top_preds:
            print(top_movie[1],':',top_movie[2])
        
recommend(predictions, 10, 3)
