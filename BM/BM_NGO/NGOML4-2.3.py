#협업 필터링 3종류 모델 찾기 과정
import pandas as pd

#데이터 전처리
from surprise.model_selection import train_test_split
from sklearn.model_selection import train_test_split as split

#모델 평가
from surprise import accuracy

#협업필터링
from surprise import KNNBasic
from surprise import Reader, Dataset
from surprise import SVD

# data = pd.read_csv('협업필터링data2.csv', encoding='utf-8')
data = pd.read_csv('협업필터링data3.csv', encoding='utf-8')

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(data[['uid', '후원종류', '플릿지수']], reader)

trainset, testset = train_test_split(data,test_size=0.3 ,random_state=0)


#3. 모델 생성

model_user = {'name':[],'k':[],'min_k':[], 'rmse':[]}
model_item = {'name':[],'k':[],'min_k':[], 'rmse':[]}
model_svd = {'n_factors':[], 'rmse':[]}
model_user = pd.DataFrame(model_user)
model_item = pd.DataFrame(model_item)
model_svd = pd.DataFrame(model_svd)

for i in range(30,51,1):
    for j in range(1,4,1) : 
        model_user_cos = KNNBasic(name='cosine', user_base=True, k=i, min_k=j)
        model_user_msd = KNNBasic(name='msd', user_base=True, k=i, min_k=j)
        model_item_cos = KNNBasic(name='cosine', user_base=False, k=i, min_k=j)
        model_item_msd = KNNBasic(name='msd', user_base=False, k=i, min_k=j)

        predictions_user_cos = model_user_cos.fit(trainset).test(testset)
        predictions_user_msd = model_user_msd.fit(trainset).test(testset)
        rmse_user_cos = accuracy.rmse(predictions_user_cos)
        rmse_user_msd = accuracy.rmse(predictions_user_msd)
        predictions_item_cos = model_item_cos.fit(trainset).test(testset)
        predictions_item_msd = model_item_msd.fit(trainset).test(testset)
        rmse_item_cos = accuracy.rmse(predictions_item_cos)
        rmse_item_msd = accuracy.rmse(predictions_item_msd)


        newdata = {'name' : 'cosine', 'k':i, 'min_k':j, 'rmse':rmse_user_cos}
        model_user = model_user.append(newdata, ignore_index=True)
        newdata = {'name' : 'msd', 'k':i, 'min_k':j, 'rmse':rmse_user_msd}
        model_user = model_user.append(newdata, ignore_index=True)
        newdata = {'name' : 'cosine', 'k':i, 'min_k':j, 'rmse':rmse_item_cos}
        model_item = model_item.append(newdata, ignore_index=True)
        newdata = {'name' : 'msd', 'k':i, 'min_k':j, 'rmse':rmse_item_msd}
        model_item = model_item.append(newdata, ignore_index=True)

for k in range(60,121,1) : 
    model_svd_test = SVD(n_factors=k, random_state=0)
    predictions_svd = model_svd_test.fit(trainset).test(testset)
    rmse_svd = accuracy.rmse(predictions_svd)
    newdata = {'n_factors' : k , 'rmse':rmse_svd}
    model_svd = model_svd.append(newdata, ignore_index=True)

print(model_user)

# model_user.to_csv('협업필터링_user_모델결과.csv', encoding='utf-8-sig')
# model_item.to_csv('협업필터링_item_모델결과.csv', encoding='utf-8-sig')
# model_svd.to_csv('협업필터링_svd_모델결과.csv', encoding='utf-8-sig')

model_user.to_csv('협업필터링_user_모델결과2.csv', encoding='utf-8-sig')
model_item.to_csv('협업필터링_item_모델결과2.csv', encoding='utf-8-sig')
model_svd.to_csv('협업필터링_svd_모델결과2.csv', encoding='utf-8-sig')

# from surprise.model_selection import cross_validate
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)