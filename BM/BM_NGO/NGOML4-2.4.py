#헙업 필터링 3종류 분석 과정
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
# df = pd.read_csv('협업필터링df2.csv', encoding='utf-8')
data = pd.read_csv('협업필터링data3.csv', encoding='utf-8')

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(data[['uid', '후원종류', '플릿지수']], reader)

print(data)

trainset, testset = train_test_split(data,test_size=0.3 ,random_state=0)


#3. 모델 생성

model_user_result = {'uid':[],'후원종류':[],'기존플릿지수':[], '예측플릿지수':[]}
model_item_result = {'uid':[],'후원종류':[],'기존플릿지수':[], '예측플릿지수':[]}
model_svd_result = {'uid':[],'후원종류':[],'기존플릿지수':[], '예측플릿지수':[]}

model_user_result = pd.DataFrame(model_user_result)
model_item_result = pd.DataFrame(model_item_result)
model_svd_result = pd.DataFrame(model_svd_result)

model_user = KNNBasic(name='cosine', user_base=True, k=50, min_k=1)
predictions_user = model_user.fit(trainset).test(testset)
rmse_user = accuracy.rmse(predictions_user)
# model_user_result['uid'] = predictions_user.uid
model_user_result['uid'] = [pred.uid for pred in predictions_user]
model_user_result['후원종류'] = [pred.iid for pred in predictions_user]
model_user_result['기존플릿지수'] = [pred.r_ui for pred in predictions_user]
model_user_result['예측플릿지수'] = [pred.est for pred in predictions_user]


model_item = KNNBasic(name='cosine', user_base=False, k=50, min_k=1)      
predictions_item = model_item.fit(trainset).test(testset)
rmse_item = accuracy.rmse(predictions_item)
model_item_result['uid'] = [pred.uid for pred in predictions_item]
model_item_result['후원종류'] = [pred.iid for pred in predictions_item]
model_item_result['기존플릿지수'] = [pred.r_ui for pred in predictions_item]
model_item_result['예측플릿지수'] = [pred.est for pred in predictions_item]



 
model_svd = SVD(n_factors=119, random_state=0)
predictions_svd = model_svd.fit(trainset).test(testset)
rmse_svd = accuracy.rmse(predictions_svd)
model_svd_result['uid'] = [pred.uid for pred in predictions_svd]
model_svd_result['후원종류'] = [pred.iid for pred in predictions_svd]
model_svd_result['기존플릿지수'] = [pred.r_ui for pred in predictions_svd]
model_svd_result['예측플릿지수'] = [pred.est for pred in predictions_svd]

model_user_result.to_csv('협업필터링_user_분석결과2.csv', encoding='utf-8-sig')
model_item_result.to_csv('협업필터링_item_분석결과2.csv', encoding='utf-8-sig')
model_svd_result.to_csv('협업필터링_svd_분석결과2.csv', encoding='utf-8-sig')