#협업 필터링 3종류 사후 분석 과정
import pandas as pd

cf_user = pd.read_csv('협업필터링_user_분석결과2.csv', encoding='utf-8')
cf_item = pd.read_csv('협업필터링_item_분석결과2.csv', encoding='utf-8')
cf_svd = pd.read_csv('협업필터링_svd_분석결과2.csv', encoding='utf-8')

print(cf_user.head())
cf_user=cf_user.drop('Unnamed: 0',axis=1)
cf_item=cf_item.drop('Unnamed: 0',axis=1)
cf_svd=cf_svd.drop('Unnamed: 0',axis=1)
print(cf_user.head())