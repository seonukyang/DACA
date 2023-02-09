#가우시안 나이브 베이즈 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

df= pd.read_csv('Ashopping.csv', encoding='CP949')

X = df[['고객ID','방문빈도','거래기간','구매카테고리수']]
Y = df['이탈여부']
print(Y.head())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=25)
X_train1 = X_train[['방문빈도','거래기간','구매카테고리수']]
X_test1 = X_test[['방문빈도','거래기간','구매카테고리수']]
ct = ColumnTransformer([('scaling', StandardScaler(),['방문빈도','거래기간','구매카테고리수'])])
ct.fit(X_train1)
X_train1 = ct.transform(X_train1)
X_test1 = ct.transform(X_test1)


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train1, Y_train)
Y_pred = model.predict(X_test1)
print(Y_pred)
print(Y_pred[2])

result = X_test['고객ID']
for i in range(0,len(result),1):
    if Y_pred[i]==0 : 
        print(result.iloc[i])


#정확도 평가
print('3학습용 데이터 세트 정확도 : ', model.score(X_train1, Y_train))
print('평가용 데이터 세트 정확도 : ', model.score(X_test1, Y_test))

#정밀도, 재현율, F1 스코어
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

#분류 확률의 추정치 확인
yproba = model.predict_proba(X_test1)
print(yproba)


