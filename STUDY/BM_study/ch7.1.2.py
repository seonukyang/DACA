import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df= pd.read_csv('Ashopping.csv', encoding='CP949')

X1 = df[['총매출액','거래기간','방문빈도']]
Y1 = df['이탈여부']
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.3, random_state=0)
scaler = StandardScaler()
scaler.fit(X1_train)
X1_test = scaler.transform(X1_test)
X1_train = scaler.transform(X1_train)
from imblearn.over_sampling import SMOTE
from collections import Counter
smote = SMOTE(random_state = 0)
X1_train, Y1_train = smote.fit_sample(X1_train, Y1_train)
from sklearn.linear_model import LogisticRegression
Logr = LogisticRegression(C = 1, random_state=0)
Logr.fit(X1_train, Y1_train)


X2 = df[['총매출액','방문빈도','1회 평균매출액','거래기간','평균 구매주기']]
Y2 = df['이탈여부']
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.3, random_state=0)
scaler = StandardScaler()
scaler.fit(X2_train)
X2_test = scaler.transform(X2_test)
X2_train = scaler.transform(X2_train)
X2_train, Y2_train = SMOTE(random_state = 0).fit_sample(X2_train, Y2_train)

from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
kNN.fit(X2_train, Y2_train)

X3 = df[['총매출액','구매금액대','할인권 사용 횟수','총 할인 금액','구매유형','구매카테고리수','성별','거래기간','방문빈도','할인민감여부']]
Y3 = df['이탈여부']
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3, Y3, test_size = 0.3, random_state=0)
smote = SMOTE(random_state=0)
X3_train, Y3_train = smote.fit_sample(X3_train, Y3_train)

from sklearn.tree import DecisionTreeClassifier
Dtree = DecisionTreeClassifier(random_state=0, max_depth=3)
Dtree.fit(X3_train, Y3_train)

#모형 성능 비교
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

fpr1, tpr1, thresholds = roc_curve(Y1_test, Logr.decision_function(X1_test))
fpr2, tpr2, thresholds = roc_curve(Y2_test, kNN.predict_proba(X2_test)[:,1])
fpr3, tpr3, thresholds = roc_curve(Y3_test, Dtree.predict_proba(X3_test)[:,1])

plt.plot(fpr1, tpr1, 'o-', ms=2, label="Logr")
plt.plot(fpr2, tpr2, 'o-', ms=2, label='k-NN')
plt.plot(fpr3, tpr3, 'o-', ms=2, label='Dtree')
plt.plot([0,1], [0,1], 'k--', label='random guess')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#AUC점수 계산
Logr_auc = roc_auc_score(Y1_test, Logr.decision_function(X1_test))
kNN_auc = roc_auc_score(Y2_test, kNN.predict_proba(X2_test)[:,1])
Dtree_auc = roc_auc_score(Y3_test, Dtree.predict_proba(X3_test)[:,1])

print('로지스틱 회귀분석 AUC 점수 : ',Logr_auc)
print('k-최근접 이웃 AUC 점수 : ',kNN_auc)
print('의사결정나무 AUC 점수 : ',Dtree_auc)
