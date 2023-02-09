#앙상블 모형 비교
# 분류 예측
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
df = pd.read_csv('Ashopping.csv', encoding = 'cp949')

#3. 변수 지정 및 데이터 세트 분할
X = df[['방문빈도','1회 평균매출액','할인권 사용 횟수','총 할인 금액','거래기간','평균 구매주기','구매유형']]
Y = df['고객등급']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#4. 표준화 및 원핫인코딩
ct = ColumnTransformer([('scaling',StandardScaler(),['1회 평균매출액','방문빈도','총 할인 금액','할인권 사용 횟수',
'거래기간','평균 구매주기']),('onehot',OneHotEncoder(sparse= False, handle_unknown = 'ignore'),['구매유형'])])
ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

#5. 오버샘플링
smote = SMOTE(random_state=0)
X_train, Y_train = smote.fit_sample(X_train, Y_train)
Y_train[Y_train==2] = 0
Y_test[Y_test==2] = 0
print(Y_train)
print(Y_test)
#2) 모형 학습 및 예측
#1. 모듈 및 함수 불러오기
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#2. 단일 모형 객체 생성 (의사결정나무 모형, K-NN 모형)
dtree = DecisionTreeClassifier(random_state=0)
knn = KNeighborsClassifier()

#3. 앙상블 모형 생성
model_voting = VotingClassifier(estimators=[('K-NN',knn),('Dtree',dtree)], voting = 'soft')
model_randomforest = RandomForestClassifier(random_state = 0, n_estimators = 300, max_depth = 5)
model_gradient = GradientBoostingClassifier(random_state = 0, n_estimators = 100, max_depth=2, learning_rate=0.1)



#ROC 그래프 그리기, auc 구하기
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
all_model = [model_voting, model_randomforest, model_gradient]
model_label = ['voting','randomforest','gradientboosting']
colors = ['black','orange','blue']
linestyles = [':', '--', '-.']
for model, label, clr, ls in zip(all_model, model_label, colors, linestyles) :
    Y_pred = model.fit(X_train, Y_train).predict_proba(X_test)[:,1]
    # print(Y_test)
    # print(Y_pred)
    fpr, tpr, thresholds = roc_curve(y_true=Y_test, y_score=Y_pred)
    roc_auc = auc(x=fpr, y=tpr)

    plt.plot(fpr, tpr, color=clr, linestyle= ls, label='%s (auc = %0.2f)' % (label, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], linestyle='--', color='gray', linewidth=2)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(alpha=0.5)
    plt.xlabel('False positive Rate(FPR)')
    plt.ylabel('True Positive Rate(FPR)')
plt.show()

#ROC_AUC수치 구하기
from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = model, X=X_train, y=Y_train, cv=10, scoring='roc_auc')
print('ROC_AUC : ', score.mean(),'+/-', score.std())