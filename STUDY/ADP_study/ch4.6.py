#XGBoost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer
from xgboost import plot_importance
from xgboost import plot_importance
from sklearn.datasets import load_breast_cancer

def get_clf_eval(y_test, pred=None, pred_proba=None) : 
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print(confusion)
    print('정확도 : ', accuracy)
    print('정밀도 : ',precision)
    print('재현율 : ', recall)
    print('F1 score : ',f1)
    print('AUC : ',roc_auc)


def precision_recall_curve_plot(y_test, pred_proba_c1):
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='-', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')

    start, end = plt. xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlabel('Threshold value')
    plt.ylabel('Precision and recall value')
    plt.legend()
    plt.grid()
    plt.show()
    plt.clf()

def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for custom_threshold in thresholds : 
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임계값 : ',custom_threshold)
        get_clf_eval(y_test, custom_predict)

dataset = load_breast_cancer()
X_features = dataset.data
y_label = dataset.target
cancer_df = pd.DataFrame(data=X_features, columns = dataset.feature_names)
cancer_df['target'] = y_label

X_train, X_test, y_train, y_test  = train_test_split(X_features, y_label, test_size=0.2, random_state=156)

dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)
params = {'max_depth':3, 
            'eta':0.1,
            'objective':'binary:logistic',
            'eval_metric':'logloss',
            'early_stoppings':100}
num_rounds = 400
wlist = [(dtrain, 'train'),(dtest,'eval')]
xgb_model = xgb.train(params = params, dtrain=dtrain, num_boost_round=num_rounds,
            early_stopping_rounds=100, evals=wlist)

pred_probs = xgb_model.predict(dtest)
preds = [1 if x>0.5 else 0 for x in pred_probs]
get_clf_eval(y_test, preds, pred_probs)

 
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)
