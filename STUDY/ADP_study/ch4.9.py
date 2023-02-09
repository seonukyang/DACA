import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib

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

card_df = pd.read_csv('creditcard.csv')

card_df.drop('Time', axis=1, inplace=True)
# scaler = StandardScaler()
# card_df['Amount'] = scaler.fit_transform(card_df['Amount'].values.reshape(-1,1))
card_df['Amount'] = np.log1p(card_df['Amount'])

X_features = card_df.iloc[:,:-1]
y_target = card_df.iloc[:,-1]

X_train, X_test, y_train, y_test  = train_test_split(X_features, y_target, test_size=0.2, random_state=0)

lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
lr_pred_proba = lr_clf.predict_proba(X_test)[:,1]

get_clf_eval(y_tset, lr_pred, lr_pred_proba)

