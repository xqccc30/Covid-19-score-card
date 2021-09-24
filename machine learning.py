
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import ADASYN
import scipy
import rpy2.robjects as robjects
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from sklearn import metrics
from scipy.stats import uniform
from sklearn.ensemble import VotingClassifier

log = []
xyk = []
xyf = []
xyx = []
xys = []
b4 = []
tprs = []
aucs = []
qy = []
featr = []
featx = []
mean_fpr = np.linspace(0, 1, 100)
def SpSe(y_test, y_predict):
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_test, y_predict)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    return  TN / float(TN + FP), TP / (TP + FN), TN/(TN+FN), TP/(TP+FP)

data_test = pd.read_excel(r"C:\Users\JQ\Desktop\3-1\3-10\data_test.xlsx")
data_train = pd.read_excel(r"C:\Users\JQ\Desktop\3-1\3-10\data_traink.xlsx")
model_smote = ADASYN(random_state=2324)
x_train, x_test, y_train, y_test= data_train.drop(["Unnamed: 0", "outcome"], axis = 1), data_test.drop(["Unnamed: 0", "outcome"],axis = 1), data_train["outcome"], data_test["outcome"]
x_test.columns = x_train.columns
x_train, y_train = model_smote.fit_resample(x_train, y_train)
min_max_scaler = preprocessing.MinMaxScaler()
x_train[x_train.columns] = min_max_scaler.fit_transform(x_train)
x_test[x_test.columns] = min_max_scaler.transform(x_test)
logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=0)
distributions = dict(C= uniform(loc=0, scale=4), penalty=['l2', 'l1'])
clf = RandomizedSearchCV(logistic, distributions, random_state=0,scoring="roc_auc")
search = clf.fit(x_train, y_train)
search.best_params_
clf = search.best_estimator_
log.append(clf.score(x_test, y_test))
k_range = [2,3, 5, 7, 9, 11,13,15]
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=5)

    cv_score = np.mean(scores)
    print('k={}，验证集上的准确率={:.3f}'.format(k, cv_score))
    cv_scores.append(cv_score)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
xyk.append(knn.score(x_test, y_test))
rt_n_estimators = [int(x) for x in np.linspace(1000, 7000, 18)]

rt_max_deep = [int(x) for x in np.linspace(3, 7, 5)]
rt_criterion = ['gini', 'entropy']
rt_min_samples_split = [int(x) for x in np.linspace(2, 10, 9)]
rt_min_impurity_decrease = [0.0, 0.05, 0.1]
rt_bootstrap = [True, False]
rf_grid = {'n_estimators': rt_n_estimators, 'criterion': rt_criterion, 'max_depth': rt_max_deep,
         'min_samples_split': rt_min_samples_split, 'min_samples_leaf': [1, 2, 3, 4],
         'min_impurity_decrease': [0.0, 0.05, 0.1], 'bootstrap': rt_bootstrap}


XG = XGBClassifier(random_state=0,use_label_encoder=False)
Xg_n_estimators = [int(x) for x in np.linspace(1000, 7000, 18)]
Xg_max_deep = [int(x) for x in np.linspace(3, 7, 5)]
gamma = [i * 0.1 for i in np.linspace(0.1, 1000, 201)]
rf_grid = {'n_estimators': Xg_n_estimators, 'max_depth': Xg_max_deep, 'gamma': gamma}
Xgb = RandomizedSearchCV(XG, rf_grid, n_iter=10, random_state=0, cv=5,scoring= "roc_auc")
Xgb.fit(x_train, y_train)
XG = Xgb.best_estimator_
XG.fit(x_train, y_train)
xyx.append(XG.score(x_test, y_test))

eclf = VotingClassifier(estimators=[('Rf', Rf), ("Xg", XG)], voting='soft', weights=[2, 1])
eclf.fit(x_train, y_train)
b4.append(eclf.score(x_test,y_test))

model = SVC(kernel='rbf', probability=True, tol=0.01)
param_grid = {'C': [i * 0.03 for i in np.linspace(1, 1000, 100)],
            'gamma': [i * 0.01 for i in np.linspace(0.1, 100, 100)], 'kernel': ['linear', 'poly', "rbf", "sigmoid"],
            "degree": [int(i) for i in np.linspace(3, 13, 10)]}
grid_search = RandomizedSearchCV(model, param_grid, n_iter=100, random_state=0, cv=5 ,scoring= "roc_auc")
grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
for para, val in list(best_parameters.items()):
    print(para, val)
model = grid_search.best_estimator_
model.fit(x_train,y_train)
xys.append(model.score(x_test, y_test))
qy.append(SpSe(y_test, XG.predict(x_test)))

feature_importances = [(feature, round(importance, 2)) for feature, importance in
                         zip(x_train.columns, Rf.feature_importances_)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
cumulative_importances = np.cumsum(sorted_importances)
number = np.where(cumulative_importances >=0.95)[0][0] + 1
important_feature_names = [feature[0] for feature in feature_importances[0:number]]

featr.append(important_feature_names)

feature_importances = [(feature, round(importance, 2)) for feature, importance in
                         zip(x_train.columns, XG.feature_importances_)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
cumulative_importances = np.cumsum(sorted_importances)
number = np.where(cumulative_importances >= 0.95)[0][0] + 1
important_feature_names = [feature[0] for feature in feature_importances[0:number]]
featx.append(important_feature_names)


data_train1= pd.read_csv(r"C:\Users\JQ\Desktop\3-1\3-10\trainc.csv")
data_test1= pd.read_csv(r"C:\Users\JQ\Desktop\3-1\3-10\testc.csv")
x_train1, x_test1, y_train1, y_test1= data_train1.drop(["Unnamed: 0", "outcome"], axis = 1), data_test1.drop(["Unnamed: 0", "outcome"],axis = 1), data_train1["outcome"], data_test1["outcome"]
iv = LogisticRegression(C = 1e5)
iv.fit(x_train1,y_train1)


import seaborn as sns
font1 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 18,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 10,
}
plt.figure(figsize=(8,8))
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(8,8))
viz = plot_roc_curve(knn, x_test, y_test,
                         name="KNN",
                        alpha=1, lw=1, ax=ax,color = sns.color_palette("Set2")[1])
viz = plot_roc_curve(model, x_test, y_test,
                         name="SVM",
                        alpha=1, lw=1, ax=ax,color = sns.color_palette("Set2")[2])
viz = plot_roc_curve(clf, x_test, y_test,
                         name="logistic",
                        alpha=1, lw=1, ax=ax,color = sns.color_palette("Set2")[3])
viz = plot_roc_curve(XG, x_test, y_test,
                         name="logistic",
                        alpha=1, lw=1, ax=ax,color = sns.color_palette("Set2")[4])
viz = plot_roc_curve(Rf, x_test, y_test,
                         name="RF",
                        alpha=1, lw=1, ax=ax,color = sns.color_palette("Set2")[5])
viz = plot_roc_curve(eclf, x_test, y_test,
                         name="vote",
                        alpha=1, lw=1, ax=ax,color = sns.color_palette("Set2")[6])
viz = plot_roc_curve(iv, x_test1, y_test1,
                         name="vote",
                        alpha=1, lw=1, ax=ax,color = "black")
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
       label='Chance', alpha=.8)

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       )
plt.title("Receiver operating characteristic",font1)
ax.set_xlabel('1-Specificity', font2)
ax.set_ylabel ("Sensitivity", font2)
ax.legend(loc="lower right", prop = font2)
plt.tight_layout()
plt.savefig(r"C:\Users\JQ\Desktop\3-1\3-10\roc.pdf")



'''from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
y_pred = clf.predict(x_test)
if hasattr(clf, "predict_proba"):
    prob_pos = clf.predict_proba(x_test)[:, 1]
else:  # use decision function
    prob_pos = clf.decision_function(x_test)
    prob_pos = \
        (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

clf_score = brier_score_loss(y_test, prob_pos, pos_label=1)
print("%s:" % name)
print("\tBrier: %1.3f" % (clf_score))
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % ("log", clf_score))
ax.set_ylabel("Fraction of positives")
ax.set_ylim([-0.05, 1.05])
ax.legend(loc="lower right")
ax.set_title('Calibration plots  (reliability curve)')
plt.show()'''