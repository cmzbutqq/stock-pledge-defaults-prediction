# https://www.kaggle.com/code/jiaoyouzhang/stock-pledge-default-prediction-xgboost?scriptVersionId=202751418&cellId=1
# 注意查重！
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import graphviz
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn import tree
from IPython.display import Image
import shap
import matplotlib

matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.unicode_minus"] = False

data = pd.read_csv(
    "data/train.csv", encoding="utf-8"
)
df = pd.DataFrame(data)
df = df.replace(",", "", regex=True)

X_train = np.array(df.drop(["Stock code", "IsDefault"], axis=1))
y_train = np.array(df["IsDefault"]).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train)


ctest = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "booster": "gbtree",
    "max_depth": 5,
    "eta": 0.3,
    "objective": "binary:logistic",
    "min_child_weight": 2,
    #'colsample_bytree':0.85,
}
# train model
bst = xgb.train(params, dtrain, num_boost_round=200)
# prediction
y_prec = bst.predict(ctest)
y_pred = bst.predict(dtest)
# pd.DataFrame(y_pred).to_csv('xgboost_probit.csv')
# ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("roc_auc", roc_auc)
ks_statistic = max(tpr - fpr)
print("KS value:", ks_statistic)


# ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()


y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy: %.4f" % accuracy)
print("Precision: %.4f" % precision)
print("Recall: %.4f" % recall)
print("F1-score: %.4f" % f1)

# feature_importance
xgb.plot_importance(bst, max_num_features=100, show_values=True, grid=False)
plt.rcParams["axes.labelcolor"] = "black"
plt.show()

feature_importance = bst.get_score(importance_type="weight")
for key, value in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print("%s %f" % (key, value))


xgb.plot_tree(bst, tree_index=1)  # the first decision tree
plt.figure(figsize=(12, 6))
plt.rcParams["axes.labelcolor"] = "black"
plt.show()

# SHAP
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X_test)
column_names = df.drop(["Stock code", "IsDefault"], axis=1).columns.tolist()
shap.summary_plot(shap_values, X_test, feature_names=column_names, show=True)
