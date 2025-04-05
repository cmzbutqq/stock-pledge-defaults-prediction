import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#读取文件
df_train = pd.read_csv("data/train.csv")
df_test=pd.read_csv("data/test.csv")

target = 'IsDefault'   #只有df_train中有

#df_test中有六列有大量缺失值,后续分析中需要特别注意。这里先删除这些列
df_test_no_value=['Tobin Q','Debt financing costs','Enterprise age','Goodwill impairment ratio','Asset quality index','SG&A Expense']
df_train= df_train.drop(df_test_no_value, axis=1)
df_test= df_test.drop(df_test_no_value, axis=1)

#数据集中的股票代码不重要。Z-score意义不明，先进行删除
df_train = df_train.drop(['Stock code'], axis=1)
df_test = df_test.drop(['Stock code'], axis=1)
df_train = df_train.drop(['Z-SCORE'], axis=1)
df_test = df_test.drop(['Z-SCORE'], axis=1)


# 数值型与类别型变量区分
#把训练集中的所有数值型变量的列名编为列表储存再num_cols中，把所有字符型变量的列名储存再cat_cols中
num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove(target)  # 除去目标变量
#经过调试，发现只有P/E ratio这一个变量被识别为字符型变量


df_train['P/E ratio'] = pd.to_numeric(df_train['P/E ratio'].str.replace(',', ''), errors='raise')
df_test['P/E ratio'] = pd.to_numeric(df_test['P/E ratio'].str.replace(',', ''), errors='raise')

num_cols.append('P/E ratio')


# 保证测试集和训练集的列一致（有些变量在测试集可能缺失）
df_test = df_test.reindex(columns=df_train.columns.drop(target), fill_value=0)

# 缺失值填补（这里简单地用均值填补）
df_train = df_train.fillna(df_train.mean(numeric_only=True))
df_test = df_test.fillna(df_train.mean(numeric_only=True))  # 用训练集均值填补测试集

# 对于数值形的列，让它们都进行标准化。标准方法：减去均值再除以标准差
scaler = StandardScaler()
df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
df_test[num_cols] = scaler.transform(df_test[num_cols])

#训练集中分割出目标变量和其它变量
X = df_train.drop(columns=[target])
y = df_train[target]


#把训练集分为两部分，一部分用来训练，一部分用来检验训练效果
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


#建立逻辑回归模型，其中class_weight='balanced'是自动选择权重，避免由于目标变量为0的值占大多数而影响模型功能
#random_state=42是随机种子，保证每次运行结果一致
#max_iter=1000是最大迭代次数，避免模型训练时间过长
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)     #训练模型

# 输出类别预测结果（0或1）
y_pred = model.predict(X_val)
# 输出属于类别1的概率
y_prob = model.predict_proba(X_val)[:, 1]


#汇报Classification Report,输出的结果是评估指标。
# 其中precision是精确率,预测为该类的样本中，真实为该类的比例
# recall是召回率，表示真实为改类的样本中，被正确预测的比例
# f1-score是f1值，是recall和precision的调和平均数
# support是支持数，是每个类别真实的样本数
print("Classification Report:\n", classification_report(y_val, y_pred))

#汇报 AUC Score
#举例：AUC=0.9：模型有90%的概率能够正确区分正负类。AUC=0.5：模型无区分能力（相当于随机猜测）。
print("AUC Score:", roc_auc_score(y_val, y_prob))

# ROC曲线
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='ROC Curve (AUC = %.3f)' % roc_auc_score(y_val, y_prob))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()
#ROC曲线越靠近左上角，模型效果越好


#报告的分析
#模型 很擅长发现违约（recall高），但有时把“好人”也误判成“坏人”（precision较低）。
#在某些金融场景下（比如贷款审批），这种倾向可能是可以接受的，因为宁可误杀也不放过。但要根据业务来判断




#对测试集进行预测，暂时还不需要
#test_pred_prob = model.predict_proba(df_test)[:, 1]
#df_result = pd.DataFrame({'Default_Probability': test_pred_prob})
#df_result.to_csv("逻辑回归违约预测结果.csv", index=False)