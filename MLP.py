# 导入必要库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# 读取数据并预处理
# --------------------------------------------------
def preprocess_data(df):
    # 转换所有列为字符串类型，方便统一处理
    df = df.astype(str)
    # 移除数值中的逗号和多余空格
    df = df.apply(lambda x: x.str.replace(',', '').str.strip())
    # 将空字符串转换为NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # 转换数据类型为float
    return df.astype(float)

# 读取数据集，设置股票代码为索引
train_df = pd.read_csv('data/train.csv', index_col='Stock code').pipe(preprocess_data)
test_df = pd.read_csv('data/test.csv', index_col='Stock code').pipe(preprocess_data)

# 分离特征和标签
X_train = train_df.iloc[:, :-1]  # 前60列为特征
y_train = train_df.iloc[:, -1]   # 最后一列为标签
X_test = test_df                 # 测试集无标签

# 处理缺失值
# --------------------------------------------------
# 获取需要填补的列
cols_to_impute = X_train.columns[51:58]

# 用训练集对应列的均值填补测试集缺失值
for col in cols_to_impute:
    mean_val = X_train[col].mean()
    X_test.loc[:, col] = X_test[col].fillna(mean_val)

# 特征标准化
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# 划分训练集和验证集
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# 构建神经网络模型
# --------------------------------------------------
model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(60,)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ]
)

# 设置优化器和早停法
optimizer = Adam(learning_rate=0.001)
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# 计算类别权重处理不平衡数据
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# 编译模型
model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "AUC"]
)

# 训练模型
# --------------------------------------------------
history = model.fit(
    X_train_split,
    y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    class_weight=class_weights_dict,
    verbose=1,
)

# 验证集评估
# --------------------------------------------------
y_val_pred = model.predict(X_val_split)
val_acc = accuracy_score(y_val_split, (y_val_pred > 0.5).astype(int))
val_auc = roc_auc_score(y_val_split, y_val_pred)
print(f"Validation Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")

# 测试集预测
# --------------------------------------------------
test_pred = model.predict(X_test_scaled)
result_df = pd.DataFrame(
    {"Stock code": test_df.index, "DefaultProbability": test_pred.flatten()}
)

# 保存预测结果
result_df.to_csv("predictions.csv", index=False)
print("预测结果已保存至 predictions.csv")
