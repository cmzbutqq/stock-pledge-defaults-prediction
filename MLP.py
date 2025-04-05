"""
基于神经网络的上市公司股权质押违约预测模型
包含数据预处理、模型训练、评估可视化和预测结果输出
"""

# 导入必要库
# ===============================核心库===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)
from sklearn.utils.class_weight import compute_class_weight

# ===============================可视化库===============================
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================深度学习库===============================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 配置可视化样式
plt.style.use("ggplot")
sns.set_palette("husl")


# 数据预处理模块
# ===================================================================
def preprocess_data(df):
    """
    数据预处理函数
    功能：
    1. 处理带有千分位逗号的数值字符串
    2. 转换空值为NaN
    3. 统一转换为浮点型数据

    参数：
    df -- 原始数据DataFrame

    返回：
    处理后的干净DataFrame
    """
    # 统一转为字符串类型便于处理特殊字符
    df = df.astype(str)

    # 移除数值中的逗号和空格（处理千分位表示问题）
    df = df.apply(lambda x: x.str.replace(",", "").str.strip())

    # 将空字符串转换为NaN（统一缺失值表示）
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # 转换为浮点型数据
    return df.astype(float)


# 数据加载与预处理
# ===================================================================
# 读取原始数据（注意文件路径需要根据实际情况调整）
train_df = pd.read_csv("data/train.csv", index_col="Stock code").pipe(preprocess_data)
test_df = pd.read_csv("data/test.csv", index_col="Stock code").pipe(preprocess_data)

# 特征工程
# ===================================================================
# 分离特征和标签
X_train = train_df.iloc[:, :-1]  # 前60列为特征(0-59索引)
y_train = train_df.iloc[:, -1]  # 最后一列为标签(索引60)
X_test = test_df  # 测试集无标签

# 缺失值处理
# ===================================================================
# 确定需要填补的列（根据问题描述为第52-58列，对应索引51-57）
cols_to_impute = X_train.columns[51:58]

# 使用训练集均值填补测试集缺失值
for col in cols_to_impute:
    mean_val = X_train[col].mean()
    X_test.loc[:, col] = X_test[col].fillna(mean_val)

# 特征标准化
# ===================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 在训练集上计算均值和方差
X_test_scaled = scaler.transform(X_test)  # 在测试集上应用相同变换

# 数据集划分
# ===================================================================
# 分层抽样保持类别分布
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled,
    y_train,
    test_size=0.2,
    stratify=y_train,  # 保持违约/正常比例
    random_state=42,
)


# 模型构建
# ===================================================================
def build_model(input_dim):
    """
    构建神经网络模型
    结构：
    - 输入层：64个神经元，ReLU激活
    - Dropout层：0.3丢弃率
    - 隐藏层：32个神经元，ReLU激活
    - Dropout层：0.3丢弃率
    - 输出层：1个神经元，Sigmoid激活（二分类输出）

    返回：
    编译好的模型实例
    """
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    # 配置优化器
    optimizer = Adam(learning_rate=0.001)

    # 编译模型
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "auc"]
    )
    return model


model = build_model(60)  # 输入维度60个特征

# 训练配置
# ===================================================================
# 早停法：当验证损失连续10轮不下降时停止训练，并恢复最佳权重
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# 类别权重：处理不平衡数据
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# 模型训练
# ===================================================================
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


# 训练过程可视化
# ===================================================================
def plot_training_history(history):
    """
    绘制训练过程指标变化曲线
    包含：
    - 训练集 vs 验证集的损失曲线
    - 训练集 vs 验证集的AUC曲线
    """
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # AUC曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history["auc"], label="Train AUC")
    plt.plot(history.history["val_auc"], label="Validation AUC")
    plt.title("AUC Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend()

    plt.tight_layout()
    plt.savefig("MLP/training_history.png", dpi=300)
    plt.show()


plot_training_history(history)


# 模型评估
# ===================================================================
def evaluate_model(model, X, y, threshold=0.5):
    """
    综合模型评估函数
    输出：
    - 准确率
    - AUC值
    - 混淆矩阵热力图
    - ROC曲线
    """
    # 预测概率
    y_pred_proba = model.predict(X)

    # 计算指标
    acc = accuracy_score(y, (y_pred_proba > threshold).astype(int))
    auc = roc_auc_score(y, y_pred_proba)

    print(f"\n评估结果:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y, (y_pred_proba > threshold).astype(int))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Confusion Matrix")
    plt.savefig("MLP/confusion_matrix.png", dpi=300)
    plt.show()

    # ROC曲线
    RocCurveDisplay.from_predictions(y, y_pred_proba)
    plt.title("ROC Curve")
    plt.plot([0, 1], [0, 1], "k--")  # 添加对角线参考线
    plt.savefig("MLP/roc_curve.png", dpi=300)
    plt.show()

    return acc, auc


# 在验证集上执行评估
val_acc, val_auc = evaluate_model(model, X_val_split, y_val_split)

# 测试集预测
# ===================================================================
test_pred = model.predict(X_test_scaled)
result_df = pd.DataFrame(
    {"Stock code": test_df.index, "DefaultProbability": test_pred.flatten()}
)

# 保存预测结果
result_df.to_csv("MLP/predictions.csv", index=False)
print("\n预测结果已保存至 predictions.csv")
