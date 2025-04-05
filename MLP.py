"""
基于神经网络的上市公司股权质押违约预测模型
包含数据预处理、模型训练、评估可视化和结果保存
"""

# ===============================导入必要库===============================
import os
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

# 创建保存结果的目录
os.makedirs("MLP", exist_ok=True)

# 配置可视化样式
plt.style.use("ggplot")
sns.set_palette("husl")


# ===============================数据预处理函数===============================
def preprocess_data(df):
    """处理带特殊字符的数值数据"""
    df = df.astype(str)
    df = df.apply(lambda x: x.str.replace(",", "").str.strip())
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df.astype(float)


# ===============================数据加载与预处理===============================
# 读取训练数据
train_df = pd.read_csv("data/train.csv", index_col="Stock code").pipe(preprocess_data)

# 分离特征和标签
X = train_df.iloc[:, :-1]  # 前60列为特征
y = train_df.iloc[:, -1]  # 最后一列为标签

# 分层分割数据集（80%训练+验证，20%最终测试）
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================特征标准化===============================
scaler = StandardScaler()
X_train_all_scaled = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test)

# 进一步分割训练集为训练和验证集
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_all_scaled,
    y_train_all,
    test_size=0.2,
    stratify=y_train_all,
    random_state=42,
)


# ===============================模型构建===============================
def build_model(input_dim):
    """构建多层感知机模型"""
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "auc"]
    )
    return model


model = build_model(60)

# ===============================训练配置===============================
# 早停法配置
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# 类别权重计算（处理不平衡数据）
class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train_all), y=y_train_all
)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# ===============================模型训练===============================
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


# ===============================训练过程可视化===============================
def plot_training_history(history):
    """绘制训练指标变化曲线"""
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


# ===============================模型评估函数===============================
def evaluate_model(model, X, y, threshold=0.5):
    """执行完整模型评估流程"""
    # 预测概率
    y_pred_proba = model.predict(X)

    # 计算指标
    acc = accuracy_score(y, (y_pred_proba > threshold).astype(int))
    auc = roc_auc_score(y, y_pred_proba)

    # 保存评估结果到文件
    with open("MLP/evaluation_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")

    print("\n评估结果:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")

    # 绘制混淆矩阵
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y, (y_pred_proba > threshold).astype(int))
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

    # 绘制ROC曲线
    RocCurveDisplay.from_predictions(y, y_pred_proba)
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    plt.savefig("MLP/roc_curve.png", dpi=300)
    plt.show()

    return acc, auc


# 在测试集上进行最终评估
test_acc, test_auc = evaluate_model(model, X_test_scaled, y_test)

# ===============================特征重要性可视化===============================
# （可选：添加更多可视化分析）
plt.figure(figsize=(10, 6))
plt.bar(
    range(len(model.layers[0].weights[0].numpy().mean(axis=1))),
    model.layers[0].weights[0].numpy().mean(axis=1),
)
plt.title("Feature Importance (First Layer Weights)")
plt.xlabel("Feature Index")
plt.ylabel("Average Weight")
plt.savefig("MLP/feature_importance.png", dpi=300)
plt.show()
