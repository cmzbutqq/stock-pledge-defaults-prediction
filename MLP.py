"""
基于多层感知机(MLP)的上市公司股权质押违约预测模型
包含数据预处理、模型训练、评估可视化、SHAP解释和结果保存
"""

# ===============================导入核心库===============================
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
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight  # 添加这行导入

# ===============================可视化库===============================
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================深度学习库===============================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ===============================SHAP解释库===============================
import shap

shap.initjs()

# 创建结果目录
os.makedirs("MLP", exist_ok=True)

# 配置可视化样式
plt.style.use("ggplot")
sns.set_palette("husl")


# ===============================数据预处理===============================
def preprocess_data(df):
    """
    数据预处理流程：
    1. 处理带特殊字符的数值数据
    2. 剔除ID列(第一列)
    3. 转换空值为NaN
    4. 统一转换为浮点型
    """
    # 剔除第一列(ID列)
    df = df.iloc[:, 1:]

    # 处理特殊字符
    df = df.astype(str)
    df = df.apply(lambda x: x.str.replace(",", "").str.strip())
    df = df.replace(r"^\s*$", np.nan, regex=True)

    return df.astype(float)


# ===============================数据加载与分割===============================
print("正在加载和预处理数据...")
train_df = pd.read_csv("data/train.csv").pipe(preprocess_data)

# 分离特征和标签
X = train_df.iloc[:, :-1]  # 特征列
y = train_df.iloc[:, -1]  # 标签列

# 分层分割数据集(80%训练+验证，20%测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 进一步分割训练集为训练和验证集
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
)


# ===============================模型构建===============================
def build_model(input_dim):
    """
    构建MLP模型架构：
    - 输入层: 60个特征
    - 隐藏层1: 64个神经元，ReLU激活，30% Dropout
    - 隐藏层2: 32个神经元，ReLU激活，30% Dropout
    - 输出层: 1个神经元，Sigmoid激活
    - 优化器: Adam(lr=0.001)
    - 损失函数: 二分类交叉熵
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

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "auc"],
    )
    return model


print("构建模型中...")
model = build_model(X_train.shape[1])

# ===============================模型训练===============================
# 类别权重处理不平衡数据
print("计算类别权重...")
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"类别权重: {class_weights_dict}")

# 早停法防止过拟合
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

print("开始训练模型...")
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
    """绘制训练过程中的损失和AUC曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # AUC曲线
    ax2.plot(history.history["auc"], label="Train AUC")
    ax2.plot(history.history["val_auc"], label="Validation AUC")
    ax2.set_title("Training and Validation AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("MLP/training_history.png", dpi=300)
    plt.show()


plot_training_history(history)


# ===============================模型评估===============================
def evaluate_model(model, X, y, threshold=0.5):
    """
    综合模型评估：
    1. 计算准确率、AUC、精确率、召回率、F1等指标
    2. 绘制混淆矩阵和ROC曲线
    3. 保存所有评估结果到文件
    """
    # 预测概率
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba > threshold).astype(int)
    # 计算各项指标
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred, output_dict=True)
    # 获取类别标签（处理不同类型的标签）
    class_labels = [str(cls) for cls in sorted(np.unique(y))]
    # 保存评估结果
    with open("MLP/evaluation_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n\n")
        f.write("Classification Report:\n")
        # 为每个类别写入指标
        for cls in class_labels:
            f.write(f"\nClass {cls}:\n")
            f.write(f"Precision: {cr[cls]['precision']:.4f}\n")
            f.write(f"Recall: {cr[cls]['recall']:.4f}\n")
            f.write(f"F1-score: {cr[cls]['f1-score']:.4f}\n")
            f.write(f"Support: {cr[cls]['support']}\n")
        # 写入宏观和加权平均
        f.write("\nMacro Avg:\n")
        f.write(f"Precision: {cr['macro avg']['precision']:.4f}\n")
        f.write(f"Recall: {cr['macro avg']['recall']:.4f}\n")
        f.write(f"F1-score: {cr['macro avg']['f1-score']:.4f}\n")
        f.write(f"Support: {cr['macro avg']['support']}\n")
        f.write("\nWeighted Avg:\n")
        f.write(f"Precision: {cr['weighted avg']['precision']:.4f}\n")
        f.write(f"Recall: {cr['weighted avg']['recall']:.4f}\n")
        f.write(f"F1-score: {cr['weighted avg']['f1-score']:.4f}\n")
        f.write(f"Support: {cr['weighted avg']['support']}\n")
        # 添加混淆矩阵表格
        f.write("\n\nConfusion Matrix:\n")
        f.write("-----------------\n")
        f.write("|               | Predicted 0 | Predicted 1 |\n")
        f.write("|---------------|-------------|-------------|\n")
        f.write(f"| Actual 0      | {cm[0,0]:^11} | {cm[0,1]:^11} |\n")
        f.write(f"| Actual 1      | {cm[1,0]:^11} | {cm[1,1]:^11} |\n")
        f.write("-----------------\n")
    # 打印关键指标
    print("\n评估指标:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y, y_pred))
    print("\n混淆矩阵:")
    print("-----------------")
    print("|               | Predicted 0 | Predicted 1 |")
    print("|---------------|-------------|-------------|")
    print(f"| Actual 0      | {cm[0,0]:^11} | {cm[0,1]:^11} |")
    print(f"| Actual 1      | {cm[1,0]:^11} | {cm[1,1]:^11} |")
    print("-----------------")
    # 绘制混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Default"],
        yticklabels=["Normal", "Default"],
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


print("\n在测试集上评估模型...")
test_acc, test_auc = evaluate_model(model, X_test_scaled, y_test)


# ===============================SHAP分析===============================
def perform_shap_analysis(model, X_train_sample, X_test_sample, feature_names):
    """
    执行SHAP分析：
    1. 计算特征重要性
    2. 可视化全局特征重要性
    3. 可视化个体样本解释
    4. 将特征重要性保存到evaluation_results.txt
    """
    print("\n正在进行SHAP分析...")
    try:
        # 创建解释器
        background = (
            X_train_sample
            if isinstance(X_train_sample, np.ndarray)
            else np.array(X_train_sample)
        )
        test_data = (
            X_test_sample
            if isinstance(X_test_sample, np.ndarray)
            else np.array(X_test_sample)
        )
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(test_data[:50])
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        # 计算全局特征重要性
        global_shap_values = np.abs(shap_values).mean(axis=0)
        if global_shap_values.ndim > 1:
            global_shap_values = global_shap_values.flatten()
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": global_shap_values}
        ).sort_values("importance", ascending=False)
        # 将特征重要性追加到evaluation_results.txt
        with open("MLP/evaluation_results.txt", "a") as f:
            f.write("\n\n=== SHAP特征重要性 ===\n")
            f.write("特征名称\t重要性得分\n")
            for _, row in feature_importance.iterrows():
                f.write(f"{row['feature']}\t{row['importance']:.6f}\n")
        print("特征重要性已保存到evaluation_results.txt")
        # 自定义SHAP重要性柱状图
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)  # 只显示最重要的20个特征
        plt.barh(
            top_features["feature"][::-1],
            top_features["importance"][::-1],
            color="#1f77b4",
        )
        plt.xlabel("SHAP Value (mean absolute impact on model output)")
        plt.title("Top 20 Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig("MLP/shap_feature_importance.png", dpi=300)
        plt.show()
        # 个体样本解释
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, test_data[:50], feature_names=feature_names, show=False
        )
        plt.title("Feature Impact on Model Output (SHAP)")
        plt.tight_layout()
        plt.savefig("MLP/shap_summary_plot.png", dpi=300)
        plt.show()
        # 保存SHAP值
        np.save("MLP/shap_values.npy", shap_values)
        print("SHAP分析完成!")
    except Exception as e:
        print(f"SHAP分析出错: {str(e)}")
        import traceback

        traceback.print_exc()


# 使用子样本进行分析
X_train_sample = X_train_scaled[:100]  # 背景样本
X_test_sample = X_test_scaled[:50]  # 解释样本
perform_shap_analysis(model, X_train_sample, X_test_sample, X.columns.tolist())
