"""
基于多层感知机(MLP)的上市公司股权质押违约预测模型
包含数据预处理、模型训练、评估可视化、SHAP解释、内生性检验和结果保存
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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf

# ===============================SHAP解释库===============================
import shap

# ===============================内生性检验库===============================
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

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


# ==========================变量描述性统计===============================
def generate_variable_stats(df, output_file="MLP/variable_stats.csv"):
    """
    生成变量的描述性统计信息并保存到CSV
    包括均值、标准差、最小值、最大值等
    """
    stats_df = pd.DataFrame(
        {
            "Mean": df.mean(),
            "Std": df.std(),
            "Min": df.min(),
            "25%": df.quantile(0.25),
            "50%": df.median(),
            "75%": df.quantile(0.75),
            "Max": df.max(),
        }
    )
    stats_df.to_csv(output_file)
    print(f"变量统计信息已保存到 {output_file}")
    return stats_df


# 在数据预处理后调用
print("正在生成变量统计信息...")
generate_variable_stats(X)


# ===============================模型构建===============================
def build_model(input_dim):
    model = Sequential(
        [
            Dense(
                128,
                activation="relu",
                input_shape=(input_dim,),
                kernel_regularizer=l2(0.01),
            ),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation="relu", kernel_regularizer=l2(0.005)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            "auc",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
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
# 改进的早停策略
early_stop = EarlyStopping(
    monitor="val_auc", patience=15, mode="max", restore_best_weights=True, verbose=1
)


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
    """绘制训练过程中的损失和AUC曲线，并保存结果到CSV"""
    # 创建历史数据DataFrame
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = history.epoch

    # 保存训练历史到CSV
    history_df.to_csv("MLP/training_history.csv", index=False)

    # 绘制图表（原有代码保持不变）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 损失曲线
    ax1.plot(history_df["loss"], label="Train Loss")
    ax1.plot(history_df["val_loss"], label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # AUC曲线
    ax2.plot(history_df["auc"], label="Train AUC")
    ax2.plot(history_df["val_auc"], label="Validation AUC")
    ax2.set_title("Training and Validation AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("MLP/training_history.png", dpi=300)
    plt.show()

    return history_df


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
    2. 可视化全局特征重要性（自动处理长特征名）
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

        # 保存特征重要性
        with open("MLP/evaluation_results.txt", "a") as f:
            f.write("\n\n=== SHAP特征重要性 ===\n")
            f.write("特征名称\t重要性得分\n")
            for _, row in feature_importance.iterrows():
                f.write(f"{row['feature']}\t{row['importance']:.6f}\n")
        print("特征重要性已保存到evaluation_results.txt")

        # 自定义SHAP重要性柱状图（处理长特征名）
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)

        # 方法1：截断长名称
        truncated_names = [
            name[:30] + "..." if len(name) > 30 else name
            for name in top_features["feature"][::-1]
        ]

        # 方法2：自动换行（任选一种）
        # import textwrap
        # truncated_names = ['\n'.join(textwrap.wrap(name, width=25))
        #                  for name in top_features["feature"][::-1]]

        plt.barh(truncated_names, top_features["importance"][::-1], color="#1f77b4")
        plt.xlabel("SHAP Value (mean absolute impact on model output)")
        plt.title("Top 20 Feature Importance (SHAP)")
        plt.subplots_adjust(left=0.3)  # 调整左边距
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


def robustness_test(model, X_test_scaled, y_test, n_iterations=100):
    """
    执行多维度稳健性检验：
    1. 随机分割测试集
    2. 添加随机噪声
    3. 模拟特征缺失
    4. 模拟异常值
    5. 计算性能指标的均值和标准差
    """
    print("\n正在进行多维度稳健性检验...")

    # 初始化结果字典
    metrics = {
        "original": {"accuracy": [], "auc": []},
        "noise": {"accuracy": [], "auc": []},
        "missing": {"accuracy": [], "auc": []},
        "outliers": {"accuracy": [], "auc": []},
    }

    # 确保数据是numpy数组
    X_test_scaled = np.array(X_test_scaled)
    y_test = np.array(y_test)

    # 获取特征数量
    n_features = X_test_scaled.shape[1]

    try:
        for i in range(n_iterations):
            # 1. 原始数据测试（随机分割）
            _, X_test_sample, _, y_test_sample = train_test_split(
                X_test_scaled, y_test, test_size=0.5, stratify=y_test, random_state=i
            )

            # 评估原始数据
            y_pred_proba = model.predict(X_test_sample, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int)

            metrics["original"]["accuracy"].append(
                accuracy_score(y_test_sample, y_pred)
            )
            metrics["original"]["auc"].append(
                roc_auc_score(y_test_sample, y_pred_proba)
            )

            # 2. 添加随机噪声测试
            noise_level = 0.05  # 5%的噪声水平
            X_noise = X_test_sample.copy()
            noise = np.random.normal(0, noise_level, X_noise.shape)
            X_noise = X_noise + noise

            # 评估带噪声数据
            y_pred_proba_noise = model.predict(X_noise, verbose=0)
            y_pred_noise = (y_pred_proba_noise > 0.5).astype(int)

            metrics["noise"]["accuracy"].append(
                accuracy_score(y_test_sample, y_pred_noise)
            )
            metrics["noise"]["auc"].append(
                roc_auc_score(y_test_sample, y_pred_proba_noise)
            )

            # 3. 模拟特征缺失测试
            missing_ratio = 0.1  # 10%的特征缺失
            n_missing_features = int(n_features * missing_ratio)
            missing_indices = np.random.choice(
                n_features, n_missing_features, replace=False
            )

            X_missing = X_test_sample.copy()
            X_missing[:, missing_indices] = 0  # 将缺失特征设为0

            # 评估特征缺失数据
            y_pred_proba_missing = model.predict(X_missing, verbose=0)
            y_pred_missing = (y_pred_proba_missing > 0.5).astype(int)

            metrics["missing"]["accuracy"].append(
                accuracy_score(y_test_sample, y_pred_missing)
            )
            metrics["missing"]["auc"].append(
                roc_auc_score(y_test_sample, y_pred_proba_missing)
            )

            # 4. 模拟异常值测试
            outlier_ratio = 0.05  # 5%的样本包含异常值
            n_outlier_samples = int(X_test_sample.shape[0] * outlier_ratio)
            outlier_indices = np.random.choice(
                X_test_sample.shape[0], n_outlier_samples, replace=False
            )

            X_outliers = X_test_sample.copy()
            for idx in outlier_indices:
                # 随机选择特征并添加异常值
                feature_idx = np.random.randint(0, n_features)
                X_outliers[idx, feature_idx] = (
                    X_outliers[idx, feature_idx] * 10
                )  # 放大10倍

            # 评估异常值数据
            y_pred_proba_outliers = model.predict(X_outliers, verbose=0)
            y_pred_outliers = (y_pred_proba_outliers > 0.5).astype(int)

            metrics["outliers"]["accuracy"].append(
                accuracy_score(y_test_sample, y_pred_outliers)
            )
            metrics["outliers"]["auc"].append(
                roc_auc_score(y_test_sample, y_pred_proba_outliers)
            )

            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"完成 {i+1}/{n_iterations} 次迭代")

        # 计算统计量
        robustness_results = {
            "original": {
                "mean_accuracy": np.mean(metrics["original"]["accuracy"]),
                "std_accuracy": np.std(metrics["original"]["accuracy"]),
                "mean_auc": np.mean(metrics["original"]["auc"]),
                "std_auc": np.std(metrics["original"]["auc"]),
            },
            "noise": {
                "mean_accuracy": np.mean(metrics["noise"]["accuracy"]),
                "std_accuracy": np.std(metrics["noise"]["accuracy"]),
                "mean_auc": np.mean(metrics["noise"]["auc"]),
                "std_auc": np.std(metrics["noise"]["auc"]),
            },
            "missing": {
                "mean_accuracy": np.mean(metrics["missing"]["accuracy"]),
                "std_accuracy": np.std(metrics["missing"]["accuracy"]),
                "mean_auc": np.mean(metrics["missing"]["auc"]),
                "std_auc": np.std(metrics["missing"]["auc"]),
            },
            "outliers": {
                "mean_accuracy": np.mean(metrics["outliers"]["accuracy"]),
                "std_accuracy": np.std(metrics["outliers"]["accuracy"]),
                "mean_auc": np.mean(metrics["outliers"]["auc"]),
                "std_auc": np.std(metrics["outliers"]["auc"]),
            },
            "n_iterations": n_iterations,
        }

        # 保存结果
        results_df = pd.DataFrame(
            {
                "Test Type": ["Original", "Noise", "Missing", "Outliers"],
                "Mean Accuracy": [
                    robustness_results["original"]["mean_accuracy"],
                    robustness_results["noise"]["mean_accuracy"],
                    robustness_results["missing"]["mean_accuracy"],
                    robustness_results["outliers"]["mean_accuracy"],
                ],
                "Std Accuracy": [
                    robustness_results["original"]["std_accuracy"],
                    robustness_results["noise"]["std_accuracy"],
                    robustness_results["missing"]["std_accuracy"],
                    robustness_results["outliers"]["std_accuracy"],
                ],
                "Mean AUC": [
                    robustness_results["original"]["mean_auc"],
                    robustness_results["noise"]["mean_auc"],
                    robustness_results["missing"]["mean_auc"],
                    robustness_results["outliers"]["mean_auc"],
                ],
                "Std AUC": [
                    robustness_results["original"]["std_auc"],
                    robustness_results["noise"]["std_auc"],
                    robustness_results["missing"]["std_auc"],
                    robustness_results["outliers"]["std_auc"],
                ],
            }
        )

        results_df.to_csv("MLP/robustness_test_results.csv", index=False)

        # 绘制稳健性检验结果
        plt.figure(figsize=(12, 8))

        # 准确率比较
        plt.subplot(2, 1, 1)
        test_types = ["Original", "Noise", "Missing", "Outliers"]
        accuracies = [
            robustness_results["original"]["mean_accuracy"],
            robustness_results["noise"]["mean_accuracy"],
            robustness_results["missing"]["mean_accuracy"],
            robustness_results["outliers"]["mean_accuracy"],
        ]
        acc_stds = [
            robustness_results["original"]["std_accuracy"],
            robustness_results["noise"]["std_accuracy"],
            robustness_results["missing"]["std_accuracy"],
            robustness_results["outliers"]["std_accuracy"],
        ]

        plt.bar(test_types, accuracies, yerr=acc_stds, capsize=5, color="skyblue")
        plt.title("Accuracy Comparison Across Different\nData Perturbations")
        plt.ylabel("Accuracy")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # AUC比较
        plt.subplot(2, 1, 2)
        aucs = [
            robustness_results["original"]["mean_auc"],
            robustness_results["noise"]["mean_auc"],
            robustness_results["missing"]["mean_auc"],
            robustness_results["outliers"]["mean_auc"],
        ]
        auc_stds = [
            robustness_results["original"]["std_auc"],
            robustness_results["noise"]["std_auc"],
            robustness_results["missing"]["std_auc"],
            robustness_results["outliers"]["std_auc"],
        ]

        plt.bar(test_types, aucs, yerr=auc_stds, capsize=5, color="lightgreen")
        plt.title("AUC Comparison Across Different\nData Perturbations")
        plt.ylabel("AUC")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig("MLP/robustness_test_results.png", dpi=300)
        plt.show()

        # 打印结果
        print(f"稳健性检验完成 (迭代次数={n_iterations})")
        print("\n原始数据:")
        print(
            f"准确率: {robustness_results['original']['mean_accuracy']:.4f} ± "
            f"{robustness_results['original']['std_accuracy']:.4f}"
        )
        print(
            f"AUC: {robustness_results['original']['mean_auc']:.4f} ± "
            f"{robustness_results['original']['std_auc']:.4f}"
        )

        print("\n添加噪声数据:")
        print(
            f"准确率: {robustness_results['noise']['mean_accuracy']:.4f} ± "
            f"{robustness_results['noise']['std_accuracy']:.4f}"
        )
        print(
            f"AUC: {robustness_results['noise']['mean_auc']:.4f} ± "
            f"{robustness_results['noise']['std_auc']:.4f}"
        )

        print("\n特征缺失数据:")
        print(
            f"准确率: {robustness_results['missing']['mean_accuracy']:.4f} ± "
            f"{robustness_results['missing']['std_accuracy']:.4f}"
        )
        print(
            f"AUC: {robustness_results['missing']['mean_auc']:.4f} ± "
            f"{robustness_results['missing']['std_auc']:.4f}"
        )

        print("\n异常值数据:")
        print(
            f"准确率: {robustness_results['outliers']['mean_accuracy']:.4f} ± "
            f"{robustness_results['outliers']['std_accuracy']:.4f}"
        )
        print(
            f"AUC: {robustness_results['outliers']['mean_auc']:.4f} ± "
            f"{robustness_results['outliers']['std_auc']:.4f}"
        )

        return robustness_results

    except Exception as e:
        print(f"稳健性检验过程中出现错误: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


# 在模型评估后调用
robustness_results = robustness_test(model, X_test_scaled, y_test)


# ===============================内生性检验===============================
def perform_endogeneity_tests(X, y, feature_names, output_dir="MLP"):
    """
    执行内生性检验：
    1. 多重共线性检验 (VIF)
    2. 保存检验结果
    """
    print("\n正在进行内生性检验...")

    # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)

    # 添加常数项
    X_with_const = add_constant(X)

    # 1. 多重共线性检验 (VIF)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = ["const"] + feature_names
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_const, i) for i in range(X_with_const.shape[1])
    ]

    # 保存VIF结果
    vif_data.to_csv(f"{output_dir}/vif_results.csv", index=False)

    # 绘制VIF柱状图
    plt.figure(figsize=(12, 8))

    # 处理长变量名，类似于SHAP特征重要性图表
    truncated_names = [
        name[:30] + "..." if len(name) > 30 else name for name in vif_data["Variable"]
    ]

    sns.barplot(x="VIF", y=truncated_names, data=vif_data)
    plt.title("Variance Inflation Factors")
    plt.axvline(x=5, color="r", linestyle="--", label="VIF=5(Threshold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vif_plot.png", dpi=300)
    plt.show()

    # 保存内生性检验报告
    with open(f"{output_dir}/endogeneity_test_report.txt", "w") as f:
        f.write("=== 内生性检验报告 ===\n\n")

        f.write("1. 多重共线性检验 (VIF)\n")
        f.write("-------------------\n")
        f.write("VIF > 5 表示存在严重的多重共线性问题\n\n")
        for _, row in vif_data.iterrows():
            f.write(f"{row['Variable']}: {row['VIF']:.4f}\n")

        f.write("\n2. 综合结论\n")
        f.write("-------------------\n")
        f.write("根据VIF检验结果，以下变量可能存在多重共线性问题：\n")
        high_vif_vars = vif_data[vif_data["VIF"] > 5]["Variable"].tolist()
        if high_vif_vars:
            for var in high_vif_vars:
                f.write(
                    f"- {var} (VIF: {vif_data[vif_data['Variable'] == var]['VIF'].values[0]:.4f})\n"
                )
        else:
            f.write("未发现明显的多重共线性问题。\n")

    print(f"内生性检验完成，结果已保存到 {output_dir}/endogeneity_test_report.txt")

    return {"vif_results": vif_data}


# 在模型评估后调用内生性检验
print("\n执行内生性检验...")
endogeneity_results = perform_endogeneity_tests(
    X_train_scaled, y_train, X.columns.tolist()
)
