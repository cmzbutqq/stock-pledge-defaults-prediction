"""
基于多层感知机(MLP)的上市公司股权质押违约预测模型
整合计量经济学分析：基准回归、内生性检验、机制分析、稳健性检验
"""

# ======================== 导入库 ========================
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import shap

shap.initjs()

# ======================== 配置 ========================
os.makedirs("MLP/results/econometric", exist_ok=True)  # 计量分析结果目录
plt.style.use("ggplot")
sns.set_palette("husl")


# ======================== 数据预处理 ========================
def preprocess_data(df):
    """增强型数据预处理"""
    df = df.iloc[:, 1:]  # 剔除ID列
    df = df.astype(str).apply(lambda x: x.str.replace(",", "").str.strip())
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df.astype(float)


# ======================== 计量经济学分析类 ========================
class EconometricAnalyzer:
    def __init__(self, model, X, y, feature_names):
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.results = {}

    def baseline_regression(self):
        """基准回归分析（传统逻辑回归对比）"""
        print("\n=== 基准回归分析 ===")
        X_sm = sm.add_constant(self.X)
        logit_model = sm.Logit(self.y, X_sm)
        logit_result = logit_model.fit(disp=0)

        # 保存结果
        summary = logit_result.summary2().tables[1]
        summary.to_csv("MLP/results/econometric/baseline_regression.csv")

        # 比较MLP与传统模型
        mlp_pred = self.model.predict(self.X)
        logit_pred = logit_model.predict(logit_result.params)

        comparison = pd.DataFrame(
            {
                "MLP_AUC": roc_auc_score(self.y, mlp_pred),
                "Logit_AUC": roc_auc_score(self.y, logit_pred),
            },
            index=[0],
        )
        comparison.to_csv("MLP/results/econometric/model_comparison.csv")

        self.results["baseline"] = {
            "logit_summary": summary,
            "model_comparison": comparison,
        }
        return self

    def endogeneity_test(self, instrument_idx=0, endogenous_idx=1):
        """简化版内生性检验（需要指定工具变量）"""
        print("\n=== 内生性检验 ===")
        try:
            # 工具变量法（两阶段最小二乘）
            Z = self.X[:, [instrument_idx]]  # 工具变量
            X_endog = self.X[:, [endogenous_idx]]  # 内生变量

            # 第一阶段回归
            stage1 = LinearRegression().fit(Z, X_endog)
            X_hat = stage1.predict(Z)

            # 第二阶段回归
            X_new = np.column_stack(
                [self.X[:, :endogenous_idx], X_hat, self.X[:, endogenous_idx + 1 :]]
            )
            stage2 = LinearRegression().fit(X_new, self.y)

            # 保存结果
            results = pd.DataFrame(
                {
                    "Variable": ["Instrument"] + list(self.feature_names),
                    "Coefficient": [stage1.coef_[0][0]] + list(stage2.coef_),
                }
            )
            results.to_csv("MLP/results/econometric/endogeneity_test.csv", index=False)

            self.results["endogeneity"] = results
        except Exception as e:
            print(f"内生性检验出错: {str(e)}")
        return self

    def mechanism_analysis(self, economic_vars=None):
        """经济机制分析（基于SHAP）"""
        print("\n=== 经济机制分析 ===")
        if economic_vars is None:
            economic_vars = self.feature_names[:5]  # 默认前5个变量

        # SHAP分析 - 修正提取方式
        explainer = shap.DeepExplainer(self.model, self.X[:100])
        shap_values = explainer.shap_values(self.X[:200])

        # 对于二分类问题，SHAP可能返回两个数组（每个类一个）或一个数组
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 取正类的SHAP值

        # 机制分析结果
        mechanism_results = {}
        for var in economic_vars:
            try:
                idx = list(self.feature_names).index(var)
                effect = np.sign(shap_values[:, idx].mean())
                mechanism_results[var] = {
                    "effect_direction": "Positive" if effect > 0 else "Negative",
                    "shap_mean": float(shap_values[:, idx].mean()),
                }
            except Exception as e:
                print(f"Error analyzing variable {var}: {str(e)}")
                continue

        # 保存结果
        pd.DataFrame(mechanism_results).T.to_csv(
            "MLP/results/econometric/mechanism_analysis.csv"
        )

        # 可视化
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X[:200], feature_names=self.feature_names)
        plt.savefig("MLP/results/econometric/mechanism_shap.png", dpi=300)
        plt.close()

        self.results["mechanism"] = mechanism_results
        return self

    def enhanced_robustness_test(self, n_iter=100):
        """增强型稳健性检验"""
        print("\n=== 稳健性检验 ===")
        metrics = {
            "accuracy": [],
            "auc": [],
            "precision_0": [],
            "recall_0": [],
            "precision_1": [],
            "recall_1": [],
        }

        for _ in range(n_iter):
            # 数据扰动
            X_noisy = self.X * np.random.normal(1, 0.1, self.X.shape)

            # 评估
            y_pred = self.model.predict(X_noisy)
            y_pred_class = (y_pred > 0.5).astype(int)

            # 记录指标
            metrics["accuracy"].append(accuracy_score(self.y, y_pred_class))
            metrics["auc"].append(roc_auc_score(self.y, y_pred))

            # 分类报告 - 添加零除处理
            try:
                report = classification_report(self.y, y_pred_class, output_dict=True)
                for cls in ["0", "1"]:
                    if cls in report:  # 检查类别是否存在
                        metrics[f"precision_{cls}"].append(report[cls]["precision"])
                        metrics[f"recall_{cls}"].append(report[cls]["recall"])
                    else:
                        # 如果类别不存在，使用默认值或跳过
                        metrics[f"precision_{cls}"].append(np.nan)
                        metrics[f"recall_{cls}"].append(np.nan)
            except:
                # 如果分类报告完全失败，填充所有值为NaN
                for cls in ["0", "1"]:
                    metrics[f"precision_{cls}"].append(np.nan)
                    metrics[f"recall_{cls}"].append(np.nan)

        # 保存结果 - 添加NaN处理
        robustness_results = pd.DataFrame(
            {k: [np.nanmean(v), np.nanstd(v)] for k, v in metrics.items()},
            index=["mean", "std"],
        ).T

        robustness_results.to_csv("MLP/results/econometric/robustness_test.csv")

        # 可视化
        plt.figure(figsize=(12, 6))
        robustness_results["mean"].plot(kind="bar", yerr=robustness_results["std"])
        plt.title("Robustness Test Results (Mean ± SD)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("MLP/results/econometric/robustness_plot.png", dpi=300)
        plt.close()

        self.results["robustness"] = robustness_results
        return self

    def save_all_results(self):
        """保存所有分析结果"""
        with pd.ExcelWriter("MLP/results/econometric/full_analysis.xlsx") as writer:
            sheets_created = False  # Track if any sheets were created

            for name, result in self.results.items():
                try:
                    if isinstance(result, pd.DataFrame):
                        result.to_excel(
                            writer, sheet_name=name[:31]
                        )  # Excel sheet name limit
                        sheets_created = True
                    elif isinstance(result, dict):
                        # Handle scalar values by providing an index
                        if all(
                            not isinstance(v, (dict, list, pd.DataFrame))
                            for v in result.values()
                        ):
                            df = pd.DataFrame.from_dict(
                                result, orient="index", columns=["Value"]
                            )
                        else:
                            df = pd.DataFrame(result)
                        df.to_excel(writer, sheet_name=name[:31])
                        sheets_created = True

                except Exception as e:
                    print(f"Error saving {name} sheet: {str(e)}")
                    continue

            # Create a dummy sheet if no sheets were created
            if not sheets_created:
                pd.DataFrame({"Message": ["No results available"]}).to_excel(
                    writer, sheet_name="NoResults"
                )

        print("所有计量分析结果已保存到MLP/results/econometric/")


# ======================== 主流程 ========================
if __name__ == "__main__":
    # 数据加载
    print("正在加载数据...")
    train_df = pd.read_csv("data/train.csv").pipe(preprocess_data)
    X = train_df.iloc[:, :-1].values
    y = train_df.iloc[:, -1].values
    feature_names = train_df.columns[:-1].tolist()

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 构建并训练模型
    print("训练模型中...")
    model = Sequential(
        [
            Dense(
                128,
                activation="relu",
                input_shape=(X.shape[1],),
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

    model.compile(
        optimizer=Adam(0.0005), loss="binary_crossentropy", metrics=["accuracy", "auc"]
    )

    # 类别权重
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

    # 训练
    history = model.fit(
        X_scaled,
        y,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=[
            EarlyStopping(
                patience=15, monitor="val_auc", mode="max", restore_best_weights=True
            )
        ],
        verbose=1,
    )

    # 计量经济学分析
    print("\n开始计量经济学分析...")
    analyzer = EconometricAnalyzer(model, X_scaled, y, feature_names)
    (
        analyzer.baseline_regression()
        .endogeneity_test()
        .mechanism_analysis()
        .enhanced_robustness_test()
        .save_all_results()
    )

    print("分析流程完成！所有结果保存在MLP/results/目录下")
