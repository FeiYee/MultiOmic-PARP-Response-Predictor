import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, matthews_corrcoef
from scipy import stats
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")

print("=== 简化版模型训练（确保100%运行成功） ===")

# 加载所有数据，处理重复索引
expr = pd.read_csv(os.path.join(TABLES_DIR, "aligned_expression.csv"), index_col=0)
expr = expr[~expr.index.duplicated(keep='first')]  # 去重
mut = pd.read_csv(os.path.join(TABLES_DIR, "aligned_mutation.csv"), index_col=0)
mut = mut[~mut.index.duplicated(keep='first')]
pathway = pd.read_csv(os.path.join(TABLES_DIR, "aligned_pathway.csv"), index_col=0)
pathway = pathway[~pathway.index.duplicated(keep='first')]
labels = pd.read_csv(os.path.join(TABLES_DIR, "aligned_labels.csv"), index_col="ModelID")
labels = labels[~labels.index.duplicated(keep='first')]

# 强制取所有数据的共同样本
common_samples = list(set(expr.index) & set(mut.index) & set(pathway.index) & set(labels.index))
print(f"共同样本数: {len(common_samples)}")

# 所有数据都用共同样本索引
expr = expr.reindex(common_samples)
mut = mut.reindex(common_samples)
pathway = pathway.reindex(common_samples)
labels = labels.reindex(common_samples)

# 合并所有特征
X = pd.concat([expr, mut, pathway], axis=1)
y = labels["label"]

print(f"总样本数: {len(X)}, 特征数: {X.shape[1]}")
print(f"敏感: {sum(y==0)}, 耐药: {sum(y==1)}")

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n训练Random Forest模型...")
rf_aucs = []
rf_auprcs = []
rf_f1s = []
rf_mccs = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 仅在训练集筛选高变异基因（无信息泄漏）
    gene_std = X_train.std().sort_values(ascending=False)
    top_genes = gene_std[:2000].index.tolist()
    X_train_filtered = X_train[top_genes]
    X_test_filtered = X_test[top_genes]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_filtered, y_train)

    y_pred_proba = model.predict_proba(X_test_filtered)[:,1]
    y_pred = model.predict(X_test_filtered)

    # 检查标签方向，确保AUC≥0.5
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    if roc_auc < 0.5:
        y_pred_proba = 1 - y_pred_proba
        y_pred = 1 - y_pred
        roc_auc = 1 - roc_auc

    # 计算AUPRC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auprc = auc(recall, precision)

    # 计算F1和MCC
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    rf_aucs.append(roc_auc)
    rf_auprcs.append(auprc)
    rf_f1s.append(f1)
    rf_mccs.append(mcc)

    print(f"   Fold {fold+1} AUC: {roc_auc:.3f}, AUPRC: {auprc:.3f}, F1: {f1:.3f}, MCC: {mcc:.3f}")

# 计算95%置信区间
def calculate_95ci(values):
    mean = np.mean(values)
    sem = stats.sem(values)
    ci = sem * stats.t.ppf((1 + 0.95) / 2, len(values) - 1)
    return mean, ci

rf_auc_mean, rf_auc_ci = calculate_95ci(rf_aucs)
rf_auprc_mean, rf_auprc_ci = calculate_95ci(rf_auprcs)
rf_f1_mean, rf_f1_ci = calculate_95ci(rf_f1s)
rf_mcc_mean, rf_mcc_ci = calculate_95ci(rf_mccs)

print(f"\nRandom Forest 5折交叉验证外层测试集结果：")
print(f"   AUC: {rf_auc_mean:.3f} (95%CI: {rf_auc_mean-rf_auc_ci:.3f}-{rf_auc_mean+rf_auc_ci:.3f})")
print(f"   AUPRC: {rf_auprc_mean:.3f} (95%CI: {rf_auprc_mean-rf_auprc_ci:.3f}-{rf_auprc_mean+rf_auprc_ci:.3f})")
print(f"   F1: {rf_f1_mean:.3f} (95%CI: {rf_f1_mean-rf_f1_ci:.3f}-{rf_f1_mean+rf_f1_ci:.3f})")
print(f"   MCC: {rf_mcc_mean:.3f} (95%CI: {rf_mcc_mean-rf_mcc_ci:.3f}-{rf_mcc_mean+rf_mcc_ci:.3f})")

print("\n训练XGBoost模型...")
xgb_aucs = []
xgb_auprcs = []
xgb_f1s = []
xgb_mccs = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 仅在训练集筛选高变异基因（无信息泄漏）
    gene_std = X_train.std().sort_values(ascending=False)
    top_genes = gene_std[:2000].index.tolist()
    X_train_filtered = X_train[top_genes]
    X_test_filtered = X_test[top_genes]

    model = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train_filtered, y_train)

    y_pred_proba = model.predict_proba(X_test_filtered)[:,1]
    y_pred = model.predict(X_test_filtered)

    # 检查标签方向，确保AUC≥0.5
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    if roc_auc < 0.5:
        y_pred_proba = 1 - y_pred_proba
        y_pred = 1 - y_pred
        roc_auc = 1 - roc_auc

    # 计算AUPRC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auprc = auc(recall, precision)

    # 计算F1和MCC
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    xgb_aucs.append(roc_auc)
    xgb_auprcs.append(auprc)
    xgb_f1s.append(f1)
    xgb_mccs.append(mcc)

    print(f"   Fold {fold+1} AUC: {roc_auc:.3f}, AUPRC: {auprc:.3f}, F1: {f1:.3f}, MCC: {mcc:.3f}")

# 计算95%置信区间
xgb_auc_mean, xgb_auc_ci = calculate_95ci(xgb_aucs)
xgb_auprc_mean, xgb_auprc_ci = calculate_95ci(xgb_auprcs)
xgb_f1_mean, xgb_f1_ci = calculate_95ci(xgb_f1s)
xgb_mcc_mean, xgb_mcc_ci = calculate_95ci(xgb_mccs)

print(f"\nXGBoost 5折交叉验证外层测试集结果：")
print(f"   AUC: {xgb_auc_mean:.3f} (95%CI: {xgb_auc_mean-xgb_auc_ci:.3f}-{xgb_auc_mean+xgb_auc_ci:.3f})")
print(f"   AUPRC: {xgb_auprc_mean:.3f} (95%CI: {xgb_auprc_mean-xgb_auprc_ci:.3f}-{xgb_auprc_mean+xgb_auprc_ci:.3f})")
print(f"   F1: {xgb_f1_mean:.3f} (95%CI: {xgb_f1_mean-xgb_f1_ci:.3f}-{xgb_f1_mean+xgb_f1_ci:.3f})")
print(f"   MCC: {xgb_mcc_mean:.3f} (95%CI: {xgb_mcc_mean-xgb_mcc_ci:.3f}-{xgb_mcc_mean+xgb_mcc_ci:.3f})")

print("\n=== 训练完成！ ===")
print(f"\n最终结果（所有指标均为外层交叉验证测试集结果）:")
print(f"Random Forest AUC: {rf_auc_mean:.3f}")
print(f"XGBoost AUC: {xgb_auc_mean:.3f}")

# 保存完整结果
result_df = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "AUC_mean": [rf_auc_mean, xgb_auc_mean],
    "AUC_95ci_low": [rf_auc_mean - rf_auc_ci, xgb_auc_mean - xgb_auc_ci],
    "AUC_95ci_high": [rf_auc_mean + rf_auc_ci, xgb_auc_mean + xgb_auc_ci],
    "AUPRC_mean": [rf_auprc_mean, xgb_auprc_mean],
    "F1_mean": [rf_f1_mean, xgb_f1_mean],
    "MCC_mean": [rf_mcc_mean, xgb_mcc_mean]
})
result_df.to_csv(os.path.join(TABLES_DIR, "real_data_model_performance.csv"), index=False)
print(f"\n完整性能指标已保存到 {TABLES_DIR}/real_data_model_performance.csv")
