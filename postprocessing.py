"""Postprocessing for Diabetes Body Composition Analysis"""


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef, f1_score, roc_auc_score, specificity_score, geometric_mean_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import plotly.express as px

# Split features and target
X = ctgan_result_df.drop('HasDiabetes', axis=1)
y = ctgan_result_df['HasDiabetes']

# Stratified K-Fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_ix, test_ix in kfold.split(X, y):
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)
print(y_train.value_counts())

# Correlation heatmap
corr_matrix = data.corr()
fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='Plasma',
    aspect='auto',
    labels=dict(color="Correlation"),
    title="Correlation Matrix Heatmap"
)
fig.update_layout(width=1600, height=1200, font=dict(size=10), xaxis=dict(tickangle=45))
fig.show()

# Define models
models = {
    "MLP": MLPClassifier(),
    "GB": GradientBoostingClassifier(),
    "RF": RandomForestClassifier(),
    "LR": LogisticRegression(),
    "DT": DecisionTreeClassifier(),
    "LightGBM": LGBMClassifier(),
    "XGB": XGBClassifier(),
    "ADA": AdaBoostClassifier(),
    "LDA": LinearDiscriminantAnalysis()
}

# Test model performance
def test_model_performance(models, X_train, X_test, y_train, y_test):
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, predicted) * 100
        accuracy = accuracy_score(y_test, predicted) * 100
        sensivity = recall_score(y_test, predicted) * 100
        precision = precision_score(y_test, predicted) * 100
        f1score = f1_score(y_test, predicted) * 100
        specificity = specificity_score(y_test, predicted) * 100
        mcc = matthews_corrcoef(y_test, predicted) * 100
        gmeans = geometric_mean_score(y_test, predicted) * 100
        macro_precision = precision_score(y_test, predicted, average='macro') * 100
        macro_recall = recall_score(y_test, predicted, average='macro') * 100
        macro_f1score = f1_score(y_test, predicted, average='macro') * 100
        print(f'{model_name}: AUROC - {roc_auc:.2f}%, Accuracy - {accuracy:.2f}%, Recall - {sensivity:.2f}%, Precision - {precision:.2f}%, '
              f'F1_Score - {f1score:.2f}%, Specificity - {specificity:.2f}%, MCC - {mcc:.2f}%, '
              f'Geometric Mean - {gmeans:.2f}%, Macro Precision - {macro_precision:.2f}%, Macro Recall - {macro_recall:.2f}%, '
              f'Macro F1 Score - {macro_f1score:.2f}%')

test_model_performance(models, X_train, X_test, y_train, y_test)

# TabNet Classifier
X_train_t = X_train.values if hasattr(X_train, 'values') else X_train
X_test_t = X_test.values if hasattr(X_test, 'values') else X_test
tb_cls = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=5e-4),
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax'
)
tb_cls.fit(
    X_train_t, y_train,
    eval_set=[(X_train_t, y_train), (X_test_t, y_test)],
    eval_name=['train', 'valid'],
    eval_metric=['accuracy'],
    max_epochs=100, patience=100,
    batch_size=256, drop_last=False
)

# Evaluate TabNet
preds = tb_cls.predict(X_test_t)
accuracy = accuracy_score(y_test, preds)
print(f'TabNet Validation Accuracy: {accuracy}')
y_test_proba = tb_cls.predict(X_test_t)
y_test_pred = (y_test_proba > 0.5).astype(int)

# Performance metrics
precision = precision_score(y_test, y_test_pred) * 100
recall = recall_score(y_test, y_test_pred) * 100
f1score = f1_score(y_test, y_test_pred) * 100
mcc = matthews_corrcoef(y_test, y_test_pred) * 100
gmeans = geometric_mean_score(y_test, y_test_pred) * 100
spec = specificity_score(y_test, y_test_pred) * 100
roc_auc = roc_auc_score(y_test, y_test_proba) * 100
print("TabNet Metrics:")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}, F1: {f1score:.4f}, MCC: {mcc:.4f}, Gmeans: {gmeans:.4f}, Specificity: {spec:.4f}")

# ROC Curve Visualization for MLP only
plt.figure(figsize=(10, 8))
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
y_score = mlp.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'MLP (AUC = {roc_auc:.4f})', color='blue')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Receiver Operating Characteristic (ROC) for MLP')
plt.legend(loc="lower right")
plt.show()
