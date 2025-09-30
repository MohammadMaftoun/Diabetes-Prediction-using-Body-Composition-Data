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

# Convert metrics to DataFrame for analysis and plotting
metrics_df_mlp = pd.DataFrame(metrics_all_folds['MLP'])

# Calculate mean, std, and 95% CI for key metrics
print("\nSummary of Performance Across Folds:")
for model_name, metrics_df in [('MLP', metrics_df_mlp)]:
    print(f"\n{model_name}:")
    for metric in ['roc_auc', 'accuracy', 'macro_f1score']:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        ci_95 = 1.96 * std_val / np.sqrt(5)  # 95% confidence interval
        print(f"{metric.upper()}: Mean = {mean_val:.2f}%, Std = {std_val:.2f}%, 95% CI = [{mean_val - ci_95:.2f}, {mean_val + ci_95:.2f}]")

# Plot 1: Bar plot of key metrics across folds for MLP
fig_bar = go.Figure()
for model_name, metrics_df in [('MLP', metrics_df_mlp)]:
    for metric in ['roc_auc', 'accuracy', 'macro_f1score']:
        fig_bar.add_trace(
            go.Bar(
                x=metrics_df['fold'],
                y=metrics_df[metric],
                name=f'{model_name} {metric.upper()}',
                text=[f'{val:.2f}%' for val in metrics_df[metric]],
                textposition='auto'
            )
        )
fig_bar.update_layout(
    title='MLP Performance Metrics Across Folds',
    xaxis_title='Fold',
    yaxis_title='Metric Value (%)',
    barmode='group',
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=True
)
fig_bar.show()

# Plot 2: Box plot of key metrics across folds for MLP
fig_box = go.Figure()
for model_name, metrics_df in [('MLP', metrics_df_mlp)]:
    for metric in ['roc_auc', 'accuracy', 'macro_f1score']:
        fig_box.add_trace(
            go.Box(
                y=metrics_df[metric],
                name=f'{model_name} {metric.upper()}',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            )
        )
fig_box.update_layout(
    title='Distribution of MLP Performance Metrics Across Folds',
    yaxis_title='Metric Value (%)',
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=True
)
fig_box.show()



# =============================================================================
# COMPREHENSIVE SHAP ANALYSIS - ALL FEATURES
# =============================================================================

print(f"\n🔄 Computing SHAP values for ALL {X.shape[1]} features...")
print("This will take a few minutes but will show complete feature analysis...")

# Optimized SHAP computation
background_size = 150  # Increased for better baseline
test_sample_size = 300  # Increased for better representation

# Create background dataset
background = shap.sample(X_train, background_size, random_state=42)
X_test_shap = shap.sample(X_test, test_sample_size, random_state=42)

# Create SHAP explainer
explainer = shap.KernelExplainer(model.predict_proba, background, link="logit")

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_shap)

# Handle binary classification and ensure proper dimensions
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]  # Positive class
    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
else:
    shap_values_pos = shap_values
    expected_value = explainer.expected_value

# Ensure shap_values_pos is 2D array
if shap_values_pos.ndim != 2:
    print(f"Warning: SHAP values have unexpected shape: {shap_values_pos.shape}")
    # Try to reshape if needed
    if len(shap_values_pos.shape) == 3:
        shap_values_pos = shap_values_pos.reshape(shap_values_pos.shape[0], -1)

print("✅ SHAP computation completed!")
print(f"SHAP values shape: {shap_values_pos.shape}")
print(f"Number of features: {X.shape[1]}")

# =============================================================================
# VISUALIZATION 1: COMPREHENSIVE FEATURE IMPORTANCE BAR PLOT
# =============================================================================

# Calculate mean absolute SHAP values for all features
feature_importance = np.abs(shap_values_pos).mean(axis=0)
feature_names = X.columns.tolist()

# Ensure dimensions match
if len(feature_importance) != len(feature_names):
    print(f"Dimension mismatch! Features: {len(feature_names)}, Importance: {len(feature_importance)}")
    # Take only the matching number of features
    min_len = min(len(feature_names), len(feature_importance))
    feature_names = feature_names[:min_len]
    feature_importance = feature_importance[:min_len]

print(f"Creating importance dataframe with {len(feature_names)} features")

# Create importance dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance.flatten()  # Ensure 1D array
}).sort_values('Importance', ascending=True)

# Enhanced bar plot showing ALL features
plt.figure(figsize=(14, max(8, len(feature_names) * 0.4)))
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)

plt.title(f'🎯 SHAP Feature Importance - ALL {len(feature_names)} Features\nDiabetes Prediction Model',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, importance_df['Importance'])):
    plt.text(value + max(importance_df['Importance']) * 0.01, bar.get_y() + bar.get_height()/2,
             f'{value:.4f}', ha='left', va='center', fontweight='bold', fontsize=9)

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Print ranking
print(f"\n📊 FEATURE IMPORTANCE RANKING (All {len(feature_names)} features):")
print("="*60)
for i, (_, row) in enumerate(importance_df.iloc[::-1].iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:20} | Importance: {row['Importance']:.6f}")
