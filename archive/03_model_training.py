"""
=============================================================================
Credit Card Approval Prediction System
Step 3: Machine Learning Model Training & Evaluation
=============================================================================
This script trains and evaluates 4 classification models:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. XGBoost Classifier

Each model is evaluated using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)
  - ROC-AUC Score

The best model is saved for Flask deployment.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve, f1_score)
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
os.makedirs('models', exist_ok=True)
os.makedirs('model_plots', exist_ok=True)

print("=" * 70)
print("  CREDIT CARD APPROVAL - MODEL TRAINING & EVALUATION")
print("=" * 70)

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================
print("\n📂 Loading processed data...")

# Scaled data (for Logistic Regression)
X_train_scaled = pd.read_csv('processed_data/X_train_scaled.csv')
X_test_scaled = pd.read_csv('processed_data/X_test_scaled.csv')

# Unscaled data (for tree-based models)
X_train = pd.read_csv('processed_data/X_train.csv')
X_test = pd.read_csv('processed_data/X_test.csv')

# Target
y_train = pd.read_csv('processed_data/y_train.csv').values.ravel()
y_test = pd.read_csv('processed_data/y_test.csv').values.ravel()

feature_list = joblib.load('processed_data/feature_list.pkl')

print(f"   Training samples: {X_train.shape[0]:,}")
print(f"   Test samples:     {X_test.shape[0]:,}")
print(f"   Features:         {X_train.shape[1]}")
print(f"   Target classes:   Approved (0), Rejected (1)")
print(f"   Train class distribution: Approved={sum(y_train==0):,}, Rejected={sum(y_train==1):,}")
print(f"   Test class distribution:  Approved={sum(y_test==0):,}, Rejected={sum(y_test==1):,}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def evaluate_model(model, X_test_data, y_test_data, model_name):
    """Comprehensive model evaluation with metrics and plots."""
    y_pred = model.predict(X_test_data)
    
    # Probability predictions (for ROC)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_data)[:, 1]
    else:
        y_prob = y_pred.astype(float)
    
    # Metrics
    acc = accuracy_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test_data, y_prob)
    cm = confusion_matrix(y_test_data, y_pred)
    report = classification_report(y_test_data, y_pred, target_names=['Approved', 'Rejected'])
    
    print(f"\n{'─'*50}")
    print(f"  {model_name} - EVALUATION RESULTS")
    print(f"{'─'*50}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Approved  Rejected")
    print(f"  Approved   {cm[0][0]:>7,}   {cm[0][1]:>7,}")
    print(f"  Rejected   {cm[1][0]:>7,}   {cm[1][1]:>7,}")
    print(f"\n  Classification Report:")
    print(report)
    
    return {
        'name': model_name,
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, model_name, filename):
    """Plot a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', 
                xticklabels=['Approved', 'Rejected'],
                yticklabels=['Approved', 'Rejected'],
                annot_kws={"size": 16, "fontweight": "bold"},
                linewidths=2, linecolor='white', ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'model_plots/{filename}', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Store all results
# ============================================================================
all_results = []

# ╔════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 1: LOGISTIC REGRESSION                                        ║
# ╚════════════════════════════════════════════════════════════════════════╝
print(f"\n{'='*70}")
print("  MODEL 1: LOGISTIC REGRESSION")
print(f"{'='*70}")

start_time = time.time()
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    class_weight='balanced',   # Handle class imbalance
    C=1.0
)
lr_model.fit(X_train_scaled, y_train)
lr_time = time.time() - start_time

print(f"\n   ⏱️  Training time: {lr_time:.2f} seconds")
print(f"   Parameters: solver=lbfgs, max_iter=1000, class_weight=balanced")

lr_results = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
lr_results['train_time'] = lr_time
all_results.append(lr_results)

# Save model
joblib.dump(lr_model, 'models/logistic_regression.pkl')
print(f"   ✅ Model saved: models/logistic_regression.pkl")

# Confusion Matrix
plot_confusion_matrix(lr_results['confusion_matrix'], 'Logistic Regression', '01_lr_confusion_matrix.png')
print(f"   ✅ Plot saved: model_plots/01_lr_confusion_matrix.png")

# Feature Importance (coefficients)
fig, ax = plt.subplots(figsize=(12, 8))
coef_df = pd.DataFrame({
    'Feature': feature_list,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=True)
colors = ['#e74c3c' if c < 0 else '#27ae60' for c in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
ax.set_title('Logistic Regression - Feature Coefficients', fontsize=14, fontweight='bold')
ax.set_xlabel('Coefficient Value')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('model_plots/02_lr_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Plot saved: model_plots/02_lr_feature_importance.png")


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 2: DECISION TREE CLASSIFIER                                   ║
# ╚════════════════════════════════════════════════════════════════════════╝
print(f"\n{'='*70}")
print("  MODEL 2: DECISION TREE CLASSIFIER")
print(f"{'='*70}")

start_time = time.time()
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    criterion='gini'
)
dt_model.fit(X_train, y_train)
dt_time = time.time() - start_time

print(f"\n   ⏱️  Training time: {dt_time:.2f} seconds")
print(f"   Parameters: max_depth=10, min_samples_split=20, min_samples_leaf=10")
print(f"   Tree depth: {dt_model.get_depth()}, Leaves: {dt_model.get_n_leaves()}")

dt_results = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
dt_results['train_time'] = dt_time
all_results.append(dt_results)

# Save model
joblib.dump(dt_model, 'models/decision_tree.pkl')
print(f"   ✅ Model saved: models/decision_tree.pkl")

# Confusion Matrix
plot_confusion_matrix(dt_results['confusion_matrix'], 'Decision Tree', '03_dt_confusion_matrix.png')
print(f"   ✅ Plot saved: model_plots/03_dt_confusion_matrix.png")

# Feature Importance
fig, ax = plt.subplots(figsize=(12, 8))
fi_df = pd.DataFrame({
    'Feature': feature_list,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi_df)))
ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='white')
ax.set_title('Decision Tree - Feature Importance', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance (Gini)')
plt.tight_layout()
plt.savefig('model_plots/04_dt_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Plot saved: model_plots/04_dt_feature_importance.png")


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 3: RANDOM FOREST CLASSIFIER                                   ║
# ╚════════════════════════════════════════════════════════════════════════╝
print(f"\n{'='*70}")
print("  MODEL 3: RANDOM FOREST CLASSIFIER")
print(f"{'='*70}")

start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    max_features='sqrt'
)
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

print(f"\n   ⏱️  Training time: {rf_time:.2f} seconds")
print(f"   Parameters: n_estimators=200, max_depth=15, max_features=sqrt")

rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
rf_results['train_time'] = rf_time
all_results.append(rf_results)

# Save model
joblib.dump(rf_model, 'models/random_forest.pkl')
print(f"   ✅ Model saved: models/random_forest.pkl")

# Confusion Matrix
plot_confusion_matrix(rf_results['confusion_matrix'], 'Random Forest', '05_rf_confusion_matrix.png')
print(f"   ✅ Plot saved: model_plots/05_rf_confusion_matrix.png")

# Feature Importance
fig, ax = plt.subplots(figsize=(12, 8))
fi_df = pd.DataFrame({
    'Feature': feature_list,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(fi_df)))
ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='white')
ax.set_title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('model_plots/06_rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Plot saved: model_plots/06_rf_feature_importance.png")


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 4: XGBOOST CLASSIFIER                                         ║
# ╚════════════════════════════════════════════════════════════════════════╝
print(f"\n{'='*70}")
print("  MODEL 4: XGBOOST CLASSIFIER")
print(f"{'='*70}")

# Calculate scale_pos_weight for class imbalance
neg_count = sum(y_train == 0)
pos_count = sum(y_train == 1)
scale_pos = neg_count / pos_count
print(f"\n   Class imbalance ratio: {scale_pos:.2f} (neg/pos)")

start_time = time.time()
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1,
    reg_alpha=0.1,
    reg_lambda=1.0
)
xgb_model.fit(X_train, y_train, verbose=False)
xgb_time = time.time() - start_time

print(f"   ⏱️  Training time: {xgb_time:.2f} seconds")
print(f"   Parameters: n_estimators=300, max_depth=8, lr=0.1, subsample=0.8")

xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
xgb_results['train_time'] = xgb_time
all_results.append(xgb_results)

# Save model
joblib.dump(xgb_model, 'models/xgboost.pkl')
print(f"   ✅ Model saved: models/xgboost.pkl")

# Confusion Matrix
plot_confusion_matrix(xgb_results['confusion_matrix'], 'XGBoost', '07_xgb_confusion_matrix.png')
print(f"   ✅ Plot saved: model_plots/07_xgb_confusion_matrix.png")

# Feature Importance
fig, ax = plt.subplots(figsize=(12, 8))
fi_df = pd.DataFrame({
    'Feature': feature_list,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=True)
colors = plt.cm.magma(np.linspace(0.2, 0.9, len(fi_df)))
ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='white')
ax.set_title('XGBoost - Feature Importance', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('model_plots/08_xgb_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Plot saved: model_plots/08_xgb_feature_importance.png")


# ╔════════════════════════════════════════════════════════════════════════╗
# ║  COMPARATIVE ANALYSIS                                                 ║
# ╚════════════════════════════════════════════════════════════════════════╝
print(f"\n{'='*70}")
print("  COMPARATIVE MODEL ANALYSIS")
print(f"{'='*70}")

# Comparison table
print(f"\n{'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10} {'Time (s)':>10}")
print("─" * 65)
for r in all_results:
    print(f"{r['name']:<25} {r['accuracy']:>10.4f} {r['f1_score']:>10.4f} {r['roc_auc']:>10.4f} {r['train_time']:>10.2f}")

# Find best model
best_by_accuracy = max(all_results, key=lambda x: x['accuracy'])
best_by_f1 = max(all_results, key=lambda x: x['f1_score'])
best_by_roc = max(all_results, key=lambda x: x['roc_auc'])

print(f"\n   🏆 Best by Accuracy: {best_by_accuracy['name']} ({best_by_accuracy['accuracy']:.4f})")
print(f"   🏆 Best by F1-Score: {best_by_f1['name']} ({best_by_f1['f1_score']:.4f})")
print(f"   🏆 Best by ROC-AUC:  {best_by_roc['name']} ({best_by_roc['roc_auc']:.4f})")

# ============================================================================
# COMPARATIVE PLOTS
# ============================================================================

# 1. Metrics Comparison Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_names = [r['name'] for r in all_results]
colors_models = ['#3498db', '#e67e22', '#27ae60', '#e74c3c']

# Accuracy
ax = axes[0]
accs = [r['accuracy'] for r in all_results]
bars = ax.bar(model_names, accs, color=colors_models, edgecolor='white', linewidth=1.5)
ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy')
ax.set_ylim(min(accs) - 0.05, max(accs) + 0.03)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=15)

# F1-Score
ax = axes[1]
f1s = [r['f1_score'] for r in all_results]
bars = ax.bar(model_names, f1s, color=colors_models, edgecolor='white', linewidth=1.5)
ax.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-Score')
ax.set_ylim(min(f1s) - 0.05, max(f1s) + 0.03)
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=15)

# ROC-AUC
ax = axes[2]
rocs = [r['roc_auc'] for r in all_results]
bars = ax.bar(model_names, rocs, color=colors_models, edgecolor='white', linewidth=1.5)
ax.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('ROC-AUC')
ax.set_ylim(min(rocs) - 0.05, max(rocs) + 0.03)
for bar, val in zip(bars, rocs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=15)

plt.tight_layout(pad=2.0)
plt.savefig('model_plots/09_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Plot saved: model_plots/09_metrics_comparison.png")

# 2. ROC Curves (all models)
fig, ax = plt.subplots(figsize=(10, 8))
line_styles = ['-', '--', '-.', ':']

for i, r in enumerate(all_results):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, color=colors_models[i], linewidth=2.5, linestyle=line_styles[i],
            label=f"{r['name']} (AUC = {r['roc_auc']:.4f})")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves - All Models', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_plots/10_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Plot saved: model_plots/10_roc_curves.png")

# 3. All Confusion Matrices in one figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
cmaps = ['Blues', 'Oranges', 'Greens', 'Reds']

for idx, (r, cmap) in enumerate(zip(all_results, cmaps)):
    ax = axes[idx // 2, idx % 2]
    sns.heatmap(r['confusion_matrix'], annot=True, fmt=',', cmap=cmap,
                xticklabels=['Approved', 'Rejected'],
                yticklabels=['Approved', 'Rejected'],
                annot_kws={"size": 14, "fontweight": "bold"},
                linewidths=2, linecolor='white', ax=ax)
    ax.set_title(f"{r['name']}\nAcc: {r['accuracy']:.4f} | AUC: {r['roc_auc']:.4f}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model_plots/11_all_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Plot saved: model_plots/11_all_confusion_matrices.png")

# ============================================================================
# SELECT BEST MODEL FOR DEPLOYMENT
# ============================================================================
print(f"\n{'='*70}")
print("  BEST MODEL SELECTION FOR DEPLOYMENT")
print(f"{'='*70}")

# Use ROC-AUC as primary metric (handles class imbalance better)
best = max(all_results, key=lambda x: x['roc_auc'])
print(f"\n   🏆 Selected Model: {best['name']}")
print(f"      Accuracy: {best['accuracy']:.4f}")
print(f"      F1-Score: {best['f1_score']:.4f}")
print(f"      ROC-AUC:  {best['roc_auc']:.4f}")

# Map best model name to the saved model file
model_file_map = {
    'Logistic Regression': 'models/logistic_regression.pkl',
    'Decision Tree': 'models/decision_tree.pkl',
    'Random Forest': 'models/random_forest.pkl',
    'XGBoost': 'models/xgboost.pkl'
}

# Copy best model as the deployment model
best_model_path = model_file_map[best['name']]
best_model = joblib.load(best_model_path)
joblib.dump(best_model, 'models/best_model.pkl')

# Save metadata about best model
best_meta = {
    'model_name': best['name'],
    'accuracy': best['accuracy'],
    'f1_score': best['f1_score'],
    'roc_auc': best['roc_auc'],
    'needs_scaling': best['name'] == 'Logistic Regression',
    'features': feature_list
}
joblib.dump(best_meta, 'models/best_model_meta.pkl')

print(f"\n   ✅ Best model saved as: models/best_model.pkl")
print(f"   ✅ Best model metadata: models/best_model_meta.pkl")

print(f"\n{'='*70}")
print("  MODEL TRAINING & EVALUATION COMPLETE!")
print(f"{'='*70}")
print(f"\n  All 4 models trained, evaluated, and saved.")
print(f"  Best model ({best['name']}) selected for Flask deployment.")
print(f"\n  Saved files:")
print(f"   📁 models/")
for f in os.listdir('models'):
    size = os.path.getsize(f'models/{f}')
    print(f"      {f} ({size/1024:.1f} KB)")
print(f"\n   📁 model_plots/")
for f in sorted(os.listdir('model_plots')):
    print(f"      {f}")
