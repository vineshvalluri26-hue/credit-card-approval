"""
=============================================================================
Credit Card Approval Prediction System
Model Training: Decision Tree Classifier
=============================================================================
This script trains and evaluates a Decision Tree classifier
for credit card approval prediction.

Evaluation metrics:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)
  - ROC-AUC Score
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, f1_score)
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
os.makedirs('../models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 70)
print("  DECISION TREE - MODEL TRAINING & EVALUATION")
print("=" * 70)

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================
print("\nLoading processed data...")

# Decision Tree uses UNSCALED data (tree-based models don't need scaling)
X_train = pd.read_csv('../processed_data/X_train.csv')
X_test = pd.read_csv('../processed_data/X_test.csv')
y_train = pd.read_csv('../processed_data/y_train.csv').values.ravel()
y_test = pd.read_csv('../processed_data/y_test.csv').values.ravel()
feature_list = joblib.load('../processed_data/feature_list.pkl')

print(f"   Training samples: {X_train.shape[0]:,}")
print(f"   Test samples:     {X_test.shape[0]:,}")
print(f"   Features:         {X_train.shape[1]}")
print(f"   Train class distribution: Approved={sum(y_train==0):,}, Rejected={sum(y_train==1):,}")
print(f"   Test class distribution:  Approved={sum(y_test==0):,}, Rejected={sum(y_test==1):,}")

# ============================================================================
# 2. TRAIN MODEL
# ============================================================================
print(f"\n{'='*70}")
print("  TRAINING DECISION TREE CLASSIFIER")
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
train_time = time.time() - start_time

print(f"\n   Training time: {train_time:.2f} seconds")
print(f"   Parameters: max_depth=10, min_samples_split=20, min_samples_leaf=10")
print(f"   criterion=gini, class_weight=balanced")
print(f"   Tree depth: {dt_model.get_depth()}, Leaves: {dt_model.get_n_leaves()}")

# ============================================================================
# 3. EVALUATE MODEL
# ============================================================================
print(f"\n{'='*70}")
print("  EVALUATION RESULTS")
print(f"{'='*70}")

y_pred = dt_model.predict(X_test)
y_prob = dt_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Approved', 'Rejected'])

print(f"\n  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"                Predicted")
print(f"              Approved  Rejected")
print(f"  Approved   {cm[0][0]:>7,}   {cm[0][1]:>7,}")
print(f"  Rejected   {cm[1][0]:>7,}   {cm[1][1]:>7,}")
print(f"\n  Classification Report:")
print(report)

# ============================================================================
# 4. PLOTS
# ============================================================================

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=',', cmap='Oranges',
            xticklabels=['Approved', 'Rejected'],
            yticklabels=['Approved', 'Rejected'],
            annot_kws={"size": 16, "fontweight": "bold"},
            linewidths=2, linecolor='white', ax=ax)
ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax.set_title('Decision Tree\nConfusion Matrix', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('plots/dt_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: plots/dt_confusion_matrix.png")

# ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, color='#e67e22', linewidth=2.5, label=f'Decision Tree (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve - Decision Tree', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/dt_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: plots/dt_roc_curve.png")

# Feature Importance
fig, ax = plt.subplots(figsize=(12, 8))
fi_df = pd.DataFrame({
    'Feature': feature_list,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi_df)))
ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='white')
ax.set_title('Decision Tree - Feature Importance (Gini)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('plots/dt_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: plots/dt_feature_importance.png")

# ============================================================================
# 5. SAVE MODEL
# ============================================================================
joblib.dump(dt_model, '../models/decision_tree.pkl')
print(f"\nModel saved: ../models/decision_tree.pkl")

print(f"\n{'='*70}")
print("  DECISION TREE TRAINING COMPLETE!")
print(f"{'='*70}")
