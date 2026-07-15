"""
=============================================================================
Credit Card Approval Prediction System
Model Training: Logistic Regression
=============================================================================
This script trains and evaluates a Logistic Regression classifier
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
from sklearn.linear_model import LogisticRegression
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
print("  LOGISTIC REGRESSION - MODEL TRAINING & EVALUATION")
print("=" * 70)

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================
print("\nLoading processed data...")

# Logistic Regression uses SCALED data
X_train = pd.read_csv('../processed_data/X_train_scaled.csv')
X_test = pd.read_csv('../processed_data/X_test_scaled.csv')
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
print("  TRAINING LOGISTIC REGRESSION")
print(f"{'='*70}")

start_time = time.time()
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    class_weight='balanced',
    C=1.0
)
lr_model.fit(X_train, y_train)
train_time = time.time() - start_time

print(f"\n   Training time: {train_time:.2f} seconds")
print(f"   Parameters: solver=lbfgs, max_iter=1000, class_weight=balanced, C=1.0")

# ============================================================================
# 3. EVALUATE MODEL
# ============================================================================
print(f"\n{'='*70}")
print("  EVALUATION RESULTS")
print(f"{'='*70}")

y_pred = lr_model.predict(X_test)
y_prob = lr_model.predict_proba(X_test)[:, 1]

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
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues',
            xticklabels=['Approved', 'Rejected'],
            yticklabels=['Approved', 'Rejected'],
            annot_kws={"size": 16, "fontweight": "bold"},
            linewidths=2, linecolor='white', ax=ax)
ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax.set_title('Logistic Regression\nConfusion Matrix', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('plots/lr_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: plots/lr_confusion_matrix.png")

# ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, color='#3498db', linewidth=2.5, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve - Logistic Regression', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/lr_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: plots/lr_roc_curve.png")

# Feature Coefficients
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
plt.savefig('plots/lr_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved: plots/lr_feature_importance.png")

# ============================================================================
# 5. SAVE MODEL
# ============================================================================
joblib.dump(lr_model, '../models/logistic_regression.pkl')
print(f"\nModel saved: ../models/logistic_regression.pkl")

print(f"\n{'='*70}")
print("  LOGISTIC REGRESSION TRAINING COMPLETE!")
print(f"{'='*70}")
