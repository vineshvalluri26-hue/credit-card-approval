"""
=============================================================================
Credit Card Approval Prediction System
Step 2: Data Preprocessing & Feature Engineering
=============================================================================
This script performs:
  - Merging application & credit record datasets
  - Target variable creation (binary: approved/rejected)
  - Missing value handling
  - Duplicate removal
  - Categorical encoding (Label Encoding)
  - Feature engineering (age, employment years, income-to-family ratio)
  - Anomaly handling (DAYS_EMPLOYED sentinel values)
  - Feature scaling
  - Train-test split
  - Save processed dataset
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

print("=" * 70)
print("  CREDIT CARD APPROVAL - DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 70)

# ============================================================================
# 1. LOAD RAW DATASETS
# ============================================================================
print("\n📂 Loading raw datasets...")
app_df = pd.read_csv('application_record.csv')
credit_df = pd.read_csv('credit_record.csv')
print(f"   Application records: {app_df.shape}")
print(f"   Credit records: {credit_df.shape}")

# ============================================================================
# 2. TARGET VARIABLE CREATION
# ============================================================================
print(f"\n{'='*70}")
print("1. TARGET VARIABLE CREATION")
print(f"{'='*70}")
print("""
Strategy for creating binary target label:
  STATUS codes:
    C (paid off), X (no loan), 0 (1-29 days past due) → Good (0)
    1 (30-59 dpd), 2 (60-89 dpd), 3 (90-119 dpd), 
    4 (120-149 dpd), 5 (150+ dpd / write-off)         → Bad (1)
    
  We aggregate per applicant: if ANY month has status ≥ 1 (30+ days 
  past due), the applicant is labeled as "Rejected" (1), else "Approved" (0).
""")

# Map STATUS to binary: good payment (0) vs bad payment (1)
status_map = {
    'C': 0, 'X': 0, '0': 0,   # Good: paid off, no loan, or ≤29 days late
    '1': 1, '2': 1, '3': 1,   # Bad: 30+ days past due
    '4': 1, '5': 1             # Bad: severely overdue / write-offs
}
credit_df['STATUS_BINARY'] = credit_df['STATUS'].map(status_map)

# Aggregate per applicant: max of binary status (1 if ever bad)
target_df = credit_df.groupby('ID').agg(
    TARGET=('STATUS_BINARY', 'max'),
    NUM_MONTHS=('MONTHS_BALANCE', 'count'),
    MONTHS_HISTORY=('MONTHS_BALANCE', lambda x: x.max() - x.min() + 1)
).reset_index()

print(f"   Applicants with credit history: {len(target_df):,}")
print(f"\n   Target Distribution (before merge):")
target_counts = target_df['TARGET'].value_counts()
print(f"   Approved (0): {target_counts[0]:,} ({target_counts[0]/len(target_df)*100:.1f}%)")
print(f"   Rejected (1): {target_counts[1]:,} ({target_counts[1]/len(target_df)*100:.1f}%)")

# ============================================================================
# 3. MERGE DATASETS
# ============================================================================
print(f"\n{'='*70}")
print("2. MERGING DATASETS")
print(f"{'='*70}")

# Remove duplicates from application data (keep first occurrence per ID)
print(f"\n   Application records before dedup: {len(app_df):,}")
app_df = app_df.drop_duplicates(subset='ID', keep='first')
print(f"   Application records after dedup:  {len(app_df):,}")

# Inner merge on ID
df = app_df.merge(target_df, on='ID', how='inner')
print(f"\n   Merged dataset shape: {df.shape}")
print(f"   Unique applicants: {df['ID'].nunique():,}")

print(f"\n   Target Distribution (after merge):")
target_counts = df['TARGET'].value_counts()
print(f"   Approved (0): {target_counts[0]:,} ({target_counts[0]/len(df)*100:.1f}%)")
print(f"   Rejected (1): {target_counts[1]:,} ({target_counts[1]/len(df)*100:.1f}%)")

# ============================================================================
# 4. HANDLING MISSING VALUES
# ============================================================================
print(f"\n{'='*70}")
print("3. HANDLING MISSING VALUES")
print(f"{'='*70}")

print("\n   Missing values before handling:")
missing_before = df.isnull().sum()
for col in df.columns:
    if missing_before[col] > 0:
        print(f"   {col}: {missing_before[col]:,} ({missing_before[col]/len(df)*100:.1f}%)")

# OCCUPATION_TYPE: Fill with 'Unknown' (it's meaningful - pensioners/students don't have occupations)
df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')

# CNT_FAM_MEMBERS: Fill with mode (very few missing)
if df['CNT_FAM_MEMBERS'].isnull().sum() > 0:
    df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].fillna(df['CNT_FAM_MEMBERS'].mode()[0])

print("\n   Missing values after handling:")
remaining_missing = df.isnull().sum().sum()
print(f"   Total remaining: {remaining_missing}")

# ============================================================================
# 5. HANDLING ANOMALIES
# ============================================================================
print(f"\n{'='*70}")
print("4. HANDLING ANOMALIES")
print(f"{'='*70}")

# DAYS_EMPLOYED: Value 365243 is a sentinel for pensioners/unemployed
anomalous_employed = (df['DAYS_EMPLOYED'] == 365243).sum()
print(f"\n   DAYS_EMPLOYED sentinel value (365243): {anomalous_employed:,} records")
print(f"   Creating FLAG_RETIRED indicator and replacing sentinel with 0")
df['FLAG_RETIRED'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = 0

# FLAG_MOBIL: All values are 1 (no variance) → drop it
print(f"\n   FLAG_MOBIL: All values = {df['FLAG_MOBIL'].unique()} → Dropping (zero variance)")
df.drop('FLAG_MOBIL', axis=1, inplace=True)

# ============================================================================
# 6. FEATURE ENGINEERING
# ============================================================================
print(f"\n{'='*70}")
print("5. FEATURE ENGINEERING")
print(f"{'='*70}")

# Convert DAYS_BIRTH to AGE_YEARS
df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).astype(int)
print(f"\n   ✅ Created AGE_YEARS from DAYS_BIRTH")
print(f"      Range: {df['AGE_YEARS'].min()} - {df['AGE_YEARS'].max()} years")

# Convert DAYS_EMPLOYED to EMPLOYMENT_YEARS
df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED'] / 365.25).round(1)
df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(lower=0)  # Ensure non-negative
print(f"   ✅ Created EMPLOYMENT_YEARS from DAYS_EMPLOYED")
print(f"      Range: {df['EMPLOYMENT_YEARS'].min()} - {df['EMPLOYMENT_YEARS'].max():.1f} years")

# Income per family member
df['INCOME_PER_MEMBER'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
print(f"   ✅ Created INCOME_PER_MEMBER (income / family members)")

# Age group binning
df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'], 
                          bins=[0, 25, 35, 45, 55, 65, 100],
                          labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
print(f"   ✅ Created AGE_GROUP (binned age)")

# Income group binning
df['INCOME_GROUP'] = pd.cut(df['AMT_INCOME_TOTAL'],
                             bins=[0, 100000, 200000, 300000, 500000, float('inf')],
                             labels=['Low', 'Medium', 'High', 'Very High', 'Ultra High'])
print(f"   ✅ Created INCOME_GROUP (binned income)")

# Has children flag
df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype(int)
print(f"   ✅ Created HAS_CHILDREN flag")

# Drop original day-based columns (now redundant)
df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)
print(f"\n   Dropped original DAYS_BIRTH and DAYS_EMPLOYED (replaced by derived features)")

print(f"\n   Dataset shape after feature engineering: {df.shape}")

# ============================================================================
# 7. ENCODING CATEGORICAL VARIABLES
# ============================================================================
print(f"\n{'='*70}")
print("6. ENCODING CATEGORICAL VARIABLES")
print(f"{'='*70}")

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\n   Categorical columns to encode ({len(categorical_cols)}):")
for col in categorical_cols:
    print(f"   - {col}: {df[col].nunique()} unique values")

# Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"   ✅ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ============================================================================
# 8. REMOVE ID COLUMN (not a feature)
# ============================================================================
print(f"\n{'='*70}")
print("7. FINAL FEATURE SELECTION")
print(f"{'='*70}")

# Drop ID - it's an identifier, not a predictive feature
df.drop('ID', axis=1, inplace=True)
print(f"\n   Dropped 'ID' column (not a predictive feature)")

# List all final features
feature_cols = [c for c in df.columns if c != 'TARGET']
print(f"\n   Final features ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {col:<25} dtype: {df[col].dtype}")

print(f"\n   Target: TARGET")
print(f"   Final dataset shape: {df.shape}")

# ============================================================================
# 9. FEATURE SCALING
# ============================================================================
print(f"\n{'='*70}")
print("8. FEATURE SCALING")
print(f"{'='*70}")

X = df.drop('TARGET', axis=1)
y = df['TARGET']

# Split BEFORE scaling to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n   Train-Test Split (80-20, stratified):")
print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set:     {X_test.shape[0]:,} samples")
print(f"\n   Training Target Distribution:")
train_target = y_train.value_counts()
print(f"   Approved (0): {train_target[0]:,} ({train_target[0]/len(y_train)*100:.1f}%)")
print(f"   Rejected (1): {train_target[1]:,} ({train_target[1]/len(y_train)*100:.1f}%)")
print(f"\n   Test Target Distribution:")
test_target = y_test.value_counts()
print(f"   Approved (0): {test_target[0]:,} ({test_target[0]/len(y_test)*100:.1f}%)")
print(f"   Rejected (1): {test_target[1]:,} ({test_target[1]/len(y_test)*100:.1f}%)")

# StandardScaler - fit on train only
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"\n   ✅ Applied StandardScaler (fit on training data only)")
print(f"      Mean of scaled training features (should be ~0): {X_train_scaled.mean().mean():.6f}")
print(f"      Std of scaled training features (should be ~1):  {X_train_scaled.std().mean():.6f}")

# ============================================================================
# 10. SAVE PROCESSED DATA
# ============================================================================
print(f"\n{'='*70}")
print("9. SAVING PROCESSED DATA")
print(f"{'='*70}")

os.makedirs('processed_data', exist_ok=True)

# Save the complete processed dataframe
df.to_csv('processed_data/processed_dataset.csv', index=False)
print(f"   ✅ Saved: processed_data/processed_dataset.csv")

# Save train-test splits (scaled)
X_train_scaled.to_csv('processed_data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('processed_data/X_test_scaled.csv', index=False)
y_train.to_csv('processed_data/y_train.csv', index=False)
y_test.to_csv('processed_data/y_test.csv', index=False)
print(f"   ✅ Saved: X_train_scaled, X_test_scaled, y_train, y_test")

# Save train-test splits (unscaled - for tree-based models)
X_train.to_csv('processed_data/X_train.csv', index=False)
X_test.to_csv('processed_data/X_test.csv', index=False)
print(f"   ✅ Saved: X_train, X_test (unscaled for tree-based models)")

# Save scaler and label encoders for deployment
joblib.dump(scaler, 'processed_data/scaler.pkl')
joblib.dump(label_encoders, 'processed_data/label_encoders.pkl')
print(f"   ✅ Saved: scaler.pkl, label_encoders.pkl")

# Save feature list
feature_list = X_train.columns.tolist()
joblib.dump(feature_list, 'processed_data/feature_list.pkl')
print(f"   ✅ Saved: feature_list.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("  PREPROCESSING & FEATURE ENGINEERING COMPLETE!")
print(f"{'='*70}")
print(f"""
  Summary of operations performed:
  ─────────────────────────────────────
  1. Created binary TARGET from credit STATUS codes
  2. Merged application + credit datasets on ID (inner join)
  3. Removed duplicate application records
  4. Filled OCCUPATION_TYPE nulls with 'Unknown'
  5. Handled DAYS_EMPLOYED sentinel (365243) → FLAG_RETIRED
  6. Dropped FLAG_MOBIL (zero variance)
  7. Engineered features: AGE_YEARS, EMPLOYMENT_YEARS, INCOME_PER_MEMBER,
     AGE_GROUP, INCOME_GROUP, HAS_CHILDREN
  8. Label-encoded all categorical variables
  9. Dropped ID column
  10. Applied StandardScaler (fit on train only)
  11. Saved all artifacts to 'processed_data/' directory
  
  Final dataset: {df.shape[0]:,} samples × {df.shape[1]} columns
  Features: {len(feature_cols)}  |  Target: 1 (binary)
""")
