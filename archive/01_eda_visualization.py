"""
=============================================================================
Credit Card Approval Prediction System
Step 1: Exploratory Data Analysis (EDA) & Visualization
=============================================================================
This script performs comprehensive EDA on the credit card approval dataset:
  - Dataset shape, types, and summary statistics
  - Missing value analysis
  - Count plots for categorical features
  - Distribution plots for numerical features
  - Correlation heatmap
  - Target variable distribution analysis
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Create output directory for plots
os.makedirs('eda_plots', exist_ok=True)

print("=" * 70)
print("  CREDIT CARD APPROVAL PREDICTION - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ============================================================================
# 1. LOAD DATASETS
# ============================================================================
print("\n📂 Loading datasets...")
app_df = pd.read_csv('application_record.csv')
credit_df = pd.read_csv('credit_record.csv')

print(f"\n{'='*70}")
print("1. DATASET OVERVIEW")
print(f"{'='*70}")

print(f"\n📋 Application Record Dataset:")
print(f"   Shape: {app_df.shape[0]:,} rows × {app_df.shape[1]} columns")
print(f"   Unique Applicants (IDs): {app_df['ID'].nunique():,}")
print(f"   Duplicate Rows: {app_df.duplicated().sum():,}")

print(f"\n📋 Credit Record Dataset:")
print(f"   Shape: {credit_df.shape[0]:,} rows × {credit_df.shape[1]} columns")
print(f"   Unique Applicants (IDs): {credit_df['ID'].nunique():,}")

# Common IDs
common_ids = set(app_df['ID'].unique()) & set(credit_df['ID'].unique())
print(f"\n🔗 Common Applicants (Overlap): {len(common_ids):,}")

# ============================================================================
# 2. COLUMN DETAILS
# ============================================================================
print(f"\n{'='*70}")
print("2. COLUMN DETAILS - APPLICATION RECORD")
print(f"{'='*70}")
print(f"\n{'Column':<25} {'Type':<15} {'Non-Null':>10} {'Null':>10} {'Unique':>10}")
print("-" * 70)
for col in app_df.columns:
    dtype = str(app_df[col].dtype)
    non_null = app_df[col].notna().sum()
    null_count = app_df[col].isna().sum()
    unique = app_df[col].nunique()
    print(f"{col:<25} {dtype:<15} {non_null:>10,} {null_count:>10,} {unique:>10,}")

print(f"\n{'='*70}")
print("3. COLUMN DETAILS - CREDIT RECORD")
print(f"{'='*70}")
print(f"\n{'Column':<25} {'Type':<15} {'Non-Null':>10} {'Null':>10} {'Unique':>10}")
print("-" * 70)
for col in credit_df.columns:
    dtype = str(credit_df[col].dtype)
    non_null = credit_df[col].notna().sum()
    null_count = credit_df[col].isna().sum()
    unique = credit_df[col].nunique()
    print(f"{col:<25} {dtype:<15} {non_null:>10,} {null_count:>10,} {unique:>10,}")

# ============================================================================
# 3. STATISTICAL SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("4. STATISTICAL SUMMARY - NUMERICAL FEATURES")
print(f"{'='*70}")
numerical_cols = app_df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('ID')
print(app_df[numerical_cols].describe().to_string())

# ============================================================================
# 4. MISSING VALUES ANALYSIS
# ============================================================================
print(f"\n{'='*70}")
print("5. MISSING VALUES ANALYSIS")
print(f"{'='*70}")
missing = app_df.isnull().sum()
missing_pct = (app_df.isnull().sum() / len(app_df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing Count': missing.values,
    'Missing %': missing_pct.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("No missing values found!")

# Plot missing values
fig, ax = plt.subplots(figsize=(14, 6))
missing_all = app_df.isnull().sum().sort_values(ascending=False)
colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in missing_all.values]
bars = ax.bar(range(len(missing_all)), missing_all.values, color=colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(missing_all)))
ax.set_xticklabels(missing_all.index, rotation=45, ha='right', fontsize=10)
ax.set_title('Missing Values per Column', fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('Count of Missing Values', fontsize=12)
for i, (val, pct) in enumerate(zip(missing_all.values, (missing_all.values / len(app_df) * 100))):
    if val > 0:
        ax.text(i, val + 1000, f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#e74c3c')
plt.tight_layout()
plt.savefig('eda_plots/01_missing_values.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Plot saved: eda_plots/01_missing_values.png")

# ============================================================================
# 5. TARGET VARIABLE CREATION & ANALYSIS
# ============================================================================
print(f"\n{'='*70}")
print("6. CREDIT STATUS (TARGET) ANALYSIS")
print(f"{'='*70}")
print("\nSTATUS codes meaning:")
print("  C  = Loan paid off that month (Closed/Current)")
print("  X  = No loan for that month")
print("  0  = 1-29 days past due")
print("  1  = 30-59 days past due")
print("  2  = 60-89 days past due")
print("  3  = 90-119 days past due")
print("  4  = 120-149 days past due")
print("  5  = Overdue or bad debts, write-offs (150+ days)")

print("\nSTATUS Value Counts:")
status_counts = credit_df['STATUS'].value_counts()
for status, count in status_counts.items():
    pct = count / len(credit_df) * 100
    print(f"   {status}: {count:>10,}  ({pct:>6.2f}%)")

# Plot STATUS distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Count plot
colors_status = ['#27ae60', '#2ecc71', '#3498db', '#e67e22', '#e74c3c', '#c0392b', '#8e44ad', '#2c3e50']
status_order = ['C', 'X', '0', '1', '2', '3', '4', '5']
ax1 = axes[0]
counts = [status_counts.get(s, 0) for s in status_order]
bars = ax1.bar(status_order, counts, color=colors_status, edgecolor='white', linewidth=1.5)
ax1.set_title('Credit Record Status Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Status Code', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5000,
             f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Pie chart
ax2 = axes[1]
ax2.pie(counts, labels=status_order, colors=colors_status, autopct='%1.1f%%',
        startangle=140, textprops={'fontsize': 11})
ax2.set_title('Credit Status Proportions', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots/02_credit_status_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: eda_plots/02_credit_status_distribution.png")

# ============================================================================
# 6. CATEGORICAL FEATURE COUNT PLOTS
# ============================================================================
print(f"\n{'='*70}")
print("7. CATEGORICAL FEATURE ANALYSIS (COUNT PLOTS)")
print(f"{'='*70}")

categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']

for col in categorical_cols:
    vc = app_df[col].value_counts()
    print(f"\n📊 {col}:")
    for val, count in vc.items():
        pct = count / len(app_df) * 100
        print(f"   {val:<35} {count:>10,}  ({pct:>5.1f}%)")

# Gender Distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gender
ax = axes[0, 0]
gender_counts = app_df['CODE_GENDER'].value_counts()
colors_g = ['#e91e63', '#2196f3']
bars = ax.bar(gender_counts.index, gender_counts.values, color=colors_g, edgecolor='white', width=0.5)
ax.set_title('Gender Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
for bar, val in zip(bars, gender_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2000,
            f'{val:,}\n({val/len(app_df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

# Own Car
ax = axes[0, 1]
car_counts = app_df['FLAG_OWN_CAR'].value_counts()
colors_c = ['#e74c3c', '#27ae60']
bars = ax.bar(car_counts.index, car_counts.values, color=colors_c, edgecolor='white', width=0.5)
ax.set_title('Car Ownership', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
for bar, val in zip(bars, car_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2000,
            f'{val:,}\n({val/len(app_df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

# Own Realty
ax = axes[1, 0]
realty_counts = app_df['FLAG_OWN_REALTY'].value_counts()
colors_r = ['#27ae60', '#e74c3c']
bars = ax.bar(realty_counts.index, realty_counts.values, color=colors_r, edgecolor='white', width=0.5)
ax.set_title('Property Ownership', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
for bar, val in zip(bars, realty_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2000,
            f'{val:,}\n({val/len(app_df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

# Family Status
ax = axes[1, 1]
family_counts = app_df['NAME_FAMILY_STATUS'].value_counts()
colors_f = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']
bars = ax.barh(family_counts.index, family_counts.values, color=colors_f, edgecolor='white')
ax.set_title('Family Status Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Count')
for bar, val in zip(bars, family_counts.values):
    ax.text(bar.get_width() + 2000, bar.get_y() + bar.get_height()/2.,
            f'{val:,}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout(pad=2.0)
plt.savefig('eda_plots/03_categorical_count_plots_1.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Plot saved: eda_plots/03_categorical_count_plots_1.png")

# Income Type, Education, Housing
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Income Type
ax = axes[0]
income_counts = app_df['NAME_INCOME_TYPE'].value_counts()
palette1 = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']
bars = ax.barh(income_counts.index, income_counts.values, color=palette1, edgecolor='white')
ax.set_title('Income Type Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Count')
for bar, val in zip(bars, income_counts.values):
    ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2.,
            f'{val:,}', va='center', fontsize=9, fontweight='bold')

# Education Type
ax = axes[1]
edu_counts = app_df['NAME_EDUCATION_TYPE'].value_counts()
palette2 = ['#2c3e50', '#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
bars = ax.barh(edu_counts.index, edu_counts.values, color=palette2, edgecolor='white')
ax.set_title('Education Level Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Count')
for bar, val in zip(bars, edu_counts.values):
    ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2.,
            f'{val:,}', va='center', fontsize=9, fontweight='bold')

# Housing Type
ax = axes[2]
housing_counts = app_df['NAME_HOUSING_TYPE'].value_counts()
palette3 = ['#1abc9c', '#3498db', '#9b59b6', '#e67e22', '#e74c3c', '#2c3e50']
bars = ax.barh(housing_counts.index, housing_counts.values, color=palette3, edgecolor='white')
ax.set_title('Housing Type Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Count')
for bar, val in zip(bars, housing_counts.values):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2.,
            f'{val:,}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout(pad=2.0)
plt.savefig('eda_plots/04_categorical_count_plots_2.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: eda_plots/04_categorical_count_plots_2.png")

# Occupation Type
fig, ax = plt.subplots(figsize=(14, 8))
occ_counts = app_df['OCCUPATION_TYPE'].value_counts()
palette_occ = sns.color_palette("viridis", len(occ_counts))
bars = ax.barh(occ_counts.index[::-1], occ_counts.values[::-1], color=palette_occ, edgecolor='white')
ax.set_title('Occupation Type Distribution', fontsize=16, fontweight='bold')
ax.set_xlabel('Count', fontsize=12)
for bar, val in zip(bars, occ_counts.values[::-1]):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2.,
            f'{val:,}', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/05_occupation_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: eda_plots/05_occupation_type.png")

# ============================================================================
# 7. NUMERICAL FEATURE DISTRIBUTION PLOTS
# ============================================================================
print(f"\n{'='*70}")
print("8. NUMERICAL FEATURE DISTRIBUTIONS")
print(f"{'='*70}")

# Age Distribution (convert DAYS_BIRTH to years)
app_df['AGE_YEARS'] = (-app_df['DAYS_BIRTH'] / 365.25).astype(int)

# Employment Years (handle anomalous positive values - pensioners/unemployed get 365243)
app_df['EMPLOYMENT_YEARS'] = app_df['DAYS_EMPLOYED'].apply(
    lambda x: 0 if x > 0 else round(-x / 365.25, 1)
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Age Distribution
ax = axes[0, 0]
ax.hist(app_df['AGE_YEARS'], bins=50, color='#3498db', edgecolor='white', alpha=0.85)
ax.axvline(app_df['AGE_YEARS'].mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f"Mean: {app_df['AGE_YEARS'].mean():.1f}")
ax.axvline(app_df['AGE_YEARS'].median(), color='#f39c12', linestyle='--', linewidth=2, label=f"Median: {app_df['AGE_YEARS'].median():.1f}")
ax.set_title('Age Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Age (Years)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=10)

# Income Distribution
ax = axes[0, 1]
income_capped = app_df['AMT_INCOME_TOTAL'].clip(upper=app_df['AMT_INCOME_TOTAL'].quantile(0.99))
ax.hist(income_capped, bins=50, color='#27ae60', edgecolor='white', alpha=0.85)
ax.axvline(app_df['AMT_INCOME_TOTAL'].mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f"Mean: {app_df['AMT_INCOME_TOTAL'].mean():,.0f}")
ax.axvline(app_df['AMT_INCOME_TOTAL'].median(), color='#f39c12', linestyle='--', linewidth=2, label=f"Median: {app_df['AMT_INCOME_TOTAL'].median():,.0f}")
ax.set_title('Annual Income Distribution (99th percentile capped)', fontsize=14, fontweight='bold')
ax.set_xlabel('Annual Income')
ax.set_ylabel('Frequency')
ax.legend(fontsize=10)

# Employment Years Distribution (only employed people)
ax = axes[1, 0]
employed = app_df[app_df['EMPLOYMENT_YEARS'] > 0]['EMPLOYMENT_YEARS']
ax.hist(employed, bins=50, color='#9b59b6', edgecolor='white', alpha=0.85)
ax.axvline(employed.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f"Mean: {employed.mean():.1f}")
ax.set_title('Employment Duration (Employed Applicants)', fontsize=14, fontweight='bold')
ax.set_xlabel('Years Employed')
ax.set_ylabel('Frequency')
ax.legend(fontsize=10)

# Children Count Distribution
ax = axes[1, 1]
child_counts = app_df['CNT_CHILDREN'].value_counts().sort_index()
bars = ax.bar(child_counts.index, child_counts.values, color='#e67e22', edgecolor='white')
ax.set_title('Number of Children Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Children')
ax.set_ylabel('Count')
ax.set_xticks(range(0, min(child_counts.index.max()+1, 10)))

plt.tight_layout(pad=2.0)
plt.savefig('eda_plots/06_numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: eda_plots/06_numerical_distributions.png")

# Family Members Distribution
fig, ax = plt.subplots(figsize=(10, 6))
fam_counts = app_df['CNT_FAM_MEMBERS'].value_counts().sort_index()
bars = ax.bar(fam_counts.index, fam_counts.values, color='#1abc9c', edgecolor='white')
ax.set_title('Family Members Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Family Members')
ax.set_ylabel('Count')
for bar, val in zip(bars, fam_counts.values):
    if val > 5000:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1000,
                f'{val:,}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/07_family_members.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: eda_plots/07_family_members.png")

# ============================================================================
# 8. KDE / BOX PLOTS FOR KEY NUMERICAL FEATURES
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Income by Gender
ax = axes[0]
for gender, color, label in [('M', '#2196f3', 'Male'), ('F', '#e91e63', 'Female')]:
    data = app_df[app_df['CODE_GENDER'] == gender]['AMT_INCOME_TOTAL'].clip(upper=500000)
    ax.hist(data, bins=50, alpha=0.5, color=color, label=label, density=True)
ax.set_title('Income Distribution by Gender', fontsize=14, fontweight='bold')
ax.set_xlabel('Annual Income')
ax.set_ylabel('Density')
ax.legend()

# Age by Gender
ax = axes[1]
for gender, color, label in [('M', '#2196f3', 'Male'), ('F', '#e91e63', 'Female')]:
    data = app_df[app_df['CODE_GENDER'] == gender]['AGE_YEARS']
    ax.hist(data, bins=50, alpha=0.5, color=color, label=label, density=True)
ax.set_title('Age Distribution by Gender', fontsize=14, fontweight='bold')
ax.set_xlabel('Age (Years)')
ax.set_ylabel('Density')
ax.legend()

# Income by Education
ax = axes[2]
edu_order = ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree']
income_by_edu = [app_df[app_df['NAME_EDUCATION_TYPE'] == e]['AMT_INCOME_TOTAL'].clip(upper=500000).values for e in edu_order]
bp = ax.boxplot(income_by_edu, labels=['Lower\nSec', 'Secondary', 'Incomplete\nHigher', 'Higher\nEdu', 'Academic'], 
                patch_artist=True, showmeans=True, showfliers=False)
colors_box = ['#e74c3c', '#e67e22', '#f1c40f', '#27ae60', '#3498db']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('Income by Education Level', fontsize=14, fontweight='bold')
ax.set_ylabel('Annual Income')

plt.tight_layout(pad=2.0)
plt.savefig('eda_plots/08_distributions_by_category.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: eda_plots/08_distributions_by_category.png")

# ============================================================================
# 9. CORRELATION HEATMAP
# ============================================================================
print(f"\n{'='*70}")
print("9. CORRELATION ANALYSIS")
print(f"{'='*70}")

numerical_for_corr = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
                       'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS']
corr_matrix = app_df[numerical_for_corr].corr()
print("\nCorrelation Matrix:")
print(corr_matrix.round(3).to_string())

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
            annot=True, fmt='.3f', square=True, linewidths=1,
            cbar_kws={"shrink": 0.8}, ax=ax,
            annot_kws={"size": 11, "fontweight": "bold"})
ax.set_title('Correlation Heatmap - Numerical Features', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda_plots/09_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Plot saved: eda_plots/09_correlation_heatmap.png")

# ============================================================================
# 10. ANOMALY OBSERVATIONS
# ============================================================================
print(f"\n{'='*70}")
print("10. KEY OBSERVATIONS & ANOMALIES")
print(f"{'='*70}")

positive_employed = (app_df['DAYS_EMPLOYED'] > 0).sum()
print(f"\n⚠️  DAYS_EMPLOYED anomalies:")
print(f"   Records with positive DAYS_EMPLOYED (365243): {positive_employed:,}")
print(f"   These likely represent pensioners/unemployed (sentinel value)")
print(f"   Percentage: {positive_employed/len(app_df)*100:.1f}%")

print(f"\n📊 OCCUPATION_TYPE missing values:")
print(f"   Missing: {app_df['OCCUPATION_TYPE'].isna().sum():,} ({app_df['OCCUPATION_TYPE'].isna().sum()/len(app_df)*100:.1f}%)")
print(f"   This is the ONLY column with missing values")

print(f"\n📊 Income outliers:")
q99 = app_df['AMT_INCOME_TOTAL'].quantile(0.99)
q01 = app_df['AMT_INCOME_TOTAL'].quantile(0.01)
print(f"   1st percentile: {q01:,.0f}")
print(f"   99th percentile: {q99:,.0f}")
print(f"   Maximum: {app_df['AMT_INCOME_TOTAL'].max():,.0f}")

print(f"\n📊 Age range: {app_df['AGE_YEARS'].min()} - {app_df['AGE_YEARS'].max()} years")
print(f"   Mean age: {app_df['AGE_YEARS'].mean():.1f} years")

# Clean up temp columns
app_df.drop(['AGE_YEARS', 'EMPLOYMENT_YEARS'], axis=1, inplace=True)

print(f"\n{'='*70}")
print("  EDA COMPLETE! All plots saved in 'eda_plots/' directory.")
print(f"{'='*70}")
print("\nGenerated plots:")
for f in sorted(os.listdir('eda_plots')):
    print(f"   📈 eda_plots/{f}")
