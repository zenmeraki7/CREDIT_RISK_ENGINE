"""
Extract Test Samples from 60K Dataset
Creates sample CSV with only the features needed for prediction

Author: Zen Meraki
Date: January 2025
"""

import pandas as pd
import joblib
import numpy as np

print("=" * 70)
print("🧪 EXTRACTING TEST SAMPLES FROM 60K DATASET")
print("=" * 70)

# =============================================================================
# 1. LOAD MODEL ASSETS
# =============================================================================
print("\n[1/5] Loading model assets...")
try:
    assets = joblib.load('credit_risk_assets.pkl')
    top_features = assets['features']
    target_le = assets['target_le']
    print(f"✅ Model loaded successfully")
    print(f"   - Features needed: {len(top_features)}")
    print(f"   - Target classes: {list(target_le.classes_)}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# =============================================================================
# 2. DISPLAY TOP 15 FEATURES
# =============================================================================
print("\n[2/5] Top 15 Features the Model Needs:")
print("-" * 70)
for i, feat in enumerate(top_features[:15], 1):
    print(f"{i:2d}. {feat}")

# =============================================================================
# 3. LOAD ORIGINAL DATASET
# =============================================================================
print("\n[3/5] Loading 60K dataset...")
try:
    # Try different paths
    paths = [
        'train_60k_rule_accepted.csv',
        '../DATA/processed/train_60k_rule_accepted.csv',
        'DATA/processed/train_60k_rule_accepted.csv'
    ]
    
    df = None
    for path in paths:
        try:
            df = pd.read_csv(path)
            print(f"✅ Loaded from: {path}")
            print(f"   - Total rows: {len(df):,}")
            print(f"   - Total columns: {len(df.columns)}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError("Could not find dataset. Please specify path.")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Solution: Run this script from the correct directory or update the path")
    exit(1)

# =============================================================================
# 4. EXTRACT SAMPLES FOR TESTING
# =============================================================================
print("\n[4/5] Extracting diverse test samples...")

# Get samples for each decision type
samples = []
decision_col = 'loan_decision' if 'loan_decision' in df.columns else 'TARGET'

if decision_col not in df.columns:
    print("⚠️ Warning: Could not find decision column. Using random samples.")
    sample_df = df.sample(n=min(20, len(df)), random_state=42)
else:
    # Get balanced samples from each class
    for decision_class in target_le.classes_:
        class_df = df[df[decision_col] == decision_class]
        n_samples = min(5, len(class_df))
        class_samples = class_df.sample(n=n_samples, random_state=42)
        samples.append(class_samples)
        print(f"   - {decision_class}: {n_samples} samples")
    
    sample_df = pd.concat(samples, ignore_index=True)

# =============================================================================
# 5. CREATE TEST CSV WITH ONLY NEEDED FEATURES
# =============================================================================
print("\n[5/5] Creating test CSV...")

# Add customer_id if not present
if 'customer_id' not in sample_df.columns:
    sample_df.insert(0, 'customer_id', [f'CUST_{i:04d}' for i in range(len(sample_df))])

# Select only the features the model needs + customer_id + target
columns_to_keep = ['customer_id'] + top_features

# Add target column if it exists
if decision_col in sample_df.columns:
    columns_to_keep.append(decision_col)

# Filter columns that actually exist in the dataset
available_columns = [col for col in columns_to_keep if col in sample_df.columns]
missing_columns = [col for col in columns_to_keep if col not in sample_df.columns]

if missing_columns:
    print(f"\n⚠️ Warning: {len(missing_columns)} features not found in dataset:")
    for col in missing_columns[:5]:  # Show first 5
        print(f"   - {col}")
    if len(missing_columns) > 5:
        print(f"   ... and {len(missing_columns) - 5} more")

# Create final test CSV
test_df = sample_df[available_columns].copy()

# Save to CSV
output_file = 'test_samples.csv'
test_df.to_csv(output_file, index=False)

print(f"\n✅ Test samples created: {output_file}")
print(f"   - Samples: {len(test_df)}")
print(f"   - Features: {len(available_columns) - 2}")  # -2 for customer_id and target

# =============================================================================
# 6. DISPLAY SAMPLE PREVIEW
# =============================================================================
print("\n" + "=" * 70)
print("📋 SAMPLE PREVIEW (First 3 rows)")
print("=" * 70)

# Show just key columns for preview
preview_cols = ['customer_id']
preview_cols += [col for col in top_features[:5] if col in test_df.columns]
if decision_col in test_df.columns:
    preview_cols.append(decision_col)

print(test_df[preview_cols].head(3).to_string(index=False))

# =============================================================================
# 7. SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("📊 DATASET SUMMARY")
print("=" * 70)

if decision_col in test_df.columns:
    decision_counts = test_df[decision_col].value_counts()
    print("\nDecision Distribution:")
    for decision, count in decision_counts.items():
        print(f"   {decision}: {count} ({count/len(test_df)*100:.1f}%)")

# Show statistics for key numeric features
numeric_cols = test_df.select_dtypes(include=[np.number]).columns[:5]
if len(numeric_cols) > 0:
    print("\nKey Numeric Features (sample statistics):")
    print(test_df[numeric_cols].describe().round(2).to_string())

# =============================================================================
# 8. CREATE DETAILED FEATURE MAPPING
# =============================================================================
print("\n" + "=" * 70)
print("📝 FEATURE MAPPING FOR STREAMLIT")
print("=" * 70)
print("\nCopy this to update your Streamlit form:\n")

for i, feat in enumerate(top_features[:15], 1):
    if feat in test_df.columns:
        if test_df[feat].dtype == 'object':
            unique_vals = test_df[feat].unique()[:3]
            print(f"{i:2d}. {feat:<30} (Categorical: {list(unique_vals)})")
        else:
            min_val = test_df[feat].min()
            max_val = test_df[feat].max()
            median_val = test_df[feat].median()
            print(f"{i:2d}. {feat:<30} (Numeric: {min_val:.1f} - {max_val:.1f}, median: {median_val:.1f})")
    else:
        print(f"{i:2d}. {feat:<30} ⚠️ NOT IN DATASET")

# =============================================================================
# 9. NEXT STEPS
# =============================================================================
print("\n" + "=" * 70)
print("🚀 NEXT STEPS")
print("=" * 70)
print("""
1. ✅ Test CSV created: test_samples.csv
2. 📊 Upload this CSV to your Streamlit app (Batch Processing)
3. 🧪 Or use individual rows to test Single Assessment
4. 🔍 Compare predictions with actual decisions in the CSV

To test in Streamlit:
   - Go to 'Batch Processing' page
   - Upload test_samples.csv
   - Click 'Process All Applications'
   - Compare predicted vs actual decisions

To test individual cases:
   - Go to 'Single Assessment' page
   - Copy values from test_samples.csv
   - Enter them in the form
   - Check if prediction matches actual decision
""")

print("\n" + "=" * 70)
print("✨ SAMPLE CSV READY FOR TESTING!")
print("=" * 70)