"""
Synthetic Data Generator Based on Actual Processed Data
Learns distributions from credit_analysis_final.csv and generates 30K rows
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("🚀 Generating Synthetic Data from Actual Dataset...")
print("="*60)

# ============================================================================
# STEP 1: LOAD ACTUAL DATA
# ============================================================================
print("\n📂 Loading actual processed data...")

# Load the actual data
actual_file = Path('DATA/processed/credit_analysis_final.csv')

if not actual_file.exists():
    print(f"❌ Error: {actual_file} not found!")
    print("\nPlease make sure you have run the cleaning script first.")
    exit(1)

df_actual = pd.read_csv(actual_file)

print(f"✅ Loaded {len(df_actual):,} actual customers")
print(f"   Columns: {df_actual.shape[1]}")

# ============================================================================
# STEP 2: LEARN DISTRIBUTIONS FROM ACTUAL DATA
# ============================================================================
print("\n📊 Learning distributions from actual data...")

# Categorical distributions
bureau_risk_dist = df_actual['bureau_risk_flag'].value_counts(normalize=True).to_dict()
salary_stability_dist = df_actual['salary_stability_flag'].value_counts(normalize=True).to_dict()
liquidity_dist = df_actual['liquidity_flag'].value_counts(normalize=True).to_dict()
decision_dist = df_actual['loan_decision'].value_counts(normalize=True).to_dict()

# Numeric distributions (by risk category)
numeric_stats = {}
for risk in df_actual['bureau_risk_flag'].unique():
    subset = df_actual[df_actual['bureau_risk_flag'] == risk]
    numeric_stats[risk] = {
        'bureau_score': (subset['bureau_score'].mean(), subset['bureau_score'].std()),
        'dpd_15': (subset['dpd_15_count_6m'].mean(), subset['dpd_15_count_6m'].std()),
        'dpd_30': (subset['dpd_30_count_6m'].mean(), subset['dpd_30_count_6m'].std()),
        'dpd_90': (subset['dpd_90_count_6m'].mean(), subset['dpd_90_count_6m'].std()),
        'active_loans': (subset['active_loans_count'].mean(), subset['active_loans_count'].std()),
        'emi': (subset['total_emi_monthly'].mean(), subset['total_emi_monthly'].std()),
        'salary': (subset['avg_salary_6m'].mean(), subset['avg_salary_6m'].std()),
        'balance': (subset['avg_monthly_balance_6m'].mean(), subset['avg_monthly_balance_6m'].std()),
    }

print(f"   Learned distributions for {len(bureau_risk_dist)} risk categories")

# ============================================================================
# STEP 3: GENERATE SYNTHETIC DATA
# ============================================================================
print("\n🎲 Generating 30,000 synthetic customers...")

NUM_CUSTOMERS = 30000
np.random.seed(42)

# Customer IDs
customer_ids = [f"CUST_{i:06d}" for i in range(1, NUM_CUSTOMERS + 1)]

# Generate bureau risk distribution
bureau_risk_flags = np.random.choice(
    list(bureau_risk_dist.keys()),
    size=NUM_CUSTOMERS,
    p=list(bureau_risk_dist.values())
)

# Initialize lists
bureau_scores = []
dpd_15_counts = []
dpd_30_counts = []
dpd_90_counts = []
dpd_30_3m = []
active_loans = []
total_emi = []
total_credit = []
total_debit = []
avg_balance = []
net_surplus = []
bounce_count = []
avg_salary = []
salary_txn = []
salary_cv = []
salary_date_std = []
salary_consistent = []
salary_missing = []
salary_stability = []
liquidity_flags = []
hard_reject_flags = []
risk_scores = []
loan_decisions = []
decision_reasons = []

# Generate for each customer
for i, risk in enumerate(bureau_risk_flags):
    stats = numeric_stats[risk]
    
    # Bureau score (normal distribution, clipped)
    bureau_score = int(np.clip(
        np.random.normal(stats['bureau_score'][0], stats['bureau_score'][1]),
        300, 900
    ))
    bureau_scores.append(bureau_score)
    
    # DPD counts (Poisson-like distribution)
    dpd_15 = int(max(0, np.random.normal(stats['dpd_15'][0], stats['dpd_15'][1])))
    dpd_30 = int(max(0, np.random.normal(stats['dpd_30'][0], stats['dpd_30'][1])))
    dpd_90 = int(max(0, np.random.normal(stats['dpd_90'][0], stats['dpd_90'][1])))
    
    dpd_15_counts.append(dpd_15)
    dpd_30_counts.append(dpd_30)
    dpd_90_counts.append(dpd_90)
    
    # Recent DPD (last 3 months)
    if dpd_30 > 0:
        dpd_3m = np.random.choice([0, 1], p=[0.6, 0.4])
    else:
        dpd_3m = 0
    dpd_30_3m.append(dpd_3m)
    
    # Active loans and EMI
    loans = int(max(1, np.random.normal(stats['active_loans'][0], stats['active_loans'][1])))
    emi = int(max(5000, np.random.normal(stats['emi'][0], stats['emi'][1])))
    
    active_loans.append(loans)
    total_emi.append(emi)
    
    # Salary
    salary = int(max(20000, np.random.normal(stats['salary'][0], stats['salary'][1])))
    avg_salary.append(salary)
    
    # Salary metrics
    if risk == 'LOW':
        cv = round(np.random.uniform(0.03, 0.10), 2)
        date_std = round(np.random.uniform(1.5, 3.5), 1)
        consistent = 1
        missing = 0
        stability = 'STABLE'
    elif risk == 'MEDIUM':
        cv = round(np.random.uniform(0.08, 0.18), 2)
        date_std = round(np.random.uniform(2.5, 6.0), 1)
        consistent = np.random.choice([0, 1], p=[0.1, 0.9])
        missing = np.random.choice([0, 1], p=[0.8, 0.2])
        stability = np.random.choice(['STABLE', 'MODERATE'], p=[0.6, 0.4])
    else:  # HIGH
        cv = round(np.random.uniform(0.15, 0.35), 2)
        date_std = round(np.random.uniform(5.0, 10.0), 1)
        consistent = np.random.choice([0, 1], p=[0.3, 0.7])
        missing = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
        stability = 'UNSTABLE'
    
    salary_cv.append(cv)
    salary_date_std.append(date_std)
    salary_consistent.append(consistent)
    salary_missing.append(missing)
    salary_txn.append(6 - missing)
    salary_stability.append(stability)
    
    # Balance
    balance = int(max(0, np.random.normal(stats['balance'][0], stats['balance'][1])))
    avg_balance.append(balance)
    
    # Cash flow
    credit = salary * 6 + np.random.randint(-20000, 20001)
    debit = emi * 6 + np.random.randint(100000, 200001)
    
    total_credit.append(credit)
    total_debit.append(debit)
    net_surplus.append(credit - debit)
    
    # Liquidity
    if balance >= emi * 1.5:
        liq = 'ADEQUATE'
        bounce = 0
    elif balance >= emi:
        liq = 'MODERATE'
        bounce = np.random.choice([0, 1], p=[0.8, 0.2])
    else:
        liq = 'LOW'
        bounce = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
    
    liquidity_flags.append(liq)
    bounce_count.append(bounce)
    
    # Risk score calculation
    score = 100
    score -= dpd_30 * 10
    score -= dpd_15 * 3
    score -= 20 if risk == 'HIGH' else 0
    score -= 10 if risk == 'MEDIUM' else 0
    score -= 10 if cv > 0.20 else 0
    score -= 8 if date_std > 7 else 0
    score -= 15 if consistent == 0 else 0
    score -= missing * 5
    score -= 15 if liq == 'LOW' else 0
    score -= 7 if liq == 'MODERATE' else 0
    score -= bounce * 10
    
    # Hard reject
    hard_rej = (
        dpd_90 > 0 or
        dpd_3m > 1 or
        (consistent == 0 and liq == 'LOW')
    )
    
    if hard_rej:
        score = 0
        decision = 'REJECT'
        reason = 'Critical risk factors detected'
        hard_flag = 1
    else:
        hard_flag = 0
        score = max(0, min(100, score))
        
        if score >= 75:
            decision = 'APPROVE'
            reason = 'Strong profile'
        elif score >= 55:
            decision = 'REVIEW'
            reason = 'Manual review needed'
        else:
            decision = 'REJECT'
            reason = 'Risk too high'
    
    hard_reject_flags.append(hard_flag)
    risk_scores.append(score)
    loan_decisions.append(decision)
    decision_reasons.append(reason)

# ============================================================================
# STEP 4: CREATE DATAFRAME
# ============================================================================
print("\n📦 Building final dataset...")

df_synthetic = pd.DataFrame({
    'customer_id': customer_ids,
    'bureau_score': bureau_scores,
    'dpd_15_count_6m': dpd_15_counts,
    'dpd_30_count_6m': dpd_30_counts,
    'dpd_90_count_6m': dpd_90_counts,
    'dpd_30_count_3m': dpd_30_3m,
    'active_loans_count': active_loans,
    'total_emi_monthly': total_emi,
    'total_credit_6m': total_credit,
    'total_debit_6m': total_debit,
    'avg_monthly_balance_6m': avg_balance,
    'net_cash_surplus_6m': net_surplus,
    'inward_bounce_count_3m': bounce_count,
    'avg_salary_6m': avg_salary,
    'salary_txn_count_6m': salary_txn,
    'salary_amount_cv': salary_cv,
    'salary_date_std': salary_date_std,
    'salary_creditor_consistent': salary_consistent,
    'salary_missing_months': salary_missing,
    'salary_stability_flag': salary_stability,
    'liquidity_flag': liquidity_flags,
    'bureau_risk_flag': bureau_risk_flags,
    'hard_reject_flag': hard_reject_flags,
    'risk_score': risk_scores,
    'loan_decision': loan_decisions,
    'decision_reason': decision_reasons
})

# ============================================================================
# STEP 5: STATISTICS
# ============================================================================
print("\n" + "="*60)
print("✅ SYNTHETIC DATASET GENERATED!")
print("="*60)

print(f"\n📊 Dataset Comparison:")
print(f"   Actual customers:    {len(df_actual):,}")
print(f"   Synthetic customers: {len(df_synthetic):,}")
print(f"   Columns:             {df_synthetic.shape[1]}")

print(f"\n📈 Decision Distribution Comparison:")
print(f"{'Decision':<12} {'Actual':<12} {'Synthetic':<12}")
print("-" * 40)
for decision in ['APPROVE', 'REVIEW', 'REJECT']:
    actual_pct = (df_actual['loan_decision'] == decision).mean() * 100
    synth_pct = (df_synthetic['loan_decision'] == decision).mean() * 100
    print(f"{decision:<12} {actual_pct:>6.1f}%      {synth_pct:>6.1f}%")

print(f"\n🏷️ Bureau Risk Distribution:")
print(f"{'Risk':<12} {'Actual':<12} {'Synthetic':<12}")
print("-" * 40)
for risk in ['LOW', 'MEDIUM', 'HIGH']:
    actual_pct = (df_actual['bureau_risk_flag'] == risk).mean() * 100
    synth_pct = (df_synthetic['bureau_risk_flag'] == risk).mean() * 100
    print(f"{risk:<12} {actual_pct:>6.1f}%      {synth_pct:>6.1f}%")

# ============================================================================
# STEP 6: SAVE FILES
# ============================================================================
print("\n💾 Saving files...")

output_dir = Path('DATA/processed')
output_dir.mkdir(exist_ok=True, parents=True)

# CSV
csv_file = output_dir / 'credit_dataset_30k.csv'
df_synthetic.to_csv(csv_file, index=False)
print(f"   ✅ CSV: {csv_file}")

# Excel
xlsx_file = output_dir / 'credit_dataset_30k.xlsx'
df_synthetic.to_excel(xlsx_file, index=False, sheet_name='Credit_Data')
print(f"   ✅ Excel: {xlsx_file}")

print("\n" + "="*60)
print("🎉 READY FOR USE!")
print("="*60)
print(f"\nSynthetic dataset preserves distributions from actual data")
print(f"while generating {NUM_CUSTOMERS:,} new unique customers.")