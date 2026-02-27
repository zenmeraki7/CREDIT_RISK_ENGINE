# """
# Reason Code Generation System
# Generates human-readable explanations for decisions
# """

# APPROVAL_REASONS = {
#     'high_bureau': 'Excellent credit score ({score})',
#     'stable_employment': 'Stable employment history ({tenure} months)',
#     'low_foir': 'Affordable EMI burden (FOIR: {foir}%)',
#     'clean_payment': 'Clean payment history (No DPD)',
#     'strong_income': 'Strong monthly income (₹{income:,})',
#     'low_utilization': 'Low credit utilization ({util}%)',
#     'long_credit_history': 'Long credit history',
#     'low_inquiries': 'Minimal recent credit inquiries'
# }

# REJECTION_REASONS = {
#     'low_bureau': 'Credit score below minimum ({score} < 550)',
    
#     'high_foir': 'EMI burden too high (FOIR: {foir}% > 50%)',
#     'severe_dpd': 'Severe payment delays ({dpd} instances of 90+ DPD)',
#     'low_income': 'Income below minimum threshold (₹{income:,} < ₹15,000)',
#     'short_employment': 'Insufficient employment tenure ({tenure} months < 6)',
#     'bankruptcy': 'Active bankruptcy detected',
#     'kyc_failed': 'KYC verification not completed',
#     'high_utilization': 'High credit utilization ({util}% > 80%)',
#     'age_invalid': 'Age outside acceptable range ({age} years)'
# }

# REVIEW_REASONS = {
#     'borderline_bureau': 'Credit score in borderline range ({score})',
#     'moderate_foir': 'EMI burden moderate (FOIR: {foir}%)',
#     'mixed_signals': 'Mixed credit indicators requiring human review',
#     'recent_employment': 'Recent employment change requiring verification',
#     'high_loan_amount': 'Large loan amount requiring additional review'
# }


# def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
#     """
#     Generate top 3 reason codes for the decision
    
#     Args:
#         decision: APPROVE/REJECT/REVIEW
#         customer_data: Dict with customer details
#         affordability_data: Output from calculate_affordability()
#         policy_checks: Dict of policy check results
        
#     Returns:
#         List of 3 reason strings
#     """
#     reasons = []
    
#     bureau_score = customer_data.get('bureau_score', 0)
#     foir = affordability_data.get('foir_percentage', 0)
#     dpd_90 = customer_data.get('dpd_90_count_6m', 0)
#     income = customer_data.get('avg_salary_6m', 0)
#     employment_tenure = customer_data.get('employment_tenure_months', 0)
#     credit_util = customer_data.get('credit_utilization_pct', 0)
#     age = customer_data.get('age', 0)
    
#     # --- APPROVAL REASONS ---
#     if decision == "APPROVE":
#         # Check bureau score
#         if bureau_score >= 750:
#             reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
        
#         # Check employment
#         if employment_tenure >= 24:
#             reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
        
#         # Check FOIR
#         if foir <= 40:
#             reasons.append(APPROVAL_REASONS['low_foir'].format(foir=foir))
        
#         # Check DPD
#         if dpd_90 == 0:
#             reasons.append(APPROVAL_REASONS['clean_payment'])
        
#         # Check income
#         if income >= 75000:
#             reasons.append(APPROVAL_REASONS['strong_income'].format(income=income))
        
#         # Check utilization
#         if credit_util <= 30:
#             reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))
    
#     # --- REJECTION REASONS ---
#     elif decision == "REJECT":
#         # Check failed policy checks
#         for check_name, check_result in policy_checks.items():
#             if '❌' in str(check_result):
#                 if 'bureau' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
#                 elif 'dpd' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
#                 elif 'income' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['low_income'].format(income=income))
#                 elif 'tenure' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
#                 elif 'kyc' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['kyc_failed'])
#                 elif 'bankruptcy' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['bankruptcy'])
#                 elif 'age' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))
        
#         # Check FOIR
#         if foir > 50:
#             reasons.append(REJECTION_REASONS['high_foir'].format(foir=foir))
        
#         # Check utilization
#         if credit_util > 80:
#             reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
    
#     # --- REVIEW REASONS ---
#     elif decision == "REVIEW":
#         if 650 <= bureau_score < 700:
#             reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
        
#         if 40 < foir <= 50:
#             reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=foir))
        
#         if employment_tenure < 12:
#             reasons.append(REVIEW_REASONS['recent_employment'])
        
#         if not reasons:
#             reasons.append(REVIEW_REASONS['mixed_signals'])
    
#     # Return top 3 reasons
#     return reasons[:3] if reasons else ['Decision based on model assessment']




"""
Reason Code Generation System
Generates human-readable explanations for decisions
"""

APPROVAL_REASONS = {
    'high_bureau': 'Excellent credit score ({score})',
    'stable_employment': 'Stable employment history ({tenure} months)',
    'low_foir': 'Affordable EMI burden (FOIR: {foir}%)',
    'clean_payment': 'Clean payment history (No DPD)',
    'strong_income': 'Strong monthly income (₹{income:,})',
    'low_utilization': 'Low credit utilization ({util}%)',
    'long_credit_history': 'Long credit history',
    'low_inquiries': 'Minimal recent credit inquiries'
}

REJECTION_REASONS = {
    'low_bureau': 'Credit score below minimum ({score} < 550)',
    'high_foir': 'EMI burden too high (FOIR: {foir}% > 45%)',   # Bug fix 2: was 50%, now 45%
    'severe_dpd': 'Severe payment delays ({dpd} instances of 90+ DPD)',
    'moderate_dpd': 'Frequent payment delays ({dpd} instances of 30+ DPD)',  # Bug fix 1: new
    'low_income': 'Income below minimum threshold (₹{income:,} < ₹15,000)',
    'short_employment': 'Insufficient employment tenure ({tenure} months < 6)',
    'bankruptcy': 'Active bankruptcy detected',
    'kyc_failed': 'KYC verification not completed',
    'high_utilization': 'High credit utilization ({util}% > 80%)',
    'age_invalid': 'Age outside acceptable range ({age} years)',
    'high_dependents': 'High number of dependents ({deps}) reducing net disposable income'  # Bug fix 3: new
}

REVIEW_REASONS = {
    'borderline_bureau': 'Credit score in borderline range ({score})',
    'moderate_foir': 'EMI burden moderate (FOIR: {foir}%)',
    'mixed_signals': 'Mixed credit indicators requiring human review',
    'recent_employment': 'Recent employment change requiring verification',
    'high_loan_amount': 'Large loan amount requiring additional review',
    'moderate_dpd': 'Recent 30-day payment delays requiring review ({dpd} instances)'  # Bug fix 1: new
}


def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
    """
    Generate top 3 reason codes for the decision
    
    Args:
        decision: APPROVE/REJECT/REVIEW
        customer_data: Dict with customer details
        affordability_data: Output from calculate_affordability()
        policy_checks: Dict of policy check results
        
    Returns:
        List of 3 reason strings
    """
    reasons = []
    
    bureau_score = customer_data.get('bureau_score', 0)
    foir = affordability_data.get('foir_percentage', 0)
    dpd_90 = customer_data.get('dpd_90_count_6m', 0)
    dpd_30 = customer_data.get('dpd_30_count_6m', 0)   # Bug fix 1: added dpd_30
    income = customer_data.get('avg_salary_6m', 0)
    employment_tenure = customer_data.get('employment_tenure_months', 0)
    credit_util = customer_data.get('credit_utilization_pct', 0)
    age = customer_data.get('age', 0)
    dependents = customer_data.get('number_of_dependents', 0)  # Bug fix 3: added dependents
    
    # --- APPROVAL REASONS ---
    if decision == "APPROVE":
        if bureau_score >= 750:
            reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
        
        if employment_tenure >= 24:
            reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
        
        if foir <= 40:
            reasons.append(APPROVAL_REASONS['low_foir'].format(foir=foir))
        
        if dpd_90 == 0 and dpd_30 == 0:
            reasons.append(APPROVAL_REASONS['clean_payment'])
        
        if income >= 75000:
            reasons.append(APPROVAL_REASONS['strong_income'].format(income=income))
        
        if credit_util <= 30:
            reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))
    
    # --- REJECTION REASONS ---
    elif decision == "REJECT":
        for check_name, check_result in policy_checks.items():
            if '❌' in str(check_result):
                if 'bureau' in check_name.lower():
                    reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
                elif 'dpd' in check_name.lower():
                    reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
                elif 'income' in check_name.lower():
                    reasons.append(REJECTION_REASONS['low_income'].format(income=income))
                elif 'tenure' in check_name.lower():
                    reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
                elif 'kyc' in check_name.lower():
                    reasons.append(REJECTION_REASONS['kyc_failed'])
                elif 'bankruptcy' in check_name.lower():
                    reasons.append(REJECTION_REASONS['bankruptcy'])
                elif 'age' in check_name.lower():
                    reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))
        
        # Bug fix 2: FOIR rejection threshold now 45% to match policy gate (was 50%)
        if foir > 45:
            reasons.append(REJECTION_REASONS['high_foir'].format(foir=round(foir, 1)))
        
        if credit_util > 80:
            reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
        
        # Bug fix 1: add reason for 30-day DPDs even if not 90-day
        if dpd_30 >= 3 and dpd_90 == 0:
            reasons.append(REJECTION_REASONS['moderate_dpd'].format(dpd=dpd_30))
        
        # Bug fix 3: add dependents reason if high
        if dependents >= 4:
            reasons.append(REJECTION_REASONS['high_dependents'].format(deps=dependents))
    
    # --- REVIEW REASONS ---
    elif decision == "REVIEW":
        if 650 <= bureau_score < 700:
            reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
        
        # Bug fix 2: REVIEW range for FOIR is 40-45% (was 40-50%)
        if 40 < foir <= 45:
            reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=round(foir, 1)))
        
        if employment_tenure < 12:
            reasons.append(REVIEW_REASONS['recent_employment'])
        
        # Bug fix 1: flag moderate DPDs (30-day) in REVIEW
        if dpd_30 >= 1 and dpd_90 == 0:
            reasons.append(REVIEW_REASONS['moderate_dpd'].format(dpd=dpd_30))
        
        # Bug fix 3: flag dependents in REVIEW if moderately high
        if 2 <= dependents < 4:
            reasons.append(REJECTION_REASONS['high_dependents'].format(deps=dependents))
        
        if not reasons:
            reasons.append(REVIEW_REASONS['mixed_signals'])
    
    # Return top 3 reasons
    return reasons[:3] if reasons else ['Decision based on model assessment']
