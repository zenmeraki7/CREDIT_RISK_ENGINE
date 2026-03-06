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
Generates human-readable explanations for decisions.

Author: Zen Meraki
Version: 2.3 - Added moderate 90+ DPD reason for REVIEW
"""

APPROVAL_REASONS = {
    'high_bureau':          'Excellent credit score ({score})',
    'stable_employment':    'Stable employment history ({tenure} months)',
    'low_foir':             'Affordable EMI burden (FOIR: {foir}%)',
    'clean_payment':        'Clean payment history (No DPD)',
    'strong_income':        'Strong monthly income (₹{income:,})',
    'low_utilization':      'Low credit utilization ({util}%)',
    'long_credit_history':  'Long credit history',
    'low_inquiries':        'Minimal recent credit inquiries ({inq} in last 3 months)',
    'long_history':         'Long and positive credit history ({years}+ years)',
    'good_mix':             'Healthy mix of secured and unsecured loans',
    'excellent_util':       'Very low credit utilization ({util}%) – well below warning levels',
}

REJECTION_REASONS = {
    'low_bureau':       'Credit score below minimum ({score} < 550)',
    'high_foir':        'EMI burden too high (FOIR: {foir}% > 50%)',
    'severe_dpd':       'Severe payment delays ({dpd} instances of 90+ DPD)',
    'moderate_dpd':     'Frequent payment delays ({dpd} instances of 30+ DPD)',
    'low_income':       'Income below minimum threshold (₹{income:,} < ₹15,000)',
    'short_employment': 'Insufficient employment tenure ({tenure} months < 6)',
    'short_vintage':    'Insufficient business vintage ({vintage} years < 2)',
    'bankruptcy':       'Active bankruptcy detected',
    'kyc_failed':       'KYC verification not completed',
    'fraud_flag':       'Fraud flag present on application',
    'high_utilization': 'High credit utilization ({util}% > 80%)',
    'age_invalid':      'Age outside acceptable range ({age} years, must be 24–70)',
    'high_dependents':  'High number of dependents ({deps}) reducing net disposable income',
    'too_many_inquiries': 'Excessive credit inquiries ({inq} in 3 months) – indicates credit hungry',
    'high_lti':         'Loan amount too high relative to income ({ratio:.1f}x annual income)',
    'negative_net':     'Negative net disposable income after existing EMIs',
    'recent_default':   'Recent default on another loan (within last 12 months)',
}

REVIEW_REASONS = {
    'borderline_bureau':   'Credit score in borderline range ({score})',
    'moderate_foir':       'EMI burden moderate (FOIR: {foir}%)',
    'mixed_signals':       'Mixed credit indicators requiring human review',
    'recent_employment':   'Recent employment change requiring verification',
    'high_loan_amount':    'Large loan amount requiring additional underwriting review',
    'moderate_dpd':        'Recent 30-day payment delays requiring review ({dpd} instances)',
    'moderate_dependents': 'Moderate number of dependents ({deps}) may affect repayment',
    'borderline_lti':      'Loan-to-income ratio is borderline ({ratio:.1f}x) – manual check needed',
    'high_util_review':    'Credit utilization is high ({util}%) – may indicate stress',
    'unstable_income':     'Income pattern shows high volatility – verify stability',
    'mixed_product':       'First product enquiry type differs from requested loan – possible mismatch',
    'moderate_90_dpd':     'Multiple 90+ DPD instances ({dpd}) – requires review',   # NEW
}


def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
    """
    Generate top 3 reason codes for the decision.

    Args:
        decision:          'APPROVE' | 'REJECT' | 'REVIEW'
        customer_data:     dict with customer fields
        affordability_data: output from calculate_affordability()
        policy_checks:     dict of policy gate results

    Returns:
        List of up to 3 human-readable reason strings.
    """
    reasons = []

    bureau_score      = customer_data.get('bureau_score', 0)
    foir              = affordability_data.get('foir_percentage', 0)
    dpd_90            = customer_data.get('dpd_90_count_6m', 0)
    dpd_30            = customer_data.get('dpd_30_count_6m', 0)
    income            = customer_data.get('avg_salary_6m', 0)
    employment_tenure = customer_data.get('employment_tenure_months', 0)
    business_vintage  = customer_data.get('business_vintage_years', 0)
    employment_type   = customer_data.get('employment_type', 'Salaried')
    credit_util       = customer_data.get('credit_utilization_pct', 0)
    age               = customer_data.get('age', 0)
    dependents        = customer_data.get('dependents', 0)
    loan_amount       = customer_data.get('loan_amount', 0)
    annual_income     = customer_data.get('AMT_INCOME_TOTAL', income * 12)
    recent_inquiries  = customer_data.get('recent_inquiries_3m', 0)
    net_disposable    = affordability_data.get('net_disposable', 0)
    first_prod        = customer_data.get('first_prod_enq2', None)
    loan_type         = "Personal Loan"

    # ── APPROVAL ─────────────────────────────────────────────────────────────
    if decision == "APPROVE":
        if bureau_score >= 750:
            reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
        if employment_tenure >= 24:
            reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
        if foir <= 40:
            reasons.append(APPROVAL_REASONS['low_foir'].format(foir=round(foir, 1)))
        if dpd_90 == 0 and dpd_30 == 0:
            reasons.append(APPROVAL_REASONS['clean_payment'])
        if income >= 75000:
            reasons.append(APPROVAL_REASONS['strong_income'].format(income=income))
        if credit_util <= 30:
            reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))
        if recent_inquiries <= 2:
            reasons.append(APPROVAL_REASONS['low_inquiries'].format(inq=recent_inquiries))
        if bureau_score >= 800 and customer_data.get('age', 0) > 30:
            reasons.append(APPROVAL_REASONS['long_history'].format(years=max(1, customer_data.get('age', 0)-25)))
        if customer_data.get('CC_Flag', 0) == 1 and customer_data.get('PL_Flag', 0) == 1:
            reasons.append(APPROVAL_REASONS['good_mix'])
        if credit_util <= 20:
            reasons.append(APPROVAL_REASONS['excellent_util'].format(util=credit_util))

    # ── REJECTION ────────────────────────────────────────────────────────────
    elif decision == "REJECT":
        # Policy gate failures
        for check_name, check_result in policy_checks.items():
            if '❌' in str(check_result):
                cn = check_name.lower()
                if 'bureau' in cn:
                    reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
                elif 'dpd' in cn:
                    reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
                elif 'income' in cn:
                    reasons.append(REJECTION_REASONS['low_income'].format(income=income))
                elif 'tenure' in cn:
                    if employment_type == 'Salaried':
                        reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
                    else:
                        reasons.append(REJECTION_REASONS['short_vintage'].format(vintage=business_vintage))
                elif 'kyc' in cn:
                    reasons.append(REJECTION_REASONS['kyc_failed'])
                elif 'bankruptcy' in cn:
                    reasons.append(REJECTION_REASONS['bankruptcy'])
                elif 'fraud' in cn:
                    reasons.append(REJECTION_REASONS['fraud_flag'])
                elif 'age' in cn:
                    reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))

        if foir > 50:
            reasons.append(REJECTION_REASONS['high_foir'].format(foir=round(foir, 1)))
        if credit_util > 80:
            reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
        if dpd_30 >= 3 and dpd_90 == 0:
            reasons.append(REJECTION_REASONS['moderate_dpd'].format(dpd=dpd_30))
        if dependents >= 4:
            reasons.append(REJECTION_REASONS['high_dependents'].format(deps=dependents))
        if recent_inquiries > 8:
            reasons.append(REJECTION_REASONS['too_many_inquiries'].format(inq=recent_inquiries))
        lti_ratio = loan_amount / annual_income if annual_income > 0 else 99
        if lti_ratio > 5:
            reasons.append(REJECTION_REASONS['high_lti'].format(ratio=lti_ratio))
        if net_disposable < 0:
            reasons.append(REJECTION_REASONS['negative_net'])
        if dpd_90 > 0:
            reasons.append(REJECTION_REASONS['recent_default'])

    # ── REVIEW ───────────────────────────────────────────────────────────────
    elif decision == "REVIEW":
        if 650 <= bureau_score < 700:
            reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
        if 40 < foir <= 50:
            reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=round(foir, 1)))
        if employment_tenure < 12:
            reasons.append(REVIEW_REASONS['recent_employment'])

        if dpd_30 >= 1 and dpd_90 == 0:
            reasons.append(REVIEW_REASONS['moderate_dpd'].format(dpd=dpd_30))

        # New: moderate 90+ DPD (2–5 instances)
        if 1 < dpd_90 <= 5:
            reasons.append(REVIEW_REASONS['moderate_90_dpd'].format(dpd=dpd_90))

        if 2 <= dependents < 4:
            reasons.append(REVIEW_REASONS['moderate_dependents'].format(deps=dependents))

        lti_ratio = loan_amount / annual_income if annual_income > 0 else 99
        if 3 < lti_ratio <= 5:
            reasons.append(REVIEW_REASONS['borderline_lti'].format(ratio=lti_ratio))
        if 60 < credit_util <= 80:
            reasons.append(REVIEW_REASONS['high_util_review'].format(util=credit_util))
        stability = customer_data.get('salary_stability_flag', 'STABLE')
        if stability in ('MODERATE', 'UNSTABLE'):
            reasons.append(REVIEW_REASONS['unstable_income'])
        if first_prod and first_prod not in ('None', loan_type):
            reasons.append(REVIEW_REASONS['mixed_product'])

        if not reasons:
            reasons.append(REVIEW_REASONS['mixed_signals'])

    # Deduplicate and limit to 3
    seen = set()
    unique = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique[:3] if unique else ['Decision based on comprehensive model assessment']
