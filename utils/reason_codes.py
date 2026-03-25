

 
"""
Reason Code Generation System
Generates human-readable explanations for credit decisions.
 
Author: Zen Meraki
Version: 8.7 — Aligned with tiered DPD 90+ gate (0-1 pass / 2-5 review / >5 reject)
"""
 
# =============================================================================
# REASON CODE TEMPLATES
# =============================================================================
 
APPROVAL_REASONS = {
    'high_bureau':         'Excellent credit score ({score})',
    'stable_employment':   'Stable employment history ({tenure} months)',
    'low_foir':            'Affordable EMI burden (FOIR: {foir:.1f}%)',
    'clean_payment':       'Clean payment history (No 90+ DPD in last 6 months)',
    'strong_income':       'Strong monthly income (Rs.{income:,})',
    'low_utilization':     'Low credit utilization ({util}%)',
    'low_inquiries':       'Minimal recent credit inquiries ({inq} in 3M)',
}
 
REJECTION_REASONS = {
    'low_bureau':          'Credit score below minimum ({score} < 550)',
    'high_foir':           'EMI burden too high (FOIR: {foir:.1f}% > 50%)',
    'severe_dpd':          'Severe payment delays ({dpd} instances of 90+ DPD -- exceeds limit of 5)',
    'low_income':          'Income below minimum threshold (Rs.{income:,} < Rs.15,000)',
    'short_employment':    'Insufficient employment tenure ({tenure} months < 6 months required)',
    'short_vintage':       'Insufficient business vintage ({vintage} years < 2 years required)',
    'bankruptcy':          'Active bankruptcy detected',
    'kyc_failed':          'KYC verification not completed',
    'rbi_consent_missing': 'RBI consent not obtained -- mandatory per Digital Lending Guidelines 2022',
    'high_utilization':    'High credit utilization ({util}% > 80%)',
    'age_invalid':         'Age outside acceptable range ({age} years -- must be 24-70)',
    'fraud':               'Fraud flag detected on application',
    'net_surplus_critical':'Net cash surplus critically negative (Rs.{surplus:,})',
}
 
REVIEW_REASONS = {
    'borderline_bureau':   'Credit score in borderline range ({score} -- manual review required)',
    'moderate_foir':       'EMI burden elevated (FOIR: {foir:.1f}% -- within limit but requires review)',
    'dpd_moderate':        '{dpd} instance(s) of 90+ DPD detected (1-5 range -- review required)',
    'high_active_loans':   'High number of active loans ({loans} -- 5+ triggers review)',
    'unstable_salary':     'Unstable salary pattern -- manual verification recommended',
    'high_dependents':     'High number of dependents ({deps} > 5 -- review recommended)',
    'recent_employment':   'Recent employment -- tenure {tenure} months requires verification',
    'moderate_utilization':'Credit utilization elevated ({util}% -- approaching high-risk threshold)',
    'high_inquiries':      'High recent inquiries ({inq} in 3M -- possible credit hunger)',
    'mixed_signals':       'Mixed credit indicators -- comprehensive manual review recommended',
    'ml_model_review':     'ML model flagged application for review (confidence: {confidence:.1f}%) -- secondary assessment recommended',
    'ml_low_confidence':   'Model prediction below high-confidence threshold ({confidence:.1f}%) -- manual verification recommended',
    'multi_account_exposure': 'Multiple active credit accounts ({loans}) with moderate combined exposure -- review recommended',
    'income_vs_exposure':  'Strong income profile but combined credit exposure warrants review',
}
 
 
def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
    """
    Generate up to 3 reason codes explaining the credit decision.
 
    DPD 90+ tiers (aligned with make_hybrid_decision_enhanced):
      0-1  = pass (APPROVE eligible)
      2-5  = review flag
      >5   = hard REJECT
 
    Args:
        decision:           'APPROVE' | 'REVIEW' | 'REJECT'
        customer_data:      dict with customer fields
        affordability_data: output of calculate_affordability()
        policy_checks:      dict of policy gate results
 
    Returns:
        List of up to 3 reason strings.
    """
    reasons = []
 
    bureau_score      = customer_data.get('bureau_score', 0)
    foir              = affordability_data.get('foir_percentage', 0)
    dpd_90            = customer_data.get('dpd_90_count_6m', 0)
    income            = customer_data.get('avg_salary_6m', 0)
    employment_type   = customer_data.get('employment_type', 'Salaried')
    employment_tenure = customer_data.get('employment_tenure_months', 0)
    business_vintage  = customer_data.get('business_vintage_years', 0)
    credit_util       = customer_data.get('credit_utilization_pct', 0)
    age               = customer_data.get('age', 0)
    active_loans      = customer_data.get('active_loans_count', 0)
    dependents        = customer_data.get('dependents', 0)
    recent_inquiries  = customer_data.get('recent_inquiries_3m', 0)
    salary_stability  = customer_data.get('salary_stability_flag', 'STABLE')
    net_surplus       = customer_data.get('net_cash_surplus_6m', 0)
 
    def _failed(key):
        return 'X' in str(policy_checks.get(key, '')) or 'FAILED' in str(policy_checks.get(key, '')).upper() or (
            len(str(policy_checks.get(key, ''))) > 0 and str(policy_checks.get(key, ''))[0:2] in ['\u274c', '❌']
        ) or '\u274c' in str(policy_checks.get(key, ''))
 
    # =========================================================================
    # APPROVE
    # =========================================================================
    if decision == "APPROVE":
        if bureau_score >= 750:
            reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
 
        if employment_type == 'Salaried' and employment_tenure >= 24:
            reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
        elif employment_type in ('Self-Employed', 'Business') and business_vintage >= 5:
            reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=int(business_vintage * 12)))
 
        if foir <= 40:
            reasons.append(APPROVAL_REASONS['low_foir'].format(foir=foir))
 
        if dpd_90 <= 1:
            reasons.append(APPROVAL_REASONS['clean_payment'])
 
        if income >= 75000:
            reasons.append(APPROVAL_REASONS['strong_income'].format(income=int(income)))
 
        if credit_util <= 30:
            reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))
 
        if recent_inquiries <= 1:
            reasons.append(APPROVAL_REASONS['low_inquiries'].format(inq=recent_inquiries))
 
    # =========================================================================
    # REJECT
    # =========================================================================
    elif decision == "REJECT":
        if _failed('age'):
            reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))
        if _failed('kyc'):
            reasons.append(REJECTION_REASONS['kyc_failed'])
        if _failed('rbi_consent'):
            reasons.append(REJECTION_REASONS['rbi_consent_missing'])
        if _failed('bankruptcy'):
            reasons.append(REJECTION_REASONS['bankruptcy'])
        if _failed('fraud'):
            reasons.append(REJECTION_REASONS['fraud'])
        if _failed('income'):
            reasons.append(REJECTION_REASONS['low_income'].format(income=int(income)))
        if _failed('tenure'):
            if employment_type == 'Salaried':
                reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
            else:
                reasons.append(REJECTION_REASONS['short_vintage'].format(vintage=business_vintage))
        if _failed('bureau'):
            reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
        if _failed('dpd') or dpd_90 > 5:
            reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
        if _failed('foir') or foir > 50:
            reasons.append(REJECTION_REASONS['high_foir'].format(foir=foir))
        if credit_util > 80:
            reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
        if net_surplus < -100000:
            reasons.append(REJECTION_REASONS['net_surplus_critical'].format(surplus=int(net_surplus)))
        if not reasons:
            reasons.append('Application rejected based on overall risk model assessment')
 
    # =========================================================================
    # REVIEW
    # =========================================================================
    elif decision == "REVIEW":
        # FIX R-1: dpd_90 == 1 also sets dpd_review_flag in make_hybrid_decision_enhanced()
        # and can produce a REVIEW outcome, but the old threshold (1 < dpd_90) skipped it,
        # causing the case to fall through silently to 'mixed_signals'.
        # Extended range to 1 <= dpd_90 <= 5 so all tiered DPD review cases get a specific code.
        if 1 <= dpd_90 <= 5:
            reasons.append(REVIEW_REASONS['dpd_moderate'].format(dpd=dpd_90))
        if 550 <= bureau_score < 700:
            reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
        if 40 < foir <= 50:
            reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=foir))
        if active_loans >= 5:
            reasons.append(REVIEW_REASONS['high_active_loans'].format(loans=int(active_loans)))
        if salary_stability == 'UNSTABLE':
            reasons.append(REVIEW_REASONS['unstable_salary'])
        if dependents > 5:
            reasons.append(REVIEW_REASONS['high_dependents'].format(deps=dependents))
        if employment_type == 'Salaried' and 6 <= employment_tenure < 12:
            reasons.append(REVIEW_REASONS['recent_employment'].format(tenure=employment_tenure))
        if 60 < credit_util <= 80:
            reasons.append(REVIEW_REASONS['moderate_utilization'].format(util=credit_util))
        if recent_inquiries > 5:
            reasons.append(REVIEW_REASONS['high_inquiries'].format(inq=recent_inquiries))
        # ML model returned REVIEW with no specific policy trigger
        # This happens when the model's combined feature assessment flags the
        # application despite individually clean metrics (e.g. Arjun Mehta scenario:
        # score 821, 0 DPD, but model returned REVIEW due to feature interactions)
        if not reasons:
            ml_confidence = customer_data.get('ml_confidence', 0) or customer_data.get('confidence', 0)
            if ml_confidence > 0 and ml_confidence < 70:
                reasons.append(REVIEW_REASONS['ml_low_confidence'].format(confidence=ml_confidence))
            elif active_loans >= 3 and recent_inquiries >= 3:
                reasons.append(REVIEW_REASONS['multi_account_exposure'].format(loans=int(active_loans)))
                if bureau_score >= 750:
                    reasons.append(REVIEW_REASONS['income_vs_exposure'])
            elif bureau_score >= 750 and dpd_90 == 0:
                # Clean profile flagged by ML — give an informative reason
                reasons.append(
                    f'Excellent credit profile (score {bureau_score}, 0 DPD) — ML model '
                    f'flagged for secondary review based on combined feature assessment. '
                    f'Likely to convert to APPROVE at Stage 2.'
                )
            else:
                reasons.append(REVIEW_REASONS['mixed_signals'])
 
    return reasons[:3] if reasons else ['Decision based on comprehensive model assessment']
 
