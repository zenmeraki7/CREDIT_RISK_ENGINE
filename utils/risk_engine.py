# import streamlit as st

# @st.cache_data
# def calculate_risk_score(
#     bureau_score, dpd_15, dpd_30, dpd_90, active_loans,
#     total_emi, avg_salary, net_surplus, bounces,
#     salary_stability, liquidity_flag, bureau_risk_flag, missing_months
# ):
#     risk_score = 0

#     if bureau_score < 450: risk_score += 30
#     elif bureau_score < 500: risk_score += 25
#     elif bureau_score < 600: risk_score += 20
#     elif bureau_score < 650: risk_score += 15
#     elif bureau_score < 700: risk_score += 10
#     elif bureau_score < 750: risk_score += 5

#     risk_score += min(dpd_90 * 15, 30)
#     risk_score += min(dpd_30 * 8, 20)
#     risk_score += min(dpd_15 * 3, 10)

#     risk_score += min(active_loans * 2, 20)

#     emi_ratio = total_emi / (avg_salary + 1)
#     if emi_ratio > 0.7: risk_score += 25
#     elif emi_ratio > 0.6: risk_score += 20
#     elif emi_ratio > 0.5: risk_score += 15
#     elif emi_ratio > 0.4: risk_score += 10
#     elif emi_ratio > 0.3: risk_score += 5

#     if net_surplus < -100000: risk_score += 20
#     elif net_surplus < -50000: risk_score += 15
#     elif net_surplus < 0: risk_score += 10

#     risk_score += min(bounces * 5, 15)
#     risk_score += min(missing_months * 5, 15)

#     return min(risk_score, 100)


# def make_loan_decision(risk_score, bureau_score, dpd_90):
#     if bureau_score < 450:
#         return "REJECT", "Bureau score critically low"
#     if dpd_90 > 5:
#         return "REJECT", "Severe delinquencies"

#     if risk_score >= 75:
#         return "REJECT", "High risk"
#     elif risk_score >= 60:
#         return "MANUAL_REVIEW", "Medium-high risk"
#     elif risk_score >= 45:
#         return "MANUAL_REVIEW", "Borderline"
#     else:
#         return "APPROVE", "Low risk"


import streamlit as st

@st.cache_data
def calculate_risk_score(
    bureau_score, dpd_15, dpd_30, dpd_90, active_loans,
    total_emi, avg_salary, net_surplus, bounces,
    salary_stability, liquidity_flag, bureau_risk_flag, missing_months
):
    risk_score = 0

    if bureau_score < 450: risk_score += 30
    elif bureau_score < 500: risk_score += 25
    elif bureau_score < 600: risk_score += 20
    elif bureau_score < 650: risk_score += 15
    elif bureau_score < 700: risk_score += 10
    elif bureau_score < 750: risk_score += 5

    risk_score += min(dpd_90 * 15, 30)
    risk_score += min(dpd_30 * 8, 20)
    risk_score += min(dpd_15 * 3, 10)

    risk_score += min(active_loans * 2, 20)

    emi_ratio = total_emi / (avg_salary + 1)
    if emi_ratio > 0.7: risk_score += 25
    elif emi_ratio > 0.6: risk_score += 20
    elif emi_ratio > 0.5: risk_score += 15
    elif emi_ratio > 0.4: risk_score += 10
    elif emi_ratio > 0.3: risk_score += 5

    if net_surplus < -100000: risk_score += 20
    elif net_surplus < -50000: risk_score += 15
    elif net_surplus < 0: risk_score += 10

    risk_score += min(bounces * 5, 15)
    risk_score += min(missing_months * 5, 15)

    # Bug fix 1: salary_stability, liquidity_flag, bureau_risk_flag were accepted but never used
    if salary_stability == 'UNSTABLE':
        risk_score += 10
    elif salary_stability == 'VARIABLE':
        risk_score += 5

    if liquidity_flag == 'HIGH':
        risk_score += 10
    elif liquidity_flag == 'MODERATE':
        risk_score += 5

    if bureau_risk_flag == 'HIGH':
        risk_score += 15
    elif bureau_risk_flag == 'MEDIUM':
        risk_score += 8

    return min(risk_score, 100)


def make_loan_decision(risk_score, bureau_score, dpd_90, net_surplus=0):
    # Hard rejects
    if bureau_score < 450:
        return "REJECT", "Bureau score critically low"
    if dpd_90 > 5:
        return "REJECT", "Severe delinquencies"
    # Bug fix 3: hard reject on deeply negative net surplus (no minimum income viability)
    if net_surplus < -50000:
        return "REJECT", "Net cash surplus critically negative"

    if risk_score >= 75:
        return "REJECT", "High risk"
    elif risk_score >= 60:
        return "REVIEW", "Medium-high risk"   # Bug fix 2: was "MANUAL_REVIEW", system expects "REVIEW"
    elif risk_score >= 45:
        return "REVIEW", "Borderline"         # Bug fix 2: was "MANUAL_REVIEW"
    else:
        return "APPROVE", "Low risk"


# =============================================================================
# NEW FUNCTIONS REQUIRED BY test.py (new version)
# =============================================================================

def calculate_final_risk_score(bureau_score, ml_confidence, foir,
                                dpd_90=0, dpd_30=0, net_surplus=0,
                                active_loans=0, bounces=0, missing_months=0):
    """
    Calculate final risk score (0-100) combining bureau, ML confidence, and behavioral factors.
    Called by make_hybrid_decision_enhanced in test.py.
    """
    risk = 0

    # Bureau score component (0-30)
    if bureau_score < 450: risk += 30
    elif bureau_score < 500: risk += 25
    elif bureau_score < 600: risk += 20
    elif bureau_score < 650: risk += 15
    elif bureau_score < 700: risk += 10
    elif bureau_score < 750: risk += 5

    # ML confidence penalty — low confidence = higher risk (0-15)
    if ml_confidence < 50: risk += 15
    elif ml_confidence < 65: risk += 10
    elif ml_confidence < 75: risk += 5

    # FOIR component (0-20)
    if foir > 70: risk += 20
    elif foir > 60: risk += 15
    elif foir > 50: risk += 10
    elif foir > 45: risk += 7
    elif foir > 40: risk += 3

    # Delinquency (0-20)
    risk += min(dpd_90 * 10, 20)
    risk += min(dpd_30 * 4, 10)

    # Net surplus (0-10)
    if net_surplus < -100000: risk += 10
    elif net_surplus < -50000: risk += 7
    elif net_surplus < 0: risk += 4

    # Behavioral (0-15)
    risk += min(bounces * 5, 10)
    risk += min(missing_months * 5, 10)
    risk += min(active_loans * 1, 5)

    return min(int(risk), 100)


def fill_missing_ml_fields(customer_dict):
    """
    Fill critical Stage 1 ML model fields that the form may not collect.
    Derives smart defaults from fields that ARE collected.
    Modifies customer_dict in-place.
    """
    dpd_90 = customer_dict.get('dpd_90_count_6m', 0)
    dpd_30 = customer_dict.get('dpd_30_count_6m', 0)

    defaults = {
        'inward_bounce_count_3m': 0,
        'salary_missing_months': 0,
        'salary_amount_cv': 0.1,
        'salary_creditor_consistent': 1.0,
        'salary_txn_count_6m': 6,
        'payment_discipline_flag': 'GOOD',
        'liquidity_flag': 'LOW',
        'cashflow_health': 'MODERATE',
        'bureau_risk_flag': 'LOW',
        'hard_reject_flag': 0,
        'total_dpd_count': dpd_90 + dpd_30,
        'max_dpd_6m': 90 if dpd_90 > 0 else (30 if dpd_30 > 0 else 0),
        'recent_payment_stress': 1 if dpd_90 > 0 else 0,
        'total_late_15_6m': 0,
        'total_late_30_6m': customer_dict.get('total_late_30_6m', dpd_30),
        'total_late_60_6m': 0,
        'total_late_90_6m': customer_dict.get('total_late_90_6m', dpd_90),
        'total_emi_monthly': customer_dict.get('existing_emi', 0),
        # max_utilization maps credit_utilization_pct — ML model was trained on this name
        'max_utilization': customer_dict.get('credit_utilization_pct', 0),
    }
    for field, default_val in defaults.items():
        if field not in customer_dict:
            customer_dict[field] = default_val


def clean_sentinel_values(data_dict):
    """
    Clean -99999 sentinel values from CIBIL dataset before passing to model.
    92.8% of CIBIL records have CC_utilization = -99999 (no credit card).
    Replaces all negative sentinel values with 0.
    Returns cleaned copy of dict.
    """
    cleaned = dict(data_dict)
    sentinel_fields = [
        # Utilization fields
        'CC_utilization', 'PL_utilization', 'max_unsec_exposure_inPct',
        'pct_currentBal_all_TL', 'pct_of_active_TLs_ever',
        'credit_utilization_pct', 'max_utilization',
        # Time-since fields (no event = -99999)
        'time_since_recent_payment', 'time_since_first_deliquency',
        'time_since_recent_deliquency', 'time_since_recent_enq',
        # Enquiry fields (no product = -99999)
        'CC_enq', 'CC_enq_L6m', 'CC_enq_L12m',
        'PL_enq', 'PL_enq_L6m', 'PL_enq_L12m',
        'tot_enq'
    ]
    for field in sentinel_fields:
        val = cleaned.get(field)
        if val is not None and (val < 0 or val == -99999):
            cleaned[field] = 0
    return cleaned


def validate_cibil_identity(stage1_customer, extraction_result):
    """
    Validate that CIBIL PDF belongs to same applicant as Stage 1.
    Returns list of warning strings (empty = no issues).
    """
    warnings = []

    s1_age = stage1_customer.get('age', 0)
    s2_age = extraction_result.get('AGE', 0)
    if s1_age and s2_age and abs(s1_age - s2_age) > 5:
        warnings.append(f"Age mismatch: Application age {s1_age} vs CIBIL age {s2_age} (difference > 5 years)")

    s1_income = stage1_customer.get('avg_salary_6m', 0)
    s2_income = extraction_result.get('NETMONTHLYINCOME', 0)
    if s1_income and s2_income:
        ratio = max(s1_income, s2_income) / max(min(s1_income, s2_income), 1)
        if ratio > 5:
            warnings.append(f"Income mismatch: Application ₹{s1_income:,} vs CIBIL ₹{s2_income:,} (ratio {ratio:.1f}x)")

    s1_bureau = stage1_customer.get('bureau_score', 0)
    s2_bureau = extraction_result.get('Credit_Score', 0)
    if s1_bureau and s2_bureau and abs(s1_bureau - s2_bureau) > 100:
        warnings.append(f"Bureau score mismatch: Application {s1_bureau} vs CIBIL {s2_bureau} (difference > 100)")

    return warnings
