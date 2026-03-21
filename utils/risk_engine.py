

# """
# Risk Engine
# Calculates Risk Score (0-100, higher = more risky), PD multipliers,
# and provides utility functions for sentinel cleaning and ML field filling.

# Author: Zen Meraki
# Version: 2.0 - FIXED STABILITY/LIQUIDITY VALUES, CORRECT RISK SCORE
# """


# # =============================================================================
# # LEGACY FUNCTIONS — FALLBACK ONLY. DO NOT USE IN PRODUCTION.
# # These are retained as a safety net if the ML model fails to load.
# # Active production path: calculate_final_risk_score() + make_hybrid_decision_enhanced()
# # =============================================================================

# def calculate_risk_score(
#         bureau_score: float,
#         dpd_15: int,
#         dpd_30: int,
#         dpd_90: int,
#         active_loans: int,
#         total_emi: float,
#         avg_salary: float,
#         net_surplus: float,
#         bounces: int,
#         salary_stability: str,
#         liquidity_flag: str,
#         bureau_risk_flag: str,
#         missing_months: int
# ) -> int:
#     """
#     *** FALLBACK ONLY — DO NOT USE IN PRODUCTION ***
#     Legacy rule-based risk score (0-100). Higher score = more risky.
#     Active production function: calculate_final_risk_score() in this module.
#     """
#     score = 0

#     # Bureau score penalties
#     if   bureau_score < 450: score += 30
#     elif bureau_score < 500: score += 25
#     elif bureau_score < 600: score += 20
#     elif bureau_score < 650: score += 15
#     elif bureau_score < 700: score += 10
#     elif bureau_score < 750: score += 5

#     # Delinquency penalties (your original design)
#     score += min(dpd_90 * 15, 30)
#     score += min(dpd_30 * 8,  20)
#     score += min(dpd_15 * 3,  10)

#     # Active loans
#     score += min(active_loans * 2, 20)

#     # EMI/income ratio (FOIR-like)
#     emi_ratio = total_emi / (avg_salary + 1)
#     if   emi_ratio > 0.7: score += 25
#     elif emi_ratio > 0.6: score += 20
#     elif emi_ratio > 0.5: score += 15
#     elif emi_ratio > 0.4: score += 10
#     elif emi_ratio > 0.3: score += 5

#     # Net surplus
#     if   net_surplus < -100000: score += 20
#     elif net_surplus < -50000:  score += 15
#     elif net_surplus < 0:       score += 10

#     # Behavioral
#     score += min(bounces * 5,        15)
#     score += min(missing_months * 5, 15)

#     # -------------------------------------------------------------------------
#     # Salary stability — FIXED: uses actual training data values
#     # Training data: MODERATE (85.8%), STABLE (12.1%), UNSTABLE (2.1%)
#     # Form dropdown: STABLE, MODERATE, UNSTABLE
#     # -------------------------------------------------------------------------
#     if   salary_stability == 'UNSTABLE': score += 10
#     elif salary_stability == 'MODERATE': score += 5   # FIX: was 'VARIABLE' (never triggered)

#     # -------------------------------------------------------------------------
#     # Liquidity flag — FIXED: uses actual training data & form values
#     # Training data: LOW (87.7%), ADEQUATE (11.9%), MODERATE (0.4%)
#     # Form dropdown: ADEQUATE, LOW
#     # -------------------------------------------------------------------------
#     if   liquidity_flag == 'LOW':      score += 10   # FIX: was 'HIGH' (never triggered)
#     elif liquidity_flag == 'ADEQUATE': score += 5    # FIX: was 'MODERATE'

#     # Bureau risk flag
#     if   bureau_risk_flag == 'HIGH':   score += 15
#     elif bureau_risk_flag == 'MEDIUM': score += 8

#     return min(int(score), 100)


# def make_loan_decision(
#         risk_score: int,
#         bureau_score: float,
#         dpd_90: int,
#         net_surplus: float = 0
# ) -> tuple:
#     """
#     *** FALLBACK ONLY — DO NOT USE IN PRODUCTION ***
#     Legacy standalone decision function used only when ML model is unavailable.
#     Active production function: make_hybrid_decision_enhanced() in app.py.
#     Returns (decision, reason).
#     """
#     # Hard rejects
#     if bureau_score < 450:
#         return "REJECT", "Bureau score critically low"
#     if dpd_90 > 5:
#         return "REJECT", "Severe delinquencies"
#     if net_surplus < -50000:
#         return "REJECT", "Net cash surplus critically negative"

#     # Score bands (your original design)
#     if   risk_score >= 75: return "REJECT", "High risk"
#     elif risk_score >= 60: return "REVIEW", "Medium-high risk"
#     elif risk_score >= 45: return "REVIEW", "Borderline"
#     else:                  return "APPROVE", "Low risk"


# # =============================================================================
# # COMPOSITE RISK SCORE (0-100) — Used by make_hybrid_decision_enhanced
# # This is the ACTIVE risk score shown in the UI.
# # =============================================================================

# def calculate_final_risk_score(
#         bureau_score: float,
#         ml_confidence: float,
#         foir: float,
#         dpd_90: int = 0,
#         dpd_30: int = 0,
#         net_surplus: float = 0,
#         active_loans: int = 0,
#         bounces: int = 0,
#         missing_months: int = 0
# ) -> int:
#     """
#     Composite risk score (0-100, higher = more risky).
#     Combines bureau score, ML confidence penalty, FOIR, delinquency,
#     net surplus, and behavioral signals.

#     This is the score displayed in the Assessment UI.
#     """
#     risk = 0

#     # Bureau score component (0-30 pts)
#     if   bureau_score < 450: risk += 30
#     elif bureau_score < 500: risk += 25
#     elif bureau_score < 600: risk += 20
#     elif bureau_score < 650: risk += 15
#     elif bureau_score < 700: risk += 10
#     elif bureau_score < 750: risk += 5

#     # ML confidence penalty — low confidence = higher risk (0-15 pts)
#     if   ml_confidence < 50: risk += 15
#     elif ml_confidence < 65: risk += 10
#     elif ml_confidence < 75: risk += 5

#     # FOIR component (0-20 pts) — uses your original 50% cap
#     if   foir > 70: risk += 20
#     elif foir > 60: risk += 15
#     elif foir > 50: risk += 10
#     elif foir > 45: risk += 7
#     elif foir > 40: risk += 3

#     # Delinquency (0-20 pts)
#     risk += min(dpd_90 * 10, 20)
#     risk += min(dpd_30 * 4,  10)

#     # Net surplus (0-10 pts)
#     if   net_surplus < -100000: risk += 10
#     elif net_surplus < -50000:  risk += 7
#     elif net_surplus < 0:       risk += 4

#     # Behavioral (0-15 pts)
#     risk += min(bounces * 5,        10)
#     risk += min(missing_months * 5, 10)
#     risk += min(active_loans * 1,    5)

#     return min(int(risk), 100)


# # =============================================================================
# # UTILITY FUNCTIONS
# # =============================================================================

# def fill_missing_ml_fields(customer_dict: dict) -> None:
#     """
#     Fill critical Stage 1 ML model fields that the form may not collect.
#     Derives smart defaults from fields that ARE collected.
#     Modifies customer_dict in-place.
#     """
#     dpd_90 = customer_dict.get('dpd_90_count_6m', 0)
#     dpd_30 = customer_dict.get('dpd_30_count_6m', 0)

#     defaults = {
#         'inward_bounce_count_3m':   0,
#         'salary_missing_months':    0,
#         'salary_amount_cv':         0.1,
#         'salary_creditor_consistent': 1.0,
#         'salary_txn_count_6m':      6,
#         'payment_discipline_flag':  'GOOD',
#         'liquidity_flag':           'LOW',       # Most common value in training data
#         'cashflow_health':          'MODERATE',
#         'bureau_risk_flag':         'LOW',
#         'hard_reject_flag':         0,
#         'total_dpd_count':          dpd_90 + dpd_30,
#         'max_dpd_6m':               90 if dpd_90 > 0 else (30 if dpd_30 > 0 else 0),
#         'recent_payment_stress':    1 if dpd_90 > 0 else 0,
#         'total_late_15_6m':         0,
#         'total_late_30_6m':         customer_dict.get('total_late_30_6m', dpd_30),
#         'total_late_60_6m':         0,
#         'total_late_90_6m':         customer_dict.get('total_late_90_6m', dpd_90),
#         'total_emi_monthly':        customer_dict.get('existing_emi', 0),
#         # max_utilization = credit_utilization_pct (ML model uses this name)
#         'max_utilization':          customer_dict.get('credit_utilization_pct', 0),
#     }
#     for field, default_val in defaults.items():
#         if field not in customer_dict:
#             customer_dict[field] = default_val


# def clean_sentinel_values(data_dict: dict) -> dict:
#     """
#     Clean -99999 sentinel values from CIBIL dataset before passing to model.
#     92.8% of CIBIL records have CC_utilization = -99999 (no credit card).
#     Replaces all negative sentinel values with 0.
#     Returns cleaned copy of dict.
#     """
#     cleaned = dict(data_dict)
#     sentinel_fields = [
#         # Utilization fields
#         'CC_utilization', 'PL_utilization', 'max_unsec_exposure_inPct',
#         'pct_currentBal_all_TL', 'pct_of_active_TLs_ever',
#         'credit_utilization_pct', 'max_utilization',
#         # Time-since fields (no event = -99999)
#         'time_since_recent_payment', 'time_since_first_deliquency',
#         'time_since_recent_deliquency', 'time_since_recent_enq',
#         # Enquiry fields (no product = -99999)
#         'CC_enq', 'CC_enq_L6m', 'CC_enq_L12m',
#         'PL_enq', 'PL_enq_L6m', 'PL_enq_L12m',
#         'tot_enq',
#     ]
#     for field in sentinel_fields:
#         val = cleaned.get(field)
#         if val is not None and (val < 0 or val == -99999):
#             cleaned[field] = 0
#     return cleaned


# def validate_cibil_identity(stage1_customer: dict, extraction_result: dict) -> list:
#     """
#     Validate that CIBIL PDF belongs to the same applicant as Stage 1.
#     Returns list of warning strings (empty list = no issues).
#     """
#     warnings = []

#     s1_age = stage1_customer.get('age', 0)
#     s2_age = extraction_result.get('AGE', 0)
#     if s1_age and s2_age and abs(s1_age - s2_age) > 5:
#         warnings.append(
#             f"Age mismatch: Application age {s1_age} vs CIBIL age {s2_age} "
#             f"(difference > 5 years)"
#         )

#     s1_income = stage1_customer.get('avg_salary_6m', 0)
#     s2_income = extraction_result.get('NETMONTHLYINCOME', 0)
#     if s1_income and s2_income:
#         ratio = max(s1_income, s2_income) / max(min(s1_income, s2_income), 1)
#         # FIX M3: Tightened from 5x to 2x — a 5x mismatch threshold was too loose,
#         # allowing e.g. ₹20,000 on application vs ₹99,000 on CIBIL to pass silently.
#         # 2x catches meaningful income inflation while allowing for normal rounding/timing differences.
#         if ratio > 2:
#             warnings.append(
#                 f"Income mismatch: Application ₹{s1_income:,} vs CIBIL ₹{s2_income:,} "
#                 f"(ratio {ratio:.1f}x)"
#             )

#     s1_bureau = stage1_customer.get('bureau_score', 0)
#     s2_bureau = extraction_result.get('Credit_Score', 0)
#     if s1_bureau and s2_bureau and abs(s1_bureau - s2_bureau) > 100:
#         warnings.append(
#             f"Bureau score mismatch: Application {s1_bureau} vs CIBIL {s2_bureau} "
#             f"(difference > 100)"
#         )

#     return warnings




"""
Risk Engine
Calculates Risk Score (0-100, higher = more risky), PD multipliers,
and provides utility functions for sentinel cleaning and ML field filling.
 
Author: Zen Meraki
Version: 2.0 - FIXED STABILITY/LIQUIDITY VALUES, CORRECT RISK SCORE
"""
 
 
# =============================================================================
# LEGACY FUNCTIONS — FALLBACK ONLY. DO NOT USE IN PRODUCTION.
# These are retained as a safety net if the ML model fails to load.
# Active production path: calculate_final_risk_score() + make_hybrid_decision_enhanced()
# =============================================================================
 
def calculate_risk_score(
        bureau_score: float,
        dpd_15: int,
        dpd_30: int,
        dpd_90: int,
        active_loans: int,
        total_emi: float,
        avg_salary: float,
        net_surplus: float,
        bounces: int,
        salary_stability: str,
        liquidity_flag: str,
        bureau_risk_flag: str,
        missing_months: int
) -> int:
    """
    *** FALLBACK ONLY — DO NOT USE IN PRODUCTION ***
    Legacy rule-based risk score (0-100). Higher score = more risky.
    Active production function: calculate_final_risk_score() in this module.
    """
    score = 0
 
    # Bureau score penalties
    if   bureau_score < 450: score += 30
    elif bureau_score < 500: score += 25
    elif bureau_score < 600: score += 20
    elif bureau_score < 650: score += 15
    elif bureau_score < 700: score += 10
    elif bureau_score < 750: score += 5
 
    # Delinquency penalties (your original design)
    score += min(dpd_90 * 15, 30)
    score += min(dpd_30 * 8,  20)
    score += min(dpd_15 * 3,  10)
 
    # Active loans
    score += min(active_loans * 2, 20)
 
    # EMI/income ratio (FOIR-like)
    emi_ratio = total_emi / (avg_salary + 1)
    if   emi_ratio > 0.7: score += 25
    elif emi_ratio > 0.6: score += 20
    elif emi_ratio > 0.5: score += 15
    elif emi_ratio > 0.4: score += 10
    elif emi_ratio > 0.3: score += 5
 
    # Net surplus
    if   net_surplus < -100000: score += 20
    elif net_surplus < -50000:  score += 15
    elif net_surplus < 0:       score += 10
 
    # Behavioral
    score += min(bounces * 5,        15)
    score += min(missing_months * 5, 15)
 
    # -------------------------------------------------------------------------
    # Salary stability — FIXED: uses actual training data values
    # Training data: MODERATE (85.8%), STABLE (12.1%), UNSTABLE (2.1%)
    # Form dropdown: STABLE, MODERATE, UNSTABLE
    # -------------------------------------------------------------------------
    if   salary_stability == 'UNSTABLE': score += 10
    elif salary_stability == 'MODERATE': score += 5   # FIX: was 'VARIABLE' (never triggered)
 
    # -------------------------------------------------------------------------
    # Liquidity flag — FIXED: uses actual training data & form values
    # Training data: LOW (87.7%), ADEQUATE (11.9%), MODERATE (0.4%)
    # Form dropdown: ADEQUATE, LOW
    # -------------------------------------------------------------------------
    if   liquidity_flag == 'LOW':      score += 10   # FIX: was 'HIGH' (never triggered)
    elif liquidity_flag == 'ADEQUATE': score += 5    # FIX: was 'MODERATE'
 
    # Bureau risk flag
    if   bureau_risk_flag == 'HIGH':   score += 15
    elif bureau_risk_flag == 'MEDIUM': score += 8
 
    return min(int(score), 100)
 
 
def _make_loan_decision_fallback(
        risk_score: int,
        bureau_score: float,
        dpd_90: int,
        net_surplus: float = 0
) -> tuple:
    """
    *** PRIVATE FALLBACK — DO NOT CALL DIRECTLY IN PRODUCTION ***
    Renamed from make_loan_decision() to signal internal-only use (M2 fix).
 
    Used only when the ML model fails to load. Active production path is
    make_hybrid_decision_enhanced() in test.py.
 
    WARNING: net_surplus < -50000 = REJECT rule here does NOT match the
    active engine, which does not hard-reject on surplus alone. If this
    fallback fires unexpectedly, decisions will differ from the main path.
    Returns (decision, reason).
    """
    # Hard rejects
    if bureau_score < 450:
        return "REJECT", "Bureau score critically low"
    if dpd_90 > 5:
        return "REJECT", "Severe delinquencies"
    if net_surplus < -50000:
        return "REJECT", "Net cash surplus critically negative"
 
    # Score bands (your original design)
    if   risk_score >= 75: return "REJECT", "High risk"
    elif risk_score >= 60: return "REVIEW", "Medium-high risk"
    elif risk_score >= 45: return "REVIEW", "Borderline"
    else:                  return "APPROVE", "Low risk"
 
 
# =============================================================================
# COMPOSITE RISK SCORE (0-100) — Used by make_hybrid_decision_enhanced
# This is the ACTIVE risk score shown in the UI.
# =============================================================================
 
def calculate_final_risk_score(
        bureau_score: float,
        ml_confidence: float,
        foir: float,
        dpd_90: int = 0,
        dpd_30: int = 0,
        net_surplus: float = 0,
        active_loans: int = 0,
        bounces: int = 0,
        missing_months: int = 0
) -> int:
    """
    Composite risk score (0-100, higher = more risky).
    Combines bureau score, ML confidence penalty, FOIR, delinquency,
    net surplus, and behavioral signals.
 
    This is the score displayed in the Assessment UI.
    """
    risk = 0
 
    # Bureau score component (0-30 pts)
    if   bureau_score < 450: risk += 30
    elif bureau_score < 500: risk += 25
    elif bureau_score < 600: risk += 20
    elif bureau_score < 650: risk += 15
    elif bureau_score < 700: risk += 10
    elif bureau_score < 750: risk += 5
 
    # FIX 3: ML confidence penalty removed.
    # It double-counted the bureau/DPD signal: a borderline bureau=630 already adds
    # +15 to the risk score; the ML model is also uncertain about bureau=630 and would
    # have added +10 more — penalising the same underlying feature twice.
    # ml_confidence is still used downstream for display and PD adjustment, just not
    # as an additive component of the rule-based risk score.
    # (Removed: if ml_confidence < 50: risk += 15 / < 65: +10 / < 75: +5)
 
    # FOIR component (0-20 pts) — uses your original 50% cap
    if   foir > 70: risk += 20
    elif foir > 60: risk += 15
    elif foir > 50: risk += 10
    elif foir > 45: risk += 7
    elif foir > 40: risk += 3
 
    # Delinquency (0-20 pts)
    risk += min(dpd_90 * 10, 20)
    risk += min(dpd_30 * 4,  10)
 
    # Net surplus (0-10 pts)
    if   net_surplus < -100000: risk += 10
    elif net_surplus < -50000:  risk += 7
    elif net_surplus < 0:       risk += 4
 
    # Behavioral (0-15 pts)
    risk += min(bounces * 5,        10)
    risk += min(missing_months * 5, 10)
    risk += min(active_loans * 1,    5)
 
    return min(int(risk), 100)
 
 
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
 
def fill_missing_ml_fields(customer_dict: dict) -> None:
    """
    Fill critical Stage 1 ML model fields that the form may not collect.
    Derives smart defaults from fields that ARE collected.
    Modifies customer_dict in-place.
 
    BUG 1 FIX: AMT_ANNUITY and AMT_INCOME_TOTAL are two of the 15 model
    features but were never mapped from the form inputs. The model was
    silently receiving 0 for both on every prediction.
      - AMT_ANNUITY    = the proposed loan EMI (loan_amount is the principal;
                         we use the calculated new_emi from affordability if
                         available, otherwise fall back to loan_amount itself
                         as a conservative proxy)
      - AMT_INCOME_TOTAL = annual income (avg_salary_6m × 12)
 
    BUG 4 FIX: hard_reject_flag removed — it is not in the 15 model features
    and setting it here is misleading dead weight.
    """
    # FIX 1: round DPD counts — synthetic jitter produces floats like 0.9904 and 1.982.
    # All downstream comparisons (> 0, > 5, * 10) require integer counts.
    dpd_90 = int(round(float(customer_dict.get('dpd_90_count_6m', 0) or 0)))
    dpd_30 = int(round(float(customer_dict.get('dpd_30_count_6m', 0) or 0)))
 
    # ── BUG 1 FIX: map form fields → model feature names ──────────────────
    # AMT_ANNUITY: use the pre-calculated new_emi if affordability has already
    # been run (most common path), otherwise fall back to loan_amount as proxy.
    if 'AMT_ANNUITY' not in customer_dict:
        customer_dict['AMT_ANNUITY'] = (
            customer_dict.get('new_emi')          # set by calculate_affordability()
            or customer_dict.get('loan_amount', 0) # fallback: principal as proxy
        )
 
    # AMT_INCOME_TOTAL: annual income derived from monthly salary
    if 'AMT_INCOME_TOTAL' not in customer_dict:
        customer_dict['AMT_INCOME_TOTAL'] = customer_dict.get('avg_salary_6m', 0) * 12
 
    defaults = {
        'inward_bounce_count_3m':     0,
        'salary_missing_months':      0,
        'salary_amount_cv':           0.1,
        'salary_creditor_consistent': 1.0,
        'salary_txn_count_6m':        6,
        'payment_discipline_flag':    'GOOD',
        'liquidity_flag':             'LOW',      # Most common value in training data (87.7%)
        'cashflow_health':            'MODERATE',
        'bureau_risk_flag':           'LOW',
        # hard_reject_flag removed — not a model feature (Bug 4 fix)
        'total_dpd_count':            dpd_90 + dpd_30,
        'max_dpd_6m':                 90 if dpd_90 > 0 else (30 if dpd_30 > 0 else 0),
        'recent_payment_stress':      1 if dpd_90 > 0 else 0,
        'total_late_15_6m':           0,
        'total_late_30_6m':           customer_dict.get('total_late_30_6m', dpd_30),
        'total_late_60_6m':           0,
        'total_late_90_6m':           customer_dict.get('total_late_90_6m', dpd_90),
        'total_emi_monthly':          customer_dict.get('existing_emi', 0),
        'max_utilization':            customer_dict.get('credit_utilization_pct', 0),
    }
    for field, default_val in defaults.items():
        if field not in customer_dict:
            customer_dict[field] = default_val
 
 
def clean_sentinel_values(data_dict: dict) -> dict:
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
        'tot_enq',
    ]
    for field in sentinel_fields:
        val = cleaned.get(field)
        if val is not None and (val < 0 or val == -99999):
            cleaned[field] = 0
    return cleaned
 
 
def validate_cibil_identity(stage1_customer: dict, extraction_result: dict) -> list:
    """
    Validate that CIBIL PDF belongs to the same applicant as Stage 1.
    Returns list of warning strings (empty list = no issues).
    """
    warnings = []
 
    s1_age = stage1_customer.get('age', 0)
    s2_age = extraction_result.get('AGE', 0)
    if s1_age and s2_age and abs(s1_age - s2_age) > 5:
        warnings.append(
            f"Age mismatch: Application age {s1_age} vs CIBIL age {s2_age} "
            f"(difference > 5 years)"
        )
 
    s1_income = stage1_customer.get('avg_salary_6m', 0)
    s2_income = extraction_result.get('NETMONTHLYINCOME', 0)
    if s1_income and s2_income:
        ratio = max(s1_income, s2_income) / max(min(s1_income, s2_income), 1)
        # FIX M3: Tightened from 5x to 2x — a 5x mismatch threshold was too loose,
        # allowing e.g. ₹20,000 on application vs ₹99,000 on CIBIL to pass silently.
        # 2x catches meaningful income inflation while allowing for normal rounding/timing differences.
        if ratio > 2:
            warnings.append(
                f"Income mismatch: Application ₹{s1_income:,} vs CIBIL ₹{s2_income:,} "
                f"(ratio {ratio:.1f}x)"
            )
 
    s1_bureau = stage1_customer.get('bureau_score', 0)
    s2_bureau = extraction_result.get('Credit_Score', 0)
    if s1_bureau and s2_bureau and abs(s1_bureau - s2_bureau) > 100:
        warnings.append(
            f"Bureau score mismatch: Application {s1_bureau} vs CIBIL {s2_bureau} "
            f"(difference > 100)"
        )
 
    return warnings
