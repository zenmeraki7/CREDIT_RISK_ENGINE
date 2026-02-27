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
