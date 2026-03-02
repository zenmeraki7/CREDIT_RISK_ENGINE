"""
Affordability Calculation Engine
Calculates FOIR, EMI, Net Disposable Income, and Loan-to-Income ratio.

Author: Zen Meraki
Version: 2.0 - ALIGNED TO ORIGINAL DESIGN
"""

# =============================================================================
# FOIR THRESHOLDS — Restored to original design
# Your design: 0-40% excellent, 40-50% acceptable, >50% over-leveraged
# =============================================================================
FOIR_HARD_LIMIT    = 50.0   # Hard reject threshold (your original: >50% = over-leveraged)
FOIR_REVIEW_LIMIT  = 45.0   # Review/orange zone upper boundary
FOIR_ACCEPTABLE    = 40.0   # Acceptable/yellow zone upper boundary
FOIR_COMFORTABLE   = 35.0   # Comfortable limit (recommended)
MIN_NET_DISPOSABLE = 10000  # Minimum net disposable income required (₹10,000/month)


def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    """
    Calculate EMI using the standard reducing-balance method.

    Formula: EMI = P × r × (1+r)^n / ((1+r)^n - 1)
    where r = monthly rate, n = tenure months

    Returns 0 for invalid inputs.
    """
    if principal <= 0 or tenure_months <= 0:
        return 0.0

    monthly_rate = annual_rate / (12.0 * 100.0)

    if monthly_rate == 0:
        return round(principal / tenure_months, 2)

    emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure_months) / \
          ((1 + monthly_rate) ** tenure_months - 1)

    return round(emi, 2)


def calculate_affordability(
        monthly_income: float,
        loan_amount: float,
        interest_rate: float,
        tenure_months: int,
        existing_emi: float
) -> dict:
    """
    Calculate comprehensive affordability metrics.

    FOIR bands (your original design):
        ≤ 35%   → Excellent  (green)
        ≤ 40%   → Acceptable (yellow)
        ≤ 45%   → High – Review Required (orange)
        ≤ 50%   → Elevated  (orange-red)
        > 50%   → Over-leveraged – REJECT (red)

    Returns dict with all FOIR components.
    """
    new_emi          = calculate_emi(loan_amount, interest_rate, tenure_months)
    total_emi        = new_emi + existing_emi
    foir_percentage  = (total_emi / monthly_income * 100) if monthly_income > 0 else 0.0
    net_disposable   = monthly_income - total_emi
    max_allowed_emi  = monthly_income * (FOIR_HARD_LIMIT / 100)    # 50% cap
    recommended_emi  = monthly_income * (FOIR_COMFORTABLE / 100)   # 35% comfort

    affordable            = foir_percentage <= FOIR_HARD_LIMIT
    within_recommended    = foir_percentage <= FOIR_COMFORTABLE
    net_disposable_ok     = net_disposable >= MIN_NET_DISPOSABLE

    # Status label — matches your original green/yellow/red design
    if foir_percentage <= FOIR_COMFORTABLE:
        status       = "Excellent"
        status_color = "green"
    elif foir_percentage <= FOIR_ACCEPTABLE:
        status       = "Acceptable"
        status_color = "yellow"
    elif foir_percentage <= FOIR_REVIEW_LIMIT:
        status       = "High - Review Required"
        status_color = "orange"
    elif foir_percentage <= FOIR_HARD_LIMIT:
        status       = "Elevated - Underwriter Required"
        status_color = "orange"
    else:
        status       = "Over-leveraged"
        status_color = "red"

    return {
        'monthly_income':          monthly_income,
        'new_emi':                 new_emi,
        'existing_emi':            existing_emi,
        'total_emi':               total_emi,
        'foir_percentage':         round(foir_percentage, 2),
        'net_disposable':          round(net_disposable, 2),
        'net_disposable_sufficient': net_disposable_ok,
        'max_allowed_emi':         round(max_allowed_emi, 2),
        'recommended_emi':         round(recommended_emi, 2),
        'affordable':              affordable,
        'within_recommended':      within_recommended,
        'status':                  status,
        'status_color':            status_color,
        'emi_headroom':            round(max_allowed_emi - total_emi, 2),
    }


def get_affordability_message(affordability_data: dict) -> str:
    """Return a human-readable one-liner for the affordability status."""
    foir = affordability_data['foir_percentage']

    if foir <= FOIR_COMFORTABLE:
        return f"✅ EMI well within comfortable limits ({foir:.1f}% of income)"
    elif foir <= FOIR_ACCEPTABLE:
        return f"⚠️ EMI acceptable but elevated ({foir:.1f}% of income). Consider reducing loan amount."
    elif foir <= FOIR_REVIEW_LIMIT:
        return f"⚠️ EMI is high ({foir:.1f}% of income). Manual review required."
    elif foir <= FOIR_HARD_LIMIT:
        return f"⚠️ EMI is elevated ({foir:.1f}% of income). Underwriter sign-off required."
    else:
        return f"❌ EMI exceeds maximum limit ({foir:.1f}% > {FOIR_HARD_LIMIT}%). Loan not affordable."


def check_loan_to_income(loan_amount: float, annual_income: float) -> dict:
    """
    Loan-to-Income ratio check.
    ≤ 3x  → ok
    ≤ 5x  → high (review)
    > 5x  → extreme (reject or refer)
    """
    ratio = round(loan_amount / annual_income, 2) if annual_income > 0 else 99.0

    if ratio <= 3:
        return {'ratio': ratio, 'status': 'ok',
                'message': f"Loan-to-income ratio {ratio}x is acceptable"}
    elif ratio <= 5:
        return {'ratio': ratio, 'status': 'high',
                'message': f"Loan-to-income ratio {ratio}x is high — review recommended"}
    else:
        return {'ratio': ratio, 'status': 'extreme',
                'message': f"Loan-to-income ratio {ratio}x is extreme — high default risk"}


def check_net_disposable(net_disposable: float, minimum: float = 10000.0) -> dict:
    """
    Net disposable income check.
    ≥ minimum → ok
    ≥ 0       → low (warning)
    < 0       → critical
    """
    if net_disposable >= minimum:
        return {'net_disposable': net_disposable, 'status': 'ok', 'minimum': minimum,
                'message': f"Net disposable ₹{net_disposable:,.0f} is above minimum ₹{minimum:,.0f}"}
    elif net_disposable >= 0:
        return {'net_disposable': net_disposable, 'status': 'low', 'minimum': minimum,
                'message': f"Net disposable ₹{net_disposable:,.0f} is below minimum ₹{minimum:,.0f}"}
    else:
        return {'net_disposable': net_disposable, 'status': 'critical', 'minimum': minimum,
                'message': f"Net disposable ₹{net_disposable:,.0f} is negative — over-leveraged"}
