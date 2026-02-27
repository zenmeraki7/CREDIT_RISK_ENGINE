# """
# Affordability Calculation Engine
# Calculates FOIR, EMI, and Net Disposable Income
# """

# def calculate_emi(principal, annual_rate, tenure_months):
#     """Calculate EMI using reducing balance method"""
#     if principal <= 0 or tenure_months <= 0:
#         return 0
    
#     monthly_rate = annual_rate / (12 * 100)
    
#     if monthly_rate == 0:
#         return principal / tenure_months
    
#     emi = (principal * monthly_rate * (1 + monthly_rate)**tenure_months) / \
#           ((1 + monthly_rate)**tenure_months - 1)
    
#     return round(emi, 2)


# def calculate_affordability(monthly_income, loan_amount, interest_rate, 
#                            tenure_months, existing_emi):
#     """
#     Calculate comprehensive affordability metrics
#     Returns dictionary with all FOIR components
#     """
    
#     # Calculate new EMI
#     new_emi = calculate_emi(loan_amount, interest_rate, tenure_months)
    
#     # Total EMI burden
#     total_emi = new_emi + existing_emi
    
#     # FOIR calculation
#     foir_percentage = (total_emi / monthly_income) * 100 if monthly_income > 0 else 0
    
#     # Net disposable income
#     net_disposable = monthly_income - total_emi
    
#     # Maximum allowed EMI (at 50% FOIR)
#     max_allowed_emi = monthly_income * 0.50
    
#     # Recommended EMI (at 40% FOIR)
#     recommended_emi = monthly_income * 0.40
    
#     # Affordability status
#     affordable = foir_percentage <= 50
#     within_recommended = foir_percentage <= 40
    
#     # Status determination
#     if foir_percentage <= 40:
#         status = "Excellent"
#         status_color = "green"
#     elif foir_percentage <= 50:
#         status = "Acceptable"
#         status_color = "yellow"
#     else:
#         status = "Over-leveraged"
#         status_color = "red"
    
#     return {
#         'monthly_income': monthly_income,
#         'new_emi': new_emi,
#         'existing_emi': existing_emi,
#         'total_emi': total_emi,
#         'foir_percentage': round(foir_percentage, 2),
#         'net_disposable': net_disposable,
#         'max_allowed_emi': max_allowed_emi,
#         'recommended_emi': recommended_emi,
#         'affordable': affordable,
#         'within_recommended': within_recommended,
#         'status': status,
#         'status_color': status_color,
#         'emi_headroom': max_allowed_emi - total_emi
#     }


# def get_affordability_message(affordability_data):
#     """Generate human-readable affordability message"""
#     foir = affordability_data['foir_percentage']
    
#     if foir <= 40:
#         return f"✅ EMI is well within comfortable limits ({foir:.1f}% of income)"
#     elif foir <= 50:
#         return f"⚠️ EMI is acceptable but high ({foir:.1f}% of income). Consider reducing loan amount."
#     else:
#         return f"❌ EMI exceeds maximum limit ({foir:.1f}% > 50%). Loan not affordable."



"""
Affordability Calculation Engine
Calculates FOIR, EMI, and Net Disposable Income
"""

# Bug fix 1: FOIR policy gate is 45% (matches test.py), not 50%
FOIR_HARD_LIMIT = 45.0       # Hard reject threshold (matches test.py policy gate)
FOIR_REVIEW_LIMIT = 40.0     # Review threshold
FOIR_COMFORTABLE = 35.0      # Comfortable limit
MIN_NET_DISPOSABLE = 10000   # Minimum net disposable income required


def calculate_emi(principal, annual_rate, tenure_months):
    """Calculate EMI using reducing balance method"""
    if principal <= 0 or tenure_months <= 0:
        return 0
    
    monthly_rate = annual_rate / (12 * 100)
    
    if monthly_rate == 0:
        return principal / tenure_months
    
    emi = (principal * monthly_rate * (1 + monthly_rate)**tenure_months) / \
          ((1 + monthly_rate)**tenure_months - 1)
    
    return round(emi, 2)


def calculate_affordability(monthly_income, loan_amount, interest_rate, 
                           tenure_months, existing_emi):
    """
    Calculate comprehensive affordability metrics
    Returns dictionary with all FOIR components
    """
    
    # Calculate new EMI
    new_emi = calculate_emi(loan_amount, interest_rate, tenure_months)
    
    # Total EMI burden
    total_emi = new_emi + existing_emi
    
    # FOIR calculation
    foir_percentage = (total_emi / monthly_income) * 100 if monthly_income > 0 else 0
    
    # Net disposable income
    net_disposable = monthly_income - total_emi
    
    # Bug fix 1: max_allowed_emi aligned to 45% FOIR (was 50%, inconsistent with test.py policy gate)
    max_allowed_emi = monthly_income * (FOIR_HARD_LIMIT / 100)
    
    # Recommended EMI (at 35% FOIR - comfortable)
    recommended_emi = monthly_income * (FOIR_COMFORTABLE / 100)
    
    # Bug fix 1: affordable flag uses 45% threshold to match policy gate
    affordable = foir_percentage <= FOIR_HARD_LIMIT
    within_recommended = foir_percentage <= FOIR_COMFORTABLE
    
    # Status determination
    if foir_percentage <= FOIR_COMFORTABLE:
        status = "Excellent"
        status_color = "green"
    elif foir_percentage <= FOIR_REVIEW_LIMIT:
        status = "Acceptable"
        status_color = "yellow"
    elif foir_percentage <= FOIR_HARD_LIMIT:
        status = "High - Review Required"
        status_color = "orange"
    else:
        status = "Over-leveraged"
        status_color = "red"
    
    # Bug fix 2: add net_disposable_sufficient flag so callers can check minimum threshold
    net_disposable_sufficient = net_disposable >= MIN_NET_DISPOSABLE
    
    return {
        'monthly_income': monthly_income,
        'new_emi': new_emi,
        'existing_emi': existing_emi,
        'total_emi': total_emi,
        'foir_percentage': round(foir_percentage, 2),
        'net_disposable': net_disposable,
        'net_disposable_sufficient': net_disposable_sufficient,   # Bug fix 2: new flag
        'max_allowed_emi': max_allowed_emi,
        'recommended_emi': recommended_emi,
        'affordable': affordable,
        'within_recommended': within_recommended,
        'status': status,
        'status_color': status_color,
        'emi_headroom': max_allowed_emi - total_emi
    }


def get_affordability_message(affordability_data):
    """Generate human-readable affordability message"""
    foir = affordability_data['foir_percentage']
    
    if foir <= FOIR_COMFORTABLE:
        return f"✅ EMI is well within comfortable limits ({foir:.1f}% of income)"
    elif foir <= FOIR_REVIEW_LIMIT:
        return f"⚠️ EMI is acceptable but elevated ({foir:.1f}% of income). Consider reducing loan amount."
    elif foir <= FOIR_HARD_LIMIT:
        return f"⚠️ EMI is high ({foir:.1f}% of income). Manual review required."
    else:
        return f"❌ EMI exceeds maximum limit ({foir:.1f}% > {FOIR_HARD_LIMIT}%). Loan not affordable."


def check_loan_to_income(loan_amount, annual_income):
    """
    Check loan-to-income ratio.
    Returns dict with ratio, status, and message.
    Status: 'ok', 'high', 'extreme'
    """
    ratio = round(loan_amount / annual_income, 2) if annual_income > 0 else 99
    if ratio <= 3:
        status = 'ok'
        message = f"Loan-to-income ratio {ratio}x is acceptable"
    elif ratio <= 5:
        status = 'high'
        message = f"Loan-to-income ratio {ratio}x is high — review recommended"
    else:
        status = 'extreme'
        message = f"Loan-to-income ratio {ratio}x is extreme — high default risk"
    return {'ratio': ratio, 'status': status, 'message': message}


def check_net_disposable(net_disposable, minimum=10000):
    """
    Check net disposable income against minimum threshold.
    Returns dict with status and message.
    Status: 'ok', 'low', 'critical'
    """
    if net_disposable >= minimum:
        status = 'ok'
        message = f"Net disposable ₹{net_disposable:,} is above minimum"
    elif net_disposable >= 0:
        status = 'low'
        message = f"Net disposable ₹{net_disposable:,} is below recommended minimum ₹{minimum:,}"
    else:
        status = 'critical'
        message = f"Net disposable ₹{net_disposable:,} is negative — over-leveraged"
    return {'net_disposable': net_disposable, 'status': status, 'message': message, 'minimum': minimum}
