"""
Affordability Calculation Engine
Calculates FOIR, EMI, and Net Disposable Income
"""

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
    
    # Maximum allowed EMI (at 50% FOIR)
    max_allowed_emi = monthly_income * 0.50
    
    # Recommended EMI (at 40% FOIR)
    recommended_emi = monthly_income * 0.40
    
    # Affordability status
    affordable = foir_percentage <= 50
    within_recommended = foir_percentage <= 40
    
    # Status determination
    if foir_percentage <= 40:
        status = "Excellent"
        status_color = "green"
    elif foir_percentage <= 50:
        status = "Acceptable"
        status_color = "yellow"
    else:
        status = "Over-leveraged"
        status_color = "red"
    
    return {
        'monthly_income': monthly_income,
        'new_emi': new_emi,
        'existing_emi': existing_emi,
        'total_emi': total_emi,
        'foir_percentage': round(foir_percentage, 2),
        'net_disposable': net_disposable,
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
    
    if foir <= 40:
        return f"✅ EMI is well within comfortable limits ({foir:.1f}% of income)"
    elif foir <= 50:
        return f"⚠️ EMI is acceptable but high ({foir:.1f}% of income). Consider reducing loan amount."
    else:
        return f"❌ EMI exceeds maximum limit ({foir:.1f}% > 50%). Loan not affordable."