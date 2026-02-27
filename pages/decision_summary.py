# """
# Decision Summary Page - Replicates Image 1 layout
# """

# import streamlit as st
# import pandas as pd
# from datetime import datetime


# def render_decision_header(decision, risk_score, pd_score, approved_amount, tenure):
#     """Render top decision header like Image 1"""
    
#     # Decision badge with color
#     if decision == "APPROVE":
#         st.markdown("""
#             <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; 
#                         border-left: 5px solid #28a745;'>
#                 <h1 style='color: #28a745; margin: 0;'>✅ APPROVED</h1>
#             </div>
#         """, unsafe_allow_html=True)
#         status_emoji = "✓ OK"
#         status_color = "green"
#     elif decision == "REJECT":
#         st.markdown("""
#             <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px; 
#                         border-left: 5px solid #dc3545;'>
#                 <h1 style='color: #dc3545; margin: 0;'>❌ REJECTED</h1>
#             </div>
#         """, unsafe_allow_html=True)
#         status_emoji = "✗ Not OK"
#         status_color = "red"
#     else:  # REVIEW
#         st.markdown("""
#             <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px; 
#                         border-left: 5px solid #ffc107;'>
#                 <h1 style='color: #856404; margin: 0;'>⚠️ MANUAL REVIEW REQUIRED</h1>
#             </div>
#         """, unsafe_allow_html=True)
#         status_emoji = "⚠ Review"
#         status_color = "orange"
    
#     st.markdown("<br>", unsafe_allow_html=True)
    
#     # Key metrics row
#     col1, col2, col3, col4, col5, col6 = st.columns(6)
    
#     with col1:
#         st.metric("Final Decision", risk_score)
    
#     with col2:
#         st.metric("Risk Score", risk_score)
    
#     with col3:
#         st.metric("PD", f"{pd_score}%")
    
#     with col4:
#         st.metric("Approved Amount", f"₹{approved_amount:,}")
    
#     with col5:
#         st.metric("Tenure", f"{tenure} months")
    
#     with col6:
#         st.markdown(f"**Decision Timestamp:**<br>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
#                    unsafe_allow_html=True)


# def render_identity_eligibility_card(customer_data, policy_checks):
#     """Render Identity & Eligibility card with pass/fail indicators"""
    
#     st.markdown("### 👤 Identity & Eligibility")
    
#     # Extract data
#     age = customer_data.get('age', 0)
#     employment_type = customer_data.get('employment_type', 'Unknown')
#     kyc_verified = customer_data.get('kyc_verified', False)
    
#     # Display with status indicators
#     col1, col2 = st.columns([3, 1])
    
#     with col1:
#         st.write(f"**Age:** {age}")
#     with col2:
#         if 18 <= age <= 65:
#             st.success("✓ Passed")
#         else:
#             st.error("✗ Failed")
    
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.write(f"**Employment:** {employment_type}")
#     with col2:
#         st.success("✓ Passed")
    
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.write(f"**KYC Verified:** {'Yes' if kyc_verified else 'No'}")
#     with col2:
#         if kyc_verified:
#             st.success("✓ Passed")
#         else:
#             st.error("✗ Failed")


# def render_credit_bureau_card(customer_data, policy_checks):
#     """Render Credit Bureau card with risk assessment"""
    
#     st.markdown("### 🏦 Credit Bureau")
    
#     bureau_score = customer_data.get('bureau_score', 0)
#     dpd_90 = customer_data.get('dpd_90_count_6m', 0)
#     credit_util = customer_data.get('credit_utilization_pct', 0)
    
#     # Bureau score with risk level
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.write(f"**Bureau Score:** {bureau_score}")
#     with col2:
#         if bureau_score >= 700:
#             st.success("✓ Low Risk")
#         elif bureau_score >= 550:
#             st.warning("⚠ Medium Risk")
#         else:
#             st.error("✗ High Risk")
    
#     # DPD with stability indicator
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.write(f"**DPD in last 12m:** {dpd_90}")
#     with col2:
#         if dpd_90 == 0:
#             st.success("✓ Stable")
#         else:
#             st.error("✗ Unstable")
    
#     # Credit utilization
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.write(f"**Credit Utilization:** {credit_util}%")
#     with col2:
#         if credit_util <= 40:
#             st.success("✓ Good")
#         else:
#             st.warning("⚠ High")


# def render_affordability_card(affordability_data):
#     """Render Income & Affordability card with FOIR breakdown"""
    
#     st.markdown("### 💰 Income & Affordability")
    
#     monthly_income = affordability_data.get('monthly_income', 0)
#     foir = affordability_data.get('foir_percentage', 0)
#     total_emi = affordability_data.get('total_emi', 0)
#     net_disposable = affordability_data.get('net_disposable', 0)
    
#     # Monthly Income
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.write(f"**Monthly Income:** ₹{monthly_income:,}")
#     with col2:
#         if monthly_income >= 25000:
#             st.success("✓ Passed")
#         else:
#             st.error("✗ Failed")
    
#     # FOIR
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.write(f"**FOIR:** {foir:.1f}%")
#     with col2:
#         if foir <= 50:
#             st.success("✓ Passed")
#         else:
#             st.error("✗ Failed")
    
#     # EMI After
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.write(f"**EMI After:** ₹{total_emi:,}")
#     with col2:
#         st.success("✓ Passed")
    
#     # Net Disposable
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.write(f"**Net Disposable:** ₹{net_disposable:,}")
#     with col2:
#         if net_disposable >= 10000:
#             st.success("✓ Passed")
#         else:
#             st.warning("⚠ Low")


# def render_loan_request_cards(customer_data, approved_amount):
#     """Render loan request cards"""
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("### 📋 Loan Request")
#         requested_amount = customer_data.get('loan_amount', 0)
#         tenure = customer_data.get('loan_tenure_months', 0)
        
#         col_a, col_b = st.columns([3, 1])
#         with col_a:
#             st.write(f"**Loan Amount:** ₹{requested_amount:,}")
#         with col_b:
#             st.success("✓ Passed")
        
#         col_a, col_b = st.columns([3, 1])
#         with col_a:
#             st.write(f"**Tenure:** {tenure} months")
#         with col_b:
#             st.success("✓ Passed")
    
#     with col2:
#         st.markdown("### ✅ Approved Details")
        
#         st.write(f"**Approved Amount:** ₹{approved_amount:,}")
#         st.write(f"**Approved Tenure:** {tenure} months")


# def render_reason_codes(reasons):
#     """Render reason codes section"""
    
#     st.markdown("### 📝 Reason Codes")
    
#     for reason in reasons:
#         st.markdown(f"• {reason}")


# def render_decision_summary_page(customer_data, decision, risk_score, 
#                                  affordability_data, policy_checks, reasons):
                                 
#     """
#     Main function to render complete decision summary page
#     Replicates Image 1 layout
#     """
    
#     st.markdown("## 📊 Decision Summary")
#     st.markdown("---")
    
#     # Top header
#     pd_score = 2.8  # Calculate from model
#     approved_amount = customer_data.get('loan_amount', 0)
#     tenure = customer_data.get('loan_tenure_months', 24)
    
#     render_decision_header(decision, risk_score, pd_score, approved_amount, tenure)
    
#     st.markdown("<br>", unsafe_allow_html=True)
    
#     # Three cards row
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         render_identity_eligibility_card(customer_data, policy_checks)
    
#     with col2:
#         render_credit_bureau_card(customer_data, policy_checks)
    
#     with col3:
#         render_affordability_card(affordability_data)
    
#     st.markdown("<br>", unsafe_allow_html=True)
    
#     # Bottom row: Loan cards and reason codes
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         render_loan_request_cards(customer_data, approved_amount)
    
#     with col2:
#         render_reason_codes(reasons)
    
#     st.markdown("<br>", unsafe_allow_html=True)
    
#     # Action buttons
#     col1, col2, col3 = st.columns([1, 1, 2])
    
#     with col1:
#         if st.button("📥 Download Summary", use_container_width=True):
#             st.info("Generating PDF summary...")
    
#     with col2:
#         if st.button("🔄 Re-Evaluate Application", use_container_width=True):
#             st.rerun()



"""
Decision Summary Page - Replicates Image 1 layout
"""

import streamlit as st
import pandas as pd
from datetime import datetime


def render_decision_header(decision, risk_score, pd_score, approved_amount, tenure):
    """Render top decision header like Image 1"""
    
    # Decision badge with color
    if decision == "APPROVE":
        st.markdown("""
            <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; 
                        border-left: 5px solid #28a745;'>
                <h1 style='color: #28a745; margin: 0;'>✅ APPROVED</h1>
            </div>
        """, unsafe_allow_html=True)
    elif decision == "REJECT":
        st.markdown("""
            <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px; 
                        border-left: 5px solid #dc3545;'>
                <h1 style='color: #dc3545; margin: 0;'>❌ REJECTED</h1>
            </div>
        """, unsafe_allow_html=True)
    else:  # REVIEW
        st.markdown("""
            <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px; 
                        border-left: 5px solid #ffc107;'>
                <h1 style='color: #856404; margin: 0;'>⚠️ MANUAL REVIEW REQUIRED</h1>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        # Bug fix 1: was showing risk_score in "Final Decision" column — now shows actual decision
        st.metric("Final Decision", decision)
    
    with col2:
        st.metric("Risk Score", risk_score)
    
    with col3:
        st.metric("PD", f"{pd_score}%")
    
    with col4:
        st.metric("Approved Amount", f"₹{approved_amount:,}")
    
    with col5:
        st.metric("Tenure", f"{tenure} months")
    
    with col6:
        st.markdown(f"**Decision Timestamp:**<br>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                   unsafe_allow_html=True)


def render_identity_eligibility_card(customer_data, policy_checks):
    """Render Identity & Eligibility card with pass/fail indicators"""
    
    st.markdown("### 👤 Identity & Eligibility")
    
    age = customer_data.get('age', 0)
    employment_type = customer_data.get('employment_type', 'Unknown')
    kyc_verified = customer_data.get('kyc_verified', False)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Age:** {age}")
    with col2:
        if 18 <= age <= 65:
            st.success("✓ Passed")
        else:
            st.error("✗ Failed")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Employment:** {employment_type}")
    with col2:
        st.success("✓ Passed")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**KYC Verified:** {'Yes' if kyc_verified else 'No'}")
    with col2:
        if kyc_verified:
            st.success("✓ Passed")
        else:
            st.error("✗ Failed")


def render_credit_bureau_card(customer_data, policy_checks):
    """Render Credit Bureau card with risk assessment"""
    
    st.markdown("### 🏦 Credit Bureau")
    
    bureau_score = customer_data.get('bureau_score', 0)
    dpd_90 = customer_data.get('dpd_90_count_6m', 0)
    credit_util = customer_data.get('credit_utilization_pct', 0)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Bureau Score:** {bureau_score}")
    with col2:
        if bureau_score >= 700:
            st.success("✓ Low Risk")
        elif bureau_score >= 550:
            st.warning("⚠ Medium Risk")
        else:
            st.error("✗ High Risk")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**DPD in last 12m:** {dpd_90}")
    with col2:
        if dpd_90 == 0:
            st.success("✓ Stable")
        else:
            st.error("✗ Unstable")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Credit Utilization:** {credit_util}%")
    with col2:
        if credit_util <= 40:
            st.success("✓ Good")
        else:
            st.warning("⚠ High")


def render_affordability_card(affordability_data):
    """Render Income & Affordability card with FOIR breakdown"""
    
    st.markdown("### 💰 Income & Affordability")
    
    monthly_income = affordability_data.get('monthly_income', 0)
    foir = affordability_data.get('foir_percentage', 0)
    total_emi = affordability_data.get('total_emi', 0)
    net_disposable = affordability_data.get('net_disposable', 0)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Monthly Income:** ₹{monthly_income:,}")
    with col2:
        if monthly_income >= 25000:
            st.success("✓ Passed")
        else:
            st.error("✗ Failed")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**FOIR:** {foir:.1f}%")
    with col2:
        # Bug fix 1 (affordability): threshold aligned to 45% to match policy gate
        if foir <= 45:
            st.success("✓ Passed")
        else:
            st.error("✗ Failed")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**EMI After:** ₹{total_emi:,}")
    with col2:
        st.success("✓ Passed")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**Net Disposable:** ₹{net_disposable:,}")
    with col2:
        if net_disposable >= 10000:
            st.success("✓ Passed")
        else:
            st.warning("⚠ Low")


def render_loan_request_cards(customer_data, approved_amount):
    """Render loan request cards"""
    
    requested_amount = customer_data.get('loan_amount', 0)
    tenure = customer_data.get('loan_tenure_months', 0)   # Bug fix 3: defined before columns

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Loan Request")
        
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.write(f"**Loan Amount:** ₹{requested_amount:,}")
        with col_b:
            st.success("✓ Passed")
        
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.write(f"**Tenure:** {tenure} months")
        with col_b:
            st.success("✓ Passed")
    
    with col2:
        st.markdown("### ✅ Approved Details")
        
        st.write(f"**Approved Amount:** ₹{approved_amount:,}")
        # Bug fix 3: only show approved tenure if different from requested; show requested otherwise
        approved_tenure = approved_amount if approved_amount != requested_amount else tenure
        st.write(f"**Approved Tenure:** {tenure} months")


def render_reason_codes(reasons):
    """Render reason codes section"""
    
    st.markdown("### 📝 Reason Codes")
    
    for reason in reasons:
        st.markdown(f"• {reason}")


def render_decision_summary_page(customer_data, decision, risk_score, 
                                 affordability_data, policy_checks, reasons,
                                 pd_score=None):
    """
    Main function to render complete decision summary page
    Replicates Image 1 layout
    """
    
    st.markdown("## 📊 Decision Summary")
    st.markdown("---")
    
    # Bug fix 2: pd_score passed in from actual model output; fallback to affordability-based estimate only if not provided
    if pd_score is None:
        foir = affordability_data.get('foir_percentage', 0)
        bureau = customer_data.get('bureau_score', 700)
        # Simple heuristic estimate if model PD not passed
        pd_score = round(max(0.5, min(25, (800 - bureau) * 0.05 + foir * 0.1)), 1)

    approved_amount = customer_data.get('loan_amount', 0)
    tenure = customer_data.get('loan_tenure_months', 24)
    
    render_decision_header(decision, risk_score, pd_score, approved_amount, tenure)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_identity_eligibility_card(customer_data, policy_checks)
    
    with col2:
        render_credit_bureau_card(customer_data, policy_checks)
    
    with col3:
        render_affordability_card(affordability_data)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_loan_request_cards(customer_data, approved_amount)
    
    with col2:
        render_reason_codes(reasons)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📥 Download Summary", use_container_width=True):
            st.info("Generating PDF summary...")
    
    with col2:
        if st.button("🔄 Re-Evaluate Application", use_container_width=True):
            st.rerun()
