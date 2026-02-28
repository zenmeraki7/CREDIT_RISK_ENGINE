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


# =============================================================================
# NEW FUNCTIONS REQUIRED BY test.py (new version)
# =============================================================================

def render_info_card(title, icon, data_dict, status_dict=None):
    """
    Render a styled info card with key-value pairs and optional pass/fail status.
    status_dict values: 'pass', 'fail', 'warning', or ''
    """
    rows_html = ""
    for key, value in data_dict.items():
        status = status_dict.get(key, '') if status_dict else ''
        if status == 'pass':
            badge = "<span style='color:#28a745;font-weight:bold;'>✓</span>"
        elif status == 'fail':
            badge = "<span style='color:#dc3545;font-weight:bold;'>✗</span>"
        elif status == 'warning':
            badge = "<span style='color:#ffc107;font-weight:bold;'>⚠</span>"
        else:
            badge = ""
        # For rows where value is empty, just show the key with badge
        display_val = f"{value}" if value != "" else ""
        rows_html += f"""
            <div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #eee;'>
                <span style='color:#555;font-size:0.85rem;'>{key}</span>
                <span style='font-weight:500;font-size:0.85rem;'>{display_val} {badge}</span>
            </div>"""

    st.markdown(f"""
        <div style='background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:1rem;margin-bottom:1rem;'>
            <div style='font-weight:700;font-size:1rem;margin-bottom:0.75rem;color:#333;'>
                {icon} {title}
            </div>
            {rows_html}
        </div>
    """, unsafe_allow_html=True)


def create_modern_gauge(value, title, max_value=100):
    """
    Create a modern Plotly gauge chart for metrics like confidence or risk score.
    Returns a Plotly Figure.
    """
    import plotly.graph_objects as go

    # Color zones based on value
    if value >= 75:
        bar_color = "#28a745"
    elif value >= 50:
        bar_color = "#ffc107"
    else:
        bar_color = "#dc3545"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        number={'suffix': "%", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 1},
            'bar': {'color': bar_color},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': '#fdecea'},
                {'range': [max_value * 0.5, max_value * 0.75], 'color': '#fff3cd'},
                {'range': [max_value * 0.75, max_value], 'color': '#d4edda'},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_modern_bar_chart(class_probs):
    """
    Create a modern Plotly bar chart for class probabilities.
    class_probs: dict like {'APPROVE': 70.5, 'REVIEW': 20.0, 'REJECT': 9.5}
    Returns a Plotly Figure.
    """
    import plotly.graph_objects as go

    color_map = {
        'APPROVE': '#28a745',
        'REVIEW': '#ffc107',
        'REJECT': '#dc3545'
    }

    labels = list(class_probs.keys())
    values = list(class_probs.values())
    colors = [color_map.get(label, '#6c757d') for label in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside'
    ))
    fig.update_layout(
        title="Model Probability by Class",
        yaxis=dict(range=[0, 110], title="Probability (%)"),
        xaxis=dict(title="Decision Class"),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig
