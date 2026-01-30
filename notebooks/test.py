
"""
Credit Risk Assessment Dashboard - Hybrid ML + Rule-Based System
Run with: streamlit run app.py

Author: Zen Meraki  
Date: January 2025
VERSION: 6.0 - Added comprehensive policy rules (age, employment, KYC, etc.)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .approved {
        color: #28a745;
        font-weight: bold;
        font-size: 2rem;
    }
    .rejected {
        color: #dc3545;
        font-weight: bold;
        font-size: 2rem;
    }
    .review {
        color: #ffc107;
        font-weight: bold;
        font-size: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .policy-check {
        padding: 0.5rem;
        margin: 0.2rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
    .check-pass {
        background-color: #d4edda;
        color: #155724;
    }
    .check-fail {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD TRAINED MODEL ASSETS
# =============================================================================

@st.cache_resource
def load_model_assets():
    """Load the trained model and preprocessing assets"""
    try:
        possible_paths = [
            'credit_risk_assets.pkl',
            'notebooks/credit_risk_assets.pkl',
            '../notebooks/credit_risk_assets.pkl'
        ]
        
        assets = None
        for path in possible_paths:
            try:
                assets = joblib.load(path)
                st.sidebar.info(f"✅ Loaded from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if assets is None:
            raise FileNotFoundError("Could not find credit_risk_assets.pkl")
        return {
            'model': assets['model'],
            'features': assets['features'],
            'le_map': assets['le_map'],
            'target_le': assets['target_le'],
            'loaded': True,
            'error': None
        }
    except FileNotFoundError:
        return {
            'loaded': False,
            'error': 'credit_risk_assets.pkl not found. Please run the training script first.'
        }
    except Exception as e:
        return {
            'loaded': False,
            'error': f'Error loading model: {str(e)}'
        }

# Load assets
ASSETS = load_model_assets()

if not ASSETS['loaded']:
    st.error(f"❌ {ASSETS['error']}")
    st.info("Please ensure 'credit_risk_assets.pkl' is in the same directory as this app.")
    st.stop()

MODEL = ASSETS['model']
TOP_FEATURES = ASSETS['features']
LE_MAP = ASSETS['le_map']
TARGET_LE = ASSETS['target_le']

# =============================================================================
# HYBRID DECISION ENGINE WITH COMPREHENSIVE POLICY RULES
# =============================================================================

def make_hybrid_decision(customer_dict):
    """
    Implements comprehensive policy-based credit assessment
    Layer 1: Hard Policy Gates (Mandatory checks)
    Layer 2: ML Risk Assessment
    Layer 3: Affordability Overlay
    """
    
    policy_checks = {}
    
    # ==========================================
    # LAYER 1: HARD POLICY GATES
    # ==========================================
    
    # --- Section 2.1: Identity & Eligibility ---
    age = customer_dict.get('age', 0)
    employment_type = customer_dict.get('employment_type', 'Salaried')
    kyc_verified = customer_dict.get('kyc_verified', True)
    bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
    fraud_flag = customer_dict.get('fraud_flag', False)
    
    # Age validation
    if employment_type in ['Salaried']:
        age_min, age_max = 18, 65
    else:  # Self-employed/Business
        age_min, age_max = 18, 70
    
    if age < age_min or age > age_max:
        policy_checks['age'] = f"❌ Age {age} (Required: {age_min}-{age_max} for {employment_type})"
        return "REJECT", f"Policy Gate: Age outside allowed range ({age_min}-{age_max})", 0, {}, policy_checks
    policy_checks['age'] = f"✅ Age {age} (Valid)"
    
    # KYC verification
    if not kyc_verified:
        policy_checks['kyc'] = "❌ KYC Not Verified"
        return "REJECT", "Policy Gate: KYC verification required", 0, {}, policy_checks
    policy_checks['kyc'] = "✅ KYC Verified"
    
    # Bankruptcy/Fraud checks
    if bankruptcy_flag:
        policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
        return "REJECT", "Policy Gate: Active bankruptcy flag", 0, {}, policy_checks
    policy_checks['bankruptcy'] = "✅ No Bankruptcy"
    
    if fraud_flag:
        policy_checks['fraud'] = "❌ Fraud Flag"
        return "REJECT", "Policy Gate: Fraud flag detected", 0, {}, policy_checks
    policy_checks['fraud'] = "✅ No Fraud History"
    
    # --- Section 2.2: Income & Employment ---
    monthly_income = customer_dict.get('avg_salary_6m', 0)
    employment_tenure = customer_dict.get('employment_tenure_months', 0)
    business_vintage = customer_dict.get('business_vintage_years', 0)
    
    # Minimum income requirement
    if monthly_income < 15000:
        policy_checks['income'] = f"❌ Income ₹{monthly_income:,.0f} (Min: ₹15,000)"
        return "REJECT", "Policy Gate: Monthly income below ₹15,000", 0, {}, policy_checks
    policy_checks['income'] = f"✅ Income ₹{monthly_income:,.0f}"
    
    # Employment stability
    if employment_type == 'Salaried' and employment_tenure < 6:
        policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
        return "REJECT", "Policy Gate: Salaried employees need 6+ months tenure", 0, {}, policy_checks
    elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
        policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
        return "REJECT", "Policy Gate: Self-employed need 2+ years business vintage", 0, {}, policy_checks
    
    if employment_type == 'Salaried':
        policy_checks['tenure'] = f"✅ Tenure {employment_tenure} months"
    else:
        policy_checks['tenure'] = f"✅ Business Vintage {business_vintage} years"
    
    # --- Section 2.3: Credit Bureau ---
    bureau_score = customer_dict.get('bureau_score', 0)
    dpd_90 = customer_dict.get('dpd_90_count_6m', 0)
    credit_utilization = customer_dict.get('credit_utilization_pct', 0)
    recent_inquiries = customer_dict.get('recent_inquiries_3m', 0)
    
    # Hard credit score cutoff
    if bureau_score < 550:
        policy_checks['bureau'] = f"❌ Bureau Score {bureau_score} (Min: 550)"
        return "REJECT", "Policy Gate: Bureau score below 550", 0, {}, policy_checks
    policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"
    
    # Severe delinquency check
    if dpd_90 > 0:
        policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD"
        return "REJECT", "Policy Gate: Severe delinquency (90+ DPD) in last 6 months", 0, {}, policy_checks
    policy_checks['dpd'] = "✅ No 90+ DPD"
    
    # Credit utilization warning (not rejection)
    if credit_utilization > 80:
        policy_checks['utilization'] = f"⚠️ High utilization {credit_utilization}%"
    else:
        policy_checks['utilization'] = f"✅ Utilization {credit_utilization}%"
    
    # Recent inquiries warning
    if recent_inquiries > 5:
        policy_checks['inquiries'] = f"⚠️ {recent_inquiries} recent inquiries"
    else:
        policy_checks['inquiries'] = f"✅ {recent_inquiries} inquiries"
    
    # ==========================================
    # LAYER 2: ML RISK PREDICTION
    # ==========================================
    
    # Prepare data for ML model
    input_df = pd.DataFrame([customer_dict])
    
    # Ensure all required features exist
    for col in TOP_FEATURES:
        if col not in input_df.columns:
            if col in LE_MAP:
                input_df[col] = "Unknown"
            else:
                input_df[col] = 0
    
    # Apply Label Encoding
    for col, le in LE_MAP.items():
        if col in input_df.columns:
            val = str(input_df[col].values[0])
            try:
                input_df[col] = le.transform([val])[0]
            except ValueError:
                input_df[col] = 0
    
    # Select only training features
    final_input = input_df[TOP_FEATURES]
    
    # Get ML prediction
    pred_idx = MODEL.predict(final_input)[0]
    ml_decision = TARGET_LE.inverse_transform([pred_idx])[0]
    
    # Get confidence scores
    try:
        pred_proba = MODEL.predict_proba(final_input)[0]
        confidence = max(pred_proba) * 100
        class_probs = {
            cls: prob * 100 
            for cls, prob in zip(TARGET_LE.classes_, pred_proba)
        }
    except:
        confidence = 75.0
        class_probs = {ml_decision: 100.0}
    
    # ==========================================
    # LAYER 3: AFFORDABILITY OVERLAY
    # ==========================================
    
    income = customer_dict.get('avg_salary_6m', 1)
    emi = customer_dict.get('total_emi_monthly', 0)
    loan_amount = customer_dict.get('loan_amount', 0)
    loan_tenure = customer_dict.get('loan_tenure_months', 12)
    
    # Calculate proposed EMI
    if loan_amount > 0 and loan_tenure > 0:
        # Simple EMI calculation (can be enhanced with interest rate)
        monthly_rate = 0.12 / 12  # Assuming 12% annual interest
        proposed_emi = (loan_amount * monthly_rate * (1 + monthly_rate)**loan_tenure) / ((1 + monthly_rate)**loan_tenure - 1)
        total_emi_with_loan = emi + proposed_emi
    else:
        total_emi_with_loan = emi
        proposed_emi = 0
    
    # DTI ratio check
    if income > 0:
        dti_ratio = total_emi_with_loan / income
        policy_checks['dti'] = f"DTI: {dti_ratio:.1%} (EMI: ₹{total_emi_with_loan:,.0f} / Income: ₹{income:,.0f})"
        
        # Change APPROVE to REVIEW if DTI too high
        if ml_decision == "APPROVE" and dti_ratio > 0.45:
            return "REVIEW", f"Affordability: DTI ratio {dti_ratio:.1%} exceeds 45% limit", confidence, class_probs, policy_checks
    
    return ml_decision, "Decision based on Model Risk Score", confidence, class_probs, policy_checks

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_confidence_gauge(confidence, decision):
    """Create gauge chart for prediction confidence"""
    color = {
        'APPROVE': 'green',
        'REVIEW': 'orange',
        'REJECT': 'red'
    }.get(decision, 'gray')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Prediction Confidence", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "lightyellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_probability_chart(class_probs):
    """Create bar chart for class probabilities"""
    df = pd.DataFrame({
        'Decision': list(class_probs.keys()),
        'Probability': list(class_probs.values())
    })
    
    colors = {'APPROVE': '#28a745', 'REVIEW': '#ffc107', 'REJECT': '#dc3545'}
    df['Color'] = df['Decision'].map(colors)
    
    fig = px.bar(df, x='Decision', y='Probability',
                 title='Decision Probabilities',
                 color='Decision',
                 color_discrete_map=colors)
    fig.update_layout(showlegend=False, yaxis_title='Probability (%)')
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🏦 Credit Risk Engine")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "👤 Single Assessment", "📊 Batch Processing", "📈 Model Info", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.success(f"""
**Model Status:** ✅ Loaded

**Training Data:** 60K applications

**Top Features:** {len(TOP_FEATURES)}

**Decision Classes:**
- {', '.join(TARGET_LE.classes_)}

**Model Type:** Random Forest
""")

# Display top 5 features
with st.sidebar.expander("🎯 Top 5 Features"):
    for i, feat in enumerate(TOP_FEATURES[:5], 1):
        st.text(f"{i}. {feat}")

# =============================================================================
# HOME PAGE
# =============================================================================

if page == "🏠 Home":
    st.markdown('<p class="main-header">💳 Hybrid Credit Risk System</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### ML-Powered + Rule-Based Lending Decisions
    
    **Decision Flow:**
    1. 🚨 **Hard Policy Gates** - Age, KYC, Employment, Credit Bureau
    2. 🤖 **ML Risk Assessment** - Random Forest classification  
    3. 💰 **Affordability Check** - DTI ratio validation
    
    **Key Strengths:**
    - ✅ Trained on **60,000 real loan applications**
    - ✅ Comprehensive policy compliance checks
    - ⚡ Real-time hybrid decisions (<1 second)
    - 📊 Explainable AI with confidence scores
    - 🛡️ Regulatory compliance through hard rules
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Features Used", len(TOP_FEATURES))
    col2.metric("Decision Types", len(TARGET_LE.classes_))
    col3.metric("Model", "Random Forest")
    
    st.markdown("---")
    st.markdown("### 📋 Policy Checks Implemented")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Identity & Eligibility:**
        - Age limits (18-65 salaried, 18-70 self-employed)
        - KYC verification requirement
        - Bankruptcy & fraud checks
        
        **Income & Employment:**
        - Minimum income: ₹15,000/month
        - Salaried: 6+ months tenure
        - Self-employed: 2+ years vintage
        """)
    
    with col2:
        st.info("""
        **Credit Bureau:**
        - Minimum bureau score: 550
        - No 90+ DPD in last 6 months
        - Credit utilization monitoring
        - Recent inquiry tracking
        
        **Affordability:**
        - Maximum DTI ratio: 45%
        - EMI calculation with loan proposal
        """)

# =============================================================================
# SINGLE ASSESSMENT PAGE
# =============================================================================

elif page == "👤 Single Assessment":
    st.markdown('<p class="main-header">👤 Individual Credit Assessment</p>', unsafe_allow_html=True)
    
    st.info("💡 Hybrid Decision: Hard Policy Gates → ML Model → Affordability Check")
    
    with st.form("assessment_form"):
        st.markdown("### 📋 Customer Information")
        
        # Section 1: Identity & Eligibility
        st.markdown("#### 👤 Identity & Eligibility")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 18, 80, 35)
            employment_type = st.selectbox("Employment Type", 
                                          ['Salaried', 'Self-Employed', 'Business'])
        
        with col2:
            kyc_verified = st.selectbox("KYC Verified", ['Yes', 'No']) == 'Yes'
            bankruptcy_flag = st.selectbox("Bankruptcy Flag", ['No', 'Yes']) == 'Yes'
        
        with col3:
            fraud_flag = st.selectbox("Fraud Flag", ['No', 'Yes']) == 'Yes'
            if employment_type == 'Salaried':
                employment_tenure = st.number_input("Employment Tenure (months)", 0, 600, 24)
                business_vintage = 0
            else:
                business_vintage = st.number_input("Business Vintage (years)", 0, 50, 3)
                employment_tenure = 0
        
        st.markdown("---")
        
        # Section 2: Credit Bureau
        st.markdown("#### 🏦 Credit Bureau")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bureau_score = st.number_input("Bureau Score", 300, 900, 720, 10)
            dpd_90_6m = st.number_input("DPD 90+ (6M)", 0, 20, 0)
        
        with col2:
            credit_utilization = st.number_input("Credit Utilization (%)", 0, 100, 30)
            recent_inquiries = st.number_input("Recent Inquiries (3M)", 0, 20, 2)
        
        with col3:
            active_loans = st.number_input("Active Loans", 0, 10, 1)
            total_emi = st.number_input("Current Total EMI (₹)", 0, 200000, 15000, 1000)
        
        st.markdown("---")
        
        # Section 3: Income & Salary
        st.markdown("#### 💰 Income & Financial")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_salary = st.number_input("Monthly Income (₹)", 0, 1000000, 50000, 5000)
            amt_income = st.number_input("Total Annual Income (₹)", 0, 10000000, 600000, 10000)
        
        with col2:
            net_surplus = st.number_input("Net Cash Surplus 6M (₹)", -100000, 500000, 20000, 5000)
            salary_stability = st.selectbox("Salary Stability", ['STABLE', 'MODERATE', 'UNSTABLE'])
        
        with col3:
            st.markdown("**Loan Request**")
            loan_amount = st.number_input("Requested Loan Amount (₹)", 0, 5000000, 200000, 10000)
            loan_tenure = st.number_input("Loan Tenure (months)", 3, 360, 24)
        
        st.markdown("---")
        
        # Section 4: Other Metrics
        st.markdown("#### 📊 Other Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            amt_annuity = st.number_input("Requested EMI (₹)", 0, 200000, 12000, 1000)
        
        submitted = st.form_submit_button("🔍 Assess Credit Risk", 
                                          use_container_width=True, type="primary")
    
    if submitted:
        # Prepare customer data
        customer_data = {
            # Identity & Eligibility
            'age': age,
            'employment_type': employment_type,
            'kyc_verified': kyc_verified,
            'bankruptcy_flag': bankruptcy_flag,
            'fraud_flag': fraud_flag,
            'employment_tenure_months': employment_tenure,
            'business_vintage_years': business_vintage,
            
            # Credit Bureau
            'bureau_score': bureau_score,
            'dpd_90_count_6m': dpd_90_6m,
            'credit_utilization_pct': credit_utilization,
            'recent_inquiries_3m': recent_inquiries,
            'active_loans_count': active_loans,
            'total_emi_monthly': total_emi,
            
            # Income & Financial
            'avg_salary_6m': avg_salary,
            'AMT_INCOME_TOTAL': amt_income,
            'net_cash_surplus_6m': net_surplus,
            'salary_stability_flag': salary_stability,
            'loan_amount': loan_amount,
            'loan_tenure_months': loan_tenure,
            
            # Other
            'AMT_ANNUITY': amt_annuity
        }
        
        # Get decision
        decision, reason, confidence, class_probs, policy_checks = make_hybrid_decision(customer_data)
        
        st.markdown("---")
        st.markdown("## 📊 Assessment Results")
        
        # Display policy gate checks
        st.markdown("### 🛡️ Policy Gate Checks")
        
        cols = st.columns(3)
        check_items = list(policy_checks.items())
        
        for idx, (check_name, check_result) in enumerate(check_items):
            col_idx = idx % 3
            with cols[col_idx]:
                if '✅' in check_result:
                    st.markdown(f'<div class="policy-check check-pass">{check_result}</div>', 
                              unsafe_allow_html=True)
                elif '❌' in check_result:
                    st.markdown(f'<div class="policy-check check-fail">{check_result}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.warning(check_result)
        
        st.markdown("---")
        
        # Display final decision
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if decision == "APPROVE":
                st.markdown('<p class="approved">✅ APPROVED</p>', unsafe_allow_html=True)
                st.success(f"**Reason:** {reason}")
            elif decision == "REJECT":
                st.markdown('<p class="rejected">❌ REJECTED</p>', unsafe_allow_html=True)
                st.error(f"**Reason:** {reason}")
            else:
                st.markdown('<p class="review">⚠️ REVIEW REQUIRED</p>', unsafe_allow_html=True)
                st.warning(f"**Reason:** {reason}")
            
            st.metric("Model Confidence", f"{confidence:.1f}%")
            
            # Calculate and display DTI
            if avg_salary > 0 and loan_amount > 0:
                monthly_rate = 0.12 / 12
                proposed_emi = (loan_amount * monthly_rate * (1 + monthly_rate)**loan_tenure) / ((1 + monthly_rate)**loan_tenure - 1)
                total_emi_with_loan = total_emi + proposed_emi
                dti = (total_emi_with_loan / avg_salary) * 100
                
                st.metric("Current EMI", f"₹{total_emi:,.0f}")
                st.metric("Proposed EMI", f"₹{proposed_emi:,.0f}")
                st.metric("Total EMI", f"₹{total_emi_with_loan:,.0f}")
                st.metric("Debt-to-Income Ratio", f"{dti:.1f}%")
        
        with col2:
            st.plotly_chart(create_confidence_gauge(confidence, decision), 
                          use_container_width=True)
        
        # Show probability breakdown
        st.plotly_chart(create_probability_chart(class_probs), 
                       use_container_width=True)

# =============================================================================
# BATCH PROCESSING PAGE (same as before)
# =============================================================================

elif page == "📊 Batch Processing":
    st.markdown('<p class="main-header">📊 Batch Credit Assessment</p>', unsafe_allow_html=True)
    
    st.info("💡 Upload CSV with customer data for bulk processing")
    
    st.warning("**Note:** Batch processing requires all policy fields (age, employment_type, kyc_verified, etc.)")
    
    uploaded_file = st.file_uploader("📤 Upload Customer Data CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} applications")
            st.dataframe(df.head(10))
            
            if st.button("🚀 Process All Applications", use_container_width=True, type="primary"):
                with st.spinner("Processing with hybrid engine..."):
                    results = []
                    
                    for idx, row in df.iterrows():
                        customer_dict = row.to_dict()
                        decision, reason, confidence, class_probs, policy_checks = make_hybrid_decision(customer_dict)
                        
                        results.append({
                            'decision': decision,
                            'reason': reason,
                            'confidence': round(confidence, 2),
                            'approve_prob': round(class_probs.get('APPROVE', 0), 2),
                            'review_prob': round(class_probs.get('REVIEW', 0), 2),
                            'reject_prob': round(class_probs.get('REJECT', 0), 2)
                        })
                    
                    results_df = pd.DataFrame(results)
                    output_df = pd.concat([df, results_df], axis=1)
                    
                    st.success("✅ Processing Complete!")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    approved = (output_df['decision'] == 'APPROVE').sum()
                    rejected = (output_df['decision'] == 'REJECT').sum()
                    review = (output_df['decision'] == 'REVIEW').sum()
                    
                    col1.metric("Total", len(df))
                    col2.metric("Approved", approved, f"{approved/len(df)*100:.1f}%")
                    col3.metric("Rejected", rejected, f"{rejected/len(df)*100:.1f}%")
                    col4.metric("Review", review, f"{review/len(df)*100:.1f}%")
                    
                    st.dataframe(output_df, use_container_width=True)
                    
                    # Download button
                    csv_out = output_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv_out,
                        f"credit_decisions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True,
                        type="primary"
                    )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# =============================================================================
# MODEL INFO PAGE
# =============================================================================

elif page == "📈 Model Info":
    st.markdown('<p class="main-header">📈 Model Information</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "Random Forest")
    col2.metric("Features", len(TOP_FEATURES))
    col3.metric("Classes", len(TARGET_LE.classes_))
    
    st.markdown("---")
    
    st.markdown("### 🛡️ Decision Logic")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Layer 1: Policy Gates**
        - Age validation
        - KYC verification
        - Employment stability
        - Minimum income
        - Bureau score
        - Delinquency history
        """)
    
    with col2:
        st.info("""
        **Layer 2: ML Model**
        - Random Forest prediction
        - Confidence scoring
        - Class probabilities
        - Feature importance
        """)
    
    with col3:
        st.info("""
        **Layer 3: Affordability**
        - DTI ratio check
        - EMI calculation
        - APPROVE → REVIEW if DTI > 45%
        """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Feature Ranking")
    
    feature_df = pd.DataFrame({
        'Rank': range(1, len(TOP_FEATURES) + 1),
        'Feature': TOP_FEATURES
    })
    
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

# =============================================================================
# ABOUT PAGE
# =============================================================================

elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    ## Hybrid Credit Risk Assessment System
    
    **Version:** 6.0 - Comprehensive Policy Rules  
    **Developer:** Zen Meraki  
    **Date:** January 2025
    
    ### System Architecture
    
    **Three-Layer Decision Engine:**
    1. **Hard Policy Gates** - Comprehensive eligibility and compliance checks
    2. **ML Risk Model** - Random Forest classifier with {len(TOP_FEATURES)} features
    3. **Affordability Overlay** - DTI ratio and EMI validation
    
    ### Policy Rules Implemented
    
    **Identity & Eligibility:**
    - Age: 18-65 (salaried), 18-70 (self-employed)
    - KYC verification requirement
    - Bankruptcy and fraud checks
    
    **Income & Employment:**
    - Minimum monthly income: ₹15,000
    - Salaried: Minimum 6 months tenure
    - Self-employed: Minimum 2 years business vintage
    
    **Credit Bureau:**
    - Minimum bureau score: 550
    - Zero tolerance for 90+ DPD
    - Credit utilization monitoring
    - Recent inquiry tracking
    
    **Affordability:**
    - Maximum DTI ratio: 45%
    - Proposed EMI calculation
    - Total debt service analysis
    
    ### Technology Stack
    - **Framework:** Streamlit
    - **ML Library:** Scikit-learn
    - **Visualization:** Plotly
    - **Data Processing:** Pandas, NumPy
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><p>© 2025 Hybrid Credit Risk System v6.0 | Zen Meraki</p></div>", 
    unsafe_allow_html=True)
