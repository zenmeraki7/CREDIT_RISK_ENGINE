

# # # """
# # # Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
# # # Enhanced with Modern UI/UX Design
# # # Run with: streamlit run test.py

# # # Author: Zen Meraki  
# # # Date: January 2025
# # # VERSION: 8.0 - Sage Green & Yellow Professional Interface
# # # """

# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # import plotly.graph_objects as go
# # # import plotly.express as px
# # # import joblib
# # # import warnings
# # # from datetime import datetime
# # # import hashlib
# # # warnings.filterwarnings('ignore')

# # # # =============================================================================
# # # # PAGE CONFIGURATION
# # # # =============================================================================

# # # st.set_page_config(
# # #     page_title="Credit Risk Assessment",
# # #     page_icon="💳",
# # #     layout="wide",
# # #     initial_sidebar_state="expanded"
# # # )

# # # # =============================================================================
# # # # SAGE GREEN AND YELLOW THEME CSS
# # # # =============================================================================

# # # st.markdown("""
# # #     <style>
# # #     /* Import Google Fonts */
# # #     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
# # #     /* Global Styles */
# # #     * {
# # #         font-family: 'Inter', sans-serif;
# # #     }
    
# # #     /* Color Variables */
# # #     :root {
# # #         --fern-green: #587042;
# # #         --sage: #A9B494;
# # #         --cosmic-latte: #FAF7E6;
# # #         --jasmine: #F8DE8C;
# # #         --saffron: #F6C531;
# # #         --dark-fern: #486032;
# # #         --light-sage: #D4DBC4;
# # #     }
    
# # #     /* Main Background */
# # #     .main {
# # #         background-color: #FFFFFF;
# # #     }
    
# # #     .block-container {
# # #         padding-top: 2rem;
# # #         padding-bottom: 2rem;
# # #         max-width: 1400px;
# # #         background-color: #FFFFFF;
# # #     }
    
# # #     /* Headers */
# # #     .main-header {
# # #         font-size: 2.5rem;
# # #         font-weight: 700;
# # #         color: var(--fern-green);
# # #         text-align: center;
# # #         padding: 1.5rem 0;
# # #         margin-bottom: 1rem;
# # #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
# # #         -webkit-background-clip: text;
# # #         -webkit-text-fill-color: transparent;
# # #         background-clip: text;
# # #     }
    
# # #     .section-header {
# # #         font-size: 1.5rem;
# # #         font-weight: 600;
# # #         color: var(--fern-green);
# # #         margin-top: 2rem;
# # #         margin-bottom: 1rem;
# # #         padding-bottom: 0.5rem;
# # #         border-bottom: 2px solid var(--sage);
# # #     }
    
# # #     /* Decision Cards */
# # #     .decision-card {
# # #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
# # #         padding: 2rem;
# # #         border-radius: 16px;
# # #         box-shadow: 0 10px 40px rgba(88, 112, 66, 0.2);
# # #         margin-bottom: 2rem;
# # #         color: white;
# # #     }
    
# # #     .decision-card-approved {
# # #         background: linear-gradient(135deg, var(--fern-green) 0%, #7A9E4D 100%);
# # #         box-shadow: 0 10px 40px rgba(88, 112, 66, 0.3);
# # #     }
    
# # #     .decision-card-rejected {
# # #         background: linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%);
# # #         box-shadow: 0 10px 40px rgba(211, 47, 47, 0.2);
# # #     }
    
# # #     .decision-card-review {
# # #         background: linear-gradient(135deg, var(--saffron) 0%, var(--jasmine) 100%);
# # #         box-shadow: 0 10px 40px rgba(246, 197, 49, 0.3);
# # #     }
    
# # #     .decision-title {
# # #         font-size: 2.5rem;
# # #         font-weight: 700;
# # #         margin: 0;
# # #         color: white;
# # #         display: flex;
# # #         align-items: center;
# # #         gap: 1rem;
# # #     }
    
# # #     .decision-subtitle {
# # #         font-size: 1.1rem;
# # #         margin-top: 0.5rem;
# # #         opacity: 0.9;
# # #     }
    
# # #     /* Info Cards */
# # #     .info-card {
# # #         background: white;
# # #         border-radius: 12px;
# # #         padding: 1.5rem;
# # #         box-shadow: 0 4px 12px rgba(88, 112, 66, 0.08);
# # #         border: 1px solid var(--sage);
# # #         margin-bottom: 1rem;
# # #         transition: all 0.3s ease;
# # #     }
    
# # #     .info-card:hover {
# # #         box-shadow: 0 10px 25px rgba(88, 112, 66, 0.15);
# # #         transform: translateY(-2px);
# # #         border-color: var(--fern-green);
# # #     }
    
# # #     .info-card-title {
# # #         font-size: 1.1rem;
# # #         font-weight: 600;
# # #         color: var(--fern-green);
# # #         margin-bottom: 1rem;
# # #         display: flex;
# # #         align-items: center;
# # #         gap: 0.5rem;
# # #     }
    
# # #     .info-card-content {
# # #         color: #5A5A5A;
# # #         line-height: 1.6;
# # #     }
    
# # #     /* Metric Cards */
# # #     .metric-card {
# # #         background: linear-gradient(135deg, var(--cosmic-latte) 0%, #F5F2E0 100%);
# # #         border-radius: 12px;
# # #         padding: 1.5rem;
# # #         border-left: 4px solid var(--fern-green);
# # #         margin-bottom: 1rem;
# # #     }
    
# # #     .metric-label {
# # #         font-size: 0.875rem;
# # #         font-weight: 500;
# # #         color: var(--fern-green);
# # #         text-transform: uppercase;
# # #         letter-spacing: 0.05em;
# # #         margin-bottom: 0.5rem;
# # #     }
    
# # #     .metric-value {
# # #         font-size: 2rem;
# # #         font-weight: 700;
# # #         color: var(--fern-green);
# # #     }
    
# # #     /* Status Badges */
# # #     .status-badge {
# # #         display: inline-flex;
# # #         align-items: center;
# # #         padding: 0.5rem 1rem;
# # #         border-radius: 20px;
# # #         font-weight: 600;
# # #         font-size: 0.875rem;
# # #         gap: 0.5rem;
# # #     }
    
# # #     .badge-pass {
# # #         background: #E8F5E9;
# # #         color: var(--fern-green);
# # #         border: 1px solid var(--sage);
# # #     }
    
# # #     .badge-fail {
# # #         background: #FFEBEE;
# # #         color: #D32F2F;
# # #         border: 1px solid #FFCDD2;
# # #     }
    
# # #     .badge-warning {
# # #         background: linear-gradient(135deg, var(--cosmic-latte) 0%, var(--jasmine) 100%);
# # #         color: #7B5800;
# # #         border: 1px solid var(--jasmine);
# # #     }
    
# # #     .badge-info {
# # #         background: #E3F2FD;
# # #         color: #1565C0;
# # #         border: 1px solid #90CAF9;
# # #     }
    
# # #     /* Data Row */
# # #     .data-row {
# # #         display: flex;
# # #         justify-content: space-between;
# # #         align-items: center;
# # #         padding: 0.75rem 0;
# # #         border-bottom: 1px solid var(--sage);
# # #     }
    
# # #     .data-row:last-child {
# # #         border-bottom: none;
# # #     }
    
# # #     .data-label {
# # #         font-weight: 500;
# # #         color: #5A5A5A;
# # #     }
    
# # #     .data-value {
# # #         font-weight: 600;
# # #         color: var(--fern-green);
# # #     }
    
# # #     /* Reason Items */
# # #     .reason-item {
# # #         background: linear-gradient(135deg, #F9F7EB 0%, #F5F2E0 100%);
# # #         padding: 1rem 1.25rem;
# # #         border-radius: 8px;
# # #         border-left: 4px solid var(--saffron);
# # #         margin-bottom: 0.75rem;
# # #         color: #7B5800;
# # #         font-weight: 500;
# # #         display: flex;
# # #         align-items: center;
# # #         gap: 0.75rem;
# # #     }
    
# # #     .reason-icon {
# # #         font-size: 1.25rem;
# # #         color: var(--fern-green);
# # #     }
    
# # #     /* Buttons */
# # #     .stButton > button {
# # #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--dark-fern) 100%);
# # #         color: white;
# # #         border: none;
# # #         border-radius: 8px;
# # #         padding: 0.75rem 1.5rem;
# # #         font-weight: 600;
# # #         transition: all 0.3s ease;
# # #         box-shadow: 0 4px 6px rgba(88, 112, 66, 0.2);
# # #     }
    
# # #     .stButton > button:hover {
# # #         box-shadow: 0 10px 20px rgba(88, 112, 66, 0.3);
# # #         transform: translateY(-2px);
# # #         background: linear-gradient(135deg, var(--dark-fern) 0%, var(--fern-green) 100%);
# # #     }
    
# # #     /* Form Inputs */
# # #     .stNumberInput > div > div > input,
# # #     .stSelectbox > div > div > select {
# # #         border-radius: 8px;
# # #         border: 1px solid var(--sage);
# # #         padding: 0.75rem;
# # #         font-size: 1rem;
# # #         background-color: white;
# # #     }
    
# # #     .stNumberInput > div > div > input:focus,
# # #     .stSelectbox > div > div > select:focus {
# # #         border-color: var(--fern-green);
# # #         box-shadow: 0 0 0 3px rgba(88, 112, 66, 0.1);
# # #     }
    
# # #     /* Tabs */
# # #     .stTabs [data-baseweb="tab-list"] {
# # #         gap: 2rem;
# # #         background-color: white;
# # #         padding: 1rem;
# # #         border-radius: 12px;
# # #         box-shadow: 0 2px 4px rgba(88, 112, 66, 0.05);
# # #     }
    
# # #     .stTabs [data-baseweb="tab"] {
# # #         height: 3rem;
# # #         padding: 0 1.5rem;
# # #         background-color: transparent;
# # #         border-radius: 8px;
# # #         color: #718096;
# # #         font-weight: 600;
# # #         transition: all 0.3s ease;
# # #     }
    
# # #     .stTabs [aria-selected="true"] {
# # #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
# # #         color: white;
# # #     }
    
# # #     .stTabs [data-baseweb="tab"]:hover {
# # #         background-color: var(--cosmic-latte);
# # #     }
    
# # #     /* Sidebar */
# # #     [data-testid="stSidebar"] {
# # #         background: linear-gradient(180deg, var(--cosmic-latte) 0%, #F5F2E0 100%);
# # #         border-right: 1px solid var(--sage);
# # #     }
    
# # #     [data-testid="stSidebar"] .element-container {
# # #         color: var(--fern-green);
# # #     }
    
# # #     /* Expander */
# # #     .streamlit-expanderHeader {
# # #         background-color: var(--cosmic-latte);
# # #         border-radius: 8px;
# # #         padding: 0.75rem;
# # #         font-weight: 600;
# # #         color: var(--fern-green);
# # #         border: 1px solid var(--sage);
# # #     }
    
# # #     /* Alerts */
# # #     .stAlert {
# # #         border-radius: 12px;
# # #         border: none;
# # #         padding: 1rem 1.5rem;
# # #     }
    
# # #     /* Success Alert */
# # #     [data-baseweb="notification"] {
# # #         background-color: #E8F5E9;
# # #         border-left: 4px solid var(--fern-green);
# # #         border-radius: 8px;
# # #     }
    
# # #     /* Info Alert */
# # #     .info-box {
# # #         background: linear-gradient(135deg, #F9F7EB 0%, var(--cosmic-latte) 100%);
# # #         border-left: 4px solid var(--sage);
# # #         border-radius: 8px;
# # #         padding: 1.25rem;
# # #         margin: 1rem 0;
# # #         color: var(--fern-green);
# # #         border: 1px solid var(--sage);
# # #     }
    
# # #     /* Warning Alert */
# # #     .warning-box {
# # #         background: linear-gradient(135deg, var(--cosmic-latte) 0%, var(--jasmine) 100%);
# # #         border-left: 4px solid var(--saffron);
# # #         border-radius: 8px;
# # #         padding: 1.25rem;
# # #         margin: 1rem 0;
# # #         color: #7B5800;
# # #         border: 1px solid var(--jasmine);
# # #     }
    
# # #     /* Dataframe */
# # #     .dataframe {
# # #         border-radius: 12px;
# # #         overflow: hidden;
# # #         box-shadow: 0 4px 6px rgba(88, 112, 66, 0.05);
# # #         border: 1px solid var(--sage);
# # #     }
    
# # #     /* Metric Container */
# # #     [data-testid="stMetricValue"] {
# # #         font-size: 2rem;
# # #         font-weight: 700;
# # #         color: var(--fern-green);
# # #     }
    
# # #     [data-testid="stMetricLabel"] {
# # #         font-size: 0.875rem;
# # #         font-weight: 600;
# # #         color: var(--fern-green);
# # #         text-transform: uppercase;
# # #         letter-spacing: 0.05em;
# # #         opacity: 0.8;
# # #     }
    
# # #     /* Progress Bar */
# # #     .stProgress > div > div > div {
# # #         background: linear-gradient(90deg, var(--fern-green) 0%, var(--sage) 100%);
# # #     }
    
# # #     /* Divider */
# # #     hr {
# # #         margin: 2rem 0;
# # #         border: none;
# # #         border-top: 2px solid var(--sage);
# # #     }
    
# # #     /* Custom Scrollbar */
# # #     ::-webkit-scrollbar {
# # #         width: 10px;
# # #         height: 10px;
# # #     }
    
# # #     ::-webkit-scrollbar-track {
# # #         background: var(--cosmic-latte);
# # #     }
    
# # #     ::-webkit-scrollbar-thumb {
# # #         background: var(--sage);
# # #         border-radius: 5px;
# # #     }
    
# # #     ::-webkit-scrollbar-thumb:hover {
# # #         background: var(--fern-green);
# # #     }
    
# # #     /* Icon Styles */
# # #     .icon {
# # #         font-size: 1.5rem;
# # #         margin-right: 0.5rem;
# # #         color: var(--fern-green);
# # #     }
    
# # #     /* Card Grid */
# # #     .card-grid {
# # #         display: grid;
# # #         grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
# # #         gap: 1.5rem;
# # #         margin: 1.5rem 0;
# # #     }
    
# # #     /* Feature Badge */
# # #     .feature-badge {
# # #         display: inline-block;
# # #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
# # #         color: white;
# # #         padding: 0.25rem 0.75rem;
# # #         border-radius: 12px;
# # #         font-size: 0.75rem;
# # #         font-weight: 600;
# # #         text-transform: uppercase;
# # #         letter-spacing: 0.05em;
# # #     }
    
# # #     /* Timeline */
# # #     .timeline-item {
# # #         position: relative;
# # #         padding-left: 2rem;
# # #         padding-bottom: 1.5rem;
# # #         border-left: 2px solid var(--sage);
# # #     }
    
# # #     .timeline-item:last-child {
# # #         border-left: none;
# # #     }
    
# # #     .timeline-dot {
# # #         position: absolute;
# # #         left: -6px;
# # #         top: 0;
# # #         width: 12px;
# # #         height: 12px;
# # #         border-radius: 50%;
# # #         background: var(--fern-green);
# # #     }
    
# # #     /* Stat Card */
# # #     .stat-card {
# # #         background: white;
# # #         border-radius: 12px;
# # #         padding: 1.5rem;
# # #         box-shadow: 0 4px 12px rgba(88, 112, 66, 0.08);
# # #         border-top: 4px solid var(--fern-green);
# # #         text-align: center;
# # #         border: 1px solid var(--sage);
# # #         transition: all 0.3s ease;
# # #     }
    
# # #     .stat-card:hover {
# # #         box-shadow: 0 10px 25px rgba(88, 112, 66, 0.15);
# # #         transform: translateY(-2px);
# # #     }
    
# # #     .stat-number {
# # #         font-size: 2.5rem;
# # #         font-weight: 700;
# # #         color: var(--fern-green);
# # #         margin-bottom: 0.5rem;
# # #     }
    
# # #     .stat-label {
# # #         font-size: 0.875rem;
# # #         font-weight: 600;
# # #         color: var(--fern-green);
# # #         text-transform: uppercase;
# # #         letter-spacing: 0.05em;
# # #         opacity: 0.8;
# # #     }
    
# # #     /* Chart styling */
# # #     .js-plotly-plot .plotly {
# # #         background-color: white !important;
# # #     }
    
# # #     /* Radio buttons */
# # #     .stRadio > div {
# # #         background-color: white;
# # #         padding: 0.5rem;
# # #         border-radius: 8px;
# # #         border: 1px solid var(--sage);
# # #     }
    
# # #     .stRadio > div[data-baseweb="radio"] label {
# # #         color: var(--fern-green);
# # #     }
    
# # #     /* Checkbox */
# # #     .stCheckbox > label {
# # #         color: var(--fern-green);
# # #     }
    
# # #     /* Form submit button */
# # #     div[data-testid="stForm"] button {
# # #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--dark-fern) 100%);
# # #         color: white;
# # #         border: none;
# # #         border-radius: 8px;
# # #         padding: 0.75rem 1.5rem;
# # #         font-weight: 600;
# # #         transition: all 0.3s ease;
# # #         box-shadow: 0 4px 6px rgba(88, 112, 66, 0.2);
# # #     }
    
# # #     div[data-testid="stForm"] button:hover {
# # #         box-shadow: 0 10px 20px rgba(88, 112, 66, 0.3);
# # #         transform: translateY(-2px);
# # #         background: linear-gradient(135deg, var(--dark-fern) 0%, var(--fern-green) 100%);
# # #     }
    
# # #     /* Table styling */
# # #     .stTable {
# # #         border: 1px solid var(--sage);
# # #         border-radius: 8px;
# # #     }
    
# # #     /* Container borders */
# # #     .stApp {
# # #         background-color: white;
# # #     }
# # #     </style>
# # # """, unsafe_allow_html=True)

# # # # =============================================================================
# # # # LOAD TRAINED MODEL ASSETS
# # # # =============================================================================

# # # @st.cache_resource
# # # def load_model_assets():
# # #     """Load the trained model and preprocessing assets"""
# # #     try:
# # #         possible_paths = [
# # #             'credit_risk_assets.pkl',
# # #             'notebooks/credit_risk_assets.pkl',
# # #             '../notebooks/credit_risk_assets.pkl'
# # #         ]
        
# # #         assets = None
# # #         for path in possible_paths:
# # #             try:
# # #                 assets = joblib.load(path)
# # #                 break
# # #             except FileNotFoundError:
# # #                 continue
        
# # #         if assets is None:
# # #             raise FileNotFoundError("Could not find credit_risk_assets.pkl")
# # #         return {
# # #             'model': assets['model'],
# # #             'features': assets['features'],
# # #             'le_map': assets['le_map'],
# # #             'target_le': assets['target_le'],
# # #             'loaded': True,
# # #             'error': None
# # #         }
# # #     except FileNotFoundError:
# # #         return {
# # #             'loaded': False,
# # #             'error': 'credit_risk_assets.pkl not found. Please run the training script first.'
# # #         }
# # #     except Exception as e:
# # #         return {
# # #             'loaded': False,
# # #             'error': f'Error loading model: {str(e)}'
# # #         }

# # # # Load assets
# # # ASSETS = load_model_assets()

# # # if not ASSETS['loaded']:
# # #     st.error(f"❌ {ASSETS['error']}")
# # #     st.info("Please ensure 'credit_risk_assets.pkl' is in the same directory as this app.")
# # #     st.stop()

# # # MODEL = ASSETS['model']
# # # TOP_FEATURES = ASSETS['features']
# # # LE_MAP = ASSETS['le_map']
# # # TARGET_LE = ASSETS['target_le']

# # # # =============================================================================
# # # # AFFORDABILITY CALCULATION ENGINE
# # # # =============================================================================

# # # def calculate_emi(principal, annual_rate, tenure_months):
# # #     """Calculate EMI using reducing balance method"""
# # #     if principal <= 0 or tenure_months <= 0:
# # #         return 0
    
# # #     monthly_rate = annual_rate / (12 * 100)
    
# # #     if monthly_rate == 0:
# # #         return principal / tenure_months
    
# # #     emi = (principal * monthly_rate * (1 + monthly_rate)**tenure_months) / \
# # #           ((1 + monthly_rate)**tenure_months - 1)
    
# # #     return round(emi, 2)


# # # def calculate_affordability(monthly_income, loan_amount, interest_rate, 
# # #                            tenure_months, existing_emi):
# # #     """Calculate comprehensive affordability metrics"""
    
# # #     new_emi = calculate_emi(loan_amount, interest_rate, tenure_months)
# # #     total_emi = new_emi + existing_emi
# # #     foir_percentage = (total_emi / monthly_income) * 100 if monthly_income > 0 else 0
# # #     net_disposable = monthly_income - total_emi
# # #     max_allowed_emi = monthly_income * 0.50
# # #     recommended_emi = monthly_income * 0.40
# # #     affordable = foir_percentage <= 50
# # #     within_recommended = foir_percentage <= 40
    
# # #     if foir_percentage <= 40:
# # #         status = "Excellent"
# # #         status_color = "green"
# # #     elif foir_percentage <= 50:
# # #         status = "Acceptable"
# # #         status_color = "yellow"
# # #     else:
# # #         status = "Over-leveraged"
# # #         status_color = "red"
    
# # #     return {
# # #         'monthly_income': monthly_income,
# # #         'new_emi': new_emi,
# # #         'existing_emi': existing_emi,
# # #         'total_emi': total_emi,
# # #         'foir_percentage': round(foir_percentage, 2),
# # #         'net_disposable': net_disposable,
# # #         'max_allowed_emi': max_allowed_emi,
# # #         'recommended_emi': recommended_emi,
# # #         'affordable': affordable,
# # #         'within_recommended': within_recommended,
# # #         'status': status,
# # #         'status_color': status_color,
# # #         'emi_headroom': max_allowed_emi - total_emi
# # #     }

# # # # =============================================================================
# # # # REASON CODE GENERATION SYSTEM
# # # # =============================================================================

# # # APPROVAL_REASONS = {
# # #     'high_bureau': 'Excellent credit score ({score})',
# # #     'stable_employment': 'Stable employment history ({tenure} months)',
# # #     'low_foir': 'Affordable EMI burden (FOIR: {foir}%)',
# # #     'clean_payment': 'Clean payment history (No DPD)',
# # #     'strong_income': 'Strong monthly income (₹{income:,})',
# # #     'low_utilization': 'Low credit utilization ({util}%)',
# # # }

# # # REJECTION_REASONS = {
# # #     'low_bureau': 'Credit score below minimum ({score} < 550)',
# # #     'high_foir': 'EMI burden too high (FOIR: {foir}% > 50%)',
# # #     'severe_dpd': 'Severe payment delays ({dpd} instances of 90+ DPD)',
# # #     'low_income': 'Income below minimum threshold (₹{income:,} < ₹15,000)',
# # #     'short_employment': 'Insufficient employment tenure ({tenure} months < 6)',
# # #     'bankruptcy': 'Active bankruptcy detected',
# # #     'kyc_failed': 'KYC verification not completed',
# # #     'high_utilization': 'High credit utilization ({util}% > 80%)',
# # #     'age_invalid': 'Age outside acceptable range ({age} years)'
# # # }

# # # REVIEW_REASONS = {
# # #     'borderline_bureau': 'Credit score in borderline range ({score})',
# # #     'moderate_foir': 'EMI burden moderate (FOIR: {foir}%)',
# # #     'mixed_signals': 'Mixed credit indicators requiring human review',
# # #     'recent_employment': 'Recent employment change requiring verification',
# # # }


# # # def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
# # #     """Generate top 3 reason codes for the decision"""
# # #     reasons = []
    
# # #     bureau_score = customer_data.get('bureau_score', 0)
# # #     foir = affordability_data.get('foir_percentage', 0)
# # #     dpd_90 = customer_data.get('dpd_90_count_6m', 0)
# # #     income = customer_data.get('avg_salary_6m', 0)
# # #     employment_tenure = customer_data.get('employment_tenure_months', 0)
# # #     credit_util = customer_data.get('credit_utilization_pct', 0)
# # #     age = customer_data.get('age', 0)
    
# # #     if decision == "APPROVE":
# # #         if bureau_score >= 750:
# # #             reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
# # #         if employment_tenure >= 24:
# # #             reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
# # #         if foir <= 40:
# # #             reasons.append(APPROVAL_REASONS['low_foir'].format(foir=round(foir, 1)))
# # #         if dpd_90 == 0:
# # #             reasons.append(APPROVAL_REASONS['clean_payment'])
# # #         if income >= 75000:
# # #             reasons.append(APPROVAL_REASONS['strong_income'].format(income=income))
# # #         if credit_util <= 30:
# # #             reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))
    
# # #     elif decision == "REJECT":
# # #         for check_name, check_result in policy_checks.items():
# # #             if '❌' in str(check_result):
# # #                 if 'bureau' in check_name.lower():
# # #                     reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
# # #                 elif 'dpd' in check_name.lower():
# # #                     reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
# # #                 elif 'income' in check_name.lower():
# # #                     reasons.append(REJECTION_REASONS['low_income'].format(income=income))
# # #                 elif 'tenure' in check_name.lower():
# # #                     reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
# # #                 elif 'kyc' in check_name.lower():
# # #                     reasons.append(REJECTION_REASONS['kyc_failed'])
# # #                 elif 'bankruptcy' in check_name.lower():
# # #                     reasons.append(REJECTION_REASONS['bankruptcy'])
# # #                 elif 'age' in check_name.lower():
# # #                     reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))
        
# # #         if foir > 50:
# # #             reasons.append(REJECTION_REASONS['high_foir'].format(foir=round(foir, 1)))
# # #         if credit_util > 80:
# # #             reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
    
# # #     elif decision == "REVIEW":
# # #         if 650 <= bureau_score < 700:
# # #             reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
# # #         if 40 < foir <= 50:
# # #             reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=round(foir, 1)))
# # #         if employment_tenure < 12:
# # #             reasons.append(REVIEW_REASONS['recent_employment'])
# # #         if not reasons:
# # #             reasons.append(REVIEW_REASONS['mixed_signals'])
    
# # #     return reasons[:3] if reasons else ['Decision based on model assessment']

# # # # =============================================================================
# # # # RISK SCORE CALCULATION
# # # # =============================================================================

# # # def calculate_final_risk_score(bureau_score, ml_confidence, foir):
# # #     """Calculate final risk score (0-1000)"""
# # #     bureau_points = (bureau_score / 900) * 400
# # #     ml_points = (ml_confidence / 100) * 400
# # #     foir_points = max(0, (1 - foir/50) * 200)
# # #     total_score = int(bureau_points + ml_points + foir_points)
# # #     return min(max(total_score, 0), 1000)

# # # # =============================================================================
# # # # ENHANCED HYBRID DECISION ENGINE
# # # # =============================================================================

# # # def make_hybrid_decision_enhanced(customer_dict):
# # #     """Enhanced decision engine with complete data"""
    
# # #     policy_checks = {}
    
# # #     # Policy Gates
# # #     age = customer_dict.get('age', 0)
# # #     employment_type = customer_dict.get('employment_type', 'Salaried')
# # #     kyc_verified = customer_dict.get('kyc_verified', True)
# # #     bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
# # #     fraud_flag = customer_dict.get('fraud_flag', False)
    
# # #     if employment_type in ['Salaried']:
# # #         age_min, age_max = 18, 65
# # #     else:
# # #         age_min, age_max = 18, 70
    
# # #     if age < age_min or age > age_max:
# # #         policy_checks['age'] = f"❌ Age {age} (Required: {age_min}-{age_max})"
# # #         return {
# # #             'decision': "REJECT",
# # #             'reason': f"Policy Gate: Age outside allowed range",
# # #             'confidence': 0,
# # #             'class_probs': {'REJECT': 100},
# # #             'policy_checks': policy_checks,
# # #             'risk_score': 0,
# # #             'pd_percentage': 100.0
# # #         }
# # #     policy_checks['age'] = f"✅ Age {age} (Valid)"
    
# # #     if not kyc_verified:
# # #         policy_checks['kyc'] = "❌ KYC Not Verified"
# # #         return {
# # #             'decision': "REJECT",
# # #             'reason': "Policy Gate: KYC verification required",
# # #             'confidence': 0,
# # #             'class_probs': {'REJECT': 100},
# # #             'policy_checks': policy_checks,
# # #             'risk_score': 0,
# # #             'pd_percentage': 100.0
# # #         }
# # #     policy_checks['kyc'] = "✅ KYC Verified"
    
# # #     if bankruptcy_flag:
# # #         policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
# # #         return {
# # #             'decision': "REJECT",
# # #             'reason': "Policy Gate: Active bankruptcy",
# # #             'confidence': 0,
# # #             'class_probs': {'REJECT': 100},
# # #             'policy_checks': policy_checks,
# # #             'risk_score': 0,
# # #             'pd_percentage': 100.0
# # #         }
# # #     policy_checks['bankruptcy'] = "✅ No Bankruptcy"
    
# # #     if fraud_flag:
# # #         policy_checks['fraud'] = "❌ Fraud Flag"
# # #         return {
# # #             'decision': "REJECT",
# # #             'reason': "Policy Gate: Fraud detected",
# # #             'confidence': 0,
# # #             'class_probs': {'REJECT': 100},
# # #             'policy_checks': policy_checks,
# # #             'risk_score': 0,
# # #             'pd_percentage': 100.0
# # #         }
# # #     policy_checks['fraud'] = "✅ No Fraud History"
    
# # #     monthly_income = customer_dict.get('avg_salary_6m', 0)
# # #     employment_tenure = customer_dict.get('employment_tenure_months', 0)
# # #     business_vintage = customer_dict.get('business_vintage_years', 0)
    
# # #     if monthly_income < 15000:
# # #         policy_checks['income'] = f"❌ Income ₹{monthly_income:,.0f} (Min: ₹15,000)"
# # #         return {
# # #             'decision': "REJECT",
# # #             'reason': "Policy Gate: Income below minimum",
# # #             'confidence': 0,
# # #             'class_probs': {'REJECT': 100},
# # #             'policy_checks': policy_checks,
# # #             'risk_score': 0,
# # #             'pd_percentage': 100.0
# # #         }
# # #     policy_checks['income'] = f"✅ Income ₹{monthly_income:,.0f}"
    
# # #     if employment_type == 'Salaried' and employment_tenure < 6:
# # #         policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
# # #         return {
# # #             'decision': "REJECT",
# # #             'reason': "Policy Gate: Insufficient tenure",
# # #             'confidence': 0,
# # #             'class_probs': {'REJECT': 100},
# # #             'policy_checks': policy_checks,
# # #             'risk_score': 0,
# # #             'pd_percentage': 100.0
# # #         }
# # #     elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
# # #         policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
# # #         return {
# # #             'decision': "REJECT",
# # #             'reason': "Policy Gate: Insufficient business vintage",
# # #             'confidence': 0,
# # #             'class_probs': {'REJECT': 100},
# # #             'policy_checks': policy_checks,
# # #             'risk_score': 0,
# # #             'pd_percentage': 100.0
# # #         }
    
# # #     if employment_type == 'Salaried':
# # #         policy_checks['tenure'] = f"✅ Tenure {employment_tenure} months"
# # #     else:
# # #         policy_checks['tenure'] = f"✅ Business Vintage {business_vintage} years"
    
# # #     bureau_score = customer_dict.get('bureau_score', 0)
# # #     dpd_90 = customer_dict.get('dpd_90_count_6m', 0)
# # #     credit_utilization = customer_dict.get('credit_utilization_pct', 0)
# # #     recent_inquiries = customer_dict.get('recent_inquiries_3m', 0)
    
# # #     if bureau_score < 550:
# # #         policy_checks['bureau'] = f"❌ Bureau Score {bureau_score} (Min: 550)"
# # #         return {
# # #             'decision': "REJECT",
# # #             'reason': "Policy Gate: Bureau score too low",
# # #             'confidence': 0,
# # #             'class_probs': {'REJECT': 100},
# # #             'policy_checks': policy_checks,
# # #             'risk_score': 0,
# # #             'pd_percentage': 100.0
# # #         }
# # #     policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"
    
# # #     if dpd_90 > 0:
# # #         policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD"
# # #         return {
# # #             'decision': "REJECT",
# # #             'reason': "Policy Gate: Severe delinquency",
# # #             'confidence': 0,
# # #             'class_probs': {'REJECT': 100},
# # #             'policy_checks': policy_checks,
# # #             'risk_score': 0,
# # #             'pd_percentage': 100.0
# # #         }
# # #     policy_checks['dpd'] = "✅ No 90+ DPD"
    
# # #     if credit_utilization > 80:
# # #         policy_checks['utilization'] = f"⚠️ High utilization {credit_utilization}%"
# # #     else:
# # #         policy_checks['utilization'] = f"✅ Utilization {credit_utilization}%"
    
# # #     if recent_inquiries > 5:
# # #         policy_checks['inquiries'] = f"⚠️ {recent_inquiries} recent inquiries"
# # #     else:
# # #         policy_checks['inquiries'] = f"✅ {recent_inquiries} inquiries"
    
# # #     # ML Prediction
# # #     input_df = pd.DataFrame([customer_dict])
    
# # #     for col in TOP_FEATURES:
# # #         if col not in input_df.columns:
# # #             if col in LE_MAP:
# # #                 input_df[col] = "Unknown"
# # #             else:
# # #                 input_df[col] = 0
    
# # #     for col, le in LE_MAP.items():
# # #         if col in input_df.columns:
# # #             val = str(input_df[col].values[0])
# # #             try:
# # #                 input_df[col] = le.transform([val])[0]
# # #             except ValueError:
# # #                 input_df[col] = 0
    
# # #     final_input = input_df[TOP_FEATURES]
    
# # #     pred_idx = MODEL.predict(final_input)[0]
# # #     ml_decision = TARGET_LE.inverse_transform([pred_idx])[0]
    
# # #     try:
# # #         pred_proba = MODEL.predict_proba(final_input)[0]
# # #         confidence = max(pred_proba) * 100
# # #         class_probs = {
# # #             cls: prob * 100 
# # #             for cls, prob in zip(TARGET_LE.classes_, pred_proba)
# # #         }
# # #     except:
# # #         confidence = 75.0
# # #         class_probs = {ml_decision: 100.0}
    
# # #     # Affordability
# # #     loan_amount = customer_dict.get('loan_amount', 0)
# # #     loan_tenure = customer_dict.get('loan_tenure_months', 12)
# # #     interest_rate = customer_dict.get('interest_rate', 10.5)
# # #     existing_emi = customer_dict.get('existing_emi', 0)
    
# # #     affordability_data = calculate_affordability(
# # #         monthly_income=monthly_income,
# # #         loan_amount=loan_amount,
# # #         interest_rate=interest_rate,
# # #         tenure_months=loan_tenure,
# # #         existing_emi=existing_emi
# # #     )
    
# # #     foir = affordability_data['foir_percentage']
    
# # #     if ml_decision == "APPROVE" and foir > 45:
# # #         ml_decision = "REVIEW"
    
# # #     risk_score = calculate_final_risk_score(bureau_score, confidence, foir)
# # #     pd_percentage = max(0, min(100, (1 - confidence/100) * 10))
    
# # #     return {
# # #         'decision': ml_decision,
# # #         'reason': "Decision based on comprehensive assessment",
# # #         'confidence': confidence,
# # #         'class_probs': class_probs,
# # #         'policy_checks': policy_checks,
# # #         'risk_score': risk_score,
# # #         'pd_percentage': round(pd_percentage, 2),
# # #         'affordability_data': affordability_data
# # #     }

# # # # =============================================================================
# # # # MODERN UI COMPONENTS
# # # # =============================================================================

# # # def render_decision_header(decision_data, customer_data):
# # #     """Render modern decision header"""
    
# # #     decision = decision_data['decision']
# # #     risk_score = decision_data['risk_score']
# # #     pd_score = decision_data['pd_percentage']
# # #     approved_amount = customer_data.get('loan_amount', 0)
# # #     tenure = customer_data.get('loan_tenure_months', 24)
# # #     app_id = customer_data.get('application_id', 'N/A')
# # #     timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
# # #     # Decision card with appropriate styling
# # #     if decision == "APPROVE":
# # #         card_class = "decision-card decision-card-approved"
# # #         icon = "✓"
# # #         subtitle = "Application Approved Successfully"
# # #     elif decision == "REJECT":
# # #         card_class = "decision-card decision-card-rejected"
# # #         icon = "✗"
# # #         subtitle = "Application Not Approved"
# # #     else:
# # #         card_class = "decision-card decision-card-review"
# # #         icon = "⚠"
# # #         subtitle = "Requires Manual Review"
    
# # #     st.markdown(f"""
# # #         <div class="{card_class}">
# # #             <div class="decision-title">
# # #                 <span>{icon}</span>
# # #                 <span>{decision}</span>
# # #             </div>
# # #             <div class="decision-subtitle">{subtitle}</div>
# # #         </div>
# # #     """, unsafe_allow_html=True)
    
# # #     # Metrics grid
# # #     col1, col2, col3, col4, col5 = st.columns(5)
    
# # #     with col1:
# # #         st.markdown(f"""
# # #             <div class="stat-card">
# # #                 <div class="stat-number">{risk_score}</div>
# # #                 <div class="stat-label">Risk Score</div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     with col2:
# # #         st.markdown(f"""
# # #             <div class="stat-card">
# # #                 <div class="stat-number">{pd_score}%</div>
# # #                 <div class="stat-label">PD Score</div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     with col3:
# # #         st.markdown(f"""
# # #             <div class="stat-card">
# # #                 <div class="stat-number">₹{approved_amount:,.0f}</div>
# # #                 <div class="stat-label">Loan Amount</div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     with col4:
# # #         st.markdown(f"""
# # #             <div class="stat-card">
# # #                 <div class="stat-number">{tenure}</div>
# # #                 <div class="stat-label">Tenure (Months)</div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     with col5:
# # #         st.markdown(f"""
# # #             <div class="stat-card">
# # #                 <div class="stat-number">{decision_data['confidence']:.0f}%</div>
# # #                 <div class="stat-label">Confidence</div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     # Application info
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         st.markdown(f"""
# # #             <div class="info-box">
# # #                 <strong>📋 Application ID:</strong> {app_id}
# # #             </div>
# # #         """, unsafe_allow_html=True)
# # #     with col2:
# # #         st.markdown(f"""
# # #             <div class="info-box">
# # #                 <strong>🕐 Decision Timestamp:</strong> {timestamp}
# # #             </div>
# # #         """, unsafe_allow_html=True)


# # # def render_info_card(title, icon, data_dict, status_dict=None):
# # #     """Render modern info card with data"""
    
# # #     st.markdown(f"""
# # #         <div class="info-card">
# # #             <div class="info-card-title">
# # #                 <span class="icon">{icon}</span>
# # #                 <span>{title}</span>
# # #             </div>
# # #             <div class="info-card-content">
# # #     """, unsafe_allow_html=True)
    
# # #     for label, value in data_dict.items():
# # #         status = ""
# # #         if status_dict and label in status_dict:
# # #             if status_dict[label] == "pass":
# # #                 status = '<span class="status-badge badge-pass">✓ Passed</span>'
# # #             elif status_dict[label] == "fail":
# # #                 status = '<span class="status-badge badge-fail">✗ Failed</span>'
# # #             elif status_dict[label] == "warning":
# # #                 status = '<span class="status-badge badge-warning">⚠ Warning</span>'
        
# # #         st.markdown(f"""
# # #             <div class="data-row">
# # #                 <span class="data-label">{label}</span>
# # #                 <span class="data-value">{value} {status}</span>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     st.markdown("""
# # #             </div>
# # #         </div>
# # #     """, unsafe_allow_html=True)


# # # def render_reason_codes(reasons):
# # #     """Render reason codes in modern style"""
    
# # #     st.markdown("""
# # #         <div class="info-card">
# # #             <div class="info-card-title">
# # #                 <span class="icon">📝</span>
# # #                 <span>Decision Reasons</span>
# # #             </div>
# # #             <div class="info-card-content">
# # #     """, unsafe_allow_html=True)
    
# # #     for i, reason in enumerate(reasons, 1):
# # #         st.markdown(f"""
# # #             <div class="reason-item">
# # #                 <span class="reason-icon">{i}.</span>
# # #                 <span>{reason}</span>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     st.markdown("""
# # #             </div>
# # #         </div>
# # #     """, unsafe_allow_html=True)


# # # def create_modern_gauge(value, title, max_value=100):
# # #     """Create modern gauge chart"""
    
# # #     if value <= 50:
# # #         color = "#f56565"
# # #     elif value <= 75:
# # #         color = "#ed8936"
# # #     else:
# # #         color = "#48bb78"
    
# # #     fig = go.Figure(go.Indicator(
# # #         mode="gauge+number",
# # #         value=value,
# # #         title={'text': title, 'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}},
# # #         number={'suffix': "%", 'font': {'size': 36, 'color': '#2d3748', 'family': 'Inter'}},
# # #         gauge={
# # #             'axis': {'range': [0, max_value], 'tickfont': {'size': 12, 'color': '#718096'}},
# # #             'bar': {'color': color, 'thickness': 0.75},
# # #             'bgcolor': 'white',
# # #             'borderwidth': 0,
# # #             'steps': [
# # #                 {'range': [0, 50], 'color': '#fed7d7'},
# # #                 {'range': [50, 75], 'color': '#feebc8'},
# # #                 {'range': [75, 100], 'color': '#c6f6d5'}
# # #             ],
# # #         }
# # #     ))
    
# # #     fig.update_layout(
# # #         height=250,
# # #         margin=dict(l=20, r=20, t=50, b=20),
# # #         paper_bgcolor='white',
# # #         font={'family': 'Inter', 'color': '#2d3748'}
# # #     )
    
# # #     return fig


# # # def create_modern_bar_chart(class_probs):
# # #     """Create modern probability bar chart"""
    
# # #     df = pd.DataFrame({
# # #         'Decision': list(class_probs.keys()),
# # #         'Probability': list(class_probs.values())
# # #     })
    
# # #     colors = {'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}
    
# # #     fig = px.bar(
# # #         df, 
# # #         x='Decision', 
# # #         y='Probability',
# # #         title='Decision Probabilities',
# # #         color='Decision',
# # #         color_discrete_map=colors,
# # #         text='Probability'
# # #     )
    
# # #     fig.update_traces(
# # #         texttemplate='%{text:.1f}%',
# # #         textposition='outside',
# # #         marker_line_width=0
# # #     )
    
# # #     fig.update_layout(
# # #         showlegend=False,
# # #         yaxis_title='Probability (%)',
# # #         xaxis_title='',
# # #         height=300,
# # #         margin=dict(l=20, r=20, t=50, b=20),
# # #         paper_bgcolor='white',
# # #         plot_bgcolor='white',
# # #         font={'family': 'Inter', 'color': '#2d3748'},
# # #         yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]},
# # #         xaxis={'tickfont': {'size': 14, 'color': '#2d3748'}}
# # #     )
    
# # #     return fig

# # # # =============================================================================
# # # # SIDEBAR
# # # # =============================================================================

# # # with st.sidebar:
# # #     st.markdown("# 🏦 Credit Risk Engine")
# # #     st.markdown("---")
    
# # #     page = st.radio(
# # #         "**Navigation**",
# # #         ["🏠 Home", "👤 Assessment", "📊 Batch Process", "📈 Model Info", "ℹ️ About"],
# # #         label_visibility="collapsed"
# # #     )
    
# # #     st.markdown("---")
    
# # #     st.markdown(f"""
# # #         <div class="info-card">
# # #             <div class="info-card-title">System Status</div>
# # #             <div class="info-card-content">
# # #                 <div class="data-row">
# # #                     <span class="data-label">Model</span>
# # #                     <span class="data-value">✅ Loaded</span>
# # #                 </div>
# # #                 <div class="data-row">
# # #                     <span class="data-label">Version</span>
# # #                     <span class="data-value">8.0</span>
# # #                 </div>
# # #                 <div class="data-row">
# # #                     <span class="data-label">Features</span>
# # #                     <span class="data-value">{len(TOP_FEATURES)}</span>
# # #                 </div>
# # #                 <div class="data-row">
# # #                     <span class="data-label">Type</span>
# # #                     <span class="data-value">Random Forest</span>
# # #                 </div>
# # #             </div>
# # #         </div>
# # #     """, unsafe_allow_html=True)
    
# # #     with st.expander("🎯 **Top Features**"):
# # #         for i, feat in enumerate(TOP_FEATURES[:5], 1):
# # #             st.markdown(f"`{i}.` {feat}")

# # # # =============================================================================
# # # # HOME PAGE
# # # # =============================================================================

# # # if page == "🏠 Home":
# # #     st.markdown('<p class="main-header">Credit Risk Assessment Platform</p>', unsafe_allow_html=True)
    
# # #     st.markdown("""
# # #         <div class="info-box">
# # #             <h3 style="margin-top: 0;">🎯 AI-Powered Lending Decisions</h3>
# # #             <p style="margin-bottom: 0;">
# # #                 Comprehensive credit risk evaluation combining hard policy rules, 
# # #                 machine learning models, and affordability analysis for accurate lending decisions.
# # #             </p>
# # #         </div>
# # #     """, unsafe_allow_html=True)
    
# # #     st.markdown("<br>", unsafe_allow_html=True)
    
# # #     # Feature cards
# # #     col1, col2, col3 = st.columns(3)
    
# # #     with col1:
# # #         st.markdown("""
# # #             <div class="info-card">
# # #                 <div class="info-card-title">
# # #                     <span class="icon">🛡️</span>
# # #                     <span>Policy Gates</span>
# # #                 </div>
# # #                 <div class="info-card-content">
# # #                     <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
# # #                         <li>Age & KYC verification</li>
# # #                         <li>Employment stability</li>
# # #                         <li>Minimum income checks</li>
# # #                         <li>Credit bureau thresholds</li>
# # #                         <li>Bankruptcy & fraud detection</li>
# # #                     </ul>
# # #                 </div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     with col2:
# # #         st.markdown("""
# # #             <div class="info-card">
# # #                 <div class="info-card-title">
# # #                     <span class="icon">🤖</span>
# # #                     <span>ML Assessment</span>
# # #                 </div>
# # #                 <div class="info-card-content">
# # #                     <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
# # #                         <li>Random Forest classifier</li>
# # #                         <li>60K+ training samples</li>
# # #                         <li>Confidence scoring</li>
# # #                         <li>Multi-class prediction</li>
# # #                         <li>Feature importance</li>
# # #                     </ul>
# # #                 </div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     with col3:
# # #         st.markdown("""
# # #             <div class="info-card">
# # #                 <div class="info-card-title">
# # #                     <span class="icon">💰</span>
# # #                     <span>Affordability</span>
# # #                 </div>
# # #                 <div class="info-card-content">
# # #                     <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
# # #                         <li>EMI calculation</li>
# # #                         <li>FOIR analysis (max 50%)</li>
# # #                         <li>Net disposable income</li>
# # #                         <li>Debt burden assessment</li>
# # #                         <li>Affordability scoring</li>
# # #                     </ul>
# # #                 </div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     st.markdown("<br>", unsafe_allow_html=True)
    
# # #     # Stats
# # #     col1, col2, col3, col4 = st.columns(4)
    
# # #     with col1:
# # #         st.metric("🎯 Accuracy", "85%", "+2%")
    
# # #     with col2:
# # #         st.metric("⚡ Avg Response", "1.2s", "-0.3s")
    
# # #     with col3:
# # #         st.metric("📊 Features", len(TOP_FEATURES))
    
# # #     with col4:
# # #         st.metric("🔄 Version", "8.0", "Latest")
    
# # #     st.markdown("<br>", unsafe_allow_html=True)
    
# # #     st.markdown("""
# # #         <div class="warning-box">
# # #             <strong>🆕 New in Version 8.0:</strong><br>
# # #             • Sage Green & Yellow Professional Theme<br>
# # #             • Enhanced visual hierarchy and readability<br>
# # #             • Improved decision summary cards<br>
# # #             • Modern charts and gauges<br>
# # #             • Responsive layout optimization
# # #         </div>
# # #     """, unsafe_allow_html=True)

# # # # =============================================================================
# # # # ASSESSMENT PAGE
# # # # =============================================================================

# # # elif page == "👤 Assessment":
# # #     st.markdown('<p class="main-header">Credit Assessment</p>', unsafe_allow_html=True)
    
# # #     st.markdown("""
# # #         <div class="info-box">
# # #             💡 Complete the form below to assess credit risk. All fields are required for accurate evaluation.
# # #         </div>
# # #     """, unsafe_allow_html=True)
    
# # #     with st.form("assessment_form"):
# # #         # Identity & Eligibility
# # #         st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
# # #         col1, col2, col3 = st.columns(3)
        
# # #         with col1:
# # #             age = st.number_input("Age", 18, 80, 35, help="Customer's age in years")
# # #             employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business'])
        
# # #         with col2:
# # #             kyc_verified = st.selectbox("KYC Verified", ['Yes', 'No']) == 'Yes'
# # #             bankruptcy_flag = st.selectbox("Bankruptcy Flag", ['No', 'Yes']) == 'Yes'
        
# # #         with col3:
# # #             fraud_flag = st.selectbox("Fraud Flag", ['No', 'Yes']) == 'Yes'
# # #             if employment_type == 'Salaried':
# # #                 employment_tenure = st.number_input("Employment Tenure (months)", 0, 600, 24)
# # #                 business_vintage = 0
# # #             else:
# # #                 business_vintage = st.number_input("Business Vintage (years)", 0, 50, 3)
# # #                 employment_tenure = 0
        
# # #         # Credit Bureau
# # #         st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
# # #         col1, col2, col3 = st.columns(3)
        
# # #         with col1:
# # #             bureau_score = st.number_input("Bureau Score", 300, 900, 720, 10)
# # #             dpd_90_6m = st.number_input("DPD 90+ (Last 6M)", 0, 20, 0)
        
# # #         with col2:
# # #             credit_utilization = st.number_input("Credit Utilization (%)", 0, 100, 30)
# # #             recent_inquiries = st.number_input("Recent Inquiries (3M)", 0, 20, 2)
        
# # #         with col3:
# # #             active_loans = st.number_input("Active Loans", 0, 10, 1)
# # #             existing_emi = st.number_input("Existing Total EMI (₹)", 0, 200000, 15000, 1000)
        
# # #         # Income & Financial
# # #         st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
# # #         col1, col2, col3, col4 = st.columns(4)
        
# # #         with col1:
# # #             avg_salary = st.number_input("Monthly Income (₹)", 0, 1000000, 50000, 5000)
# # #             amt_income = st.number_input("Annual Income (₹)", 0, 10000000, 600000, 10000)
        
# # #         with col2:
# # #             net_surplus = st.number_input("Net Cash Surplus (₹)", -100000, 500000, 20000, 5000)
# # #             salary_stability = st.selectbox("Salary Stability", ['STABLE', 'MODERATE', 'UNSTABLE'])
        
# # #         with col3:
# # #             loan_amount = st.number_input("Loan Amount (₹)", 0, 5000000, 180000, 10000)
# # #             loan_tenure = st.number_input("Tenure (months)", 3, 360, 24)
        
# # #         with col4:
# # #             interest_rate = st.number_input("Interest Rate (%)", 8.0, 20.0, 10.5, 0.5)
# # #             amt_annuity = st.number_input("Requested EMI (₹)", 0, 200000, 8500, 500)
        
# # #         st.markdown("<br>", unsafe_allow_html=True)
# # #         submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)
    
# # #     if submitted:
# # #         # Generate application ID
# # #         timestamp = datetime.now()
# # #         app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
        
# # #         # Prepare data
# # #         customer_data = {
# # #             'age': age,
# # #             'employment_type': employment_type,
# # #             'kyc_verified': kyc_verified,
# # #             'bankruptcy_flag': bankruptcy_flag,
# # #             'fraud_flag': fraud_flag,
# # #             'employment_tenure_months': employment_tenure,
# # #             'business_vintage_years': business_vintage,
# # #             'bureau_score': bureau_score,
# # #             'dpd_90_count_6m': dpd_90_6m,
# # #             'credit_utilization_pct': credit_utilization,
# # #             'recent_inquiries_3m': recent_inquiries,
# # #             'active_loans_count': active_loans,
# # #             'avg_salary_6m': avg_salary,
# # #             'AMT_INCOME_TOTAL': amt_income,
# # #             'net_cash_surplus_6m': net_surplus,
# # #             'salary_stability_flag': salary_stability,
# # #             'loan_amount': loan_amount,
# # #             'loan_tenure_months': loan_tenure,
# # #             'interest_rate': interest_rate,
# # #             'existing_emi': existing_emi,
# # #             'AMT_ANNUITY': amt_annuity,
# # #             'application_id': app_id,
# # #             'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S")
# # #         }
        
# # #         # Get decision
# # #         with st.spinner("🔄 Processing assessment..."):
# # #             decision_data = make_hybrid_decision_enhanced(customer_data)
        
# # #         # Generate reasons
# # #         reasons = generate_reason_codes(
# # #             decision=decision_data['decision'],
# # #             customer_data=customer_data,
# # #             affordability_data=decision_data.get('affordability_data', {}),
# # #             policy_checks=decision_data['policy_checks']
# # #         )
        
# # #         customer_data['reason_codes'] = reasons
        
# # #         # Tabs
# # #         tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])
        
# # #         with tab1:
# # #             st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
            
# # #             col1, col2 = st.columns(2)
            
# # #             with col1:
# # #                 render_info_card(
# # #                     "👤 Identity", 
# # #                     "👤",
# # #                     {
# # #                         "Age": age,
# # #                         "Employment": employment_type,
# # #                         "KYC Status": "Verified" if kyc_verified else "Not Verified",
# # #                         "Tenure": f"{employment_tenure} months" if employment_type == 'Salaried' else f"{business_vintage} years"
# # #                     }
# # #                 )
                
# # #                 render_info_card(
# # #                     "💰 Financial", 
# # #                     "💰",
# # #                     {
# # #                         "Monthly Income": f"₹{avg_salary:,}",
# # #                         "Annual Income": f"₹{amt_income:,}",
# # #                         "Net Surplus": f"₹{net_surplus:,}",
# # #                         "Stability": salary_stability
# # #                     }
# # #                 )
            
# # #             with col2:
# # #                 render_info_card(
# # #                     "🏦 Credit Bureau", 
# # #                     "🏦",
# # #                     {
# # #                         "Bureau Score": bureau_score,
# # #                         "DPD 90+": dpd_90_6m,
# # #                         "Utilization": f"{credit_utilization}%",
# # #                         "Recent Inquiries": recent_inquiries,
# # #                         "Existing EMI": f"₹{existing_emi:,}"
# # #                     }
# # #                 )
                
# # #                 render_info_card(
# # #                     "📋 Loan Request", 
# # #                     "📋",
# # #                     {
# # #                         "Amount": f"₹{loan_amount:,}",
# # #                         "Tenure": f"{loan_tenure} months",
# # #                         "Interest Rate": f"{interest_rate}%",
# # #                         "Requested EMI": f"₹{amt_annuity:,}"
# # #                     }
# # #                 )
        
# # #         with tab2:
# # #             st.markdown('<p class="section-header">Decision Summary</p>', unsafe_allow_html=True)
            
# # #             render_decision_header(decision_data, customer_data)
            
# # #             st.markdown("<br>", unsafe_allow_html=True)
            
# # #             col1, col2, col3 = st.columns(3)
            
# # #             with col1:
# # #                 # Identity card
# # #                 age_pass = 18 <= age <= 65
# # #                 kyc_pass = kyc_verified
                
# # #                 render_info_card(
# # #                     "Identity & Eligibility",
# # #                     "👤",
# # #                     {
# # #                         f"Age: {age}": "",
# # #                         f"Employment: {employment_type}": "",
# # #                         f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""
# # #                     },
# # #                     {
# # #                         f"Age: {age}": "pass" if age_pass else "fail",
# # #                         f"Employment: {employment_type}": "pass",
# # #                         f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_pass else "fail"
# # #                     }
# # #                 )
            
# # #             with col2:
# # #                 # Credit card
# # #                 bureau_pass = bureau_score >= 550
# # #                 dpd_pass = dpd_90_6m == 0
                
# # #                 render_info_card(
# # #                     "Credit Bureau",
# # #                     "🏦",
# # #                     {
# # #                         f"Bureau Score: {bureau_score}": "",
# # #                         f"DPD 90+: {dpd_90_6m}": "",
# # #                         f"Utilization: {credit_utilization}%": ""
# # #                     },
# # #                     {
# # #                         f"Bureau Score: {bureau_score}": "pass" if bureau_pass else "fail",
# # #                         f"DPD 90+: {dpd_90_6m}": "pass" if dpd_pass else "fail",
# # #                         f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"
# # #                     }
# # #                 )
            
# # #             with col3:
# # #                 # Affordability card
# # #                 affordability = decision_data.get('affordability_data', {})
# # #                 foir = affordability.get('foir_percentage', 0)
# # #                 total_emi = affordability.get('total_emi', 0)
# # #                 net_disp = affordability.get('net_disposable', 0)
                
# # #                 render_info_card(
# # #                     "Affordability",
# # #                     "💰",
# # #                     {
# # #                         f"Monthly Income: ₹{avg_salary:,}": "",
# # #                         f"FOIR: {foir:.1f}%": "",
# # #                         f"Total EMI: ₹{total_emi:,}": "",
# # #                         f"Net Disposable: ₹{net_disp:,}": ""
# # #                     },
# # #                     {
# # #                         f"Monthly Income: ₹{avg_salary:,}": "pass",
# # #                         f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
# # #                         f"Total EMI: ₹{total_emi:,}": "pass",
# # #                         f"Net Disposable: ₹{net_disp:,}": "pass" if net_disp >= 10000 else "warning"
# # #                     }
# # #                 )
            
# # #             st.markdown("<br>", unsafe_allow_html=True)
            
# # #             # Reason codes
# # #             render_reason_codes(reasons)
            
# # #             st.markdown("<br>", unsafe_allow_html=True)
            
# # #             # Action buttons
# # #             col1, col2, col3 = st.columns([1, 1, 2])
# # #             with col1:
# # #                 if st.button("📥 Download Report", use_container_width=True):
# # #                     st.info("📄 Report generation coming soon...")
# # #             with col2:
# # #                 if st.button("🔄 Re-Evaluate", use_container_width=True):
# # #                     st.rerun()
        
# # #         with tab3:
# # #             st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
            
# # #             col1, col2 = st.columns(2)
            
# # #             with col1:
# # #                 fig1 = create_modern_gauge(decision_data['confidence'], "Model Confidence")
# # #                 st.plotly_chart(fig1, use_container_width=True)
            
# # #             with col2:
# # #                 fig2 = create_modern_bar_chart(decision_data['class_probs'])
# # #                 st.plotly_chart(fig2, use_container_width=True)
            
# # #             st.markdown("<br>", unsafe_allow_html=True)
            
# # #             # Policy checks
# # #             st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
            
# # #             policy_df = pd.DataFrame([
# # #                 {'Check': k, 'Result': v} 
# # #                 for k, v in decision_data['policy_checks'].items()
# # #             ])
# # #             st.dataframe(policy_df, use_container_width=True, hide_index=True)
        
# # #         with tab4:
# # #             st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
            
# # #             audit_log = {
# # #                 'application_id': app_id,
# # #                 'timestamp': timestamp.isoformat(),
# # #                 'decision': decision_data['decision'],
# # #                 'risk_score': decision_data['risk_score'],
# # #                 'pd_percentage': decision_data['pd_percentage'],
# # #                 'confidence': round(decision_data['confidence'], 2),
# # #                 'model_version': '8.0',
# # #                 'reason_codes': reasons,
# # #                 'affordability': affordability
# # #             }
            
# # #             st.json(audit_log)
            
# # #             import json
# # #             audit_json = json.dumps(audit_log, indent=2)
# # #             st.download_button(
# # #                 "📥 Download Audit Log",
# # #                 audit_json,
# # #                 f"audit_{app_id}.json",
# # #                 "application/json",
# # #                 use_container_width=True
# # #             )

# # # # =============================================================================
# # # # OTHER PAGES (Batch, Model Info, About) - Keep simplified
# # # # =============================================================================

# # # elif page == "📊 Batch Process":
# # #     st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
    
# # #     st.markdown("""
# # #         <div class="info-box">
# # #             📤 Upload a CSV file with customer data for bulk credit assessment
# # #         </div>
# # #     """, unsafe_allow_html=True)
    
# # #     st.info("Feature coming soon...")

# # # elif page == "📈 Model Info":
# # #     st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
    
# # #     col1, col2, col3 = st.columns(3)
    
# # #     with col1:
# # #         st.markdown("""
# # #             <div class="stat-card">
# # #                 <div class="stat-number">RF</div>
# # #                 <div class="stat-label">Model Type</div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     with col2:
# # #         st.markdown(f"""
# # #             <div class="stat-card">
# # #                 <div class="stat-number">{len(TOP_FEATURES)}</div>
# # #                 <div class="stat-label">Features</div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     with col3:
# # #         st.markdown(f"""
# # #             <div class="stat-card">
# # #                 <div class="stat-number">{len(TARGET_LE.classes_)}</div>
# # #                 <div class="stat-label">Classes</div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     st.markdown("<br>", unsafe_allow_html=True)
    
# # #     st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
    
# # #     feature_df = pd.DataFrame({
# # #         'Rank': range(1, min(21, len(TOP_FEATURES) + 1)),
# # #         'Feature': TOP_FEATURES[:20]
# # #     })
    
# # #     st.dataframe(feature_df, use_container_width=True, hide_index=True)

# # # elif page == "ℹ️ About":
# # #     st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
    
# # #     st.markdown("""
# # #         <div class="info-card">
# # #             <div class="info-card-title">
# # #                 <span class="icon">🏦</span>
# # #                 <span>Credit Risk Assessment Platform</span>
# # #             </div>
# # #             <div class="info-card-content">
# # #                 <p><strong>Version:</strong> 8.0 - Sage Green & Yellow Theme</p>
# # #                 <p><strong>Developer:</strong> Zen Meraki</p>
# # #                 <p><strong>Date:</strong> January 2025</p>
# # #                 <br>
# # #                 <p>
# # #                     A comprehensive credit risk evaluation system combining hard policy rules,
# # #                     machine learning models, and affordability analysis for accurate and compliant
# # #                     lending decisions.
# # #                 </p>
# # #             </div>
# # #         </div>
# # #     """, unsafe_allow_html=True)
    
# # #     st.markdown("<br>", unsafe_allow_html=True)
    
# # #     col1, col2 = st.columns(2)
    
# # #     with col1:
# # #         st.markdown("""
# # #             <div class="info-card">
# # #                 <div class="info-card-title">
# # #                     <span class="icon">🎯</span>
# # #                     <span>Key Features</span>
# # #                 </div>
# # #                 <div class="info-card-content">
# # #                     <ul style="margin: 0; padding-left: 1.25rem;">
# # #                         <li>Three-layer decision engine</li>
# # #                         <li>Real-time risk assessment</li>
# # #                         <li>FOIR calculation & validation</li>
# # #                         <li>Automated reason generation</li>
# # #                         <li>Complete audit trail</li>
# # #                         <li>Professional UI/UX</li>
# # #                     </ul>
# # #                 </div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
    
# # #     with col2:
# # #         st.markdown("""
# # #             <div class="info-card">
# # #                 <div class="info-card-title">
# # #                     <span class="icon">🛠️</span>
# # #                     <span>Technology Stack</span>
# # #                 </div>
# # #                 <div class="info-card-content">
# # #                     <ul style="margin: 0; padding-left: 1.25rem;">
# # #                         <li>Streamlit (UI Framework)</li>
# # #                         <li>Scikit-learn (ML)</li>
# # #                         <li>Plotly (Visualizations)</li>
# # #                         <li>Pandas (Data Processing)</li>
# # #                         <li>Python 3.8+</li>
# # #                     </ul>
# # #                 </div>
# # #             </div>
# # #         """, unsafe_allow_html=True)

# # # st.markdown("---")
# # # st.markdown("<div style='text-align: center; color: gray;'><p>© 2025 Hybrid Credit Risk System v6.0 | Zen Meraki</p></div>", 
# # #     unsafe_allow_html=True)






# # """
# # Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
# # Enhanced with Modern UI/UX Design
# # Run with: streamlit run test.py

# # Author: Zen Meraki  
# # Date: January 2025
# # VERSION: 8.0 - Sage Green & Yellow Professional Interface
# # """

# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import plotly.graph_objects as go
# # import plotly.express as px
# # import joblib
# # import warnings
# # from datetime import datetime
# # import hashlib
# # import io
# # import base64
# # from typing import Dict, List, Any
# # import json
# # warnings.filterwarnings('ignore')

# # # =============================================================================
# # # PAGE CONFIGURATION
# # # =============================================================================

# # st.set_page_config(
# #     page_title="Credit Risk Assessment",
# #     page_icon="💳",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # =============================================================================
# # # SAGE GREEN AND YELLOW THEME CSS
# # # =============================================================================

# # st.markdown("""
# #     <style>
# #     /* Import Google Fonts */
# #     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
# #     /* Global Styles */
# #     * {
# #         font-family: 'Inter', sans-serif;
# #     }
    
# #     /* Color Variables */
# #     :root {
# #         --fern-green: #587042;
# #         --sage: #A9B494;
# #         --cosmic-latte: #FAF7E6;
# #         --jasmine: #F8DE8C;
# #         --saffron: #F6C531;
# #         --dark-fern: #486032;
# #         --light-sage: #D4DBC4;
# #     }
    
# #     /* Main Background */
# #     .main {
# #         background-color: #FFFFFF;
# #     }
    
# #     .block-container {
# #         padding-top: 2rem;
# #         padding-bottom: 2rem;
# #         max-width: 1400px;
# #         background-color: #FFFFFF;
# #     }
    
# #     /* Headers */
# #     .main-header {
# #         font-size: 2.5rem;
# #         font-weight: 700;
# #         color: var(--fern-green);
# #         text-align: center;
# #         padding: 1.5rem 0;
# #         margin-bottom: 1rem;
# #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
# #         -webkit-background-clip: text;
# #         -webkit-text-fill-color: transparent;
# #         background-clip: text;
# #     }
    
# #     .section-header {
# #         font-size: 1.5rem;
# #         font-weight: 600;
# #         color: var(--fern-green);
# #         margin-top: 2rem;
# #         margin-bottom: 1rem;
# #         padding-bottom: 0.5rem;
# #         border-bottom: 2px solid var(--sage);
# #     }
    
# #     /* Decision Cards */
# #     .decision-card {
# #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
# #         padding: 2rem;
# #         border-radius: 16px;
# #         box-shadow: 0 10px 40px rgba(88, 112, 66, 0.2);
# #         margin-bottom: 2rem;
# #         color: white;
# #     }
    
# #     .decision-card-approved {
# #         background: linear-gradient(135deg, var(--fern-green) 0%, #7A9E4D 100%);
# #         box-shadow: 0 10px 40px rgba(88, 112, 66, 0.3);
# #     }
    
# #     .decision-card-rejected {
# #         background: linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%);
# #         box-shadow: 0 10px 40px rgba(211, 47, 47, 0.2);
# #     }
    
# #     .decision-card-review {
# #         background: linear-gradient(135deg, var(--saffron) 0%, var(--jasmine) 100%);
# #         box-shadow: 0 10px 40px rgba(246, 197, 49, 0.3);
# #     }
    
# #     .decision-title {
# #         font-size: 2.5rem;
# #         font-weight: 700;
# #         margin: 0;
# #         color: white;
# #         display: flex;
# #         align-items: center;
# #         gap: 1rem;
# #     }
    
# #     .decision-subtitle {
# #         font-size: 1.1rem;
# #         margin-top: 0.5rem;
# #         opacity: 0.9;
# #     }
    
# #     /* Info Cards */
# #     .info-card {
# #         background: white;
# #         border-radius: 12px;
# #         padding: 1.5rem;
# #         box-shadow: 0 4px 12px rgba(88, 112, 66, 0.08);
# #         border: 1px solid var(--sage);
# #         margin-bottom: 1rem;
# #         transition: all 0.3s ease;
# #     }
    
# #     .info-card:hover {
# #         box-shadow: 0 10px 25px rgba(88, 112, 66, 0.15);
# #         transform: translateY(-2px);
# #         border-color: var(--fern-green);
# #     }
    
# #     .info-card-title {
# #         font-size: 1.1rem;
# #         font-weight: 600;
# #         color: var(--fern-green);
# #         margin-bottom: 1rem;
# #         display: flex;
# #         align-items: center;
# #         gap: 0.5rem;
# #     }
    
# #     .info-card-content {
# #         color: #5A5A5A;
# #         line-height: 1.6;
# #     }
    
# #     /* Metric Cards */
# #     .metric-card {
# #         background: linear-gradient(135deg, var(--cosmic-latte) 0%, #F5F2E0 100%);
# #         border-radius: 12px;
# #         padding: 1.5rem;
# #         border-left: 4px solid var(--fern-green);
# #         margin-bottom: 1rem;
# #     }
    
# #     .metric-label {
# #         font-size: 0.875rem;
# #         font-weight: 500;
# #         color: var(--fern-green);
# #         text-transform: uppercase;
# #         letter-spacing: 0.05em;
# #         margin-bottom: 0.5rem;
# #     }
    
# #     .metric-value {
# #         font-size: 2rem;
# #         font-weight: 700;
# #         color: var(--fern-green);
# #     }
    
# #     /* Status Badges */
# #     .status-badge {
# #         display: inline-flex;
# #         align-items: center;
# #         padding: 0.5rem 1rem;
# #         border-radius: 20px;
# #         font-weight: 600;
# #         font-size: 0.875rem;
# #         gap: 0.5rem;
# #     }
    
# #     .badge-pass {
# #         background: #E8F5E9;
# #         color: var(--fern-green);
# #         border: 1px solid var(--sage);
# #     }
    
# #     .badge-fail {
# #         background: #FFEBEE;
# #         color: #D32F2F;
# #         border: 1px solid #FFCDD2;
# #     }
    
# #     .badge-warning {
# #         background: linear-gradient(135deg, var(--cosmic-latte) 0%, var(--jasmine) 100%);
# #         color: #7B5800;
# #         border: 1px solid var(--jasmine);
# #     }
    
# #     .badge-info {
# #         background: #E3F2FD;
# #         color: #1565C0;
# #         border: 1px solid #90CAF9;
# #     }
    
# #     /* Data Row */
# #     .data-row {
# #         display: flex;
# #         justify-content: space-between;
# #         align-items: center;
# #         padding: 0.75rem 0;
# #         border-bottom: 1px solid var(--sage);
# #     }
    
# #     .data-row:last-child {
# #         border-bottom: none;
# #     }
    
# #     .data-label {
# #         font-weight: 500;
# #         color: #5A5A5A;
# #     }
    
# #     .data-value {
# #         font-weight: 600;
# #         color: var(--fern-green);
# #     }
    
# #     /* Reason Items */
# #     .reason-item {
# #         background: linear-gradient(135deg, #F9F7EB 0%, #F5F2E0 100%);
# #         padding: 1rem 1.25rem;
# #         border-radius: 8px;
# #         border-left: 4px solid var(--saffron);
# #         margin-bottom: 0.75rem;
# #         color: #7B5800;
# #         font-weight: 500;
# #         display: flex;
# #         align-items: center;
# #         gap: 0.75rem;
# #     }
    
# #     .reason-icon {
# #         font-size: 1.25rem;
# #         color: var(--fern-green);
# #     }
    
# #     /* Buttons */
# #     .stButton > button {
# #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--dark-fern) 100%);
# #         color: white;
# #         border: none;
# #         border-radius: 8px;
# #         padding: 0.75rem 1.5rem;
# #         font-weight: 600;
# #         transition: all 0.3s ease;
# #         box-shadow: 0 4px 6px rgba(88, 112, 66, 0.2);
# #     }
    
# #     .stButton > button:hover {
# #         box-shadow: 0 10px 20px rgba(88, 112, 66, 0.3);
# #         transform: translateY(-2px);
# #         background: linear-gradient(135deg, var(--dark-fern) 0%, var(--fern-green) 100%);
# #     }
    
# #     /* Form Inputs */
# #     .stNumberInput > div > div > input,
# #     .stSelectbox > div > div > select {
# #         border-radius: 8px;
# #         border: 1px solid var(--sage);
# #         padding: 0.75rem;
# #         font-size: 1rem;
# #         background-color: white;
# #     }
    
# #     .stNumberInput > div > div > input:focus,
# #     .stSelectbox > div > div > select:focus {
# #         border-color: var(--fern-green);
# #         box-shadow: 0 0 0 3px rgba(88, 112, 66, 0.1);
# #     }
    
# #     /* Tabs */
# #     .stTabs [data-baseweb="tab-list"] {
# #         gap: 2rem;
# #         background-color: white;
# #         padding: 1rem;
# #         border-radius: 12px;
# #         box-shadow: 0 2px 4px rgba(88, 112, 66, 0.05);
# #     }
    
# #     .stTabs [data-baseweb="tab"] {
# #         height: 3rem;
# #         padding: 0 1.5rem;
# #         background-color: transparent;
# #         border-radius: 8px;
# #         color: #718096;
# #         font-weight: 600;
# #         transition: all 0.3s ease;
# #     }
    
# #     .stTabs [aria-selected="true"] {
# #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
# #         color: white;
# #     }
    
# #     .stTabs [data-baseweb="tab"]:hover {
# #         background-color: var(--cosmic-latte);
# #     }
    
# #     /* Sidebar */
# #     [data-testid="stSidebar"] {
# #         background: linear-gradient(180deg, var(--cosmic-latte) 0%, #F5F2E0 100%);
# #         border-right: 1px solid var(--sage);
# #     }
    
# #     [data-testid="stSidebar"] .element-container {
# #         color: var(--fern-green);
# #     }
    
# #     /* Expander */
# #     .streamlit-expanderHeader {
# #         background-color: var(--cosmic-latte);
# #         border-radius: 8px;
# #         padding: 0.75rem;
# #         font-weight: 600;
# #         color: var(--fern-green);
# #         border: 1px solid var(--sage);
# #     }
    
# #     /* Alerts */
# #     .stAlert {
# #         border-radius: 12px;
# #         border: none;
# #         padding: 1rem 1.5rem;
# #     }
    
# #     /* Success Alert */
# #     [data-baseweb="notification"] {
# #         background-color: #E8F5E9;
# #         border-left: 4px solid var(--fern-green);
# #         border-radius: 8px;
# #     }
    
# #     /* Info Alert */
# #     .info-box {
# #         background: linear-gradient(135deg, #F9F7EB 0%, var(--cosmic-latte) 100%);
# #         border-left: 4px solid var(--sage);
# #         border-radius: 8px;
# #         padding: 1.25rem;
# #         margin: 1rem 0;
# #         color: var(--fern-green);
# #         border: 1px solid var(--sage);
# #     }
    
# #     /* Warning Alert */
# #     .warning-box {
# #         background: linear-gradient(135deg, var(--cosmic-latte) 0%, var(--jasmine) 100%);
# #         border-left: 4px solid var(--saffron);
# #         border-radius: 8px;
# #         padding: 1.25rem;
# #         margin: 1rem 0;
# #         color: #7B5800;
# #         border: 1px solid var(--jasmine);
# #     }
    
# #     /* Dataframe */
# #     .dataframe {
# #         border-radius: 12px;
# #         overflow: hidden;
# #         box-shadow: 0 4px 6px rgba(88, 112, 66, 0.05);
# #         border: 1px solid var(--sage);
# #     }
    
# #     /* Metric Container */
# #     [data-testid="stMetricValue"] {
# #         font-size: 2rem;
# #         font-weight: 700;
# #         color: var(--fern-green);
# #     }
    
# #     [data-testid="stMetricLabel"] {
# #         font-size: 0.875rem;
# #         font-weight: 600;
# #         color: var(--fern-green);
# #         text-transform: uppercase;
# #         letter-spacing: 0.05em;
# #         opacity: 0.8;
# #     }
    
# #     /* Progress Bar */
# #     .stProgress > div > div > div {
# #         background: linear-gradient(90deg, var(--fern-green) 0%, var(--sage) 100%);
# #     }
    
# #     /* Divider */
# #     hr {
# #         margin: 2rem 0;
# #         border: none;
# #         border-top: 2px solid var(--sage);
# #     }
    
# #     /* Custom Scrollbar */
# #     ::-webkit-scrollbar {
# #         width: 10px;
# #         height: 10px;
# #     }
    
# #     ::-webkit-scrollbar-track {
# #         background: var(--cosmic-latte);
# #     }
    
# #     ::-webkit-scrollbar-thumb {
# #         background: var(--sage);
# #         border-radius: 5px;
# #     }
    
# #     ::-webkit-scrollbar-thumb:hover {
# #         background: var(--fern-green);
# #     }
    
# #     /* Icon Styles */
# #     .icon {
# #         font-size: 1.5rem;
# #         margin-right: 0.5rem;
# #         color: var(--fern-green);
# #     }
    
# #     /* Card Grid */
# #     .card-grid {
# #         display: grid;
# #         grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
# #         gap: 1.5rem;
# #         margin: 1.5rem 0;
# #     }
    
# #     /* Feature Badge */
# #     .feature-badge {
# #         display: inline-block;
# #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
# #         color: white;
# #         padding: 0.25rem 0.75rem;
# #         border-radius: 12px;
# #         font-size: 0.75rem;
# #         font-weight: 600;
# #         text-transform: uppercase;
# #         letter-spacing: 0.05em;
# #     }
    
# #     /* Timeline */
# #     .timeline-item {
# #         position: relative;
# #         padding-left: 2rem;
# #         padding-bottom: 1.5rem;
# #         border-left: 2px solid var(--sage);
# #     }
    
# #     .timeline-item:last-child {
# #         border-left: none;
# #     }
    
# #     .timeline-dot {
# #         position: absolute;
# #         left: -6px;
# #         top: 0;
# #         width: 12px;
# #         height: 12px;
# #         border-radius: 50%;
# #         background: var(--fern-green);
# #     }
    
# #     /* Stat Card */
# #     .stat-card {
# #         background: white;
# #         border-radius: 12px;
# #         padding: 1.5rem;
# #         box-shadow: 0 4px 12px rgba(88, 112, 66, 0.08);
# #         border-top: 4px solid var(--fern-green);
# #         text-align: center;
# #         border: 1px solid var(--sage);
# #         transition: all 0.3s ease;
# #     }
    
# #     .stat-card:hover {
# #         box-shadow: 0 10px 25px rgba(88, 112, 66, 0.15);
# #         transform: translateY(-2px);
# #     }
    
# #     .stat-number {
# #         font-size: 2.5rem;
# #         font-weight: 700;
# #         color: var(--fern-green);
# #         margin-bottom: 0.5rem;
# #     }
    
# #     .stat-label {
# #         font-size: 0.875rem;
# #         font-weight: 600;
# #         color: var(--fern-green);
# #         text-transform: uppercase;
# #         letter-spacing: 0.05em;
# #         opacity: 0.8;
# #     }
    
# #     /* Chart styling */
# #     .js-plotly-plot .plotly {
# #         background-color: white !important;
# #     }
    
# #     /* Radio buttons */
# #     .stRadio > div {
# #         background-color: white;
# #         padding: 0.5rem;
# #         border-radius: 8px;
# #         border: 1px solid var(--sage);
# #     }
    
# #     .stRadio > div[data-baseweb="radio"] label {
# #         color: var(--fern-green);
# #     }
    
# #     /* Checkbox */
# #     .stCheckbox > label {
# #         color: var(--fern-green);
# #     }
    
# #     /* Form submit button */
# #     div[data-testid="stForm"] button {
# #         background: linear-gradient(135deg, var(--fern-green) 0%, var(--dark-fern) 100%);
# #         color: white;
# #         border: none;
# #         border-radius: 8px;
# #         padding: 0.75rem 1.5rem;
# #         font-weight: 600;
# #         transition: all 0.3s ease;
# #         box-shadow: 0 4px 6px rgba(88, 112, 66, 0.2);
# #     }
    
# #     div[data-testid="stForm"] button:hover {
# #         box-shadow: 0 10px 20px rgba(88, 112, 66, 0.3);
# #         transform: translateY(-2px);
# #         background: linear-gradient(135deg, var(--dark-fern) 0%, var(--fern-green) 100%);
# #     }
    
# #     /* Table styling */
# #     .stTable {
# #         border: 1px solid var(--sage);
# #         border-radius: 8px;
# #     }
    
# #     /* Container borders */
# #     .stApp {
# #         background-color: white;
# #     }
    
# #     /* ================== FIXES FOR WHITE TEXT ================== */
# #     /* Input text color */
# #     .stNumberInput > div > div > input,
# #     .stSelectbox > div > div > select,
# #     .stTextInput > div > div > input {
# #         color: #333333 !important;
# #     }
    
# #     /* Radio button text */
# #     .stRadio > div > label,
# #     .stRadio > div > label > div > p {
# #         color: #333333 !important;
# #     }
    
# #     /* Checkbox text */
# #     .stCheckbox > label,
# #     .stCheckbox > label > div > p {
# #         color: #333333 !important;
# #     }
    
# #     /* Slider value text */
# #     .stSlider > div > div > div {
# #         color: #333333 !important;
# #     }
    
# #     /* Placeholder text */
# #     ::placeholder {
# #         color: #718096 !important;
# #     }
    
# #     /* Sidebar text */
# #     [data-testid="stSidebar"] p,
# #     [data-testid="stSidebar"] div,
# #     [data-testid="stSidebar"] span {
# #         color: #333333 !important;
# #     }
    
# #     /* Form labels */
# #     .stNumberInput label,
# #     .stSelectbox label,
# #     .stTextInput label,
# #     .stRadio label,
# #     .stCheckbox label {
# #         color: var(--fern-green) !important;
# #         font-weight: 600;
# #     }
    
# #     /* General text color for all elements */
# #     body, p, div, span, h1, h2, h3, h4, h5, h6 {
# #         color: #333333 !important;
# #     }
    
# #     /* Fix for text in expanders */
# #     .streamlit-expanderContent p,
# #     .streamlit-expanderContent div,
# #     .streamlit-expanderContent span {
# #         color: #333333 !important;
# #     }
    
# #     /* Fix for text in alerts */
# #     .stAlert p,
# #     .stAlert div,
# #     .stAlert span {
# #         color: inherit !important;
# #     }
# #     /* ================== END OF FIXES ================== */
    
# #     </style>
# # """, unsafe_allow_html=True)

# # # =============================================================================
# # # LOAD TRAINED MODEL ASSETS
# # # =============================================================================

# # @st.cache_resource
# # def load_model_assets():
# #     """Load the trained model and preprocessing assets"""
# #     try:
# #         possible_paths = [
# #             'credit_risk_assets.pkl',
# #             'notebooks/credit_risk_assets.pkl',
# #             '../notebooks/credit_risk_assets.pkl'
# #         ]
        
# #         assets = None
# #         for path in possible_paths:
# #             try:
# #                 assets = joblib.load(path)
# #                 break
# #             except FileNotFoundError:
# #                 continue
        
# #         if assets is None:
# #             raise FileNotFoundError("Could not find credit_risk_assets.pkl")
# #         return {
# #             'model': assets['model'],
# #             'features': assets['features'],
# #             'le_map': assets['le_map'],
# #             'target_le': assets['target_le'],
# #             'loaded': True,
# #             'error': None
# #         }
# #     except FileNotFoundError:
# #         return {
# #             'loaded': False,
# #             'error': 'credit_risk_assets.pkl not found. Please run the training script first.'
# #         }
# #     except Exception as e:
# #         return {
# #             'loaded': False,
# #             'error': f'Error loading model: {str(e)}'
# #         }

# # # Load assets
# # ASSETS = load_model_assets()

# # if not ASSETS['loaded']:
# #     st.error(f"❌ {ASSETS['error']}")
# #     st.info("Please ensure 'credit_risk_assets.pkl' is in the same directory as this app.")
# #     st.stop()

# # MODEL = ASSETS['model']
# # TOP_FEATURES = ASSETS['features']
# # LE_MAP = ASSETS['le_map']
# # TARGET_LE = ASSETS['target_le']

# # # =============================================================================
# # # AFFORDABILITY CALCULATION ENGINE
# # # =============================================================================

# # def calculate_emi(principal, annual_rate, tenure_months):
# #     """Calculate EMI using reducing balance method"""
# #     if principal <= 0 or tenure_months <= 0:
# #         return 0
    
# #     monthly_rate = annual_rate / (12 * 100)
    
# #     if monthly_rate == 0:
# #         return principal / tenure_months
    
# #     emi = (principal * monthly_rate * (1 + monthly_rate)**tenure_months) / \
# #           ((1 + monthly_rate)**tenure_months - 1)
    
# #     return round(emi, 2)


# # def calculate_affordability(monthly_income, loan_amount, interest_rate, 
# #                            tenure_months, existing_emi):
# #     """Calculate comprehensive affordability metrics"""
    
# #     new_emi = calculate_emi(loan_amount, interest_rate, tenure_months)
# #     total_emi = new_emi + existing_emi
# #     foir_percentage = (total_emi / monthly_income) * 100 if monthly_income > 0 else 0
# #     net_disposable = monthly_income - total_emi
# #     max_allowed_emi = monthly_income * 0.50
# #     recommended_emi = monthly_income * 0.40
# #     affordable = foir_percentage <= 50
# #     within_recommended = foir_percentage <= 40
    
# #     if foir_percentage <= 40:
# #         status = "Excellent"
# #         status_color = "green"
# #     elif foir_percentage <= 50:
# #         status = "Acceptable"
# #         status_color = "yellow"
# #     else:
# #         status = "Over-leveraged"
# #         status_color = "red"
    
# #     return {
# #         'monthly_income': monthly_income,
# #         'new_emi': new_emi,
# #         'existing_emi': existing_emi,
# #         'total_emi': total_emi,
# #         'foir_percentage': round(foir_percentage, 2),
# #         'net_disposable': net_disposable,
# #         'max_allowed_emi': max_allowed_emi,
# #         'recommended_emi': recommended_emi,
# #         'affordable': affordable,
# #         'within_recommended': within_recommended,
# #         'status': status,
# #         'status_color': status_color,
# #         'emi_headroom': max_allowed_emi - total_emi
# #     }

# # # =============================================================================
# # # REASON CODE GENERATION SYSTEM
# # # =============================================================================

# # APPROVAL_REASONS = {
# #     'high_bureau': 'Excellent credit score ({score})',
# #     'stable_employment': 'Stable employment history ({tenure} months)',
# #     'low_foir': 'Affordable EMI burden (FOIR: {foir}%)',
# #     'clean_payment': 'Clean payment history (No DPD)',
# #     'strong_income': 'Strong monthly income (₹{income:,})',
# #     'low_utilization': 'Low credit utilization ({util}%)',
# # }

# # REJECTION_REASONS = {
# #     'low_bureau': 'Credit score below minimum ({score} < 550)',
# #     'high_foir': 'EMI burden too high (FOIR: {foir}% > 50%)',
# #     'severe_dpd': 'Severe payment delays ({dpd} instances of 90+ DPD)',
# #     'low_income': 'Income below minimum threshold (₹{income:,} < ₹15,000)',
# #     'short_employment': 'Insufficient employment tenure ({tenure} months < 6)',
# #     'bankruptcy': 'Active bankruptcy detected',
# #     'kyc_failed': 'KYC verification not completed',
# #     'high_utilization': 'High credit utilization ({util}% > 80%)',
# #     'age_invalid': 'Age outside acceptable range ({age} years)'
# # }

# # REVIEW_REASONS = {
# #     'borderline_bureau': 'Credit score in borderline range ({score})',
# #     'moderate_foir': 'EMI burden moderate (FOIR: {foir}%)',
# #     'mixed_signals': 'Mixed credit indicators requiring human review',
# #     'recent_employment': 'Recent employment change requiring verification',
# # }


# # def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
# #     """Generate top 3 reason codes for the decision"""
# #     reasons = []
    
# #     bureau_score = customer_data.get('bureau_score', 0)
# #     foir = affordability_data.get('foir_percentage', 0)
# #     dpd_90 = customer_data.get('dpd_90_count_6m', 0)
# #     income = customer_data.get('avg_salary_6m', 0)
# #     employment_tenure = customer_data.get('employment_tenure_months', 0)
# #     credit_util = customer_data.get('credit_utilization_pct', 0)
# #     age = customer_data.get('age', 0)
    
# #     if decision == "APPROVE":
# #         if bureau_score >= 750:
# #             reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
# #         if employment_tenure >= 24:
# #             reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
# #         if foir <= 40:
# #             reasons.append(APPROVAL_REASONS['low_foir'].format(foir=round(foir, 1)))
# #         if dpd_90 == 0:
# #             reasons.append(APPROVAL_REASONS['clean_payment'])
# #         if income >= 75000:
# #             reasons.append(APPROVAL_REASONS['strong_income'].format(income=income))
# #         if credit_util <= 30:
# #             reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))
    
# #     elif decision == "REJECT":
# #         for check_name, check_result in policy_checks.items():
# #             if '❌' in str(check_result):
# #                 if 'bureau' in check_name.lower():
# #                     reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
# #                 elif 'dpd' in check_name.lower():
# #                     reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
# #                 elif 'income' in check_name.lower():
# #                     reasons.append(REJECTION_REASONS['low_income'].format(income=income))
# #                 elif 'tenure' in check_name.lower():
# #                     reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
# #                 elif 'kyc' in check_name.lower():
# #                     reasons.append(REJECTION_REASONS['kyc_failed'])
# #                 elif 'bankruptcy' in check_name.lower():
# #                     reasons.append(REJECTION_REASONS['bankruptcy'])
# #                 elif 'age' in check_name.lower():
# #                     reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))
        
# #         if foir > 50:
# #             reasons.append(REJECTION_REASONS['high_foir'].format(foir=round(foir, 1)))
# #         if credit_util > 80:
# #             reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
    
# #     elif decision == "REVIEW":
# #         if 650 <= bureau_score < 700:
# #             reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
# #         if 40 < foir <= 50:
# #             reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=round(foir, 1)))
# #         if employment_tenure < 12:
# #             reasons.append(REVIEW_REASONS['recent_employment'])
# #         if not reasons:
# #             reasons.append(REVIEW_REASONS['mixed_signals'])
    
# #     return reasons[:3] if reasons else ['Decision based on model assessment']

# # # =============================================================================
# # # RISK SCORE CALCULATION
# # # =============================================================================

# # def calculate_final_risk_score(bureau_score, ml_confidence, foir):
# #     """Calculate final risk score (0-1000)"""
# #     bureau_points = (bureau_score / 900) * 400
# #     ml_points = (ml_confidence / 100) * 400
# #     foir_points = max(0, (1 - foir/50) * 200)
# #     total_score = int(bureau_points + ml_points + foir_points)
# #     return min(max(total_score, 0), 1000)

# # # =============================================================================
# # # BATCH PREDICTION ENGINE
# # # =============================================================================

# # def process_batch_predictions(df: pd.DataFrame) -> pd.DataFrame:
# #     """Process batch predictions for multiple records"""
# #     results = []
    
# #     for idx, row in df.iterrows():
# #         customer_dict = row.to_dict()
        
# #         # Convert yes/no to boolean
# #         for key, value in customer_dict.items():
# #             if isinstance(value, str):
# #                 if value.lower() in ['yes', 'true', '1']:
# #                     customer_dict[key] = True
# #                 elif value.lower() in ['no', 'false', '0']:
# #                     customer_dict[key] = False
        
# #         # Add missing required fields with defaults
# #         required_fields = {
# #             'kyc_verified': True,
# #             'bankruptcy_flag': False,
# #             'fraud_flag': False,
# #             'dpd_90_count_6m': 0,
# #             'recent_inquiries_3m': 0,
# #             'active_loans_count': 0,
# #             'existing_emi': 0,
# #             'salary_stability_flag': 'STABLE'
# #         }
        
# #         for field, default in required_fields.items():
# #             if field not in customer_dict:
# #                 customer_dict[field] = default
        
# #         # Get decision
# #         decision_data = make_hybrid_decision_enhanced(customer_dict)
        
# #         # Generate application ID
# #         app_id = f"BATCH_{idx+1:04d}"
        
# #         # Prepare result
# #         result = {
# #             'application_id': app_id,
# #             'decision': decision_data['decision'],
# #             'risk_score': decision_data['risk_score'],
# #             'pd_percentage': decision_data['pd_percentage'],
# #             'confidence': round(decision_data['confidence'], 2),
# #             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# #         }
        
# #         # Add key customer data
# #         result.update({
# #             'age': customer_dict.get('age', ''),
# #             'employment_type': customer_dict.get('employment_type', ''),
# #             'bureau_score': customer_dict.get('bureau_score', ''),
# #             'monthly_income': customer_dict.get('avg_salary_6m', ''),
# #             'loan_amount': customer_dict.get('loan_amount', ''),
# #             'foir_percentage': decision_data.get('affordability_data', {}).get('foir_percentage', 0)
# #         })
        
# #         results.append(result)
    
# #     return pd.DataFrame(results)

# # def create_download_link(df: pd.DataFrame, filename: str = "batch_results.csv") -> str:
# #     """Create a download link for a DataFrame"""
# #     csv = df.to_csv(index=False)
# #     b64 = base64.b64encode(csv.encode()).decode()
# #     href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;">📥 Download CSV</a>'
# #     return href

# # # =============================================================================
# # # ENHANCED HYBRID DECISION ENGINE
# # # =============================================================================

# # def make_hybrid_decision_enhanced(customer_dict):
# #     """Enhanced decision engine with complete data"""
    
# #     policy_checks = {}
    
# #     # Policy Gates
# #     age = customer_dict.get('age', 0)
# #     employment_type = customer_dict.get('employment_type', 'Salaried')
# #     kyc_verified = customer_dict.get('kyc_verified', True)
# #     bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
# #     fraud_flag = customer_dict.get('fraud_flag', False)
    
# #     if employment_type in ['Salaried']:
# #         age_min, age_max = 18, 65
# #     else:
# #         age_min, age_max = 18, 70
    
# #     if age < age_min or age > age_max:
# #         policy_checks['age'] = f"❌ Age {age} (Required: {age_min}-{age_max})"
# #         return {
# #             'decision': "REJECT",
# #             'reason': f"Policy Gate: Age outside allowed range",
# #             'confidence': 0,
# #             'class_probs': {'REJECT': 100},
# #             'policy_checks': policy_checks,
# #             'risk_score': 0,
# #             'pd_percentage': 100.0
# #         }
# #     policy_checks['age'] = f"✅ Age {age} (Valid)"
    
# #     if not kyc_verified:
# #         policy_checks['kyc'] = "❌ KYC Not Verified"
# #         return {
# #             'decision': "REJECT",
# #             'reason': "Policy Gate: KYC verification required",
# #             'confidence': 0,
# #             'class_probs': {'REJECT': 100},
# #             'policy_checks': policy_checks,
# #             'risk_score': 0,
# #             'pd_percentage': 100.0
# #         }
# #     policy_checks['kyc'] = "✅ KYC Verified"
    
# #     if bankruptcy_flag:
# #         policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
# #         return {
# #             'decision': "REJECT",
# #             'reason': "Policy Gate: Active bankruptcy",
# #             'confidence': 0,
# #             'class_probs': {'REJECT': 100},
# #             'policy_checks': policy_checks,
# #             'risk_score': 0,
# #             'pd_percentage': 100.0
# #         }
# #     policy_checks['bankruptcy'] = "✅ No Bankruptcy"
    
# #     if fraud_flag:
# #         policy_checks['fraud'] = "❌ Fraud Flag"
# #         return {
# #             'decision': "REJECT",
# #             'reason': "Policy Gate: Fraud detected",
# #             'confidence': 0,
# #             'class_probs': {'REJECT': 100},
# #             'policy_checks': policy_checks,
# #             'risk_score': 0,
# #             'pd_percentage': 100.0
# #         }
# #     policy_checks['fraud'] = "✅ No Fraud History"
    
# #     monthly_income = customer_dict.get('avg_salary_6m', 0)
# #     employment_tenure = customer_dict.get('employment_tenure_months', 0)
# #     business_vintage = customer_dict.get('business_vintage_years', 0)
    
# #     if monthly_income < 15000:
# #         policy_checks['income'] = f"❌ Income ₹{monthly_income:,.0f} (Min: ₹15,000)"
# #         return {
# #             'decision': "REJECT",
# #             'reason': "Policy Gate: Income below minimum",
# #             'confidence': 0,
# #             'class_probs': {'REJECT': 100},
# #             'policy_checks': policy_checks,
# #             'risk_score': 0,
# #             'pd_percentage': 100.0
# #         }
# #     policy_checks['income'] = f"✅ Income ₹{monthly_income:,.0f}"
    
# #     if employment_type == 'Salaried' and employment_tenure < 6:
# #         policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
# #         return {
# #             'decision': "REJECT",
# #             'reason': "Policy Gate: Insufficient tenure",
# #             'confidence': 0,
# #             'class_probs': {'REJECT': 100},
# #             'policy_checks': policy_checks,
# #             'risk_score': 0,
# #             'pd_percentage': 100.0
# #         }
# #     elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
# #         policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
# #         return {
# #             'decision': "REJECT",
# #             'reason': "Policy Gate: Insufficient business vintage",
# #             'confidence': 0,
# #             'class_probs': {'REJECT': 100},
# #             'policy_checks': policy_checks,
# #             'risk_score': 0,
# #             'pd_percentage': 100.0
# #         }
    
# #     if employment_type == 'Salaried':
# #         policy_checks['tenure'] = f"✅ Tenure {employment_tenure} months"
# #     else:
# #         policy_checks['tenure'] = f"✅ Business Vintage {business_vintage} years"
    
# #     bureau_score = customer_dict.get('bureau_score', 0)
# #     dpd_90 = customer_dict.get('dpd_90_count_6m', 0)
# #     credit_utilization = customer_dict.get('credit_utilization_pct', 0)
# #     recent_inquiries = customer_dict.get('recent_inquiries_3m', 0)
    
# #     if bureau_score < 550:
# #         policy_checks['bureau'] = f"❌ Bureau Score {bureau_score} (Min: 550)"
# #         return {
# #             'decision': "REJECT",
# #             'reason': "Policy Gate: Bureau score too low",
# #             'confidence': 0,
# #             'class_probs': {'REJECT': 100},
# #             'policy_checks': policy_checks,
# #             'risk_score': 0,
# #             'pd_percentage': 100.0
# #         }
# #     policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"
    
# #     if dpd_90 > 0:
# #         policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD"
# #         return {
# #             'decision': "REJECT",
# #             'reason': "Policy Gate: Severe delinquency",
# #             'confidence': 0,
# #             'class_probs': {'REJECT': 100},
# #             'policy_checks': policy_checks,
# #             'risk_score': 0,
# #             'pd_percentage': 100.0
# #         }
# #     policy_checks['dpd'] = "✅ No 90+ DPD"
    
# #     if credit_utilization > 80:
# #         policy_checks['utilization'] = f"⚠️ High utilization {credit_utilization}%"
# #     else:
# #         policy_checks['utilization'] = f"✅ Utilization {credit_utilization}%"
    
# #     if recent_inquiries > 5:
# #         policy_checks['inquiries'] = f"⚠️ {recent_inquiries} recent inquiries"
# #     else:
# #         policy_checks['inquiries'] = f"✅ {recent_inquiries} inquiries"
    
# #     # ML Prediction
# #     input_df = pd.DataFrame([customer_dict])
    
# #     for col in TOP_FEATURES:
# #         if col not in input_df.columns:
# #             if col in LE_MAP:
# #                 input_df[col] = "Unknown"
# #             else:
# #                 input_df[col] = 0
    
# #     for col, le in LE_MAP.items():
# #         if col in input_df.columns:
# #             val = str(input_df[col].values[0])
# #             try:
# #                 input_df[col] = le.transform([val])[0]
# #             except ValueError:
# #                 input_df[col] = 0
    
# #     final_input = input_df[TOP_FEATURES]
    
# #     pred_idx = MODEL.predict(final_input)[0]
# #     ml_decision = TARGET_LE.inverse_transform([pred_idx])[0]
    
# #     try:
# #         pred_proba = MODEL.predict_proba(final_input)[0]
# #         confidence = max(pred_proba) * 100
# #         class_probs = {
# #             cls: prob * 100 
# #             for cls, prob in zip(TARGET_LE.classes_, pred_proba)
# #         }
# #     except:
# #         confidence = 75.0
# #         class_probs = {ml_decision: 100.0}
    
# #     # Affordability
# #     loan_amount = customer_dict.get('loan_amount', 0)
# #     loan_tenure = customer_dict.get('loan_tenure_months', 12)
# #     interest_rate = customer_dict.get('interest_rate', 10.5)
# #     existing_emi = customer_dict.get('existing_emi', 0)
    
# #     affordability_data = calculate_affordability(
# #         monthly_income=monthly_income,
# #         loan_amount=loan_amount,
# #         interest_rate=interest_rate,
# #         tenure_months=loan_tenure,
# #         existing_emi=existing_emi
# #     )
    
# #     foir = affordability_data['foir_percentage']
    
# #     if ml_decision == "APPROVE" and foir > 45:
# #         ml_decision = "REVIEW"
    
# #     risk_score = calculate_final_risk_score(bureau_score, confidence, foir)
# #     pd_percentage = max(0, min(100, (1 - confidence/100) * 10))
    
# #     return {
# #         'decision': ml_decision,
# #         'reason': "Decision based on comprehensive assessment",
# #         'confidence': confidence,
# #         'class_probs': class_probs,
# #         'policy_checks': policy_checks,
# #         'risk_score': risk_score,
# #         'pd_percentage': round(pd_percentage, 2),
# #         'affordability_data': affordability_data
# #     }

# # # =============================================================================
# # # MODERN UI COMPONENTS
# # # =============================================================================

# # def render_decision_header(decision_data, customer_data):
# #     """Render modern decision header"""
    
# #     decision = decision_data['decision']
# #     risk_score = decision_data['risk_score']
# #     pd_score = decision_data['pd_percentage']
# #     approved_amount = customer_data.get('loan_amount', 0)
# #     tenure = customer_data.get('loan_tenure_months', 24)
# #     app_id = customer_data.get('application_id', 'N/A')
# #     timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
# #     # Decision card with appropriate styling
# #     if decision == "APPROVE":
# #         card_class = "decision-card decision-card-approved"
# #         icon = "✓"
# #         subtitle = "Application Approved Successfully"
# #     elif decision == "REJECT":
# #         card_class = "decision-card decision-card-rejected"
# #         icon = "✗"
# #         subtitle = "Application Not Approved"
# #     else:
# #         card_class = "decision-card decision-card-review"
# #         icon = "⚠"
# #         subtitle = "Requires Manual Review"
    
# #     st.markdown(f"""
# #         <div class="{card_class}">
# #             <div class="decision-title">
# #                 <span>{icon}</span>
# #                 <span>{decision}</span>
# #             </div>
# #             <div class="decision-subtitle">{subtitle}</div>
# #         </div>
# #     """, unsafe_allow_html=True)
    
# #     # Metrics grid
# #     col1, col2, col3, col4, col5 = st.columns(5)
    
# #     with col1:
# #         st.markdown(f"""
# #             <div class="stat-card">
# #                 <div class="stat-number">{risk_score}</div>
# #                 <div class="stat-label">Risk Score</div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     with col2:
# #         st.markdown(f"""
# #             <div class="stat-card">
# #                 <div class="stat-number">{pd_score}%</div>
# #                 <div class="stat-label">PD Score</div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     with col3:
# #         st.markdown(f"""
# #             <div class="stat-card">
# #                 <div class="stat-number">₹{approved_amount:,.0f}</div>
# #                 <div class="stat-label">Loan Amount</div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     with col4:
# #         st.markdown(f"""
# #             <div class="stat-card">
# #                 <div class="stat-number">{tenure}</div>
# #                 <div class="stat-label">Tenure (Months)</div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     with col5:
# #         st.markdown(f"""
# #             <div class="stat-card">
# #                 <div class="stat-number">{decision_data['confidence']:.0f}%</div>
# #                 <div class="stat-label">Confidence</div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     # Application info
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.markdown(f"""
# #             <div class="info-box">
# #                 <strong>📋 Application ID:</strong> {app_id}
# #             </div>
# #         """, unsafe_allow_html=True)
# #     with col2:
# #         st.markdown(f"""
# #             <div class="info-box">
# #                 <strong>🕐 Decision Timestamp:</strong> {timestamp}
# #             </div>
# #         """, unsafe_allow_html=True)


# # def render_info_card(title, icon, data_dict, status_dict=None):
# #     """Render modern info card with data"""
    
# #     st.markdown(f"""
# #         <div class="info-card">
# #             <div class="info-card-title">
# #                 <span class="icon">{icon}</span>
# #                 <span>{title}</span>
# #             </div>
# #             <div class="info-card-content">
# #     """, unsafe_allow_html=True)
    
# #     for label, value in data_dict.items():
# #         status = ""
# #         if status_dict and label in status_dict:
# #             if status_dict[label] == "pass":
# #                 status = '<span class="status-badge badge-pass">✓ Passed</span>'
# #             elif status_dict[label] == "fail":
# #                 status = '<span class="status-badge badge-fail">✗ Failed</span>'
# #             elif status_dict[label] == "warning":
# #                 status = '<span class="status-badge badge-warning">⚠ Warning</span>'
        
# #         st.markdown(f"""
# #             <div class="data-row">
# #                 <span class="data-label">{label}</span>
# #                 <span class="data-value">{value} {status}</span>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     st.markdown("""
# #             </div>
# #         </div>
# #     """, unsafe_allow_html=True)


# # def render_reason_codes(reasons):
# #     """Render reason codes in modern style"""
    
# #     st.markdown("""
# #         <div class="info-card">
# #             <div class="info-card-title">
# #                 <span class="icon">📝</span>
# #                 <span>Decision Reasons</span>
# #             </div>
# #             <div class="info-card-content">
# #     """, unsafe_allow_html=True)
    
# #     for i, reason in enumerate(reasons, 1):
# #         st.markdown(f"""
# #             <div class="reason-item">
# #                 <span class="reason-icon">{i}.</span>
# #                 <span>{reason}</span>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     st.markdown("""
# #             </div>
# #         </div>
# #     """, unsafe_allow_html=True)


# # def create_modern_gauge(value, title, max_value=100):
# #     """Create modern gauge chart"""
    
# #     if value <= 50:
# #         color = "#f56565"
# #     elif value <= 75:
# #         color = "#ed8936"
# #     else:
# #         color = "#48bb78"
    
# #     fig = go.Figure(go.Indicator(
# #         mode="gauge+number",
# #         value=value,
# #         title={'text': title, 'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}},
# #         number={'suffix': "%", 'font': {'size': 36, 'color': '#2d3748', 'family': 'Inter'}},
# #         gauge={
# #             'axis': {'range': [0, max_value], 'tickfont': {'size': 12, 'color': '#718096'}},
# #             'bar': {'color': color, 'thickness': 0.75},
# #             'bgcolor': 'white',
# #             'borderwidth': 0,
# #             'steps': [
# #                 {'range': [0, 50], 'color': '#fed7d7'},
# #                 {'range': [50, 75], 'color': '#feebc8'},
# #                 {'range': [75, 100], 'color': '#c6f6d5'}
# #             ],
# #         }
# #     ))
    
# #     fig.update_layout(
# #         height=250,
# #         margin=dict(l=20, r=20, t=50, b=20),
# #         paper_bgcolor='white',
# #         font={'family': 'Inter', 'color': '#2d3748'}
# #     )
    
# #     return fig


# # def create_modern_bar_chart(class_probs):
# #     """Create modern probability bar chart"""
    
# #     df = pd.DataFrame({
# #         'Decision': list(class_probs.keys()),
# #         'Probability': list(class_probs.values())
# #     })
    
# #     colors = {'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}
    
# #     fig = px.bar(
# #         df, 
# #         x='Decision', 
# #         y='Probability',
# #         title='Decision Probabilities',
# #         color='Decision',
# #         color_discrete_map=colors,
# #         text='Probability'
# #     )
    
# #     fig.update_traces(
# #         texttemplate='%{text:.1f}%',
# #         textposition='outside',
# #         marker_line_width=0
# #     )
    
# #     fig.update_layout(
# #         showlegend=False,
# #         yaxis_title='Probability (%)',
# #         xaxis_title='',
# #         height=300,
# #         margin=dict(l=20, r=20, t=50, b=20),
# #         paper_bgcolor='white',
# #         plot_bgcolor='white',
# #         font={'family': 'Inter', 'color': '#2d3748'},
# #         yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]},
# #         xaxis={'tickfont': {'size': 14, 'color': '#2d3748'}}
# #     )
    
# #     return fig

# # # =============================================================================
# # # SIDEBAR
# # # =============================================================================

# # with st.sidebar:
# #     st.markdown("# 🏦 Credit Risk Engine")
# #     st.markdown("---")
    
# #     page = st.radio(
# #         "**Navigation**",
# #         ["🏠 Home", "👤 Assessment", "📊 Batch Process", "📈 Model Info", "ℹ️ About"],
# #         label_visibility="collapsed"
# #     )
    
# #     st.markdown("---")
    
# #     st.markdown(f"""
# #         <div class="info-card">
# #             <div class="info-card-title">System Status</div>
# #             <div class="info-card-content">
# #                 <div class="data-row">
# #                     <span class="data-label">Model</span>
# #                     <span class="data-value">✅ Loaded</span>
# #                 </div>
# #                 <div class="data-row">
# #                     <span class="data-label">Version</span>
# #                     <span class="data-value">8.0</span>
# #                 </div>
# #                 <div class="data-row">
# #                     <span class="data-label">Features</span>
# #                     <span class="data-value">{len(TOP_FEATURES)}</span>
# #                 </div>
# #                 <div class="data-row">
# #                     <span class="data-label">Type</span>
# #                     <span class="data-value">Random Forest</span>
# #                 </div>
# #             </div>
# #         </div>
# #     """, unsafe_allow_html=True)
    
# #     with st.expander("🎯 **Top Features**"):
# #         for i, feat in enumerate(TOP_FEATURES[:5], 1):
# #             st.markdown(f"`{i}.` {feat}")

# # # =============================================================================
# # # HOME PAGE
# # # =============================================================================

# # if page == "🏠 Home":
# #     st.markdown('<p class="main-header">Credit Risk Assessment Platform</p>', unsafe_allow_html=True)
    
# #     st.markdown("""
# #         <div class="info-box">
# #             <h3 style="margin-top: 0;">🎯 AI-Powered Lending Decisions</h3>
# #             <p style="margin-bottom: 0;">
# #                 Comprehensive credit risk evaluation combining hard policy rules, 
# #                 machine learning models, and affordability analysis for accurate lending decisions.
# #             </p>
# #         </div>
# #     """, unsafe_allow_html=True)
    
# #     st.markdown("<br>", unsafe_allow_html=True)
    
# #     # Feature cards
# #     col1, col2, col3 = st.columns(3)
    
# #     with col1:
# #         st.markdown("""
# #             <div class="info-card">
# #                 <div class="info-card-title">
# #                     <span class="icon">🛡️</span>
# #                     <span>Policy Gates</span>
# #                 </div>
# #                 <div class="info-card-content">
# #                     <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
# #                         <li>Age & KYC verification</li>
# #                         <li>Employment stability</li>
# #                         <li>Minimum income checks</li>
# #                         <li>Credit bureau thresholds</li>
# #                         <li>Bankruptcy & fraud detection</li>
# #                     </ul>
# #                 </div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     with col2:
# #         st.markdown("""
# #             <div class="info-card">
# #                 <div class="info-card-title">
# #                     <span class="icon">🤖</span>
# #                     <span>ML Assessment</span>
# #                 </div>
# #                 <div class="info-card-content">
# #                     <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
# #                         <li>Random Forest classifier</li>
# #                         <li>60K+ training samples</li>
# #                         <li>Confidence scoring</li>
# #                         <li>Multi-class prediction</li>
# #                         <li>Feature importance</li>
# #                     </ul>
# #                 </div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     with col3:
# #         st.markdown("""
# #             <div class="info-card">
# #                 <div class="info-card-title">
# #                     <span class="icon">💰</span>
# #                     <span>Affordability</span>
# #                 </div>
# #                 <div class="info-card-content">
# #                     <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
# #                         <li>EMI calculation</li>
# #                         <li>FOIR analysis (max 50%)</li>
# #                         <li>Net disposable income</li>
# #                         <li>Debt burden assessment</li>
# #                         <li>Affordability scoring</li>
# #                     </ul>
# #                 </div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     st.markdown("<br>", unsafe_allow_html=True)
    
# #     # Stats
# #     col1, col2, col3, col4 = st.columns(4)
    
# #     with col1:
# #         st.metric("🎯 Accuracy", "85%", "+2%")
    
# #     with col2:
# #         st.metric("⚡ Avg Response", "1.2s", "-0.3s")
    
# #     with col3:
# #         st.metric("📊 Features", len(TOP_FEATURES))
    
# #     with col4:
# #         st.metric("🔄 Version", "8.0", "Latest")
    
# #     st.markdown("<br>", unsafe_allow_html=True)
    
# #     st.markdown("""
# #         <div class="warning-box">
# #             <strong>🆕 New in Version 8.0:</strong><br>
# #             • Sage Green & Yellow Professional Theme<br>
# #             • Enhanced visual hierarchy and readability<br>
# #             • Improved decision summary cards<br>
# #             • Modern charts and gauges<br>
# #             • Responsive layout optimization
# #         </div>
# #     """, unsafe_allow_html=True)

# # # =============================================================================
# # # ASSESSMENT PAGE
# # # =============================================================================

# # elif page == "👤 Assessment":
# #     st.markdown('<p class="main-header">Credit Assessment</p>', unsafe_allow_html=True)
    
# #     st.markdown("""
# #         <div class="info-box">
# #             💡 Complete the form below to assess credit risk. All fields are required for accurate evaluation.
# #         </div>
# #     """, unsafe_allow_html=True)
    
# #     with st.form("assessment_form"):
# #         # Identity & Eligibility
# #         st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
# #         col1, col2, col3 = st.columns(3)
        
# #         with col1:
# #             age = st.number_input("Age", 18, 80, 35, help="Customer's age in years")
# #             employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business'])
        
# #         with col2:
# #             kyc_verified = st.selectbox("KYC Verified", ['Yes', 'No']) == 'Yes'
# #             bankruptcy_flag = st.selectbox("Bankruptcy Flag", ['No', 'Yes']) == 'Yes'
        
# #         with col3:
# #             fraud_flag = st.selectbox("Fraud Flag", ['No', 'Yes']) == 'Yes'
# #             if employment_type == 'Salaried':
# #                 employment_tenure = st.number_input("Employment Tenure (months)", 0, 600, 24)
# #                 business_vintage = 0
# #             else:
# #                 business_vintage = st.number_input("Business Vintage (years)", 0, 50, 3)
# #                 employment_tenure = 0
        
# #         # Credit Bureau
# #         st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
# #         col1, col2, col3 = st.columns(3)
        
# #         with col1:
# #             bureau_score = st.number_input("Bureau Score", 300, 900, 720, 10)
# #             dpd_90_6m = st.number_input("DPD 90+ (Last 6M)", 0, 20, 0)
        
# #         with col2:
# #             credit_utilization = st.number_input("Credit Utilization (%)", 0, 100, 30)
# #             recent_inquiries = st.number_input("Recent Inquiries (3M)", 0, 20, 2)
        
# #         with col3:
# #             active_loans = st.number_input("Active Loans", 0, 10, 1)
# #             existing_emi = st.number_input("Existing Total EMI (₹)", 0, 200000, 15000, 1000)
        
# #         # Income & Financial
# #         st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
# #         col1, col2, col3, col4 = st.columns(4)
        
# #         with col1:
# #             avg_salary = st.number_input("Monthly Income (₹)", 0, 1000000, 50000, 5000)
# #             amt_income = st.number_input("Annual Income (₹)", 0, 10000000, 600000, 10000)
        
# #         with col2:
# #             net_surplus = st.number_input("Net Cash Surplus (₹)", -100000, 500000, 20000, 5000)
# #             salary_stability = st.selectbox("Salary Stability", ['STABLE', 'MODERATE', 'UNSTABLE'])
        
# #         with col3:
# #             loan_amount = st.number_input("Loan Amount (₹)", 0, 5000000, 180000, 10000)
# #             loan_tenure = st.number_input("Tenure (months)", 3, 360, 24)
        
# #         with col4:
# #             interest_rate = st.number_input("Interest Rate (%)", 8.0, 20.0, 10.5, 0.5)
# #             amt_annuity = st.number_input("Requested EMI (₹)", 0, 200000, 8500, 500)
        
# #         st.markdown("<br>", unsafe_allow_html=True)
# #         submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)
    
# #     if submitted:
# #         # Generate application ID
# #         timestamp = datetime.now()
# #         app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
        
# #         # Prepare data
# #         customer_data = {
# #             'age': age,
# #             'employment_type': employment_type,
# #             'kyc_verified': kyc_verified,
# #             'bankruptcy_flag': bankruptcy_flag,
# #             'fraud_flag': fraud_flag,
# #             'employment_tenure_months': employment_tenure,
# #             'business_vintage_years': business_vintage,
# #             'bureau_score': bureau_score,
# #             'dpd_90_count_6m': dpd_90_6m,
# #             'credit_utilization_pct': credit_utilization,
# #             'recent_inquiries_3m': recent_inquiries,
# #             'active_loans_count': active_loans,
# #             'avg_salary_6m': avg_salary,
# #             'AMT_INCOME_TOTAL': amt_income,
# #             'net_cash_surplus_6m': net_surplus,
# #             'salary_stability_flag': salary_stability,
# #             'loan_amount': loan_amount,
# #             'loan_tenure_months': loan_tenure,
# #             'interest_rate': interest_rate,
# #             'existing_emi': existing_emi,
# #             'AMT_ANNUITY': amt_annuity,
# #             'application_id': app_id,
# #             'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S")
# #         }
        
# #         # Get decision
# #         with st.spinner("🔄 Processing assessment..."):
# #             decision_data = make_hybrid_decision_enhanced(customer_data)
        
# #         # Generate reasons
# #         reasons = generate_reason_codes(
# #             decision=decision_data['decision'],
# #             customer_data=customer_data,
# #             affordability_data=decision_data.get('affordability_data', {}),
# #             policy_checks=decision_data['policy_checks']
# #         )
        
# #         customer_data['reason_codes'] = reasons
        
# #         # Tabs
# #         tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])
        
# #         with tab1:
# #             st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
            
# #             col1, col2 = st.columns(2)
            
# #             with col1:
# #                 render_info_card(
# #                     "👤 Identity", 
# #                     "👤",
# #                     {
# #                         "Age": age,
# #                         "Employment": employment_type,
# #                         "KYC Status": "Verified" if kyc_verified else "Not Verified",
# #                         "Tenure": f"{employment_tenure} months" if employment_type == 'Salaried' else f"{business_vintage} years"
# #                     }
# #                 )
                
# #                 render_info_card(
# #                     "💰 Financial", 
# #                     "💰",
# #                     {
# #                         "Monthly Income": f"₹{avg_salary:,}",
# #                         "Annual Income": f"₹{amt_income:,}",
# #                         "Net Surplus": f"₹{net_surplus:,}",
# #                         "Stability": salary_stability
# #                     }
# #                 )
            
# #             with col2:
# #                 render_info_card(
# #                     "🏦 Credit Bureau", 
# #                     "🏦",
# #                     {
# #                         "Bureau Score": bureau_score,
# #                         "DPD 90+": dpd_90_6m,
# #                         "Utilization": f"{credit_utilization}%",
# #                         "Recent Inquiries": recent_inquiries,
# #                         "Existing EMI": f"₹{existing_emi:,}"
# #                     }
# #                 )
                
# #                 render_info_card(
# #                     "📋 Loan Request", 
# #                     "📋",
# #                     {
# #                         "Amount": f"₹{loan_amount:,}",
# #                         "Tenure": f"{loan_tenure} months",
# #                         "Interest Rate": f"{interest_rate}%",
# #                         "Requested EMI": f"₹{amt_annuity:,}"
# #                     }
# #                 )
        
# #         with tab2:
# #             st.markdown('<p class="section-header">Decision Summary</p>', unsafe_allow_html=True)
            
# #             render_decision_header(decision_data, customer_data)
            
# #             st.markdown("<br>", unsafe_allow_html=True)
            
# #             col1, col2, col3 = st.columns(3)
            
# #             with col1:
# #                 # Identity card
# #                 age_pass = 18 <= age <= 65
# #                 kyc_pass = kyc_verified
                
# #                 render_info_card(
# #                     "Identity & Eligibility",
# #                     "👤",
# #                     {
# #                         f"Age: {age}": "",
# #                         f"Employment: {employment_type}": "",
# #                         f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""
# #                     },
# #                     {
# #                         f"Age: {age}": "pass" if age_pass else "fail",
# #                         f"Employment: {employment_type}": "pass",
# #                         f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_pass else "fail"
# #                     }
# #                 )
            
# #             with col2:
# #                 # Credit card
# #                 bureau_pass = bureau_score >= 550
# #                 dpd_pass = dpd_90_6m == 0
                
# #                 render_info_card(
# #                     "Credit Bureau",
# #                     "🏦",
# #                     {
# #                         f"Bureau Score: {bureau_score}": "",
# #                         f"DPD 90+: {dpd_90_6m}": "",
# #                         f"Utilization: {credit_utilization}%": ""
# #                     },
# #                     {
# #                         f"Bureau Score: {bureau_score}": "pass" if bureau_pass else "fail",
# #                         f"DPD 90+: {dpd_90_6m}": "pass" if dpd_pass else "fail",
# #                         f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"
# #                     }
# #                 )
            
# #             with col3:
# #                 # Affordability card
# #                 affordability = decision_data.get('affordability_data', {})
# #                 foir = affordability.get('foir_percentage', 0)
# #                 total_emi = affordability.get('total_emi', 0)
# #                 net_disp = affordability.get('net_disposable', 0)
                
# #                 render_info_card(
# #                     "Affordability",
# #                     "💰",
# #                     {
# #                         f"Monthly Income: ₹{avg_salary:,}": "",
# #                         f"FOIR: {foir:.1f}%": "",
# #                         f"Total EMI: ₹{total_emi:,}": "",
# #                         f"Net Disposable: ₹{net_disp:,}": ""
# #                     },
# #                     {
# #                         f"Monthly Income: ₹{avg_salary:,}": "pass",
# #                         f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
# #                         f"Total EMI: ₹{total_emi:,}": "pass",
# #                         f"Net Disposable: ₹{net_disp:,}": "pass" if net_disp >= 10000 else "warning"
# #                     }
# #                 )
            
# #             st.markdown("<br>", unsafe_allow_html=True)
            
# #             # Reason codes
# #             render_reason_codes(reasons)
            
# #             st.markdown("<br>", unsafe_allow_html=True)
            
# #             # Action buttons
# #             col1, col2, col3 = st.columns([1, 1, 2])
# #             with col1:
# #                 if st.button("📥 Download Report", use_container_width=True):
# #                     st.info("📄 Report generation coming soon...")
# #             with col2:
# #                 if st.button("🔄 Re-Evaluate", use_container_width=True):
# #                     st.rerun()
        
# #         with tab3:
# #             st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
            
# #             col1, col2 = st.columns(2)
            
# #             with col1:
# #                 fig1 = create_modern_gauge(decision_data['confidence'], "Model Confidence")
# #                 st.plotly_chart(fig1, use_container_width=True)
            
# #             with col2:
# #                 fig2 = create_modern_bar_chart(decision_data['class_probs'])
# #                 st.plotly_chart(fig2, use_container_width=True)
            
# #             st.markdown("<br>", unsafe_allow_html=True)
            
# #             # Policy checks
# #             st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
            
# #             policy_df = pd.DataFrame([
# #                 {'Check': k, 'Result': v} 
# #                 for k, v in decision_data['policy_checks'].items()
# #             ])
# #             st.dataframe(policy_df, use_container_width=True, hide_index=True)
        
# #         with tab4:
# #             st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
            
# #             audit_log = {
# #                 'application_id': app_id,
# #                 'timestamp': timestamp.isoformat(),
# #                 'decision': decision_data['decision'],
# #                 'risk_score': decision_data['risk_score'],
# #                 'pd_percentage': decision_data['pd_percentage'],
# #                 'confidence': round(decision_data['confidence'], 2),
# #                 'model_version': '8.0',
# #                 'reason_codes': reasons,
# #                 'affordability': decision_data.get('affordability_data', {})
# #             }
            
# #             st.json(audit_log)
            
# #             import json
# #             audit_json = json.dumps(audit_log, indent=2)
# #             st.download_button(
# #                 "📥 Download Audit Log",
# #                 audit_json,
# #                 f"audit_{app_id}.json",
# #                 "application/json",
# #                 use_container_width=True
# #             )

# # # =============================================================================
# # # BATCH PROCESSING PAGE
# # # =============================================================================

# # elif page == "📊 Batch Process":
# #     st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
    
# #     st.markdown("""
# #         <div class="info-box">
# #             📤 Upload a CSV file with customer data for bulk credit assessment. 
# #             The file should include all required fields for prediction.
# #         </div>
# #     """, unsafe_allow_html=True)
    
# #     # File upload
# #     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
# #     if uploaded_file is not None:
# #         try:
# #             # Read the CSV file
# #             df = pd.read_csv(uploaded_file)
            
# #             st.success(f"✅ Successfully loaded {len(df)} records")
            
# #             # Show preview
# #             with st.expander("📄 Preview Uploaded Data"):
# #                 st.dataframe(df.head(), use_container_width=True)
# #                 st.write(f"**Total Records:** {len(df)}")
# #                 st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            
# #             # Required columns check
# #             required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
# #             missing_cols = [col for col in required_cols if col not in df.columns]
            
# #             if missing_cols:
# #                 st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
# #                 st.info("Please ensure your CSV includes at least these columns: age, employment_type, avg_salary_6m, bureau_score, loan_amount")
# #             else:
# #                 # Process batch predictions
# #                 if st.button("🚀 Process Batch Predictions", use_container_width=True, type="primary"):
# #                     with st.spinner(f"🔍 Processing {len(df)} records..."):
# #                         progress_bar = st.progress(0)
                        
# #                         # Process batch
# #                         results_df = process_batch_predictions(df)
                        
# #                         progress_bar.progress(100)
                        
# #                         st.success(f"✅ Completed processing {len(results_df)} records!")
                        
# #                         # Show results
# #                         tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
                        
# #                         with tab1:
# #                             st.dataframe(results_df, use_container_width=True)
                            
# #                             # Summary statistics
# #                             col1, col2, col3, col4 = st.columns(4)
# #                             with col1:
# #                                 approved_count = len(results_df[results_df['decision'] == 'APPROVE'])
# #                                 st.metric("✅ Approved", approved_count)
# #                             with col2:
# #                                 rejected_count = len(results_df[results_df['decision'] == 'REJECT'])
# #                                 st.metric("❌ Rejected", rejected_count)
# #                             with col3:
# #                                 review_count = len(results_df[results_df['decision'] == 'REVIEW'])
# #                                 st.metric("⚠️ Review", review_count)
# #                             with col4:
# #                                 avg_risk = results_df['risk_score'].mean()
# #                                 st.metric("📊 Avg Risk Score", f"{avg_risk:.0f}")
                        
# #                         with tab2:
# #                             # Visualizations
# #                             col1, col2 = st.columns(2)
                            
# #                             with col1:
# #                                 # Decision distribution
# #                                 decision_counts = results_df['decision'].value_counts()
# #                                 fig1 = px.pie(
# #                                     values=decision_counts.values,
# #                                     names=decision_counts.index,
# #                                     title="Decision Distribution",
# #                                     color=decision_counts.index,
# #                                     color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}
# #                                 )
# #                                 st.plotly_chart(fig1, use_container_width=True)
                            
# #                             with col2:
# #                                 # Risk score distribution
# #                                 fig2 = px.histogram(
# #                                     results_df,
# #                                     x='risk_score',
# #                                     title="Risk Score Distribution",
# #                                     nbins=20,
# #                                     color_discrete_sequence=['#587042']
# #                                 )
# #                                 st.plotly_chart(fig2, use_container_width=True)
                            
# #                             # FOIR analysis
# #                             fig3 = px.scatter(
# #                                 results_df,
# #                                 x='monthly_income',
# #                                 y='loan_amount',
# #                                 color='decision',
# #                                 size='risk_score',
# #                                 title="Income vs Loan Amount (Colored by Decision)",
# #                                 hover_data=['application_id', 'foir_percentage'],
# #                                 color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}
# #                             )
# #                             st.plotly_chart(fig3, use_container_width=True)
                        
# #                         with tab3:
# #                             st.markdown("### Download Results")
                            
# #                             # Download options
# #                             col1, col2 = st.columns(2)
                            
# #                             with col1:
# #                                 # Download CSV
# #                                 csv = results_df.to_csv(index=False)
# #                                 st.download_button(
# #                                     label="📥 Download as CSV",
# #                                     data=csv,
# #                                     file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# #                                     mime="text/csv",
# #                                     use_container_width=True
# #                                 )
                            
# #                             with col2:
# #                                 # Download JSON
# #                                 json_data = results_df.to_json(orient='records', indent=2)
# #                                 st.download_button(
# #                                     label="📥 Download as JSON",
# #                                     data=json_data,
# #                                     file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
# #                                     mime="application/json",
# #                                     use_container_width=True
# #                                 )
                            
# #                             # Filtered downloads
# #                             st.markdown("---")
# #                             st.markdown("#### Filtered Downloads")
                            
# #                             col1, col2, col3 = st.columns(3)
                            
# #                             with col1:
# #                                 approved_df = results_df[results_df['decision'] == 'APPROVE']
# #                                 if len(approved_df) > 0:
# #                                     st.download_button(
# #                                         label=f"✅ Approved Only ({len(approved_df)})",
# #                                         data=approved_df.to_csv(index=False),
# #                                         file_name=f"approved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# #                                         mime="text/csv",
# #                                         use_container_width=True
# #                                     )
                            
# #                             with col2:
# #                                 rejected_df = results_df[results_df['decision'] == 'REJECT']
# #                                 if len(rejected_df) > 0:
# #                                     st.download_button(
# #                                         label=f"❌ Rejected Only ({len(rejected_df)})",
# #                                         data=rejected_df.to_csv(index=False),
# #                                         file_name=f"rejected_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# #                                         mime="text/csv",
# #                                         use_container_width=True
# #                                     )
                            
# #                             with col3:
# #                                 review_df = results_df[results_df['decision'] == 'REVIEW']
# #                                 if len(review_df) > 0:
# #                                     st.download_button(
# #                                         label=f"⚠️ Review Only ({len(review_df)})",
# #                                         data=review_df.to_csv(index=False),
# #                                         file_name=f"review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# #                                         mime="text/csv",
# #                                         use_container_width=True
# #                                     )
            
# #         except Exception as e:
# #             st.error(f"❌ Error processing file: {str(e)}")
# #             st.info("Please ensure the CSV file is properly formatted and contains the required columns.")
    
# #     else:
# #         # Show template download
# #         st.markdown("---")
# #         st.markdown("### 📋 CSV Template")
        
# #         # Create template dataframe
# #         template_data = {
# #             'age': [35, 42, 28],
# #             'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
# #             'kyc_verified': ['Yes', 'Yes', 'No'],
# #             'bankruptcy_flag': ['No', 'No', 'No'],
# #             'fraud_flag': ['No', 'No', 'No'],
# #             'employment_tenure_months': [24, 0, 18],
# #             'business_vintage_years': [0, 5, 0],
# #             'bureau_score': [720, 680, 580],
# #             'dpd_90_count_6m': [0, 1, 2],
# #             'credit_utilization_pct': [30, 45, 75],
# #             'recent_inquiries_3m': [2, 1, 5],
# #             'active_loans_count': [1, 2, 3],
# #             'avg_salary_6m': [50000, 75000, 35000],
# #             'AMT_INCOME_TOTAL': [600000, 900000, 420000],
# #             'net_cash_surplus_6m': [20000, 35000, 10000],
# #             'salary_stability_flag': ['STABLE', 'MODERATE', 'UNSTABLE'],
# #             'loan_amount': [180000, 250000, 100000],
# #             'loan_tenure_months': [24, 36, 12],
# #             'interest_rate': [10.5, 11.0, 12.0],
# #             'existing_emi': [15000, 20000, 8000],
# #             'AMT_ANNUITY': [8500, 9500, 4500]
# #         }
        
# #         template_df = pd.DataFrame(template_data)
        
# #         st.dataframe(template_df, use_container_width=True)
        
# #         # Download template
# #         csv_template = template_df.to_csv(index=False)
# #         st.download_button(
# #             label="📥 Download CSV Template",
# #             data=csv_template,
# #             file_name="credit_assessment_template.csv",
# #             mime="text/csv",
# #             use_container_width=True
# #         )

# # # =============================================================================
# # # MODEL INFO PAGE
# # # =============================================================================

# # elif page == "📈 Model Info":
# #     st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
    
# #     col1, col2, col3 = st.columns(3)
    
# #     with col1:
# #         st.markdown("""
# #             <div class="stat-card">
# #                 <div class="stat-number">RF</div>
# #                 <div class="stat-label">Model Type</div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     with col2:
# #         st.markdown(f"""
# #             <div class="stat-card">
# #                 <div class="stat-number">{len(TOP_FEATURES)}</div>
# #                 <div class="stat-label">Features</div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     with col3:
# #         st.markdown(f"""
# #             <div class="stat-card">
# #                 <div class="stat-number">{len(TARGET_LE.classes_)}</div>
# #                 <div class="stat-label">Classes</div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     st.markdown("<br>", unsafe_allow_html=True)
    
# #     st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
    
# #     feature_df = pd.DataFrame({
# #         'Rank': range(1, min(21, len(TOP_FEATURES) + 1)),
# #         'Feature': TOP_FEATURES[:20]
# #     })
    
# #     st.dataframe(feature_df, use_container_width=True, hide_index=True)

# # # =============================================================================
# # # ABOUT PAGE
# # # =============================================================================

# # elif page == "ℹ️ About":
# #     st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
    
# #     st.markdown("""
# #         <div class="info-card">
# #             <div class="info-card-title">
# #                 <span class="icon">🏦</span>
# #                 <span>Credit Risk Assessment Platform</span>
# #             </div>
# #             <div class="info-card-content">
# #                 <p><strong>Version:</strong> 8.0 - Sage Green & Yellow Theme</p>
# #                 <p><strong>Developer:</strong> Zen Meraki</p>
# #                 <p><strong>Date:</strong> January 2025</p>
# #                 <br>
# #                 <p>
# #                     A comprehensive credit risk evaluation system combining hard policy rules,
# #                     machine learning models, and affordability analysis for accurate and compliant
# #                     lending decisions.
# #                 </p>
# #             </div>
# #         </div>
# #     """, unsafe_allow_html=True)
    
# #     st.markdown("<br>", unsafe_allow_html=True)
    
# #     col1, col2 = st.columns(2)
    
# #     with col1:
# #         st.markdown("""
# #             <div class="info-card">
# #                 <div class="info-card-title">
# #                     <span class="icon">🎯</span>
# #                     <span>Key Features</span>
# #                 </div>
# #                 <div class="info-card-content">
# #                     <ul style="margin: 0; padding-left: 1.25rem;">
# #                         <li>Three-layer decision engine</li>
# #                         <li>Real-time risk assessment</li>
# #                         <li>FOIR calculation & validation</li>
# #                         <li>Automated reason generation</li>
# #                         <li>Complete audit trail</li>
# #                         <li>Professional UI/UX</li>
# #                     </ul>
# #                 </div>
# #             </div>
# #         """, unsafe_allow_html=True)
    
# #     with col2:
# #         st.markdown("""
# #             <div class="info-card">
# #                 <div class="info-card-title">
# #                     <span class="icon">🛠️</span>
# #                     <span>Technology Stack</span>
# #                 </div>
# #                 <div class="info-card-content">
# #                     <ul style="margin: 0; padding-left: 1.25rem;">
# #                         <li>Streamlit (UI Framework)</li>
# #                         <li>Scikit-learn (ML)</li>
# #                         <li>Plotly (Visualizations)</li>
# #                         <li>Pandas (Data Processing)</li>
# #                         <li>Python 3.8+</li>
# #                     </ul>
# #                 </div>
# #             </div>
# #         """, unsafe_allow_html=True)

# # st.markdown("---")
# # st.markdown("<div style='text-align: center; color: gray;'><p>© 2025 Hybrid Credit Risk System v8.0 | Zen Meraki</p></div>", 
# #     unsafe_allow_html=True)





# #fixed 



# """
# Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
# Enhanced with Modern UI/UX Design
# Run with: streamlit run test.py

# Author: Zen Meraki  
# Date: January 2026
# VERSION: 8.0 - Sage Green & Yellow Professional Interface
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import joblib
# import warnings
# from datetime import datetime
# import hashlib
# import io
# import base64
# from typing import Dict, List, Any
# import json
# warnings.filterwarnings('ignore')
# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parents[1]))

# from utils.pdf_generator import generate_decision_pdf


# # =============================================================================
# # PAGE CONFIGURATION
# # =============================================================================

# st.set_page_config(
#     page_title="Credit Risk Assessment",
#     page_icon="💳",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # =============================================================================
# # SAGE GREEN AND YELLOW THEME CSS
# # =============================================================================

# st.markdown("""
#     <style>
#     /* Import Google Fonts */
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
#     /* Global Styles */
#     * {
#         font-family: 'Inter', sans-serif;
#     }
    
#     /* Color Variables */
#     :root {
#         --fern-green: #587042;
#         --sage: #A9B494;
#         --cosmic-latte: #FAF7E6;
#         --jasmine: #F8DE8C;
#         --saffron: #F6C531;
#         --dark-fern: #486032;
#         --light-sage: #D4DBC4;
#     }
    
#     /* Main Background */
#     .main {
#         background-color: #FFFFFF;
#     }
    
#     .block-container {
#         padding-top: 2rem;
#         padding-bottom: 2rem;
#         max-width: 1400px;
#         background-color: #FFFFFF;
#     }
    
#     /* Headers */
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: var(--fern-green);
#         text-align: center;
#         padding: 1.5rem 0;
#         margin-bottom: 1rem;
#         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#     }
    
#     .section-header {
#         font-size: 1.5rem;
#         font-weight: 600;
#         color: var(--fern-green);
#         margin-top: 2rem;
#         margin-bottom: 1rem;
#         padding-bottom: 0.5rem;
#         border-bottom: 2px solid var(--sage);
#     }
    
#     /* Decision Cards */
#     .decision-card {
#         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
#         padding: 2rem;
#         border-radius: 16px;
#         box-shadow: 0 10px 40px rgba(88, 112, 66, 0.2);
#         margin-bottom: 2rem;
#         color: white;
#     }
    
#     .decision-card-approved {
#         background: linear-gradient(135deg, var(--fern-green) 0%, #7A9E4D 100%);
#         box-shadow: 0 10px 40px rgba(88, 112, 66, 0.3);
#     }
    
#     .decision-card-rejected {
#         background: linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%);
#         box-shadow: 0 10px 40px rgba(211, 47, 47, 0.2);
#     }
    
#     .decision-card-review {
#         background: linear-gradient(135deg, var(--saffron) 0%, var(--jasmine) 100%);
#         box-shadow: 0 10px 40px rgba(246, 197, 49, 0.3);
#     }
    
#     .decision-title {
#         font-size: 2.5rem;
#         font-weight: 700;
#         margin: 0;
#         color: white;
#         display: flex;
#         align-items: center;
#         gap: 1rem;
#     }
    
#     .decision-subtitle {
#         font-size: 1.1rem;
#         margin-top: 0.5rem;
#         opacity: 0.9;
#     }
    
#     /* Info Cards */
#     .info-card {
#         background: white;
#         border-radius: 12px;
#         padding: 1.5rem;
#         box-shadow: 0 4px 12px rgba(88, 112, 66, 0.08);
#         border: 1px solid var(--sage);
#         margin-bottom: 1rem;
#         transition: all 0.3s ease;
#     }
    
#     .info-card:hover {
#         box-shadow: 0 10px 25px rgba(88, 112, 66, 0.15);
#         transform: translateY(-2px);
#         border-color: var(--fern-green);
#     }
    
#     .info-card-title {
#         font-size: 1.1rem;
#         font-weight: 600;
#         color: var(--fern-green);
#         margin-bottom: 1rem;
#         display: flex;
#         align-items: center;
#         gap: 0.5rem;
#     }
    
#     .info-card-content {
#         color: #5A5A5A;
#         line-height: 1.6;
#     }
    
#     /* Metric Cards */
#     .metric-card {
#         background: linear-gradient(135deg, var(--cosmic-latte) 0%, #F5F2E0 100%);
#         border-radius: 12px;
#         padding: 1.5rem;
#         border-left: 4px solid var(--fern-green);
#         margin-bottom: 1rem;
#     }
    
#     .metric-label {
#         font-size: 0.875rem;
#         font-weight: 500;
#         color: var(--fern-green);
#         text-transform: uppercase;
#         letter-spacing: 0.05em;
#         margin-bottom: 0.5rem;
#     }
    
#     .metric-value {
#         font-size: 2rem;
#         font-weight: 700;
#         color: var(--fern-green);
#     }
    
#     /* Status Badges */
#     .status-badge {
#         display: inline-flex;
#         align-items: center;
#         padding: 0.5rem 1rem;
#         border-radius: 20px;
#         font-weight: 600;
#         font-size: 0.875rem;
#         gap: 0.5rem;
#     }
    
#     .badge-pass {
#         background: #E8F5E9;
#         color: var(--fern-green);
#         border: 1px solid var(--sage);
#     }
    
#     .badge-fail {
#         background: #FFEBEE;
#         color: #D32F2F;
#         border: 1px solid #FFCDD2;
#     }
    
#     .badge-warning {
#         background: linear-gradient(135deg, var(--cosmic-latte) 0%, var(--jasmine) 100%);
#         color: #7B5800;
#         border: 1px solid var(--jasmine);
#     }
    
#     .badge-info {
#         background: #E3F2FD;
#         color: #1565C0;
#         border: 1px solid #90CAF9;
#     }
    
#     /* Data Row */
#     .data-row {
#         display: flex;
#         justify-content: space-between;
#         align-items: center;
#         padding: 0.75rem 0;
#         border-bottom: 1px solid var(--sage);
#     }
    
#     .data-row:last-child {
#         border-bottom: none;
#     }
    
#     .data-label {
#         font-weight: 500;
#         color: #5A5A5A;
#     }
    
#     .data-value {
#         font-weight: 600;
#         color: var(--fern-green);
#     }
    
#     /* Reason Items */
#     .reason-item {
#         background: linear-gradient(135deg, #F9F7EB 0%, #F5F2E0 100%);
#         padding: 1rem 1.25rem;
#         border-radius: 8px;
#         border-left: 4px solid var(--saffron);
#         margin-bottom: 0.75rem;
#         color: #7B5800;
#         font-weight: 500;
#         display: flex;
#         align-items: center;
#         gap: 0.75rem;
#     }
    
#     .reason-icon {
#         font-size: 1.25rem;
#         color: var(--fern-green);
#     }
    
#     /* Buttons */
#     .stButton > button {
#         background: linear-gradient(135deg, var(--fern-green) 0%, var(--dark-fern) 100%);
#         color: white;
#         border: none;
#         border-radius: 8px;
#         padding: 0.75rem 1.5rem;
#         font-weight: 600;
#         transition: all 0.3s ease;
#         box-shadow: 0 4px 6px rgba(88, 112, 66, 0.2);
#     }
    
#     .stButton > button:hover {
#         box-shadow: 0 10px 20px rgba(88, 112, 66, 0.3);
#         transform: translateY(-2px);
#         background: linear-gradient(135deg, var(--dark-fern) 0%, var(--fern-green) 100%);
#     }
    
#     /* Form Inputs */
#     .stNumberInput > div > div > input,
#     .stSelectbox > div > div > select {
#         border-radius: 8px;
#         border: 1px solid var(--sage);
#         padding: 0.75rem;
#         font-size: 1rem;
#     }
    
#     .stNumberInput > div > div > input:focus,
#     .stSelectbox > div > div > select:focus {
#         border-color: var(--fern-green);
#         box-shadow: 0 0 0 3px rgba(88, 112, 66, 0.1);
#     }
    
#     /* Tabs */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 2rem;
#         background-color: white;
#         padding: 1rem;
#         border-radius: 12px;
#         box-shadow: 0 2px 4px rgba(88, 112, 66, 0.05);
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         height: 3rem;
#         padding: 0 1.5rem;
#         background-color: transparent;
#         border-radius: 8px;
#         color: #718096;
#         font-weight: 600;
#         transition: all 0.3s ease;
#     }
    
#     .stTabs [aria-selected="true"] {
#         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
#         color: white;
#     }
    
#     .stTabs [data-baseweb="tab"]:hover {
#         background-color: var(--cosmic-latte);
#     }
    
#     /* Sidebar */
#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, var(--cosmic-latte) 0%, #F5F2E0 100%);
#         border-right: 1px solid var(--sage);
#     }
    
#     [data-testid="stSidebar"] .element-container {
#         color: var(--fern-green);
#     }
    
#     /* Expander */
#     .streamlit-expanderHeader {
#         background-color: var(--cosmic-latte);
#         border-radius: 8px;
#         padding: 0.75rem;
#         font-weight: 600;
#         color: var(--fern-green);
#         border: 1px solid var(--sage);
#     }
    
#     /* Alerts */
#     .stAlert {
#         border-radius: 12px;
#         border: none;
#         padding: 1rem 1.5rem;
#     }
    
#     /* Success Alert */
#     [data-baseweb="notification"] {
#         background-color: #E8F5E9;
#         border-left: 4px solid var(--fern-green);
#         border-radius: 8px;
#     }
    
#     /* Info Alert */
#     .info-box {
#         background: linear-gradient(135deg, #F9F7EB 0%, var(--cosmic-latte) 100%);
#         border-left: 4px solid var(--sage);
#         border-radius: 8px;
#         padding: 1.25rem;
#         margin: 1rem 0;
#         color: var(--fern-green);
#         border: 1px solid var(--sage);
#     }
    
#     /* Warning Alert */
#     .warning-box {
#         background: linear-gradient(135deg, var(--cosmic-latte) 0%, var(--jasmine) 100%);
#         border-left: 4px solid var(--saffron);
#         border-radius: 8px;
#         padding: 1.25rem;
#         margin: 1rem 0;
#         color: #7B5800;
#         border: 1px solid var(--jasmine);
#     }
    
#     /* Dataframe */
#     .dataframe {
#         border-radius: 12px;
#         overflow: hidden;
#         box-shadow: 0 4px 6px rgba(88, 112, 66, 0.05);
#         border: 1px solid var(--sage);
#     }
    
#     /* Metric Container */
#     [data-testid="stMetricValue"] {
#         font-size: 2rem;
#         font-weight: 700;
#         color: var(--fern-green);
#     }
    
#     [data-testid="stMetricLabel"] {
#         font-size: 0.875rem;
#         font-weight: 600;
#         color: var(--fern-green);
#         text-transform: uppercase;
#         letter-spacing: 0.05em;
#         opacity: 0.8;
#     }
    
#     /* Progress Bar */
#     .stProgress > div > div > div {
#         background: linear-gradient(90deg, var(--fern-green) 0%, var(--sage) 100%);
#     }
    
#     /* Divider */
#     hr {
#         margin: 2rem 0;
#         border: none;
#         border-top: 2px solid var(--sage);
#     }
    
#     /* Custom Scrollbar */
#     ::-webkit-scrollbar {
#         width: 10px;
#         height: 10px;
#     }
    
#     ::-webkit-scrollbar-track {
#         background: var(--cosmic-latte);
#     }
    
#     ::-webkit-scrollbar-thumb {
#         background: var(--sage);
#         border-radius: 5px;
#     }
    
#     ::-webkit-scrollbar-thumb:hover {
#         background: var(--fern-green);
#     }
    
#     /* Icon Styles */
#     .icon {
#         font-size: 1.5rem;
#         margin-right: 0.5rem;
#         color: var(--fern-green);
#     }
    
#     /* Card Grid */
#     .card-grid {
#         display: grid;
#         grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
#         gap: 1.5rem;
#         margin: 1.5rem 0;
#     }
    
#     /* Feature Badge */
#     .feature-badge {
#         display: inline-block;
#         background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
#         color: white;
#         padding: 0.25rem 0.75rem;
#         border-radius: 12px;
#         font-size: 0.75rem;
#         font-weight: 600;
#         text-transform: uppercase;
#         letter-spacing: 0.05em;
#     }
    
#     /* Timeline */
#     .timeline-item {
#         position: relative;
#         padding-left: 2rem;
#         padding-bottom: 1.5rem;
#         border-left: 2px solid var(--sage);
#     }
    
#     .timeline-item:last-child {
#         border-left: none;
#     }
    
#     .timeline-dot {
#         position: absolute;
#         left: -6px;
#         top: 0;
#         width: 12px;
#         height: 12px;
#         border-radius: 50%;
#         background: var(--fern-green);
#     }
    
#     /* Stat Card */
#     .stat-card {
#         background: white;
#         border-radius: 12px;
#         padding: 1.5rem;
#         box-shadow: 0 4px 12px rgba(88, 112, 66, 0.08);
#         border-top: 4px solid var(--fern-green);
#         text-align: center;
#         border: 1px solid var(--sage);
#         transition: all 0.3s ease;
#     }
    
#     .stat-card:hover {
#         box-shadow: 0 10px 25px rgba(88, 112, 66, 0.15);
#         transform: translateY(-2px);
#     }
    
#     .stat-number {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: var(--fern-green);
#         margin-bottom: 0.5rem;
#     }
    
#     .stat-label {
#         font-size: 0.875rem;
#         font-weight: 600;
#         color: var(--fern-green);
#         text-transform: uppercase;
#         letter-spacing: 0.05em;
#         opacity: 0.8;
#     }
    
#     /* Chart styling */
#     .js-plotly-plot .plotly {
#         background-color: white !important;
#     }
    
#     /* Radio buttons */
#     .stRadio > div {
#         background-color: white;
#         padding: 0.5rem;
#         border-radius: 8px;
#         border: 1px solid var(--sage);
#     }
    
#     .stRadio > div[data-baseweb="radio"] label {
#         color: var(--fern-green);
#     }
    
#     /* Checkbox */
#     .stCheckbox > label {
#         color: var(--fern-green);
#     }
    
#     /* Form submit button */
#     div[data-testid="stForm"] button {
#         background: linear-gradient(135deg, var(--fern-green) 0%, var(--dark-fern) 100%);
#         color: white;
#         border: none;
#         border-radius: 8px;
#         padding: 0.75rem 1.5rem;
#         font-weight: 600;
#         transition: all 0.3s ease;
#         box-shadow: 0 4px 6px rgba(88, 112, 66, 0.2);
#     }
    
#     div[data-testid="stForm"] button:hover {
#         box-shadow: 0 10px 20px rgba(88, 112, 66, 0.3);
#         transform: translateY(-2px);
#         background: linear-gradient(135deg, var(--dark-fern) 0%, var(--fern-green) 100%);
#     }
    
#     /* Table styling */
#     .stTable {
#         border: 1px solid var(--sage);
#         border-radius: 8px;
#     }
    
#     /* Container borders */
#     .stApp {
#         background-color: white;
#     }
    
#     /* ================== FIXES FOR TEXT VISIBILITY ================== */
#     /* Input text color - WHITE on DARK background for readability */
#     .stNumberInput > div > div > input,
#     .stSelectbox > div > div > select,
#     .stTextInput > div > div > input {
#         color: #FFFFFF !important;
#         background-color: #2D3748 !important;
#     }
    
#     /* Radio button text - DARK on LIGHT background */
#     .stRadio > div > label,
#     .stRadio > div > label > div > p {
#         color: #333333 !important;
#     }
    
#     /* Checkbox text - DARK on LIGHT background */
#     .stCheckbox > label,
#     .stCheckbox > label > div > p {
#         color: #333333 !important;
#     }
    
#     /* Slider value text */
#     .stSlider > div > div > div {
#         color: #333333 !important;
#     }
    
#     /* Placeholder text - Light gray for visibility */
#     ::placeholder {
#         color: #A0AEC0 !important;
#         opacity: 1 !important;
#     }
    
#     /* Sidebar text - DARK for readability */
#     [data-testid="stSidebar"] p,
#     [data-testid="stSidebar"] div,
#     [data-testid="stSidebar"] span {
#         color: #333333 !important;
#     }
    
#     /* Form labels - Keep GREEN for branding */
#     .stNumberInput label,
#     .stSelectbox label,
#     .stTextInput label,
#     .stRadio label,
#     .stCheckbox label {
#         color: var(--fern-green) !important;
#         font-weight: 600;
#     }
    
#     /* General text color for body content */
#     .main p, .main div, .main span {
#         color: #333333 !important;
#     }
    
#     /* Fix for text in expanders */
#     .streamlit-expanderContent p,
#     .streamlit-expanderContent div,
#     .streamlit-expanderContent span {
#         color: #333333 !important;
#     }
    
#     /* Fix for text in alerts */
#     .stAlert p,
#     .stAlert div,
#     .stAlert span {
#         color: inherit !important;
#     }
    
#     /* Fix for dropdown select options - WHITE on DARK */
#     .stSelectbox select option {
#         color: #FFFFFF !important;
#         background-color: #2D3748 !important;
#     }
    
#     /* Fix for number input increment/decrement buttons */
#     .stNumberInput button {
#         color: #FFFFFF !important;
#         background-color: #4A5568 !important;
#     }
    
#     .stNumberInput button:hover {
#         background-color: var(--fern-green) !important;
#     }
    
#     /* Input focus state - Maintain WHITE text on DARK background */
#     .stNumberInput > div > div > input:focus,
#     .stSelectbox > div > div > select:focus,
#     .stTextInput > div > div > input:focus {
#         color: #FFFFFF !important;
#         background-color: #2D3748 !important;
#         border-color: var(--fern-green);
#         box-shadow: 0 0 0 3px rgba(88, 112, 66, 0.1);
#     }
    
#     /* Ensure all input fields maintain dark background with white text */
#     input[type="number"],
#     input[type="text"],
#     select {
#         color: #FFFFFF !important;
#         background-color: #2D3748 !important;
#     }
    
#     /* Fix for disabled inputs */
#     input:disabled,
#     select:disabled {
#         color: #A0AEC0 !important;
#         background-color: #1A202C !important;
#         opacity: 0.6;
#     }
    
#     /* Ensure dropdown arrow is visible */
#     .stSelectbox svg {
#         fill: #FFFFFF !important;
#     }
    
#     /* Fix for input number spinner buttons */
#     input[type="number"]::-webkit-inner-spin-button,
#     input[type="number"]::-webkit-outer-spin-button {
#         opacity: 1;
#         background-color: #4A5568;
#     }
#     /* ================== END OF FIXES ================== */
    
#     </style>
# """, unsafe_allow_html=True)

# # =============================================================================
# # LOAD TRAINED MODEL ASSETS
# # =============================================================================

# @st.cache_resource
# def load_model_assets():
#     """Load the trained model and preprocessing assets"""
#     try:
#         possible_paths = [
#             'credit_risk_assets.pkl',
#             'notebooks/credit_risk_assets.pkl',
#             '../notebooks/credit_risk_assets.pkl'
#         ]
        
#         assets = None
#         for path in possible_paths:
#             try:
#                 assets = joblib.load(path)
#                 break
#             except FileNotFoundError:
#                 continue
        
#         if assets is None:
#             raise FileNotFoundError("Could not find credit_risk_assets.pkl")
#         return {
#             'model': assets['model'],
#             'features': assets['features'],
#             'le_map': assets['le_map'],
#             'target_le': assets['target_le'],
#             'loaded': True,
#             'error': None
#         }
#     except FileNotFoundError:
#         return {
#             'loaded': False,
#             'error': 'credit_risk_assets.pkl not found. Please run the training script first.'
#         }
#     except Exception as e:
#         return {
#             'loaded': False,
#             'error': f'Error loading model: {str(e)}'
#         }

# # Load assets
# ASSETS = load_model_assets()

# if not ASSETS['loaded']:
#     st.error(f"❌ {ASSETS['error']}")
#     st.info("Please ensure 'credit_risk_assets.pkl' is in the same directory as this app.")
#     st.stop()

# MODEL = ASSETS['model']
# TOP_FEATURES = ASSETS['features']
# LE_MAP = ASSETS['le_map']
# TARGET_LE = ASSETS['target_le']

# # =============================================================================
# # AFFORDABILITY CALCULATION ENGINE
# # =============================================================================

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
#     """Calculate comprehensive affordability metrics"""
    
#     new_emi = calculate_emi(loan_amount, interest_rate, tenure_months)
#     total_emi = new_emi + existing_emi
#     foir_percentage = (total_emi / monthly_income) * 100 if monthly_income > 0 else 0
#     net_disposable = monthly_income - total_emi
#     max_allowed_emi = monthly_income * 0.50
#     recommended_emi = monthly_income * 0.40
#     affordable = foir_percentage <= 50
#     within_recommended = foir_percentage <= 40
    
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

# # =============================================================================
# # REASON CODE GENERATION SYSTEM
# # =============================================================================

# APPROVAL_REASONS = {
#     'high_bureau': 'Excellent credit score ({score})',
#     'stable_employment': 'Stable employment history ({tenure} months)',
#     'low_foir': 'Affordable EMI burden (FOIR: {foir}%)',
#     'clean_payment': 'Clean payment history (No DPD)',
#     'strong_income': 'Strong monthly income (Rs.{income:,})',
#     'low_utilization': 'Low credit utilization ({util}%)',
# }

# REJECTION_REASONS = {
#     'low_bureau': 'Credit score below minimum ({score} < 550)',
#     'high_foir': 'EMI burden too high (FOIR: {foir}% > 50%)',
#     'severe_dpd': 'Severe payment delays ({dpd} instances of 90+ DPD)',
#     'low_income': 'Income below minimum threshold (Rs.{income:,} < Rs.15,000)',
#     'short_employment': 'Insufficient employment tenure ({tenure} months < 6)',
#     'bankruptcy': 'Active bankruptcy detected',
#     'kyc_failed': 'KYC verification not completed',
#     'high_utilization': 'High credit utilization ({util}% > 80%)',
#     'age_invalid': 'Age outside acceptable range ({age} years)'
# }

# REVIEW_REASONS = {
#     'borderline_bureau': 'Credit score in borderline range ({score})',
#     'moderate_foir': 'EMI burden moderate (FOIR: {foir}%)',
#     'mixed_signals': 'Mixed credit indicators requiring human review',
#     'recent_employment': 'Recent employment change requiring verification',
# }


# def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
#     """Generate top 3 reason codes for the decision"""
#     reasons = []
    
#     bureau_score = customer_data.get('bureau_score', 0)
#     foir = affordability_data.get('foir_percentage', 0)
#     dpd_90 = customer_data.get('dpd_90_count_6m', 0)
#     income = customer_data.get('avg_salary_6m', 0)
#     employment_tenure = customer_data.get('employment_tenure_months', 0)
#     credit_util = customer_data.get('credit_utilization_pct', 0)
#     age = customer_data.get('age', 0)
    
#     if decision == "APPROVE":
#         if bureau_score >= 750:
#             reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
#         if employment_tenure >= 24:
#             reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
#         if foir <= 40:
#             reasons.append(APPROVAL_REASONS['low_foir'].format(foir=round(foir, 1)))
#         if dpd_90 == 0:
#             reasons.append(APPROVAL_REASONS['clean_payment'])
#         if income >= 75000:
#             reasons.append(APPROVAL_REASONS['strong_income'].format(income=income))
#         if credit_util <= 30:
#             reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))
    
#     elif decision == "REJECT":
#         for check_name, check_result in policy_checks.items():
#             if '❌' in str(check_result):
#                 if 'bureau' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
#                 elif 'dpd' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
#                 elif 'income' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['low_income'].format(income=income))
#                 elif 'tenure' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
#                 elif 'kyc' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['kyc_failed'])
#                 elif 'bankruptcy' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['bankruptcy'])
#                 elif 'age' in check_name.lower():
#                     reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))
        
#         if foir > 50:
#             reasons.append(REJECTION_REASONS['high_foir'].format(foir=round(foir, 1)))
#         if credit_util > 80:
#             reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
    
#     elif decision == "REVIEW":
#         if 650 <= bureau_score < 700:
#             reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
#         if 40 < foir <= 50:
#             reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=round(foir, 1)))
#         if employment_tenure < 12:
#             reasons.append(REVIEW_REASONS['recent_employment'])
#         if not reasons:
#             reasons.append(REVIEW_REASONS['mixed_signals'])
    
#     return reasons[:3] if reasons else ['Decision based on model assessment']

# # =============================================================================
# # RISK SCORE CALCULATION
# # =============================================================================

# def calculate_final_risk_score(bureau_score, ml_confidence, foir):
#     """Calculate final risk score (0-1000)"""
#     bureau_points = (bureau_score / 900) * 400
#     ml_points = (ml_confidence / 100) * 400
#     foir_points = max(0, (1 - foir/50) * 200)
#     total_score = int(bureau_points + ml_points + foir_points)
#     return min(max(total_score, 0), 1000)

# # =============================================================================
# # BATCH PREDICTION ENGINE
# # =============================================================================

# def process_batch_predictions(df: pd.DataFrame) -> pd.DataFrame:
#     """Process batch predictions for multiple records"""
#     results = []
    
#     for idx, row in df.iterrows():
#         customer_dict = row.to_dict()
        
#         # Convert yes/no to boolean
#         for key, value in customer_dict.items():
#             if isinstance(value, str):
#                 if value.lower() in ['yes', 'true', '1']:
#                     customer_dict[key] = True
#                 elif value.lower() in ['no', 'false', '0']:
#                     customer_dict[key] = False
        
#         # Add missing required fields with defaults
#         required_fields = {
#             'kyc_verified': True,
#             'bankruptcy_flag': False,
#             'fraud_flag': False,
#             'dpd_90_count_6m': 0,
#             'recent_inquiries_3m': 0,
#             'active_loans_count': 0,
#             'existing_emi': 0,
#             'salary_stability_flag': 'STABLE'
#         }
        
#         for field, default in required_fields.items():
#             if field not in customer_dict:
#                 customer_dict[field] = default
        
#         # Get decision
#         decision_data = make_hybrid_decision_enhanced(customer_dict)
        
#         # Generate application ID
#         app_id = f"BATCH_{idx+1:04d}"
        
#         # Prepare result
#         result = {
#             'application_id': app_id,
#             'decision': decision_data['decision'],
#             'risk_score': decision_data['risk_score'],
#             'pd_percentage': decision_data['pd_percentage'],
#             'confidence': round(decision_data['confidence'], 2),
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         }
        
#         # Add key customer data
#         result.update({
#             'age': customer_dict.get('age', ''),
#             'employment_type': customer_dict.get('employment_type', ''),
#             'bureau_score': customer_dict.get('bureau_score', ''),
#             'monthly_income': customer_dict.get('avg_salary_6m', ''),
#             'loan_amount': customer_dict.get('loan_amount', ''),
#             'foir_percentage': decision_data.get('affordability_data', {}).get('foir_percentage', 0)
#         })
        
#         results.append(result)
    
#     return pd.DataFrame(results)

# def create_download_link(df: pd.DataFrame, filename: str = "batch_results.csv") -> str:
#     """Create a download link for a DataFrame"""
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;">📥 Download CSV</a>'
#     return href

# # =============================================================================
# # ENHANCED HYBRID DECISION ENGINE
# # =============================================================================

# def make_hybrid_decision_enhanced(customer_dict):
#     """Enhanced decision engine with complete data"""
    
#     policy_checks = {}
    
#     # Policy Gates
#     age = customer_dict.get('age', 0)
#     employment_type = customer_dict.get('employment_type', 'Salaried')
#     kyc_verified = customer_dict.get('kyc_verified', True)
#     bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
#     fraud_flag = customer_dict.get('fraud_flag', False)
    
#     if employment_type in ['Salaried']:
#         age_min, age_max = 18, 65
#     else:
#         age_min, age_max = 18, 70
    
#     if age < age_min or age > age_max:
#         policy_checks['age'] = f"❌ Age {age} (Required: {age_min}-{age_max})"
#         return {
#             'decision': "REJECT",
#             'reason': f"Policy Gate: Age outside allowed range",
#             'confidence': 0,
#             'class_probs': {'REJECT': 100},
#             'policy_checks': policy_checks,
#             'risk_score': 0,
#             'pd_percentage': 100.0
#         }
#     policy_checks['age'] = f"✅ Age {age} (Valid)"
    
#     if not kyc_verified:
#         policy_checks['kyc'] = "❌ KYC Not Verified"
#         return {
#             'decision': "REJECT",
#             'reason': "Policy Gate: KYC verification required",
#             'confidence': 0,
#             'class_probs': {'REJECT': 100},
#             'policy_checks': policy_checks,
#             'risk_score': 0,
#             'pd_percentage': 100.0
#         }
#     policy_checks['kyc'] = "✅ KYC Verified"
    
#     if bankruptcy_flag:
#         policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
#         return {
#             'decision': "REJECT",
#             'reason': "Policy Gate: Active bankruptcy",
#             'confidence': 0,
#             'class_probs': {'REJECT': 100},
#             'policy_checks': policy_checks,
#             'risk_score': 0,
#             'pd_percentage': 100.0
#         }
#     policy_checks['bankruptcy'] = "✅ No Bankruptcy"
    
#     if fraud_flag:
#         policy_checks['fraud'] = "❌ Fraud Flag"
#         return {
#             'decision': "REJECT",
#             'reason': "Policy Gate: Fraud detected",
#             'confidence': 0,
#             'class_probs': {'REJECT': 100},
#             'policy_checks': policy_checks,
#             'risk_score': 0,
#             'pd_percentage': 100.0
#         }
#     policy_checks['fraud'] = "✅ No Fraud History"
    
#     monthly_income = customer_dict.get('avg_salary_6m', 0)
#     employment_tenure = customer_dict.get('employment_tenure_months', 0)
#     business_vintage = customer_dict.get('business_vintage_years', 0)
    
#     if monthly_income < 15000:
#         policy_checks['income'] = f"❌ Income Rs.{monthly_income:,.0f} (Min: Rs.15,000)"
#         return {
#             'decision': "REJECT",
#             'reason': "Policy Gate: Income below minimum",
#             'confidence': 0,
#             'class_probs': {'REJECT': 100},
#             'policy_checks': policy_checks,
#             'risk_score': 0,
#             'pd_percentage': 100.0
#         }
#     policy_checks['income'] = f"✅ Income Rs.{monthly_income:,.0f}"
    
#     if employment_type == 'Salaried' and employment_tenure < 6:
#         policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
#         return {
#             'decision': "REJECT",
#             'reason': "Policy Gate: Insufficient tenure",
#             'confidence': 0,
#             'class_probs': {'REJECT': 100},
#             'policy_checks': policy_checks,
#             'risk_score': 0,
#             'pd_percentage': 100.0
#         }
#     elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
#         policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
#         return {
#             'decision': "REJECT",
#             'reason': "Policy Gate: Insufficient business vintage",
#             'confidence': 0,
#             'class_probs': {'REJECT': 100},
#             'policy_checks': policy_checks,
#             'risk_score': 0,
#             'pd_percentage': 100.0
#         }
    
#     if employment_type == 'Salaried':
#         policy_checks['tenure'] = f"✅ Tenure {employment_tenure} months"
#     else:
#         policy_checks['tenure'] = f"✅ Business Vintage {business_vintage} years"
    
#     bureau_score = customer_dict.get('bureau_score', 0)
#     dpd_90 = customer_dict.get('dpd_90_count_6m', 0)
#     credit_utilization = customer_dict.get('credit_utilization_pct', 0)
#     recent_inquiries = customer_dict.get('recent_inquiries_3m', 0)
    
#     if bureau_score < 550:
#         policy_checks['bureau'] = f"❌ Bureau Score {bureau_score} (Min: 550)"
#         return {
#             'decision': "REJECT",
#             'reason': "Policy Gate: Bureau score too low",
#             'confidence': 0,
#             'class_probs': {'REJECT': 100},
#             'policy_checks': policy_checks,
#             'risk_score': 0,
#             'pd_percentage': 100.0
#         }
#     policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"
    
#     if dpd_90 > 0:
#         policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD"
#         return {
#             'decision': "REJECT",
#             'reason': "Policy Gate: Severe delinquency",
#             'confidence': 0,
#             'class_probs': {'REJECT': 100},
#             'policy_checks': policy_checks,
#             'risk_score': 0,
#             'pd_percentage': 100.0
#         }
#     policy_checks['dpd'] = "✅ No 90+ DPD"
    
#     if credit_utilization > 80:
#         policy_checks['utilization'] = f"⚠️ High utilization {credit_utilization}%"
#     else:
#         policy_checks['utilization'] = f"✅ Utilization {credit_utilization}%"
    
#     if recent_inquiries > 5:
#         policy_checks['inquiries'] = f"⚠️ {recent_inquiries} recent inquiries"
#     else:
#         policy_checks['inquiries'] = f"✅ {recent_inquiries} inquiries"
    
#     # ML Prediction
#     input_df = pd.DataFrame([customer_dict])
    
#     for col in TOP_FEATURES:
#         if col not in input_df.columns:
#             if col in LE_MAP:
#                 input_df[col] = "Unknown"
#             else:
#                 input_df[col] = 0
    
#     for col, le in LE_MAP.items():
#         if col in input_df.columns:
#             val = str(input_df[col].values[0])
#             try:
#                 input_df[col] = le.transform([val])[0]
#             except ValueError:
#                 input_df[col] = 0
    
#     final_input = input_df[TOP_FEATURES]
    
#     pred_idx = MODEL.predict(final_input)[0]
#     ml_decision = TARGET_LE.inverse_transform([pred_idx])[0]
    
#     try:
#         pred_proba = MODEL.predict_proba(final_input)[0]
#         confidence = max(pred_proba) * 100
#         class_probs = {
#             cls: prob * 100 
#             for cls, prob in zip(TARGET_LE.classes_, pred_proba)
#         }
#     except:
#         confidence = 75.0
#         class_probs = {ml_decision: 100.0}
    
#     # Affordability
#     loan_amount = customer_dict.get('loan_amount', 0)
#     loan_tenure = customer_dict.get('loan_tenure_months', 12)
#     interest_rate = customer_dict.get('interest_rate', 10.5)
#     existing_emi = customer_dict.get('existing_emi', 0)
    
#     affordability_data = calculate_affordability(
#         monthly_income=monthly_income,
#         loan_amount=loan_amount,
#         interest_rate=interest_rate,
#         tenure_months=loan_tenure,
#         existing_emi=existing_emi
#     )
    
#     foir = affordability_data['foir_percentage']
    
#     if ml_decision == "APPROVE" and foir > 45:
#         ml_decision = "REVIEW"
    
#     risk_score = calculate_final_risk_score(bureau_score, confidence, foir)
#     pd_percentage = max(0, min(100, (1 - confidence/100) * 10))
    
#     return {
#         'decision': ml_decision,
#         'reason': "Decision based on comprehensive assessment",
#         'confidence': confidence,
#         'class_probs': class_probs,
#         'policy_checks': policy_checks,
#         'risk_score': risk_score,
#         'pd_percentage': round(pd_percentage, 2),
#         'affordability_data': affordability_data
#     }

# # =============================================================================
# # MODERN UI COMPONENTS
# # =============================================================================

# def render_decision_header(decision_data, customer_data):
#     """Render modern decision header"""
    
#     decision = decision_data['decision']
#     risk_score = decision_data['risk_score']
#     pd_score = decision_data['pd_percentage']
#     approved_amount = customer_data.get('loan_amount', 0)
#     tenure = customer_data.get('loan_tenure_months', 24)
#     app_id = customer_data.get('application_id', 'N/A')
#     timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
#     # Decision card with appropriate styling
#     if decision == "APPROVE":
#         card_class = "decision-card decision-card-approved"
#         icon = "✓"
#         subtitle = "Application Approved Successfully"
#     elif decision == "REJECT":
#         card_class = "decision-card decision-card-rejected"
#         icon = "✗"
#         subtitle = "Application Not Approved"
#     else:
#         card_class = "decision-card decision-card-review"
#         icon = "⚠"
#         subtitle = "Requires Manual Review"
    
#     st.markdown(f"""
#         <div class="{card_class}">
#             <div class="decision-title">
#                 <span>{icon}</span>
#                 <span>{decision}</span>
#             </div>
#             <div class="decision-subtitle">{subtitle}</div>
#         </div>
#     """, unsafe_allow_html=True)
    
#     # Metrics grid
#     col1, col2, col3, col4, col5 = st.columns(5)
    
#     with col1:
#         st.markdown(f"""
#             <div class="stat-card">
#                 <div class="stat-number">{risk_score}</div>
#                 <div class="stat-label">Risk Score</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown(f"""
#             <div class="stat-card">
#                 <div class="stat-number">{pd_score}%</div>
#                 <div class="stat-label">PD Score</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         st.markdown(f"""
#             <div class="stat-card">
#                 <div class="stat-number">Rs.{approved_amount:,.0f}</div>
#                 <div class="stat-label">Loan Amount</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col4:
#         st.markdown(f"""
#             <div class="stat-card">
#                 <div class="stat-number">{tenure}</div>
#                 <div class="stat-label">Tenure (Months)</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col5:
#         st.markdown(f"""
#             <div class="stat-card">
#                 <div class="stat-number">{decision_data['confidence']:.0f}%</div>
#                 <div class="stat-label">Confidence</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     # Application info
#     st.markdown("<br>", unsafe_allow_html=True)
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown(f"""
#             <div class="info-box">
#                 <strong>📋 Application ID:</strong> {app_id}
#             </div>
#         """, unsafe_allow_html=True)
#     with col2:
#         st.markdown(f"""
#             <div class="info-box">
#                 <strong>🕐 Decision Timestamp:</strong> {timestamp}
#             </div>
#         """, unsafe_allow_html=True)


# def render_info_card(title, icon, data_dict, status_dict=None):
#     """Render modern info card with data"""
    
#     st.markdown(f"""
#         <div class="info-card">
#             <div class="info-card-title">
#                 <span class="icon">{icon}</span>
#                 <span>{title}</span>
#             </div>
#             <div class="info-card-content">
#     """, unsafe_allow_html=True)
    
#     for label, value in data_dict.items():
#         status = ""
#         if status_dict and label in status_dict:
#             if status_dict[label] == "pass":
#                 status = '<span class="status-badge badge-pass">✓ Passed</span>'
#             elif status_dict[label] == "fail":
#                 status = '<span class="status-badge badge-fail">✗ Failed</span>'
#             elif status_dict[label] == "warning":
#                 status = '<span class="status-badge badge-warning">⚠ Warning</span>'
        
#         st.markdown(f"""
#             <div class="data-row">
#                 <span class="data-label">{label}</span>
#                 <span class="data-value">{value} {status}</span>
#             </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown("""
#             </div>
#         </div>
#     """, unsafe_allow_html=True)


# def render_reason_codes(reasons):
#     """Render reason codes in modern style"""
    
#     st.markdown("""
#         <div class="info-card">
#             <div class="info-card-title">
#                 <span class="icon">📝</span>
#                 <span>Decision Reasons</span>
#             </div>
#             <div class="info-card-content">
#     """, unsafe_allow_html=True)
    
#     for i, reason in enumerate(reasons, 1):
#         st.markdown(f"""
#             <div class="reason-item">
#                 <span class="reason-icon">{i}.</span>
#                 <span>{reason}</span>
#             </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown("""
#             </div>
#         </div>
#     """, unsafe_allow_html=True)


# def create_modern_gauge(value, title, max_value=100):
#     """Create modern gauge chart"""
    
#     if value <= 50:
#         color = "#f56565"
#     elif value <= 75:
#         color = "#ed8936"
#     else:
#         color = "#48bb78"
    
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=value,
#         title={'text': title, 'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}},
#         number={'suffix': "%", 'font': {'size': 36, 'color': '#2d3748', 'family': 'Inter'}},
#         gauge={
#             'axis': {'range': [0, max_value], 'tickfont': {'size': 12, 'color': '#718096'}},
#             'bar': {'color': color, 'thickness': 0.75},
#             'bgcolor': 'white',
#             'borderwidth': 0,
#             'steps': [
#                 {'range': [0, 50], 'color': '#fed7d7'},
#                 {'range': [50, 75], 'color': '#feebc8'},
#                 {'range': [75, 100], 'color': '#c6f6d5'}
#             ],
#         }
#     ))
    
#     fig.update_layout(
#         height=250,
#         margin=dict(l=20, r=20, t=50, b=20),
#         paper_bgcolor='white',
#         font={'family': 'Inter', 'color': '#2d3748'}
#     )
    
#     return fig


# def create_modern_bar_chart(class_probs):
#     """Create modern probability bar chart"""
    
#     df = pd.DataFrame({
#         'Decision': list(class_probs.keys()),
#         'Probability': list(class_probs.values())
#     })
    
#     colors = {'REVIEW': '#ed8936','APPROVE': '#48bb78',  'REJECT': '#f56565'}
    
#     fig = px.bar(
#         df, 
#         x='Decision', 
#         y='Probability',
#         title='Decision Probabilities',
#         color='Decision',
#         color_discrete_map=colors,
#         text='Probability'
#     )
    
#     fig.update_traces(
#         texttemplate='%{text:.1f}%',
#         textposition='outside',
#         marker_line_width=0
#     )
    
#     fig.update_layout(
#         showlegend=False,
#         yaxis_title='Probability (%)',
#         xaxis_title='',
#         height=300,
#         margin=dict(l=20, r=20, t=50, b=20),
#         paper_bgcolor='white',
#         plot_bgcolor='white',
#         font={'family': 'Inter', 'color': '#2d3748'},
#         yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]},
#         xaxis={'tickfont': {'size': 14, 'color': '#2d3748'}}
#     )
    
#     return fig

# # =============================================================================
# # SIDEBAR
# # =============================================================================

# with st.sidebar:
#     st.markdown("# 🏦 Credit Risk Engine")
#     st.markdown("---")
    
#     page = st.radio(
#         "**Navigation**",
#         ["🏠 Home", "👤 Assessment", "📊 Batch Process", "📈 Model Info", "ℹ️ About"],
#         label_visibility="collapsed"
#     )
    
#     st.markdown("---")
    
#     st.markdown(f"""
#         <div class="info-card">
#             <div class="info-card-title">System Status</div>
#             <div class="info-card-content">
#                 <div class="data-row">
#                     <span class="data-label">Model</span>
#                     <span class="data-value">✅ Loaded</span>
#                 </div>
#                 <div class="data-row">
#                     <span class="data-label">Version</span>
#                     <span class="data-value">8.0</span>
#                 </div>
#                 <div class="data-row">
#                     <span class="data-label">Features</span>
#                     <span class="data-value">{len(TOP_FEATURES)}</span>
#                 </div>
#                 <div class="data-row">
#                     <span class="data-label">Type</span>
#                     <span class="data-value">Random Forest</span>
#                 </div>
#             </div>
#         </div>
#     """, unsafe_allow_html=True)
    
#     with st.expander("🎯 **Top Features**"):
#         for i, feat in enumerate(TOP_FEATURES[:5], 1):
#             st.markdown(f"`{i}.` {feat}")

# # =============================================================================
# # HOME PAGE
# # =============================================================================

# if page == "🏠 Home":
#     st.markdown('<p class="main-header">Credit Risk Assessment Platform</p>', unsafe_allow_html=True)
    
#     st.markdown("""
#         <div class="info-box">
#             <h3 style="margin-top: 0;">🎯 AI-Powered Lending Decisions</h3>
#             <p style="margin-bottom: 0;">
#                 Comprehensive credit risk evaluation combining hard policy rules, 
#                 machine learning models, and affordability analysis for accurate lending decisions.
#             </p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("<br>", unsafe_allow_html=True)
    
#     # Feature cards
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#             <div class="info-card">
#                 <div class="info-card-title">
#                     <span class="icon">🛡️</span>
#                     <span>Policy Gates</span>
#                 </div>
#                 <div class="info-card-content">
#                     <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
#                         <li>Age & KYC verification</li>
#                         <li>Employment stability</li>
#                         <li>Minimum income checks</li>
#                         <li>Credit bureau thresholds</li>
#                         <li>Bankruptcy & fraud detection</li>
#                     </ul>
#                 </div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#             <div class="info-card">
#                 <div class="info-card-title">
#                     <span class="icon">🤖</span>
#                     <span>ML Assessment</span>
#                 </div>
#                 <div class="info-card-content">
#                     <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
#                         <li>Random Forest classifier</li>
#                         <li>60K+ training samples</li>
#                         <li>Confidence scoring</li>
#                         <li>Multi-class prediction</li>
#                         <li>Feature importance</li>
#                     </ul>
#                 </div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         st.markdown("""
#             <div class="info-card">
#                 <div class="info-card-title">
#                     <span class="icon">💰</span>
#                     <span>Affordability</span>
#                 </div>
#                 <div class="info-card-content">
#                     <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
#                         <li>EMI calculation</li>
#                         <li>FOIR analysis (max 50%)</li>
#                         <li>Net disposable income</li>
#                         <li>Debt burden assessment</li>
#                         <li>Affordability scoring</li>
#                     </ul>
#                 </div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown("<br>", unsafe_allow_html=True)
    
#     # Stats
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("🎯 Accuracy", "85%", "+2%")
    
#     with col2:
#         st.metric("⚡ Avg Response", "1.2s", "-0.3s")
    
#     with col3:
#         st.metric("📊 Features", len(TOP_FEATURES))
    
#     with col4:
#         st.metric("🔄 Version", "8.0", "Latest")
    
#     st.markdown("<br>", unsafe_allow_html=True)
    
#     st.markdown("""
#         <div class="warning-box">
#             <strong>🆕 New in Version 8.0:</strong><br>
#             • Sage Green & Yellow Professional Theme<br>
#             • Enhanced visual hierarchy and readability<br>
#             • Improved decision summary cards<br>
#             • Modern charts and gauges<br>
#             • Responsive layout optimization
#         </div>
#     """, unsafe_allow_html=True)

# # =============================================================================
# # ASSESSMENT PAGE
# # =============================================================================

# elif page == "👤 Assessment":
#     st.markdown('<p class="main-header">Credit Assessment</p>', unsafe_allow_html=True)
    
#     st.markdown("""
#         <div class="info-box">
#             💡 Complete the form below to assess credit risk. All fields are required for accurate evaluation.
#         </div>
#     """, unsafe_allow_html=True)
    
#     with st.form("assessment_form"):
#         # Identity & Eligibility
#         st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             age = st.number_input("Age", 18, 80, 35, help="Customer's age in years")
#             employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business'])
        
#         with col2:
#             kyc_verified = st.selectbox("KYC Verified", ['Yes', 'No']) == 'Yes'
#             bankruptcy_flag = st.selectbox("Bankruptcy Flag", ['No', 'Yes']) == 'Yes'
        
#         with col3:
#             fraud_flag = st.selectbox("Fraud Flag", ['No', 'Yes']) == 'Yes'
#             if employment_type == 'Salaried':
#                 employment_tenure = st.number_input("Employment Tenure (months)", 0, 600, 24)
#                 business_vintage = 0
#             else:
#                 business_vintage = st.number_input("Business Vintage (years)", 0, 50, 3)
#                 employment_tenure = 0
        
#         # Credit Bureau
#         st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             bureau_score = st.number_input("Bureau Score", 300, 900, 720, 10)
#             dpd_90_6m = st.number_input("DPD 90+ (Last 6M)", 0, 20, 0)
        
#         with col2:
#             credit_utilization = st.number_input("Credit Utilization (%)", 0, 100, 30)
#             recent_inquiries = st.number_input("Recent Inquiries (3M)", 0, 20, 2)
        
#         with col3:
#             active_loans = st.number_input("Active Loans", 0, 10, 1)
#             existing_emi = st.number_input("Existing Total EMI (Rs.)", 0, 200000, 15000, 1000)
        
#         # Income & Financial
#         st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             avg_salary = st.number_input("Monthly Income (Rs.)", 0, 1000000, 50000, 5000)
#             amt_income = st.number_input("Annual Income (Rs.)", 0, 10000000, 600000, 10000)
        
#         with col2:
#             net_surplus = st.number_input("Net Cash Surplus (Rs.)", -100000, 500000, 20000, 5000)
#             salary_stability = st.selectbox("Salary Stability", ['STABLE', 'MODERATE', 'UNSTABLE'])
        
#         with col3:
#             loan_amount = st.number_input("Loan Amount (Rs.)", 0, 5000000, 180000, 10000)
#             loan_tenure = st.number_input("Tenure (months)", 3, 360, 24)
        
#         with col4:
#             interest_rate = st.number_input("Interest Rate (%)", 8.0, 20.0, 10.5, 0.5)
#             amt_annuity = st.number_input("Requested EMI (Rs.)", 0, 200000, 8500, 500)
        
#         st.markdown("<br>", unsafe_allow_html=True)
#         submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)
    
#     if submitted:
#         # Generate application ID
#         timestamp = datetime.now()
#         app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
        
#         # Prepare data
#         customer_data = {
#             'age': age,
#             'employment_type': employment_type,
#             'kyc_verified': kyc_verified,
#             'bankruptcy_flag': bankruptcy_flag,
#             'fraud_flag': fraud_flag,
#             'employment_tenure_months': employment_tenure,
#             'business_vintage_years': business_vintage,
#             'bureau_score': bureau_score,
#             'dpd_90_count_6m': dpd_90_6m,
#             'credit_utilization_pct': credit_utilization,
#             'recent_inquiries_3m': recent_inquiries,
#             'active_loans_count': active_loans,
#             'avg_salary_6m': avg_salary,
#             'AMT_INCOME_TOTAL': amt_income,
#             'net_cash_surplus_6m': net_surplus,
#             'salary_stability_flag': salary_stability,
#             'loan_amount': loan_amount,
#             'loan_tenure_months': loan_tenure,
#             'interest_rate': interest_rate,
#             'existing_emi': existing_emi,
#             'AMT_ANNUITY': amt_annuity,
#             'application_id': app_id,
#             'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S")
#         }
        
#         # Get decision
#         with st.spinner("🔄 Processing assessment..."):
#             decision_data = make_hybrid_decision_enhanced(customer_data)
        
#         # Generate reasons
#         reasons = generate_reason_codes(
#             decision=decision_data['decision'],
#             customer_data=customer_data,
#             affordability_data=decision_data.get('affordability_data', {}),
#             policy_checks=decision_data['policy_checks']
#         )
        
#         customer_data['reason_codes'] = reasons
        
#         # Tabs
#         tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])
        
#         with tab1:
#             st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 render_info_card(
#                     "👤 Identity", 
#                     "👤",
#                     {
#                         "Age": age,
#                         "Employment": employment_type,
#                         "KYC Status": "Verified" if kyc_verified else "Not Verified",
#                         "Tenure": f"{employment_tenure} months" if employment_type == 'Salaried' else f"{business_vintage} years"
#                     }
#                 )
                
#                 render_info_card(
#                     "💰 Financial", 
#                     "💰",
#                     {
#                         "Monthly Income": f"Rs.{avg_salary:,}",
#                         "Annual Income": f"Rs.{amt_income:,}",
#                         "Net Surplus": f"Rs.{net_surplus:,}",
#                         "Stability": salary_stability
#                     }
#                 )
            
#             with col2:
#                 render_info_card(
#                     "🏦 Credit Bureau", 
#                     "🏦",
#                     {
#                         "Bureau Score": bureau_score,
#                         "DPD 90+": dpd_90_6m,
#                         "Utilization": f"{credit_utilization}%",
#                         "Recent Inquiries": recent_inquiries,
#                         "Existing EMI": f"Rs.{existing_emi:,}"
#                     }
#                 )
                
#                 render_info_card(
#                     "📋 Loan Request", 
#                     "📋",
#                     {
#                         "Amount": f"Rs.{loan_amount:,}",
#                         "Tenure": f"{loan_tenure} months",
#                         "Interest Rate": f"{interest_rate}%",
#                         "Requested EMI": f"Rs.{amt_annuity:,}"
#                     }
#                 )
        
#         with tab2:
#             st.markdown('<p class="section-header">Decision Summary</p>', unsafe_allow_html=True)
            
#             render_decision_header(decision_data, customer_data)
            
#             st.markdown("<br>", unsafe_allow_html=True)
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 # Identity card
#                 age_pass = 18 <= age <= 65
#                 kyc_pass = kyc_verified
                
#                 render_info_card(
#                     "Identity & Eligibility",
#                     "👤",
#                     {
#                         f"Age: {age}": "",
#                         f"Employment: {employment_type}": "",
#                         f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""
#                     },
#                     {
#                         f"Age: {age}": "pass" if age_pass else "fail",
#                         f"Employment: {employment_type}": "pass",
#                         f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_pass else "fail"
#                     }
#                 )
            
#             with col2:
#                 # Credit card
#                 bureau_pass = bureau_score >= 550
#                 dpd_pass = dpd_90_6m == 0
                
#                 render_info_card(
#                     "Credit Bureau",
#                     "🏦",
#                     {
#                         f"Bureau Score: {bureau_score}": "",
#                         f"DPD 90+: {dpd_90_6m}": "",
#                         f"Utilization: {credit_utilization}%": ""
#                     },
#                     {
#                         f"Bureau Score: {bureau_score}": "pass" if bureau_pass else "fail",
#                         f"DPD 90+: {dpd_90_6m}": "pass" if dpd_pass else "fail",
#                         f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"
#                     }
#                 )
            
#             with col3:
#                 # Affordability card
#                 affordability = decision_data.get('affordability_data', {})
#                 foir = affordability.get('foir_percentage', 0)
#                 total_emi = affordability.get('total_emi', 0)
#                 net_disp = affordability.get('net_disposable', 0)
                
#                 render_info_card(
#                     "Affordability",
#                     "💰",
#                     {
#                         f"Monthly Income: Rs.{avg_salary:,}": "",
#                         f"FOIR: {foir:.1f}%": "",
#                         f"Total EMI: Rs.{total_emi:,}": "",
#                         f"Net Disposable: Rs.{net_disp:,}": ""
#                     },
#                     {
#                         f"Monthly Income: Rs.{avg_salary:,}": "pass",
#                         f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
#                         f"Total EMI: Rs.{total_emi:,}": "pass",
#                         f"Net Disposable: Rs.{net_disp:,}": "pass" if net_disp >= 10000 else "warning"
#                     }
#                 )
            
#             st.markdown("<br>", unsafe_allow_html=True)
            
#             # Reason codes
#             render_reason_codes(reasons)
            
#             st.markdown("<br>", unsafe_allow_html=True)
            
#             # Action buttons
#             col1, col2, col3 = st.columns([1, 1, 2])
            
#             with col1:
#                 try:
#                     pdf_buffer = generate_decision_pdf(
#                         decision_data=decision_data,
#                         customer_data=customer_data,
#                         affordability_data=decision_data.get('affordability_data', {}),
#                         reasons=reasons
#                     )
                    
#                     st.download_button(
#                         label="📥 Download Report (PDF)",
#                         data=pdf_buffer,
#                         file_name=f"credit_decision_{app_id}.pdf",
#                         mime="application/pdf",
#                         use_container_width=True
#                     )
#                 except Exception as e:
#                     st.error(f"Error generating PDF: {str(e)}")
#                     st.info("Please ensure reportlab is installed: pip install reportlab")
            
#             with col2:
#                 if st.button("🔄 Re-Evaluate", use_container_width=True):
#                     st.rerun()
        
#         with tab3:
#             st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 fig1 = create_modern_gauge(decision_data['confidence'], "Model Confidence")
#                 st.plotly_chart(fig1, use_container_width=True)
            
#             with col2:
#                 fig2 = create_modern_bar_chart(decision_data['class_probs'])
#                 st.plotly_chart(fig2, use_container_width=True)
            
#             st.markdown("<br>", unsafe_allow_html=True)
            
#             # Policy checks
#             st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
            
#             policy_df = pd.DataFrame([
#                 {'Check': k, 'Result': v} 
#                 for k, v in decision_data['policy_checks'].items()
#             ])
#             st.dataframe(policy_df, use_container_width=True, hide_index=True)
        
#         with tab4:
#             st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
            
#             audit_log = {
#                 'application_id': app_id,
#                 'timestamp': timestamp.isoformat(),
#                 'decision': decision_data['decision'],
#                 'risk_score': decision_data['risk_score'],
#                 'pd_percentage': decision_data['pd_percentage'],
#                 'confidence': round(decision_data['confidence'], 2),
#                 'model_version': '8.0',
#                 'reason_codes': reasons,
#                 'affordability': decision_data.get('affordability_data', {})
#             }
            
#             st.json(audit_log)
            
#             audit_json = json.dumps(audit_log, indent=2)
#             st.download_button(
#                 "📥 Download Audit Log",
#                 audit_json,
#                 f"audit_{app_id}.json",
#                 "application/json",
#                 use_container_width=True
#             )

# # =============================================================================
# # BATCH PROCESSING PAGE
# # =============================================================================

# elif page == "📊 Batch Process":
#     st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
    
#     st.markdown("""
#         <div class="info-box">
#             📤 Upload a CSV file with customer data for bulk credit assessment. 
#             The file should include all required fields for prediction.
#         </div>
#     """, unsafe_allow_html=True)
    
#     # File upload
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
#     if uploaded_file is not None:
#         try:
#             # Read the CSV file
#             df = pd.read_csv(uploaded_file)
            
#             st.success(f"✅ Successfully loaded {len(df)} records")
            
#             # Show preview
#             with st.expander("📄 Preview Uploaded Data"):
#                 st.dataframe(df.head(), use_container_width=True)
#                 st.write(f"**Total Records:** {len(df)}")
#                 st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            
#             # Required columns check
#             required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
#             missing_cols = [col for col in required_cols if col not in df.columns]
            
#             if missing_cols:
#                 st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
#                 st.info("Please ensure your CSV includes at least these columns: age, employment_type, avg_salary_6m, bureau_score, loan_amount")
#             else:
#                 # Process batch predictions
#                 if st.button("🚀 Process Batch Predictions", use_container_width=True, type="primary"):
#                     with st.spinner(f"🔍 Processing {len(df)} records..."):
#                         progress_bar = st.progress(0)
                        
#                         # Process batch
#                         results_df = process_batch_predictions(df)
                        
#                         progress_bar.progress(100)
                        
#                         st.success(f"✅ Completed processing {len(results_df)} records!")
                        
#                         # Show results
#                         tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
                        
#                         with tab1:
#                             st.dataframe(results_df, use_container_width=True)
                            
#                             # Summary statistics
#                             col1, col2, col3, col4 = st.columns(4)
#                             with col1:
#                                 approved_count = len(results_df[results_df['decision'] == 'APPROVE'])
#                                 st.metric("✅ Approved", approved_count)
#                             with col2:
#                                 rejected_count = len(results_df[results_df['decision'] == 'REJECT'])
#                                 st.metric("❌ Rejected", rejected_count)
#                             with col3:
#                                 review_count = len(results_df[results_df['decision'] == 'REVIEW'])
#                                 st.metric("⚠️ Review", review_count)
#                             with col4:
#                                 avg_risk = results_df['risk_score'].mean()
#                                 st.metric("📊 Avg Risk Score", f"{avg_risk:.0f}")
                        
#                         with tab2:
#                             # Visualizations
#                             col1, col2 = st.columns(2)
                            
#                             with col1:
#                                 # Decision distribution
#                                 decision_counts = results_df['decision'].value_counts()
#                                 fig1 = px.pie(
#                                     values=decision_counts.values,
#                                     names=decision_counts.index,
#                                     title="Decision Distribution",
#                                     color=decision_counts.index,
#                                     color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}
#                                 )
#                                 st.plotly_chart(fig1, use_container_width=True)
                            
#                             with col2:
#                                 # Risk score distribution
#                                 fig2 = px.histogram(
#                                     results_df,
#                                     x='risk_score',
#                                     title="Risk Score Distribution",
#                                     nbins=20,
#                                     color_discrete_sequence=['#587042']
#                                 )
#                                 st.plotly_chart(fig2, use_container_width=True)
                            
#                             # FOIR analysis
#                             fig3 = px.scatter(
#                                 results_df,
#                                 x='monthly_income',
#                                 y='loan_amount',
#                                 color='decision',
#                                 size='risk_score',
#                                 title="Income vs Loan Amount (Colored by Decision)",
#                                 hover_data=['application_id', 'foir_percentage'],
#                                 color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}
#                             )
#                             st.plotly_chart(fig3, use_container_width=True)
                        
#                         with tab3:
#                             st.markdown("### Download Results")
                            
#                             # Download options
#                             col1, col2 = st.columns(2)
                            
#                             with col1:
#                                 # Download CSV
#                                 csv = results_df.to_csv(index=False)
#                                 st.download_button(
#                                     label="📥 Download as CSV",
#                                     data=csv,
#                                     file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                     mime="text/csv",
#                                     use_container_width=True
#                                 )
                            
#                             with col2:
#                                 # Download JSON
#                                 json_data = results_df.to_json(orient='records', indent=2)
#                                 st.download_button(
#                                     label="📥 Download as JSON",
#                                     data=json_data,
#                                     file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                                     mime="application/json",
#                                     use_container_width=True
#                                 )
                            
#                             # Filtered downloads
#                             st.markdown("---")
#                             st.markdown("#### Filtered Downloads")
                            
#                             col1, col2, col3 = st.columns(3)
                            
#                             with col1:
#                                 approved_df = results_df[results_df['decision'] == 'APPROVE']
#                                 if len(approved_df) > 0:
#                                     st.download_button(
#                                         label=f"✅ Approved Only ({len(approved_df)})",
#                                         data=approved_df.to_csv(index=False),
#                                         file_name=f"approved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                         mime="text/csv",
#                                         use_container_width=True
#                                     )
                            
#                             with col2:
#                                 rejected_df = results_df[results_df['decision'] == 'REJECT']
#                                 if len(rejected_df) > 0:
#                                     st.download_button(
#                                         label=f"❌ Rejected Only ({len(rejected_df)})",
#                                         data=rejected_df.to_csv(index=False),
#                                         file_name=f"rejected_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                         mime="text/csv",
#                                         use_container_width=True
#                                     )
                            
#                             with col3:
#                                 review_df = results_df[results_df['decision'] == 'REVIEW']
#                                 if len(review_df) > 0:
#                                     st.download_button(
#                                         label=f"⚠️ Review Only ({len(review_df)})",
#                                         data=review_df.to_csv(index=False),
#                                         file_name=f"review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                         mime="text/csv",
#                                         use_container_width=True
#                                     )
            
#         except Exception as e:
#             st.error(f"❌ Error processing file: {str(e)}")
#             st.info("Please ensure the CSV file is properly formatted and contains the required columns.")
    
#     else:
#         # Show template download
#         st.markdown("---")
#         st.markdown("### 📋 CSV Template")
        
#         # Create template dataframe
#         template_data = {
#             'age': [35, 42, 28],
#             'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
#             'kyc_verified': ['Yes', 'Yes', 'No'],
#             'bankruptcy_flag': ['No', 'No', 'No'],
#             'fraud_flag': ['No', 'No', 'No'],
#             'employment_tenure_months': [24, 0, 18],
#             'business_vintage_years': [0, 5, 0],
#             'bureau_score': [720, 680, 580],
#             'dpd_90_count_6m': [0, 1, 2],
#             'credit_utilization_pct': [30, 45, 75],
#             'recent_inquiries_3m': [2, 1, 5],
#             'active_loans_count': [1, 2, 3],
#             'avg_salary_6m': [50000, 75000, 35000],
#             'AMT_INCOME_TOTAL': [600000, 900000, 420000],
#             'net_cash_surplus_6m': [20000, 35000, 10000],
#             'salary_stability_flag': ['STABLE', 'MODERATE', 'UNSTABLE'],
#             'loan_amount': [180000, 250000, 100000],
#             'loan_tenure_months': [24, 36, 12],
#             'interest_rate': [10.5, 11.0, 12.0],
#             'existing_emi': [15000, 20000, 8000],
#             'AMT_ANNUITY': [8500, 9500, 4500]
#         }
        
#         template_df = pd.DataFrame(template_data)
        
#         st.dataframe(template_df, use_container_width=True)
        
#         # Download template
#         csv_template = template_df.to_csv(index=False)
#         st.download_button(
#             label="📥 Download CSV Template",
#             data=csv_template,
#             file_name="credit_assessment_template.csv",
#             mime="text/csv",
#             use_container_width=True
#         )

# # =============================================================================
# # MODEL INFO PAGE
# # =============================================================================

# elif page == "📈 Model Info":
#     st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#             <div class="stat-card">
#                 <div class="stat-number">RF</div>
#                 <div class="stat-label">Model Type</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown(f"""
#             <div class="stat-card">
#                 <div class="stat-number">{len(TOP_FEATURES)}</div>
#                 <div class="stat-label">Features</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         st.markdown(f"""
#             <div class="stat-card">
#                 <div class="stat-number">{len(TARGET_LE.classes_)}</div>
#                 <div class="stat-label">Classes</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown("<br>", unsafe_allow_html=True)
    
#     st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
    
#     feature_df = pd.DataFrame({
#         'Rank': range(1, min(21, len(TOP_FEATURES) + 1)),
#         'Feature': TOP_FEATURES[:20]
#     })
    
#     st.dataframe(feature_df, use_container_width=True, hide_index=True)

# # =============================================================================
# # ABOUT PAGE
# # =============================================================================

# elif page == "ℹ️ About":
#     st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
    
#     st.markdown("""
#         <div class="info-card">
#             <div class="info-card-title">
#                 <span class="icon">🏦</span>
#                 <span>Credit Risk Assessment Platform</span>
#             </div>
#             <div class="info-card-content">
#                 <p><strong>Version:</strong> 8.0 - Sage Green & Yellow Theme</p>
#                 <p><strong>Developer:</strong> Zen Meraki</p>
#                 <p><strong>Date:</strong> January 2026</p>
#                 <br>
#                 <p>
#                     A comprehensive credit risk evaluation system combining hard policy rules,
#                     machine learning models, and affordability analysis for accurate and compliant
#                     lending decisions.
#                 </p>
#             </div>
#         </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("<br>", unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#             <div class="info-card">
#                 <div class="info-card-title">
#                     <span class="icon">🎯</span>
#                     <span>Key Features</span>
#                 </div>
#                 <div class="info-card-content">
#                     <ul style="margin: 0; padding-left: 1.25rem;">
#                         <li>Three-layer decision engine</li>
#                         <li>Real-time risk assessment</li>
#                         <li>FOIR calculation & validation</li>
#                         <li>Automated reason generation</li>
#                         <li>Complete audit trail</li>
#                         <li>Professional UI/UX</li>
#                     </ul>
#                 </div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#             <div class="info-card">
#                 <div class="info-card-title">
#                     <span class="icon">🛠️</span>
#                     <span>Technology Stack</span>
#                 </div>
#                 <div class="info-card-content">
#                     <ul style="margin: 0; padding-left: 1.25rem;">
#                         <li>Streamlit (UI Framework)</li>
#                         <li>Scikit-learn (ML)</li>
#                         <li>Plotly (Visualizations)</li>
#                         <li>Pandas (Data Processing)</li>
#                         <li>Python 3.8+</li>
#                     </ul>
#                 </div>
#             </div>
#         """, unsafe_allow_html=True)

# st.markdown("---")
# st.markdown("<div style='text-align: center; color: gray;'><p>© 2026 Hybrid Credit Risk System v8.0 | Zen Meraki</p></div>", 
#     unsafe_allow_html=True)









"""
Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
Enhanced with Modern UI/UX Design
Run with: streamlit run test.py

Author: Zen Meraki  
Date: January 2026
VERSION: 8.0 - Sage Green & Yellow Professional Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings
from datetime import datetime
import hashlib
import io
import base64
from typing import Dict, List, Any
import json
warnings.filterwarnings('ignore')
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.pdf_generator import generate_decision_pdf


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
# SAGE GREEN AND YELLOW THEME CSS
# =============================================================================

st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Color Variables */
    :root {
        --fern-green: #587042;
        --sage: #A9B494;
        --cosmic-latte: #FAF7E6;
        --jasmine: #F8DE8C;
        --saffron: #F6C531;
        --dark-fern: #486032;
        --light-sage: #D4DBC4;
    }
    
    /* Main Background */
    .main {
        background-color: #FFFFFF;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        background-color: #FFFFFF;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--fern-green);
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--fern-green);
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--sage);
    }
    
    /* Decision Cards */
    .decision-card {
        background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(88, 112, 66, 0.2);
        margin-bottom: 2rem;
        color: white;
    }
    
    .decision-card-approved {
        background: linear-gradient(135deg, var(--fern-green) 0%, #7A9E4D 100%);
        box-shadow: 0 10px 40px rgba(88, 112, 66, 0.3);
    }
    
    .decision-card-rejected {
        background: linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%);
        box-shadow: 0 10px 40px rgba(211, 47, 47, 0.2);
    }
    
    .decision-card-review {
        background: linear-gradient(135deg, var(--saffron) 0%, var(--jasmine) 100%);
        box-shadow: 0 10px 40px rgba(246, 197, 49, 0.3);
    }
    
    .decision-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .decision-subtitle {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Info Cards */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(88, 112, 66, 0.08);
        border: 1px solid var(--sage);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        box-shadow: 0 10px 25px rgba(88, 112, 66, 0.15);
        transform: translateY(-2px);
        border-color: var(--fern-green);
    }
    
    .info-card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--fern-green);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .info-card-content {
        color: #5A5A5A;
        line-height: 1.6;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--cosmic-latte) 0%, #F5F2E0 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid var(--fern-green);
        margin-bottom: 1rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--fern-green);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--fern-green);
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        gap: 0.5rem;
    }
    
    .badge-pass {
        background: #E8F5E9;
        color: var(--fern-green);
        border: 1px solid var(--sage);
    }
    
    .badge-fail {
        background: #FFEBEE;
        color: #D32F2F;
        border: 1px solid #FFCDD2;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, var(--cosmic-latte) 0%, var(--jasmine) 100%);
        color: #7B5800;
        border: 1px solid var(--jasmine);
    }
    
    .badge-info {
        background: #E3F2FD;
        color: #1565C0;
        border: 1px solid #90CAF9;
    }
    
    /* Data Row */
    .data-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid var(--sage);
    }
    
    .data-row:last-child {
        border-bottom: none;
    }
    
    .data-label {
        font-weight: 500;
        color: #5A5A5A;
    }
    
    .data-value {
        font-weight: 600;
        color: var(--fern-green);
    }
    
    /* Reason Items */
    .reason-item {
        background: linear-gradient(135deg, #F9F7EB 0%, #F5F2E0 100%);
        padding: 1rem 1.25rem;
        border-radius: 8px;
        border-left: 4px solid var(--saffron);
        margin-bottom: 0.75rem;
        color: #7B5800;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .reason-icon {
        font-size: 1.25rem;
        color: var(--fern-green);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--fern-green) 0%, var(--dark-fern) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(88, 112, 66, 0.2);
    }
    
    .stButton > button:hover {
        box-shadow: 0 10px 20px rgba(88, 112, 66, 0.3);
        transform: translateY(-2px);
        background: linear-gradient(135deg, var(--dark-fern) 0%, var(--fern-green) 100%);
    }
    
    /* Form Inputs */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 1px solid var(--sage);
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--fern-green);
        box-shadow: 0 0 0 3px rgba(88, 112, 66, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(88, 112, 66, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        background-color: transparent;
        border-radius: 8px;
        color: #718096;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
        color: white;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--cosmic-latte);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--cosmic-latte) 0%, #F5F2E0 100%);
        border-right: 1px solid var(--sage);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: var(--fern-green);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--cosmic-latte);
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: 600;
        color: var(--fern-green);
        border: 1px solid var(--sage);
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1rem 1.5rem;
    }
    
    /* Success Alert */
    [data-baseweb="notification"] {
        background-color: #E8F5E9;
        border-left: 4px solid var(--fern-green);
        border-radius: 8px;
    }
    
    /* Info Alert */
    .info-box {
        background: linear-gradient(135deg, #F9F7EB 0%, var(--cosmic-latte) 100%);
        border-left: 4px solid var(--sage);
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        color: var(--fern-green);
        border: 1px solid var(--sage);
    }
    
    /* Warning Alert */
    .warning-box {
        background: linear-gradient(135deg, var(--cosmic-latte) 0%, var(--jasmine) 100%);
        border-left: 4px solid var(--saffron);
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        color: #7B5800;
        border: 1px solid var(--jasmine);
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(88, 112, 66, 0.05);
        border: 1px solid var(--sage);
    }
    
    /* Metric Container */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--fern-green);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--fern-green);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.8;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--fern-green) 0%, var(--sage) 100%);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid var(--sage);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--cosmic-latte);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--sage);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--fern-green);
    }
    
    /* Icon Styles */
    .icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        color: var(--fern-green);
    }
    
    /* Card Grid */
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    /* Feature Badge */
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--fern-green) 0%, var(--sage) 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Timeline */
    .timeline-item {
        position: relative;
        padding-left: 2rem;
        padding-bottom: 1.5rem;
        border-left: 2px solid var(--sage);
    }
    
    .timeline-item:last-child {
        border-left: none;
    }
    
    .timeline-dot {
        position: absolute;
        left: -6px;
        top: 0;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: var(--fern-green);
    }
    
    /* Stat Card */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(88, 112, 66, 0.08);
        border-top: 4px solid var(--fern-green);
        text-align: center;
        border: 1px solid var(--sage);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        box-shadow: 0 10px 25px rgba(88, 112, 66, 0.15);
        transform: translateY(-2px);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--fern-green);
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--fern-green);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.8;
    }
    
    /* Chart styling */
    .js-plotly-plot .plotly {
        background-color: white !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: white;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid var(--sage);
    }
    
    .stRadio > div[data-baseweb="radio"] label {
        color: var(--fern-green);
    }
    
    /* Checkbox */
    .stCheckbox > label {
        color: var(--fern-green);
    }
    
    /* Form submit button */
    div[data-testid="stForm"] button {
        background: linear-gradient(135deg, var(--fern-green) 0%, var(--dark-fern) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(88, 112, 66, 0.2);
    }
    
    div[data-testid="stForm"] button:hover {
        box-shadow: 0 10px 20px rgba(88, 112, 66, 0.3);
        transform: translateY(-2px);
        background: linear-gradient(135deg, var(--dark-fern) 0%, var(--fern-green) 100%);
    }
    
    /* Table styling */
    .stTable {
        border: 1px solid var(--sage);
        border-radius: 8px;
    }
    
    /* Container borders */
    .stApp {
        background-color: white;
    }
    
    /* ================== FIXES FOR TEXT VISIBILITY ================== */
    /* Input text color - WHITE on DARK background for readability */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        color: #FFFFFF !important;
        background-color: #2D3748 !important;
    }
    
    /* Radio button text - DARK on LIGHT background */
    .stRadio > div > label,
    .stRadio > div > label > div > p {
        color: #333333 !important;
    }
    
    /* Checkbox text - DARK on LIGHT background */
    .stCheckbox > label,
    .stCheckbox > label > div > p {
        color: #333333 !important;
    }
    
    /* Slider value text */
    .stSlider > div > div > div {
        color: #333333 !important;
    }
    
    /* Placeholder text - Light gray for visibility */
    ::placeholder {
        color: #A0AEC0 !important;
        opacity: 1 !important;
    }
    
    /* Sidebar text - DARK for readability */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span {
        color: #333333 !important;
    }
    
    /* Form labels - Keep GREEN for branding */
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stRadio label,
    .stCheckbox label {
        color: var(--fern-green) !important;
        font-weight: 600;
    }
    
    /* General text color for body content */
    .main p, .main div, .main span {
        color: #333333 !important;
    }
    
    /* Fix for text in expanders */
    .streamlit-expanderContent p,
    .streamlit-expanderContent div,
    .streamlit-expanderContent span {
        color: #333333 !important;
    }
    
    /* Fix for text in alerts */
    .stAlert p,
    .stAlert div,
    .stAlert span {
        color: inherit !important;
    }
    
    /* Fix for dropdown select options - WHITE on DARK */
    .stSelectbox select option {
        color: #FFFFFF !important;
        background-color: #2D3748 !important;
    }
    
    /* Fix for number input increment/decrement buttons */
    .stNumberInput button {
        color: #FFFFFF !important;
        background-color: #4A5568 !important;
    }
    
    .stNumberInput button:hover {
        background-color: var(--fern-green) !important;
    }
    
    /* Input focus state - Maintain WHITE text on DARK background */
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus {
        color: #FFFFFF !important;
        background-color: #2D3748 !important;
        border-color: var(--fern-green);
        box-shadow: 0 0 0 3px rgba(88, 112, 66, 0.1);
    }
    
    /* Ensure all input fields maintain dark background with white text */
    input[type="number"],
    input[type="text"],
    select {
        color: #FFFFFF !important;
        background-color: #2D3748 !important;
    }
    
    /* Fix for disabled inputs */
    input:disabled,
    select:disabled {
        color: #A0AEC0 !important;
        background-color: #1A202C !important;
        opacity: 0.6;
    }
    
    /* Ensure dropdown arrow is visible */
    .stSelectbox svg {
        fill: #FFFFFF !important;
    }
    
    /* Fix for input number spinner buttons */
    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-webkit-outer-spin-button {
        opacity: 1;
        background-color: #4A5568;
    }
    /* ================== END OF FIXES ================== */
    
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
# AFFORDABILITY CALCULATION ENGINE
# =============================================================================

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
    """Calculate comprehensive affordability metrics"""
    
    new_emi = calculate_emi(loan_amount, interest_rate, tenure_months)
    total_emi = new_emi + existing_emi
    foir_percentage = (total_emi / monthly_income) * 100 if monthly_income > 0 else 0
    net_disposable = monthly_income - total_emi
    max_allowed_emi = monthly_income * 0.50
    recommended_emi = monthly_income * 0.40
    affordable = foir_percentage <= 50
    within_recommended = foir_percentage <= 40
    
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

# =============================================================================
# REASON CODE GENERATION SYSTEM
# =============================================================================

APPROVAL_REASONS = {
    'high_bureau': 'Excellent credit score ({score})',
    'stable_employment': 'Stable employment history ({tenure} months)',
    'low_foir': 'Affordable EMI burden (FOIR: {foir}%)',
    'clean_payment': 'Clean payment history (No DPD)',
    'strong_income': 'Strong monthly income (Rs.{income:,})',
    'low_utilization': 'Low credit utilization ({util}%)',
}

REJECTION_REASONS = {
    'low_bureau': 'Credit score below minimum ({score} < 550)',
    'high_foir': 'EMI burden too high (FOIR: {foir}% > 50%)',
    'severe_dpd': 'Severe payment delays ({dpd} instances of 90+ DPD)',
    'low_income': 'Income below minimum threshold (Rs.{income:,} < Rs.15,000)',
    'short_employment': 'Insufficient employment tenure ({tenure} months < 6)',
    'bankruptcy': 'Active bankruptcy detected',
    'kyc_failed': 'KYC verification not completed',
    'high_utilization': 'High credit utilization ({util}% > 80%)',
    'age_invalid': 'Age outside acceptable range ({age} years)'
}

REVIEW_REASONS = {
    'borderline_bureau': 'Credit score in borderline range ({score})',
    'moderate_foir': 'EMI burden moderate (FOIR: {foir}%)',
    'mixed_signals': 'Mixed credit indicators requiring human review',
    'recent_employment': 'Recent employment change requiring verification',
}


def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
    """Generate top 3 reason codes for the decision"""
    reasons = []
    
    bureau_score = customer_data.get('bureau_score', 0)
    foir = affordability_data.get('foir_percentage', 0)
    dpd_90 = customer_data.get('dpd_90_count_6m', 0)
    income = customer_data.get('avg_salary_6m', 0)
    employment_tenure = customer_data.get('employment_tenure_months', 0)
    credit_util = customer_data.get('credit_utilization_pct', 0)
    age = customer_data.get('age', 0)
    
    if decision == "APPROVE":
        if bureau_score >= 750:
            reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
        if employment_tenure >= 24:
            reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
        if foir <= 40:
            reasons.append(APPROVAL_REASONS['low_foir'].format(foir=round(foir, 1)))
        if dpd_90 == 0:
            reasons.append(APPROVAL_REASONS['clean_payment'])
        if income >= 75000:
            reasons.append(APPROVAL_REASONS['strong_income'].format(income=income))
        if credit_util <= 30:
            reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))
    
    elif decision == "REJECT":
        for check_name, check_result in policy_checks.items():
            if '❌' in str(check_result):
                if 'bureau' in check_name.lower():
                    reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
                elif 'dpd' in check_name.lower():
                    reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
                elif 'income' in check_name.lower():
                    reasons.append(REJECTION_REASONS['low_income'].format(income=income))
                elif 'tenure' in check_name.lower():
                    reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
                elif 'kyc' in check_name.lower():
                    reasons.append(REJECTION_REASONS['kyc_failed'])
                elif 'bankruptcy' in check_name.lower():
                    reasons.append(REJECTION_REASONS['bankruptcy'])
                elif 'age' in check_name.lower():
                    reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))
        
        if foir > 50:
            reasons.append(REJECTION_REASONS['high_foir'].format(foir=round(foir, 1)))
        if credit_util > 80:
            reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
    
    elif decision == "REVIEW":
        if 650 <= bureau_score < 700:
            reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
        if 40 < foir <= 50:
            reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=round(foir, 1)))
        if employment_tenure < 12:
            reasons.append(REVIEW_REASONS['recent_employment'])
        if not reasons:
            reasons.append(REVIEW_REASONS['mixed_signals'])
    
    return reasons[:3] if reasons else ['Decision based on model assessment']

# =============================================================================
# RISK SCORE CALCULATION
# =============================================================================

# def calculate_final_risk_score(bureau_score, ml_confidence, foir):
#     """Calculate final risk score (0-1000)"""
#     bureau_points = (bureau_score / 900) * 400
#     ml_points = (ml_confidence / 100) * 400
#     foir_points = max(0, (1 - foir/50) * 200)
#     total_score = int(bureau_points + ml_points + foir_points)
#     return min(max(total_score, 0), 1000)



def bureau_to_pd(bureau_score):
    """Convert bureau score to PD percentage"""
    if bureau_score >= 750:
        return 1.0
    elif bureau_score >= 700:
        return 2.0
    elif bureau_score >= 650:
        return 4.0
    elif bureau_score >= 600:
        return 6.5
    elif bureau_score >= 550:
        return 10.0
    else:
        return 15.0

def foir_to_pd(foir):
    """Convert FOIR to PD percentage"""
    if foir <= 30:
        return 1.5
    elif foir <= 40:
        return 3.0
    elif foir <= 50:
        return 5.5
    else:
        return 12.0

def confidence_to_pd(confidence):
    """Convert ML confidence to PD percentage"""
    if confidence >= 90:
        return 1.0
    elif confidence >= 80:
        return 2.5
    elif confidence >= 70:
        return 4.0
    elif confidence >= 60:
        return 6.0
    else:
        return 10.0

def calculate_final_pd(bureau_score, foir, confidence):
    """Calculate final Probability of Default (PD) percentage"""
    bureau_pd = bureau_to_pd(bureau_score)
    foir_pd = foir_to_pd(foir)
    ml_pd = confidence_to_pd(confidence)
    
    final_pd = (
        0.40 * bureau_pd +
        0.35 * foir_pd +
        0.25 * ml_pd
    )
    
    final_pd = max(0.5, min(final_pd, 15.0))
    return round(final_pd, 2)

def calculate_final_risk_score(bureau_score, ml_confidence, foir):
    """Calculate final risk score (0-1000)"""
    bureau_points = (bureau_score / 900) * 400
    ml_points = (ml_confidence / 100) * 400
    foir_points = max(0, (1 - foir/50) * 200)
    total_score = int(bureau_points + ml_points + foir_points)
    return min(max(total_score, 0), 1000)


# =============================================================================
# BATCH PREDICTION ENGINE
# =============================================================================

# def process_batch_predictions(df: pd.DataFrame) -> pd.DataFrame:
#     """Process batch predictions for multiple records"""
#     results = []
    
#     for idx, row in df.iterrows():
#         customer_dict = row.to_dict()
        
#         # Convert yes/no to boolean
#         for key, value in customer_dict.items():
#             if isinstance(value, str):
#                 if value.lower() in ['yes', 'true', '1']:
#                     customer_dict[key] = True
#                 elif value.lower() in ['no', 'false', '0']:
#                     customer_dict[key] = False
        
#         # Add missing required fields with defaults
#         required_fields = {
#             'kyc_verified': True,
#             'bankruptcy_flag': False,
#             'fraud_flag': False,
#             'dpd_90_count_6m': 0,
#             'recent_inquiries_3m': 0,
#             'active_loans_count': 0,
#             'existing_emi': 0,
#             'salary_stability_flag': 'STABLE'
#         }
        
#         for field, default in required_fields.items():
#             if field not in customer_dict:
#                 customer_dict[field] = default
        
#         # Get decision
#         decision_data = make_hybrid_decision_enhanced(customer_dict)
        
#         # Generate application ID
#         app_id = f"BATCH_{idx+1:04d}"
        
#         # Prepare result
#         result = {
#             'application_id': app_id,
#             'decision': decision_data['decision'],
#             'risk_score': decision_data['risk_score'],
#             'pd_percentage': decision_data['pd_percentage'],
#             'confidence': round(decision_data['confidence'], 2),
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         }
        
#         # Add key customer data
#         result.update({
#             'age': customer_dict.get('age', ''),
#             'employment_type': customer_dict.get('employment_type', ''),
#             'bureau_score': customer_dict.get('bureau_score', ''),
#             'monthly_income': customer_dict.get('avg_salary_6m', ''),
#             'loan_amount': customer_dict.get('loan_amount', ''),
#             'foir_percentage': decision_data.get('affordability_data', {}).get('foir_percentage', 0)
#         })
        
#         results.append(result)
    
#     return pd.DataFrame(results)


def process_batch_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Process batch predictions for multiple records with complete information"""
    results = []
    
    for idx, row in df.iterrows():
        customer_dict = row.to_dict()
        
        # Convert yes/no to boolean
        for key, value in customer_dict.items():
            if isinstance(value, str):
                if value.lower() in ['yes', 'true', '1']:
                    customer_dict[key] = True
                elif value.lower() in ['no', 'false', '0']:
                    customer_dict[key] = False
        
        # Add missing required fields with defaults
        required_fields = {
            'kyc_verified': True,
            'bankruptcy_flag': False,
            'fraud_flag': False,
            'dpd_90_count_6m': 0,
            'recent_inquiries_3m': 0,
            'active_loans_count': 0,
            'existing_emi': 0,
            'salary_stability_flag': 'STABLE',
            'credit_utilization_pct': 30,
            'employment_tenure_months': 24,
            'business_vintage_years': 0,
            'net_cash_surplus_6m': 20000,
            'loan_tenure_months': 24,
            'interest_rate': 10.5,
            'AMT_ANNUITY': 8500,
            'AMT_INCOME_TOTAL': 600000
        }
        
        for field, default in required_fields.items():
            if field not in customer_dict or pd.isna(customer_dict[field]):
                customer_dict[field] = default
        
        # Get decision
        try:
            decision_data = make_hybrid_decision_enhanced(customer_dict)
            
            # Generate reasons - THIS IS NEW!
            reasons = generate_reason_codes(
                decision=decision_data['decision'],
                customer_data=customer_dict,
                affordability_data=decision_data.get('affordability_data', {}),
                policy_checks=decision_data['policy_checks']
            )
            
            # Generate application ID
            app_id = f"BATCH_{idx+1:04d}"
            
            # Get affordability data
            affordability = decision_data.get('affordability_data', {})
            
            # Prepare comprehensive result with ALL columns
            result = {
                'application_id': app_id,
                'decision': decision_data['decision'],
                'risk_score': decision_data['risk_score'],
                'pd_percentage': decision_data['pd_percentage'],
                'confidence': round(decision_data['confidence'], 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                
                # ===== REASON CODES - THIS IS WHAT YOU WANTED! =====
                'reason_1': reasons[0] if len(reasons) > 0 else '',
                'reason_2': reasons[1] if len(reasons) > 1 else '',
                'reason_3': reasons[2] if len(reasons) > 2 else '',
                # ===================================================
                
                # Customer Details
                'age': customer_dict.get('age', ''),
                'employment_type': customer_dict.get('employment_type', ''),
                'bureau_score': customer_dict.get('bureau_score', ''),
                'monthly_income': customer_dict.get('avg_salary_6m', ''),
                'loan_amount': customer_dict.get('loan_amount', ''),
                'loan_tenure_months': customer_dict.get('loan_tenure_months', ''),
                'interest_rate': customer_dict.get('interest_rate', ''),
                
                # Affordability Metrics
                'new_emi': affordability.get('new_emi', 0),
                'existing_emi': affordability.get('existing_emi', 0),
                'total_emi': affordability.get('total_emi', 0),
                'foir_percentage': round(affordability.get('foir_percentage', 0), 2),
                'net_disposable': affordability.get('net_disposable', 0),
                'affordability_status': affordability.get('status', 'N/A'),
                
                # Credit Bureau Details
                'dpd_90_count': customer_dict.get('dpd_90_count_6m', 0),
                'credit_utilization': customer_dict.get('credit_utilization_pct', 0),
                'recent_inquiries': customer_dict.get('recent_inquiries_3m', 0),
                'active_loans': customer_dict.get('active_loans_count', 0),
                
                # Employment Details
                'employment_tenure': customer_dict.get('employment_tenure_months', 0),
                'business_vintage': customer_dict.get('business_vintage_years', 0),
                'salary_stability': customer_dict.get('salary_stability_flag', ''),
                
                # Policy Checks Summary
                'kyc_status': 'Verified' if customer_dict.get('kyc_verified', True) else 'Not Verified',
                'bankruptcy': 'Yes' if customer_dict.get('bankruptcy_flag', False) else 'No',
                'fraud': 'Yes' if customer_dict.get('fraud_flag', False) else 'No',
                
                # Model Probabilities
                'prob_approve': round(decision_data.get('class_probs', {}).get('APPROVE', 0), 2),
                'prob_review': round(decision_data.get('class_probs', {}).get('REVIEW', 0), 2),
                'prob_reject': round(decision_data.get('class_probs', {}).get('REJECT', 0), 2),
            }
            
        except Exception as e:
            # If processing fails, create error record
            result = {
                'application_id': f"BATCH_{idx+1:04d}",
                'decision': 'ERROR',
                'risk_score': 0,
                'pd_percentage': 0,
                'confidence': 0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'reason_1': '',
                'reason_2': '',
                'reason_3': '',
                'age': customer_dict.get('age', ''),
                'employment_type': customer_dict.get('employment_type', ''),
                'bureau_score': customer_dict.get('bureau_score', ''),
                'monthly_income': customer_dict.get('avg_salary_6m', ''),
                'loan_amount': customer_dict.get('loan_amount', ''),
                'error_message': str(e)
            }
        
        results.append(result)
    
    return pd.DataFrame(results)

def create_download_link(df: pd.DataFrame, filename: str = "batch_results.csv") -> str:
    """Create a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;">📥 Download CSV</a>'
    return href

# =============================================================================
# ENHANCED HYBRID DECISION ENGINE
# =============================================================================

def make_hybrid_decision_enhanced(customer_dict):
    """Enhanced decision engine with complete data"""
    
    policy_checks = {}
    
    # Policy Gates
    age = customer_dict.get('age', 0)
    employment_type = customer_dict.get('employment_type', 'Salaried')
    kyc_verified = customer_dict.get('kyc_verified', True)
    bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
    fraud_flag = customer_dict.get('fraud_flag', False)
    
    if employment_type in ['Salaried']:
        age_min, age_max = 18, 65
    else:
        age_min, age_max = 18, 70
    
    if age < age_min or age > age_max:
        policy_checks['age'] = f"❌ Age {age} (Required: {age_min}-{age_max})"
        return {
            'decision': "REJECT",
            'reason': f"Policy Gate: Age outside allowed range",
            'confidence': 0,
            'class_probs': {'REJECT': 100},
            'policy_checks': policy_checks,
            'risk_score': 0,
            'pd_percentage': 100.0
        }
    policy_checks['age'] = f"✅ Age {age} (Valid)"
    
    if not kyc_verified:
        policy_checks['kyc'] = "❌ KYC Not Verified"
        return {
            'decision': "REJECT",
            'reason': "Policy Gate: KYC verification required",
            'confidence': 0,
            'class_probs': {'REJECT': 100},
            'policy_checks': policy_checks,
            'risk_score': 0,
            'pd_percentage': 100.0
        }
    policy_checks['kyc'] = "✅ KYC Verified"
    
    if bankruptcy_flag:
        policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
        return {
            'decision': "REJECT",
            'reason': "Policy Gate: Active bankruptcy",
            'confidence': 0,
            'class_probs': {'REJECT': 100},
            'policy_checks': policy_checks,
            'risk_score': 0,
            'pd_percentage': 100.0
        }
    policy_checks['bankruptcy'] = "✅ No Bankruptcy"
    
    if fraud_flag:
        policy_checks['fraud'] = "❌ Fraud Flag"
        return {
            'decision': "REJECT",
            'reason': "Policy Gate: Fraud detected",
            'confidence': 0,
            'class_probs': {'REJECT': 100},
            'policy_checks': policy_checks,
            'risk_score': 0,
            'pd_percentage': 100.0
        }
    policy_checks['fraud'] = "✅ No Fraud History"
    
    monthly_income = customer_dict.get('avg_salary_6m', 0)
    employment_tenure = customer_dict.get('employment_tenure_months', 0)
    business_vintage = customer_dict.get('business_vintage_years', 0)
    
    if monthly_income < 15000:
        policy_checks['income'] = f"❌ Income Rs.{monthly_income:,.0f} (Min: Rs.15,000)"
        return {
            'decision': "REJECT",
            'reason': "Policy Gate: Income below minimum",
            'confidence': 0,
            'class_probs': {'REJECT': 100},
            'policy_checks': policy_checks,
            'risk_score': 0,
            'pd_percentage': 100.0
        }
    policy_checks['income'] = f"✅ Income Rs.{monthly_income:,.0f}"
    
    if employment_type == 'Salaried' and employment_tenure < 6:
        policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
        return {
            'decision': "REJECT",
            'reason': "Policy Gate: Insufficient tenure",
            'confidence': 0,
            'class_probs': {'REJECT': 100},
            'policy_checks': policy_checks,
            'risk_score': 0,
            'pd_percentage': 100.0
        }
    elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
        policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
        return {
            'decision': "REJECT",
            'reason': "Policy Gate: Insufficient business vintage",
            'confidence': 0,
            'class_probs': {'REJECT': 100},
            'policy_checks': policy_checks,
            'risk_score': 0,
            'pd_percentage': 100.0
        }
    
    if employment_type == 'Salaried':
        policy_checks['tenure'] = f"✅ Tenure {employment_tenure} months"
    else:
        policy_checks['tenure'] = f"✅ Business Vintage {business_vintage} years"
    
    bureau_score = customer_dict.get('bureau_score', 0)
    dpd_90 = customer_dict.get('dpd_90_count_6m', 0)
    credit_utilization = customer_dict.get('credit_utilization_pct', 0)
    recent_inquiries = customer_dict.get('recent_inquiries_3m', 0)
    
    if bureau_score < 550:
        policy_checks['bureau'] = f"❌ Bureau Score {bureau_score} (Min: 550)"
        return {
            'decision': "REJECT",
            'reason': "Policy Gate: Bureau score too low",
            'confidence': 0,
            'class_probs': {'REJECT': 100},
            'policy_checks': policy_checks,
            'risk_score': 0,
            'pd_percentage': 100.0
        }
    policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"
    
    if dpd_90 > 0:
        policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD"
        return {
            'decision': "REJECT",
            'reason': "Policy Gate: Severe delinquency",
            'confidence': 0,
            'class_probs': {'REJECT': 100},
            'policy_checks': policy_checks,
            'risk_score': 0,
            'pd_percentage': 100.0
        }
    policy_checks['dpd'] = "✅ No 90+ DPD"
    
    if credit_utilization > 80:
        policy_checks['utilization'] = f"⚠️ High utilization {credit_utilization}%"
    else:
        policy_checks['utilization'] = f"✅ Utilization {credit_utilization}%"
    
    if recent_inquiries > 5:
        policy_checks['inquiries'] = f"⚠️ {recent_inquiries} recent inquiries"
    else:
        policy_checks['inquiries'] = f"✅ {recent_inquiries} inquiries"
    
    # ML Prediction
    input_df = pd.DataFrame([customer_dict])
    
    for col in TOP_FEATURES:
        if col not in input_df.columns:
            if col in LE_MAP:
                input_df[col] = "Unknown"
            else:
                input_df[col] = 0
    
    for col, le in LE_MAP.items():
        if col in input_df.columns:
            val = str(input_df[col].values[0])
            try:
                input_df[col] = le.transform([val])[0]
            except ValueError:
                input_df[col] = 0
    
    final_input = input_df[TOP_FEATURES]
    
    pred_idx = MODEL.predict(final_input)[0]
    ml_decision = TARGET_LE.inverse_transform([pred_idx])[0]
    
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
    
    # Affordability
    loan_amount = customer_dict.get('loan_amount', 0)
    loan_tenure = customer_dict.get('loan_tenure_months', 12)
    interest_rate = customer_dict.get('interest_rate', 10.5)
    existing_emi = customer_dict.get('existing_emi', 0)
    
    affordability_data = calculate_affordability(
        monthly_income=monthly_income,
        loan_amount=loan_amount,
        interest_rate=interest_rate,
        tenure_months=loan_tenure,
        existing_emi=existing_emi
    )
    
    foir = affordability_data['foir_percentage']
    
    if ml_decision == "APPROVE" and foir > 45:
        ml_decision = "REVIEW"
    
    risk_score = calculate_final_risk_score(bureau_score, confidence, foir)
    # pd_percentage = max(0, min(100, (1 - confidence/100) * 10))
     pd_percentage = calculate_final_pd(bureau_score, foir, confidence)


    return {
        'decision': ml_decision,
        'reason': "Decision based on comprehensive assessment",
        'confidence': confidence,
        'class_probs': class_probs,
        'policy_checks': policy_checks,
        'risk_score': risk_score,
        'pd_percentage': round(pd_percentage, 2),
        'affordability_data': affordability_data
    }

# =============================================================================
# MODERN UI COMPONENTS
# =============================================================================

def render_decision_header(decision_data, customer_data):
    """Render modern decision header"""
    
    decision = decision_data['decision']
    risk_score = decision_data['risk_score']
    pd_score = decision_data['pd_percentage']
    approved_amount = customer_data.get('loan_amount', 0)
    tenure = customer_data.get('loan_tenure_months', 24)
    app_id = customer_data.get('application_id', 'N/A')
    timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Decision card with appropriate styling
    if decision == "APPROVE":
        card_class = "decision-card decision-card-approved"
        icon = "✓"
        subtitle = "Application Approved Successfully"
    elif decision == "REJECT":
        card_class = "decision-card decision-card-rejected"
        icon = "✗"
        subtitle = "Application Not Approved"
    else:
        card_class = "decision-card decision-card-review"
        icon = "⚠"
        subtitle = "Requires Manual Review"
    
    st.markdown(f"""
        <div class="{card_class}">
            <div class="decision-title">
                <span>{icon}</span>
                <span>{decision}</span>
            </div>
            <div class="decision-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics grid
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{risk_score}</div>
                <div class="stat-label">Risk Score</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{pd_score}%</div>
                <div class="stat-label">PD Score</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">Rs.{approved_amount:,.0f}</div>
                <div class="stat-label">Loan Amount</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{tenure}</div>
                <div class="stat-label">Tenure (Months)</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{decision_data['confidence']:.0f}%</div>
                <div class="stat-label">Confidence</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Application info
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="info-box">
                <strong>📋 Application ID:</strong> {app_id}
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="info-box">
                <strong>🕐 Decision Timestamp:</strong> {timestamp}
            </div>
        """, unsafe_allow_html=True)


def render_info_card(title, icon, data_dict, status_dict=None):
    """Render modern info card with data"""
    
    st.markdown(f"""
        <div class="info-card">
            <div class="info-card-title">
                <span class="icon">{icon}</span>
                <span>{title}</span>
            </div>
            <div class="info-card-content">
    """, unsafe_allow_html=True)
    
    for label, value in data_dict.items():
        status = ""
        if status_dict and label in status_dict:
            if status_dict[label] == "pass":
                status = '<span class="status-badge badge-pass">✓ Passed</span>'
            elif status_dict[label] == "fail":
                status = '<span class="status-badge badge-fail">✗ Failed</span>'
            elif status_dict[label] == "warning":
                status = '<span class="status-badge badge-warning">⚠ Warning</span>'
        
        st.markdown(f"""
            <div class="data-row">
                <span class="data-label">{label}</span>
                <span class="data-value">{value} {status}</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_reason_codes(reasons):
    """Render reason codes in modern style"""
    
    st.markdown("""
        <div class="info-card">
            <div class="info-card-title">
                <span class="icon">📝</span>
                <span>Decision Reasons</span>
            </div>
            <div class="info-card-content">
    """, unsafe_allow_html=True)
    
    for i, reason in enumerate(reasons, 1):
        st.markdown(f"""
            <div class="reason-item">
                <span class="reason-icon">{i}.</span>
                <span>{reason}</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)


def create_modern_gauge(value, title, max_value=100):
    """Create modern gauge chart"""
    
    if value <= 50:
        color = "#f56565"
    elif value <= 75:
        color = "#ed8936"
    else:
        color = "#48bb78"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}},
        number={'suffix': "%", 'font': {'size': 36, 'color': '#2d3748', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, max_value], 'tickfont': {'size': 12, 'color': '#718096'}},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': '#fed7d7'},
                {'range': [50, 75], 'color': '#feebc8'},
                {'range': [75, 100], 'color': '#c6f6d5'}
            ],
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='white',
        font={'family': 'Inter', 'color': '#2d3748'}
    )
    
    return fig


def create_modern_bar_chart(class_probs):
    """Create modern probability bar chart"""
    
    df = pd.DataFrame({
        'Decision': list(class_probs.keys()),
        'Probability': list(class_probs.values())
    })
    
    colors = {'REVIEW': '#ed8936','APPROVE': '#48bb78',  'REJECT': '#f56565'}
    
    fig = px.bar(
        df, 
        x='Decision', 
        y='Probability',
        title='Decision Probabilities',
        color='Decision',
        color_discrete_map=colors,
        text='Probability'
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        marker_line_width=0
    )
    
    fig.update_layout(
        showlegend=False,
        yaxis_title='Probability (%)',
        xaxis_title='',
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'family': 'Inter', 'color': '#2d3748'},
        yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]},
        xaxis={'tickfont': {'size': 14, 'color': '#2d3748'}}
    )
    
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("# 🏦 Credit Risk Engine")
    st.markdown("---")
    
    page = st.radio(
        "**Navigation**",
        ["🏠 Home", "👤 Assessment", "📊 Batch Process", "📈 Model Info", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown(f"""
        <div class="info-card">
            <div class="info-card-title">System Status</div>
            <div class="info-card-content">
                <div class="data-row">
                    <span class="data-label">Model</span>
                    <span class="data-value">✅ Loaded</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Version</span>
                    <span class="data-value">8.0</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Features</span>
                    <span class="data-value">{len(TOP_FEATURES)}</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Type</span>
                    <span class="data-value">Random Forest</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🎯 **Top Features**"):
        for i, feat in enumerate(TOP_FEATURES[:5], 1):
            st.markdown(f"`{i}.` {feat}")

# =============================================================================
# HOME PAGE
# =============================================================================

if page == "🏠 Home":
    st.markdown('<p class="main-header">Credit Risk Assessment Platform</p>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <h3 style="margin-top: 0;">🎯 AI-Powered Lending Decisions</h3>
            <p style="margin-bottom: 0;">
                Comprehensive credit risk evaluation combining hard policy rules, 
                machine learning models, and affordability analysis for accurate lending decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="info-card">
                <div class="info-card-title">
                    <span class="icon">🛡️</span>
                    <span>Policy Gates</span>
                </div>
                <div class="info-card-content">
                    <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
                        <li>Age & KYC verification</li>
                        <li>Employment stability</li>
                        <li>Minimum income checks</li>
                        <li>Credit bureau thresholds</li>
                        <li>Bankruptcy & fraud detection</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card">
                <div class="info-card-title">
                    <span class="icon">🤖</span>
                    <span>ML Assessment</span>
                </div>
                <div class="info-card-content">
                    <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
                        <li>Random Forest classifier</li>
                        <li>60K+ training samples</li>
                        <li>Confidence scoring</li>
                        <li>Multi-class prediction</li>
                        <li>Feature importance</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="info-card">
                <div class="info-card-title">
                    <span class="icon">💰</span>
                    <span>Affordability</span>
                </div>
                <div class="info-card-content">
                    <ul style="margin: 0; padding-left: 1.25rem; color: #5A5A5A;">
                        <li>EMI calculation</li>
                        <li>FOIR analysis (max 50%)</li>
                        <li>Net disposable income</li>
                        <li>Debt burden assessment</li>
                        <li>Affordability scoring</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", "85%", "+2%")
    
    with col2:
        st.metric("⚡ Avg Response", "1.2s", "-0.3s")
    
    with col3:
        st.metric("📊 Features", len(TOP_FEATURES))
    
    with col4:
        st.metric("🔄 Version", "8.0", "Latest")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="warning-box">
            <strong>🆕 New in Version 8.0:</strong><br>
            • Sage Green & Yellow Professional Theme<br>
            • Enhanced visual hierarchy and readability<br>
            • Improved decision summary cards<br>
            • Modern charts and gauges<br>
            • Responsive layout optimization
        </div>
    """, unsafe_allow_html=True)

# =============================================================================
# ASSESSMENT PAGE
# =============================================================================

elif page == "👤 Assessment":
    st.markdown('<p class="main-header">Credit Assessment</p>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            💡 Complete the form below to assess credit risk. All fields are required for accurate evaluation.
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("assessment_form"):
        # Identity & Eligibility
        st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", 18, 80, 35, help="Customer's age in years")
            employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business'])
        
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
        
        # Credit Bureau
        st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bureau_score = st.number_input("Bureau Score", 300, 900, 720, 10)
            dpd_90_6m = st.number_input("DPD 90+ (Last 6M)", 0, 20, 0)
        
        with col2:
            credit_utilization = st.number_input("Credit Utilization (%)", 0, 100, 30)
            recent_inquiries = st.number_input("Recent Inquiries (3M)", 0, 20, 2)
        
        with col3:
            active_loans = st.number_input("Active Loans", 0, 10, 1)
            existing_emi = st.number_input("Existing Total EMI (Rs.)", 0, 200000, 15000, 1000)
        
        # Income & Financial
        st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_salary = st.number_input("Monthly Income (Rs.)", 0, 1000000, 50000, 5000)
            amt_income = st.number_input("Annual Income (Rs.)", 0, 10000000, 600000, 10000)
        
        with col2:
            net_surplus = st.number_input("Net Cash Surplus (Rs.)", -100000, 500000, 20000, 5000)
            salary_stability = st.selectbox("Salary Stability", ['STABLE', 'MODERATE', 'UNSTABLE'])
        
        with col3:
            loan_amount = st.number_input("Loan Amount (Rs.)", 0, 5000000, 180000, 10000)
            loan_tenure = st.number_input("Tenure (months)", 3, 360, 24)
        
        with col4:
            interest_rate = st.number_input("Interest Rate (%)", 8.0, 20.0, 10.5, 0.5)
            amt_annuity = st.number_input("Requested EMI (Rs.)", 0, 200000, 8500, 500)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)
    
    if submitted:
        # Generate application ID
        timestamp = datetime.now()
        app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
        
        # Prepare data
        customer_data = {
            'age': age,
            'employment_type': employment_type,
            'kyc_verified': kyc_verified,
            'bankruptcy_flag': bankruptcy_flag,
            'fraud_flag': fraud_flag,
            'employment_tenure_months': employment_tenure,
            'business_vintage_years': business_vintage,
            'bureau_score': bureau_score,
            'dpd_90_count_6m': dpd_90_6m,
            'credit_utilization_pct': credit_utilization,
            'recent_inquiries_3m': recent_inquiries,
            'active_loans_count': active_loans,
            'avg_salary_6m': avg_salary,
            'AMT_INCOME_TOTAL': amt_income,
            'net_cash_surplus_6m': net_surplus,
            'salary_stability_flag': salary_stability,
            'loan_amount': loan_amount,
            'loan_tenure_months': loan_tenure,
            'interest_rate': interest_rate,
            'existing_emi': existing_emi,
            'AMT_ANNUITY': amt_annuity,
            'application_id': app_id,
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Get decision
        with st.spinner("🔄 Processing assessment..."):
            decision_data = make_hybrid_decision_enhanced(customer_data)
        
        # Generate reasons
        reasons = generate_reason_codes(
            decision=decision_data['decision'],
            customer_data=customer_data,
            affordability_data=decision_data.get('affordability_data', {}),
            policy_checks=decision_data['policy_checks']
        )
        
        customer_data['reason_codes'] = reasons
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])
        
        with tab1:
            st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                render_info_card(
                    "👤 Identity", 
                    "👤",
                    {
                        "Age": age,
                        "Employment": employment_type,
                        "KYC Status": "Verified" if kyc_verified else "Not Verified",
                        "Tenure": f"{employment_tenure} months" if employment_type == 'Salaried' else f"{business_vintage} years"
                    }
                )
                
                render_info_card(
                    "💰 Financial", 
                    "💰",
                    {
                        "Monthly Income": f"Rs.{avg_salary:,}",
                        "Annual Income": f"Rs.{amt_income:,}",
                        "Net Surplus": f"Rs.{net_surplus:,}",
                        "Stability": salary_stability
                    }
                )
            
            with col2:
                render_info_card(
                    "🏦 Credit Bureau", 
                    "🏦",
                    {
                        "Bureau Score": bureau_score,
                        "DPD 90+": dpd_90_6m,
                        "Utilization": f"{credit_utilization}%",
                        "Recent Inquiries": recent_inquiries,
                        "Existing EMI": f"Rs.{existing_emi:,}"
                    }
                )
                
                render_info_card(
                    "📋 Loan Request", 
                    "📋",
                    {
                        "Amount": f"Rs.{loan_amount:,}",
                        "Tenure": f"{loan_tenure} months",
                        "Interest Rate": f"{interest_rate}%",
                        "Requested EMI": f"Rs.{amt_annuity:,}"
                    }
                )
        
        with tab2:
            st.markdown('<p class="section-header">Decision Summary</p>', unsafe_allow_html=True)
            
            render_decision_header(decision_data, customer_data)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Identity card
                age_pass = 18 <= age <= 65
                kyc_pass = kyc_verified
                
                render_info_card(
                    "Identity & Eligibility",
                    "👤",
                    {
                        f"Age: {age}": "",
                        f"Employment: {employment_type}": "",
                        f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""
                    },
                    {
                        f"Age: {age}": "pass" if age_pass else "fail",
                        f"Employment: {employment_type}": "pass",
                        f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_pass else "fail"
                    }
                )
            
            with col2:
                # Credit card
                bureau_pass = bureau_score >= 550
                dpd_pass = dpd_90_6m == 0
                
                render_info_card(
                    "Credit Bureau",
                    "🏦",
                    {
                        f"Bureau Score: {bureau_score}": "",
                        f"DPD 90+: {dpd_90_6m}": "",
                        f"Utilization: {credit_utilization}%": ""
                    },
                    {
                        f"Bureau Score: {bureau_score}": "pass" if bureau_pass else "fail",
                        f"DPD 90+: {dpd_90_6m}": "pass" if dpd_pass else "fail",
                        f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"
                    }
                )
            
            with col3:
                # Affordability card
                affordability = decision_data.get('affordability_data', {})
                foir = affordability.get('foir_percentage', 0)
                total_emi = affordability.get('total_emi', 0)
                net_disp = affordability.get('net_disposable', 0)
                
                render_info_card(
                    "Affordability",
                    "💰",
                    {
                        f"Monthly Income: Rs.{avg_salary:,}": "",
                        f"FOIR: {foir:.1f}%": "",
                        f"Total EMI: Rs.{total_emi:,}": "",
                        f"Net Disposable: Rs.{net_disp:,}": ""
                    },
                    {
                        f"Monthly Income: Rs.{avg_salary:,}": "pass",
                        f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
                        f"Total EMI: Rs.{total_emi:,}": "pass",
                        f"Net Disposable: Rs.{net_disp:,}": "pass" if net_disp >= 10000 else "warning"
                    }
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Reason codes
            render_reason_codes(reasons)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                try:
                    pdf_buffer = generate_decision_pdf(
                        decision_data=decision_data,
                        customer_data=customer_data,
                        affordability_data=decision_data.get('affordability_data', {}),
                        reasons=reasons
                    )
                    
                    st.download_button(
                        label="📥 Download Report (PDF)",
                        data=pdf_buffer,
                        file_name=f"credit_decision_{app_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("Please ensure reportlab is installed: pip install reportlab")
            
            with col2:
                if st.button("🔄 Re-Evaluate", use_container_width=True):
                    st.rerun()
        
        with tab3:
            st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = create_modern_gauge(decision_data['confidence'], "Model Confidence")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # ========== FIXED BAR CHART - POLICY-AWARE ==========
                final_decision = decision_data['decision']
                
                if final_decision == "REVIEW":
                    class_probs = {"APPROVE": 0, "REVIEW": 100, "REJECT": 0}
                elif final_decision == "REJECT":
                    class_probs = {"APPROVE": 0, "REVIEW": 0, "REJECT": 100}
                else:  # APPROVE
                    class_probs = decision_data['class_probs']
                
                fig2 = create_modern_bar_chart(class_probs)
                st.plotly_chart(fig2, use_container_width=True)
                # ====================================================
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Policy checks
            st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
            
            policy_df = pd.DataFrame([
                {'Check': k, 'Result': v} 
                for k, v in decision_data['policy_checks'].items()
            ])
            st.dataframe(policy_df, use_container_width=True, hide_index=True)
        
        with tab4:
            st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
            
            audit_log = {
                'application_id': app_id,
                'timestamp': timestamp.isoformat(),
                'decision': decision_data['decision'],
                'risk_score': decision_data['risk_score'],
                'pd_percentage': decision_data['pd_percentage'],
                'confidence': round(decision_data['confidence'], 2),
                'model_version': '8.0',
                'reason_codes': reasons,
                'affordability': decision_data.get('affordability_data', {})
            }
            
            st.json(audit_log)
            
            audit_json = json.dumps(audit_log, indent=2)
            st.download_button(
                "📥 Download Audit Log",
                audit_json,
                f"audit_{app_id}.json",
                "application/json",
                use_container_width=True
            )

# =============================================================================
# BATCH PROCESSING PAGE
# =============================================================================

elif page == "📊 Batch Process":
    st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            📤 Upload a CSV file with customer data for bulk credit assessment. 
            The file should include all required fields for prediction.
        </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded {len(df)} records")
            
            # Show preview
            with st.expander("📄 Preview Uploaded Data"):
                st.dataframe(df.head(), use_container_width=True)
                st.write(f"**Total Records:** {len(df)}")
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            
            # Required columns check
            required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your CSV includes at least these columns: age, employment_type, avg_salary_6m, bureau_score, loan_amount")
            else:
                # Process batch predictions
                if st.button("🚀 Process Batch Predictions", use_container_width=True, type="primary"):
                    with st.spinner(f"🔍 Processing {len(df)} records..."):
                        progress_bar = st.progress(0)
                        
                        # Process batch
                        results_df = process_batch_predictions(df)
                        
                        progress_bar.progress(100)
                        
                        st.success(f"✅ Completed processing {len(results_df)} records!")
                        
                        # Show results
                        tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
                        
                        with tab1:
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                approved_count = len(results_df[results_df['decision'] == 'APPROVE'])
                                st.metric("✅ Approved", approved_count)
                            with col2:
                                rejected_count = len(results_df[results_df['decision'] == 'REJECT'])
                                st.metric("❌ Rejected", rejected_count)
                            with col3:
                                review_count = len(results_df[results_df['decision'] == 'REVIEW'])
                                st.metric("⚠️ Review", review_count)
                            with col4:
                                avg_risk = results_df['risk_score'].mean()
                                st.metric("📊 Avg Risk Score", f"{avg_risk:.0f}")
                        
                        with tab2:
                            # Visualizations
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Decision distribution
                                decision_counts = results_df['decision'].value_counts()
                                fig1 = px.pie(
                                    values=decision_counts.values,
                                    names=decision_counts.index,
                                    title="Decision Distribution",
                                    color=decision_counts.index,
                                    color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}
                                )
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            with col2:
                                # Risk score distribution
                                fig2 = px.histogram(
                                    results_df,
                                    x='risk_score',
                                    title="Risk Score Distribution",
                                    nbins=20,
                                    color_discrete_sequence=['#587042']
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                            
                            # FOIR analysis
                            fig3 = px.scatter(
                                results_df,
                                x='monthly_income',
                                y='loan_amount',
                                color='decision',
                                size='risk_score',
                                title="Income vs Loan Amount (Colored by Decision)",
                                hover_data=['application_id', 'foir_percentage'],
                                color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        with tab3:
                            st.markdown("### Download Results")
                            
                            # Download options
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Download CSV
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download as CSV",
                                    data=csv,
                                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                # Download JSON
                                json_data = results_df.to_json(orient='records', indent=2)
                                st.download_button(
                                    label="📥 Download as JSON",
                                    data=json_data,
                                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                            
                            # Filtered downloads
                            st.markdown("---")
                            st.markdown("#### Filtered Downloads")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                approved_df = results_df[results_df['decision'] == 'APPROVE']
                                if len(approved_df) > 0:
                                    st.download_button(
                                        label=f"✅ Approved Only ({len(approved_df)})",
                                        data=approved_df.to_csv(index=False),
                                        file_name=f"approved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            
                            with col2:
                                rejected_df = results_df[results_df['decision'] == 'REJECT']
                                if len(rejected_df) > 0:
                                    st.download_button(
                                        label=f"❌ Rejected Only ({len(rejected_df)})",
                                        data=rejected_df.to_csv(index=False),
                                        file_name=f"rejected_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            
                            with col3:
                                review_df = results_df[results_df['decision'] == 'REVIEW']
                                if len(review_df) > 0:
                                    st.download_button(
                                        label=f"⚠️ Review Only ({len(review_df)})",
                                        data=review_df.to_csv(index=False),
                                        file_name=f"review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure the CSV file is properly formatted and contains the required columns.")
    
    else:
        # Show template download
        st.markdown("---")
        st.markdown("### 📋 CSV Template")
        
        # Create template dataframe
        template_data = {
            'age': [35, 42, 28],
            'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
            'kyc_verified': ['Yes', 'Yes', 'No'],
            'bankruptcy_flag': ['No', 'No', 'No'],
            'fraud_flag': ['No', 'No', 'No'],
            'employment_tenure_months': [24, 0, 18],
            'business_vintage_years': [0, 5, 0],
            'bureau_score': [720, 680, 580],
            'dpd_90_count_6m': [0, 1, 2],
            'credit_utilization_pct': [30, 45, 75],
            'recent_inquiries_3m': [2, 1, 5],
            'active_loans_count': [1, 2, 3],
            'avg_salary_6m': [50000, 75000, 35000],
            'AMT_INCOME_TOTAL': [600000, 900000, 420000],
            'net_cash_surplus_6m': [20000, 35000, 10000],
            'salary_stability_flag': ['STABLE', 'MODERATE', 'UNSTABLE'],
            'loan_amount': [180000, 250000, 100000],
            'loan_tenure_months': [24, 36, 12],
            'interest_rate': [10.5, 11.0, 12.0],
            'existing_emi': [15000, 20000, 8000],
            'AMT_ANNUITY': [8500, 9500, 4500]
        }
        
        template_df = pd.DataFrame(template_data)
        
        st.dataframe(template_df, use_container_width=True)
        
        # Download template
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="credit_assessment_template.csv",
            mime="text/csv",
            use_container_width=True
        )

# =============================================================================
# MODEL INFO PAGE
# =============================================================================

elif page == "📈 Model Info":
    st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">RF</div>
                <div class="stat-label">Model Type</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(TOP_FEATURES)}</div>
                <div class="stat-label">Features</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(TARGET_LE.classes_)}</div>
                <div class="stat-label">Classes</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
    
    feature_df = pd.DataFrame({
        'Rank': range(1, min(21, len(TOP_FEATURES) + 1)),
        'Feature': TOP_FEATURES[:20]
    })
    
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

# =============================================================================
# ABOUT PAGE
# =============================================================================

elif page == "ℹ️ About":
    st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card">
            <div class="info-card-title">
                <span class="icon">🏦</span>
                <span>Credit Risk Assessment Platform</span>
            </div>
            <div class="info-card-content">
                <p><strong>Version:</strong> 8.0 - Sage Green & Yellow Theme</p>
                <p><strong>Developer:</strong> Zen Meraki</p>
                <p><strong>Date:</strong> January 2026</p>
                <br>
                <p>
                    A comprehensive credit risk evaluation system combining hard policy rules,
                    machine learning models, and affordability analysis for accurate and compliant
                    lending decisions.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="info-card">
                <div class="info-card-title">
                    <span class="icon">🎯</span>
                    <span>Key Features</span>
                </div>
                <div class="info-card-content">
                    <ul style="margin: 0; padding-left: 1.25rem;">
                        <li>Three-layer decision engine</li>
                        <li>Real-time risk assessment</li>
                        <li>FOIR calculation & validation</li>
                        <li>Automated reason generation</li>
                        <li>Complete audit trail</li>
                        <li>Professional UI/UX</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card">
                <div class="info-card-title">
                    <span class="icon">🛠️</span>
                    <span>Technology Stack</span>
                <div class="info-card-content">
                    <ul style="margin: 0; padding-left: 1.25rem;">
                        <li>Streamlit (UI Framework)</li>
                        <li>Scikit-learn (ML)</li>
                        <li>Plotly (Visualizations)</li>
                        <li>Pandas (Data Processing)</li>
                        <li>Python 3.8+</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><p>© 2026 Hybrid Credit Risk System v8.0 | Zen Meraki</p></div>", 
    unsafe_allow_html=True)
