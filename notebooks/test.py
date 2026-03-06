# # # # """
# # # # Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
# # # # Enhanced with Modern UI/UX Design
# # # # Run with: streamlit run test.py (from inside the notebooks folder)
# # # # Author: Zen Meraki
# # # # Date: January 2026
# # # # VERSION: 8.3 - FULLY CORRECTED
# # # # """

# # # # import streamlit as st

# # # # # =============================================================================
# # # # # PAGE CONFIGURATION – MUST BE THE VERY FIRST STREAMLIT COMMAND
# # # # # =============================================================================
# # # # st.set_page_config(
# # # #     page_title="Credit Risk Assessment",
# # # #     page_icon="💳",
# # # #     layout="wide",
# # # #     initial_sidebar_state="expanded"
# # # # )

# # # # # =============================================================================
# # # # # STANDARD LIBRARY / THIRD-PARTY IMPORTS
# # # # # =============================================================================
# # # # import pandas as pd
# # # # import numpy as np
# # # # import plotly.graph_objects as go
# # # # import plotly.express as px
# # # # import joblib
# # # # import warnings
# # # # from datetime import datetime
# # # # import hashlib
# # # # import io
# # # # import base64
# # # # from typing import Dict, List, Any, Union
# # # # import json
# # # # import sys
# # # # import os
# # # # from pathlib import Path
# # # # import re

# # # # # =============================================================================
# # # # # SUPPRESS SKLEARN FEATURE NAME WARNING
# # # # # =============================================================================
# # # # warnings.filterwarnings("ignore", message="X does not have valid feature names")

# # # # # =============================================================================
# # # # # DYNAMIC PATH RESOLUTION – MAKE ALL PROJECT MODULES IMPORTABLE
# # # # # =============================================================================
# # # # CURRENT_DIR = Path(__file__).resolve().parent          # notebooks/
# # # # PROJECT_ROOT = CURRENT_DIR.parent                      # credit_risk_engine/
# # # # POSSIBLE_LOCATIONS = [
# # # #     CURRENT_DIR,                           # notebooks/
# # # #     PROJECT_ROOT,                           # credit_risk_engine/
# # # #     PROJECT_ROOT / "loan",                   # credit_risk_engine/loan/
# # # #     PROJECT_ROOT / "utils",                   # credit_risk_engine/utils/
# # # #     PROJECT_ROOT / "notebooks",               # credit_risk_engine/notebooks/
# # # # ]

# # # # for loc in POSSIBLE_LOCATIONS:
# # # #     if loc.exists() and str(loc) not in sys.path:
# # # #         sys.path.insert(0, str(loc))

# # # # # =============================================================================
# # # # # OPTIONAL OCR DEPENDENCIES – GRACEFUL FALLBACK
# # # # # Requires system packages (packages.txt):   tesseract-ocr  poppler-utils
# # # # # Requires Python packages (requirements.txt): pytesseract pdf2image opencv-python-headless pillow
# # # # # =============================================================================
# # # # OCR_AVAILABLE = False
# # # # OCR_ERROR_MSG = ""
# # # # try:
# # # #     import pytesseract
# # # #     from pdf2image import convert_from_bytes
# # # #     import cv2
# # # #     from PIL import Image

# # # #     # Auto-detect Tesseract binary (Streamlit Cloud / Linux / Mac / Windows)
# # # #     import shutil as _shutil
# # # #     _tess_cmd = (
# # # #         _shutil.which("tesseract")
# # # #         or r"C:\Program Files\Tesseract-OCR\tesseract.exe"   # Windows fallback
# # # #     )
# # # #     if _tess_cmd:
# # # #         pytesseract.pytesseract.tesseract_cmd = _tess_cmd

# # # #     # Verify tesseract binary is actually callable
# # # #     pytesseract.get_tesseract_version()
# # # #     OCR_AVAILABLE = True

# # # # except ImportError as _e:
# # # #     OCR_ERROR_MSG = (
# # # #         f"Missing Python package: {_e}. "
# # # #         "Add to requirements.txt: pytesseract  pdf2image  opencv-python-headless  pillow"
# # # #     )
# # # # except Exception as _e:
# # # #     _name = type(_e).__name__
# # # #     if "TesseractNotFound" in _name or "tesseract" in str(_e).lower():
# # # #         OCR_ERROR_MSG = (
# # # #             "Tesseract binary not found. "
# # # #             "Streamlit Cloud → add 'tesseract-ocr' and 'poppler-utils' to packages.txt. "
# # # #             "Linux → sudo apt install tesseract-ocr poppler-utils. "
# # # #             "Mac → brew install tesseract poppler."
# # # #         )
# # # #     else:
# # # #         OCR_ERROR_MSG = f"OCR init error ({_name}): {_e}"

# # # # # =============================================================================
# # # # # IMPORT CSS – WITH FALLBACK
# # # # # =============================================================================
# # # # try:
# # # #     from css_styles import CSS
# # # # except ImportError:
# # # #     CSS = """
# # # #     <style>
# # # #         .main-header { font-size: 2rem; font-weight: bold; color: #2d3748; }
# # # #         .section-header { font-size: 1.5rem; font-weight: 600; color: #2d3748; }
# # # #         .info-box { background: #f7fafc; padding: 1rem; border-radius: 0.5rem; }
# # # #         .decision-card { padding: 2rem; border-radius: 1rem; text-align: center; margin-bottom: 1rem; }
# # # #         .decision-card-approved { background: #c6f6d5; border-left: 5px solid #48bb78; }
# # # #         .decision-card-rejected { background: #fed7d7; border-left: 5px solid #f56565; }
# # # #         .decision-card-review { background: #feebc8; border-left: 5px solid #ed8936; }
# # # #         .decision-title { font-size: 2.5rem; font-weight: bold; }
# # # #         .decision-subtitle { font-size: 1rem; opacity: 0.8; }
# # # #         .stat-card { background: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
# # # #         .stat-number { font-size: 1.8rem; font-weight: bold; color: #2d3748; }
# # # #         .stat-label { font-size: 0.875rem; color: #718096; }
# # # #         .info-card { background: white; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
# # # #         .info-card-title { font-weight: 600; margin-bottom: 0.5rem; color: #2d3748; }
# # # #         .info-card-content { font-size: 0.875rem; }
# # # #         .data-row { display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #e2e8f0; }
# # # #         .data-label { color: #4a5568; }
# # # #         .data-value { font-weight: 500; }
# # # #         .status-badge { padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; margin-left: 0.5rem; }
# # # #         .badge-pass { background: #c6f6d5; color: #22543d; }
# # # #         .badge-fail { background: #fed7d7; color: #742a2a; }
# # # #         .badge-warning { background: #feebc8; color: #744210; }
# # # #         .reason-item { padding: 0.25rem 0; }
# # # #         .reason-icon { color: #587042; font-weight: bold; margin-right: 0.5rem; }
# # # #     </style>
# # # #     """

# # # # # Apply CSS immediately after set_page_config
# # # # st.markdown(CSS, unsafe_allow_html=True)

# # # # # =============================================================================
# # # # # SESSION STATE INITIALIZATION
# # # # # =============================================================================
# # # # def init_session_state():
# # # #     if 'stage1_complete' not in st.session_state:
# # # #         st.session_state.stage1_complete = False
# # # #     if 'stage1_decision' not in st.session_state:
# # # #         st.session_state.stage1_decision = None
# # # #     if 'stage1_data' not in st.session_state:
# # # #         st.session_state.stage1_data = None
# # # #     if 'current_customer_data' not in st.session_state:
# # # #         st.session_state.current_customer_data = None
# # # #     if 'page_navigation' not in st.session_state:
# # # #         st.session_state.page_navigation = "🏠 Home"
# # # #     if 'use_two_stage' not in st.session_state:
# # # #         st.session_state.use_two_stage = False
# # # #     if 'stage2_selected_tab' not in st.session_state:
# # # #         st.session_state.stage2_selected_tab = "Manual Entry"

# # # # init_session_state()

# # # # # =============================================================================
# # # # # IMPORT BUSINESS LOGIC MODULES – WITH HELPFUL ERROR IF MISSING
# # # # # =============================================================================
# # # # try:
# # # #     from affordability_engine import calculate_emi, calculate_affordability
# # # #     from reason_codes import generate_reason_codes
# # # #     from risk_engine import calculate_final_risk_score, fill_missing_ml_fields, clean_sentinel_values, validate_cibil_identity
# # # # except ImportError as e:
# # # #     st.error(f"❌ Failed to import required modules: {e}")
# # # #     st.info("""
# # # #     Please ensure the following files are placed in one of these directories:
# # # #     - `notebooks/` (same folder as test.py)
# # # #     - `loan/` (sibling of notebooks)
# # # #     - `utils/` (containing pdf_generator.py and __init__.py)
# # # #     - The project root (`credit_risk_engine/`)

# # # #     Required files:
# # # #     - affordability_engine.py
# # # #     - reason_codes.py
# # # #     - risk_engine.py
# # # #     - utils/__init__.py
# # # #     - utils/pdf_generator.py
# # # #     """)
# # # #     st.stop()

# # # # # =============================================================================
# # # # # STAGE 2 ENGINE – ROBUST FALLBACK
# # # # # =============================================================================
# # # # try:
# # # #     import stage2_engine
# # # #     from stage2_engine import make_two_stage_decision, is_stage2_available, get_stage2_status
# # # #     STAGE2_AVAILABLE = is_stage2_available()
# # # # except ImportError:
# # # #     stage2_engine = None
# # # #     STAGE2_AVAILABLE = False
# # # #     def make_two_stage_decision(*args, **kwargs):
# # # #         raise NotImplementedError("Stage 2 engine not available")
# # # #     def is_stage2_available():
# # # #         return False
# # # #     def get_stage2_status():
# # # #         return {"error": "Stage 2 engine module not found", "available": False}

# # # # # =============================================================================
# # # # # PDF GENERATION – SAFE FALLBACK
# # # # # =============================================================================
# # # # PDF_AVAILABLE = False
# # # # generate_decision_pdf = None
# # # # generate_audit_pdf = None
# # # # try:
# # # #     from utils.pdf_generator import generate_decision_pdf, generate_audit_pdf
# # # #     PDF_AVAILABLE = True
# # # # except ImportError:
# # # #     pass

# # # # # =============================================================================
# # # # # JSON SANITIZER
# # # # # =============================================================================
# # # # def sanitize_for_json(obj: Any) -> Any:
# # # #     if obj is None or isinstance(obj, (str, int, float, bool)):
# # # #         return obj
# # # #     if isinstance(obj, set):
# # # #         return list(obj)
# # # #     if isinstance(obj, datetime):
# # # #         return obj.isoformat()
# # # #     if isinstance(obj, np.integer):
# # # #         return int(obj)
# # # #     if isinstance(obj, np.floating):
# # # #         return float(obj)
# # # #     if isinstance(obj, np.ndarray):
# # # #         return obj.tolist()
# # # #     if isinstance(obj, dict):
# # # #         return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
# # # #     if isinstance(obj, (list, tuple)):
# # # #         return [sanitize_for_json(item) for item in obj]
# # # #     try:
# # # #         json.dumps(obj)
# # # #         return obj
# # # #     except (TypeError, ValueError):
# # # #         return str(obj)

# # # # # =============================================================================
# # # # # LOAD TRAINED MODEL ASSETS (Stage 1 Random Forest)
# # # # # =============================================================================
# # # # @st.cache_resource
# # # # def load_model_assets():
# # # #     try:
# # # #         possible_paths = [
# # # #             'credit_risk_assets.pkl',
# # # #             'notebooks/credit_risk_assets.pkl',
# # # #             '../notebooks/credit_risk_assets.pkl'
# # # #         ]
# # # #         assets = None
# # # #         for path in possible_paths:
# # # #             try:
# # # #                 assets = joblib.load(path)
# # # #                 break
# # # #             except FileNotFoundError:
# # # #                 continue
# # # #         if assets is None:
# # # #             raise FileNotFoundError("Could not find credit_risk_assets.pkl")
# # # #         return {
# # # #             'model': assets['model'],
# # # #             'features': assets['features'],
# # # #             'le_map': assets['le_map'],
# # # #             'target_le': assets['target_le'],
# # # #             'loaded': True,
# # # #             'error': None
# # # #         }
# # # #     except FileNotFoundError:
# # # #         return {'loaded': False, 'error': 'credit_risk_assets.pkl not found. Please run the training script first.'}
# # # #     except Exception as e:
# # # #         return {'loaded': False, 'error': f'Error loading model: {str(e)}'}

# # # # ASSETS = load_model_assets()
# # # # if not ASSETS['loaded']:
# # # #     st.error(f"❌ {ASSETS['error']}")
# # # #     st.info("Please ensure 'credit_risk_assets.pkl' is in the same directory as this app.")
# # # #     st.stop()

# # # # MODEL = ASSETS['model']
# # # # TOP_FEATURES = ASSETS['features']
# # # # LE_MAP = ASSETS['le_map']
# # # # TARGET_LE = ASSETS['target_le']

# # # # # =============================================================================
# # # # # AFFORDABILITY CALCULATION ENGINE
# # # # # =============================================================================
# # # # def calculate_emi(principal, annual_rate, tenure_months):
# # # #     if principal <= 0 or tenure_months <= 0:
# # # #         return 0
# # # #     monthly_rate = annual_rate / (12 * 100)
# # # #     if monthly_rate == 0:
# # # #         return principal / tenure_months
# # # #     emi = (principal * monthly_rate * (1 + monthly_rate)**tenure_months) / \
# # # #           ((1 + monthly_rate)**tenure_months - 1)
# # # #     return round(emi, 2)

# # # # def calculate_affordability(monthly_income, loan_amount, interest_rate, tenure_months, existing_emi):
# # # #     new_emi = calculate_emi(loan_amount, interest_rate, tenure_months)
# # # #     total_emi = new_emi + existing_emi
# # # #     foir_percentage = (total_emi / monthly_income) * 100 if monthly_income > 0 else 0
# # # #     net_disposable = monthly_income - total_emi
# # # #     max_allowed_emi = monthly_income * 0.45
# # # #     recommended_emi = monthly_income * 0.35
# # # #     affordable = foir_percentage <= 45
# # # #     within_recommended = foir_percentage <= 35
# # # #     if foir_percentage <= 35:
# # # #         status = "Excellent"
# # # #         status_color = "green"
# # # #     elif foir_percentage <= 40:
# # # #         status = "Acceptable"
# # # #         status_color = "yellow"
# # # #     elif foir_percentage <= 45:
# # # #         status = "High - Review Required"
# # # #         status_color = "orange"
# # # #     else:
# # # #         status = "Over-leveraged"
# # # #         status_color = "red"
# # # #     return {
# # # #         'monthly_income': monthly_income,
# # # #         'new_emi': new_emi,
# # # #         'existing_emi': existing_emi,
# # # #         'total_emi': total_emi,
# # # #         'foir_percentage': round(foir_percentage, 2),
# # # #         'net_disposable': net_disposable,
# # # #         'max_allowed_emi': max_allowed_emi,
# # # #         'recommended_emi': recommended_emi,
# # # #         'affordable': affordable,
# # # #         'within_recommended': within_recommended,
# # # #         'status': status,
# # # #         'status_color': status_color,
# # # #         'emi_headroom': max_allowed_emi - total_emi
# # # #     }

# # # # # =============================================================================
# # # # # REASON CODE GENERATION SYSTEM
# # # # # =============================================================================
# # # # APPROVAL_REASONS = {
# # # #     'high_bureau': 'Excellent credit score ({score})',
# # # #     'stable_employment': 'Stable employment history ({tenure} months)',
# # # #     'low_foir': 'Affordable EMI burden (FOIR: {foir}%)',
# # # #     'clean_payment': 'Clean payment history (No DPD)',
# # # #     'strong_income': 'Strong monthly income (₹{income:,})',
# # # #     'low_utilization': 'Low credit utilization ({util}%)',
# # # # }
# # # # REJECTION_REASONS = {
# # # #     'low_bureau':       'Credit score below minimum ({score} < 550)',
# # # #     'high_foir':        'EMI burden too high (FOIR: {foir}% > 50%)',
# # # #     'severe_dpd':       'Severe payment delays ({dpd} instances of 90+ DPD)',
# # # #     'moderate_dpd':     'Frequent payment delays ({dpd} instances of 30+ DPD)',
# # # #     'low_income':       'Income below minimum threshold (₹{income:,} < ₹15,000)',
# # # #     'short_employment': 'Insufficient employment tenure ({tenure} months < 6)',
# # # #     'short_vintage':    'Insufficient business vintage ({vintage} years < 2 years)',
# # # #     'bankruptcy':       'Active bankruptcy detected',
# # # #     'kyc_failed':       'KYC verification not completed',
# # # #     'fraud_flag':       'Fraud flag present on application',
# # # #     'high_utilization': 'High credit utilization ({util}% > 80%)',
# # # #     'age_invalid':      'Age outside acceptable range ({age} years, must be 24–70)',
# # # #     'high_dependents':  'High number of dependents ({deps}) reducing net disposable income',
# # # # }
# # # # REVIEW_REASONS = {
# # # #     'borderline_bureau':  'Credit score in borderline range ({score})',
# # # #     'moderate_foir':      'EMI burden moderate (FOIR: {foir}%)',
# # # #     'mixed_signals':      'Mixed credit indicators requiring human review',
# # # #     'recent_employment':  'Recent employment change requiring verification',
# # # #     'high_loan_amount':   'Large loan amount requiring additional underwriting review',
# # # #     'moderate_dpd':       'Recent 30-day payment delays requiring review ({dpd} instances)',
# # # #     'moderate_dependents':'Moderate number of dependents ({deps}) may affect repayment',
# # # # }

# # # # def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
# # # #     reasons = []
# # # #     bureau_score      = customer_data.get('bureau_score', 0)
# # # #     foir              = affordability_data.get('foir_percentage', 0)
# # # #     dpd_90            = customer_data.get('dpd_90_count_6m', 0)
# # # #     dpd_30            = customer_data.get('dpd_30_count_6m', 0)
# # # #     income            = customer_data.get('avg_salary_6m', 0)
# # # #     employment_tenure = customer_data.get('employment_tenure_months', 0)
# # # #     business_vintage  = customer_data.get('business_vintage_years', 0)
# # # #     employment_type   = customer_data.get('employment_type', 'Salaried')
# # # #     credit_util       = customer_data.get('credit_utilization_pct', 0)
# # # #     age               = customer_data.get('age', 0)
# # # #     dependents        = customer_data.get('dependents', 0)

# # # #     if decision == "APPROVE":
# # # #         if bureau_score >= 750:
# # # #             reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
# # # #         if employment_tenure >= 24:
# # # #             reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
# # # #         if foir <= 40:
# # # #             reasons.append(APPROVAL_REASONS['low_foir'].format(foir=round(foir, 1)))
# # # #         if dpd_90 == 0 and dpd_30 == 0:
# # # #             reasons.append(APPROVAL_REASONS['clean_payment'])
# # # #         if income >= 75000:
# # # #             reasons.append(APPROVAL_REASONS['strong_income'].format(income=income))
# # # #         if credit_util <= 30:
# # # #             reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))

# # # #     elif decision == "REJECT":
# # # #         for check_name, check_result in policy_checks.items():
# # # #             if '❌' in str(check_result):
# # # #                 cn = check_name.lower()
# # # #                 if 'bureau' in cn:
# # # #                     reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
# # # #                 elif 'dpd' in cn:
# # # #                     reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
# # # #                 elif 'income' in cn:
# # # #                     reasons.append(REJECTION_REASONS['low_income'].format(income=income))
# # # #                 elif 'tenure' in cn:
# # # #                     if employment_type == 'Salaried':
# # # #                         reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
# # # #                     else:
# # # #                         reasons.append(REJECTION_REASONS['short_vintage'].format(vintage=business_vintage))
# # # #                 elif 'kyc' in cn:
# # # #                     reasons.append(REJECTION_REASONS['kyc_failed'])
# # # #                 elif 'bankruptcy' in cn:
# # # #                     reasons.append(REJECTION_REASONS['bankruptcy'])
# # # #                 elif 'fraud' in cn:
# # # #                     reasons.append(REJECTION_REASONS['fraud_flag'])
# # # #                 elif 'age' in cn:
# # # #                     reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))
# # # #         if foir > 50:
# # # #             reasons.append(REJECTION_REASONS['high_foir'].format(foir=round(foir, 1)))
# # # #         if credit_util > 80:
# # # #             reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
# # # #         if dpd_30 >= 3 and dpd_90 == 0:
# # # #             reasons.append(REJECTION_REASONS['moderate_dpd'].format(dpd=dpd_30))
# # # #         if dependents >= 4:
# # # #             reasons.append(REJECTION_REASONS['high_dependents'].format(deps=dependents))

# # # #     elif decision == "REVIEW":
# # # #         if 650 <= bureau_score < 700:
# # # #             reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
# # # #         if 40 < foir <= 50:
# # # #             reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=round(foir, 1)))
# # # #         if employment_tenure < 12:
# # # #             reasons.append(REVIEW_REASONS['recent_employment'])
# # # #         if dpd_30 >= 1 and dpd_90 == 0:
# # # #             reasons.append(REVIEW_REASONS['moderate_dpd'].format(dpd=dpd_30))
# # # #         if 2 <= dependents < 4:
# # # #             reasons.append(REVIEW_REASONS['moderate_dependents'].format(deps=dependents))
# # # #         if not reasons:
# # # #             reasons.append(REVIEW_REASONS['mixed_signals'])

# # # #     return reasons[:3] if reasons else ['Decision based on comprehensive model assessment']

# # # # # =============================================================================
# # # # # PD CALCULATION
# # # # # =============================================================================
# # # # def bureau_score_to_pd(bureau_score):
# # # #     if bureau_score >= 800:
# # # #         return 0.5 + (900 - bureau_score) / 200 * 0.5
# # # #     elif bureau_score >= 750:
# # # #         return 1.0 + (800 - bureau_score) / 50 * 1.0
# # # #     elif bureau_score >= 700:
# # # #         return 2.0 + (750 - bureau_score) / 50 * 1.5
# # # #     elif bureau_score >= 650:
# # # #         return 3.5 + (700 - bureau_score) / 50 * 2.5
# # # #     elif bureau_score >= 600:
# # # #         return 6.0 + (650 - bureau_score) / 50 * 4.0
# # # #     elif bureau_score >= 550:
# # # #         return 10.0 + (600 - bureau_score) / 50 * 5.0
# # # #     else:
# # # #         return min(25.0, 15.0 + (550 - bureau_score) / 50 * 10.0)

# # # # def foir_to_pd_adjustment(foir_percentage):
# # # #     if foir_percentage <= 30:
# # # #         return -0.75
# # # #     elif foir_percentage <= 40:
# # # #         return 0.00
# # # #     elif foir_percentage <= 45:
# # # #         return 0.75
# # # #     elif foir_percentage <= 50:
# # # #         return 1.50
# # # #     elif foir_percentage <= 55:
# # # #         return 2.25
# # # #     elif foir_percentage <= 60:
# # # #         return 3.50
# # # #     else:
# # # #         return 6.00

# # # # def delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count=0):
# # # #     if dpd_90_count >= 3:
# # # #         return 5.0
# # # #     elif dpd_90_count == 2:
# # # #         return 3.0
# # # #     elif dpd_90_count == 1:
# # # #         return 2.0
# # # #     elif dpd_30_count >= 3:
# # # #         return 1.6
# # # #     elif dpd_30_count >= 1:
# # # #         return 1.3
# # # #     else:
# # # #         return 1.0

# # # # def employment_stability_to_pd_adjustment(employment_type, tenure_months, business_vintage_years=0):
# # # #     if employment_type == 'Salaried':
# # # #         if tenure_months >= 36:
# # # #             return -0.5
# # # #         elif tenure_months >= 12:
# # # #             return 0.0
# # # #         elif tenure_months >= 6:
# # # #             return 0.5
# # # #         else:
# # # #             return 2.0
# # # #     elif employment_type in ['Self-Employed', 'Business']:
# # # #         if business_vintage_years >= 5:
# # # #             return -0.5
# # # #         elif business_vintage_years >= 2:
# # # #             return 0.0
# # # #         else:
# # # #             return 1.5
# # # #     else:
# # # #         return 1.0

# # # # def inquiry_pattern_to_pd_adjustment(recent_inquiries_3m):
# # # #     if recent_inquiries_3m <= 1:
# # # #         return -0.3
# # # #     elif recent_inquiries_3m <= 3:
# # # #         return 0.0
# # # #     elif recent_inquiries_3m <= 5:
# # # #         return 0.8
# # # #     elif recent_inquiries_3m <= 8:
# # # #         return 1.5
# # # #     else:
# # # #         return 3.0

# # # # def ml_confidence_to_pd_adjustment(ml_confidence, ml_decision):
# # # #     if ml_decision == "APPROVE":
# # # #         if ml_confidence >= 90:
# # # #             return -0.5
# # # #         elif ml_confidence >= 70:
# # # #             return 0.0
# # # #         else:
# # # #             return 0.5
# # # #     elif ml_decision == "REVIEW":
# # # #         return 1.0
# # # #     else:
# # # #         return 5.0

# # # # def calculate_final_pd(bureau_score, foir, confidence, dpd_90_count=0, dpd_30_count=0,
# # # #                        employment_type='Salaried', employment_tenure=24, business_vintage=0,
# # # #                        recent_inquiries=2, ml_decision='APPROVE'):
# # # #     base_pd = bureau_score_to_pd(bureau_score)
# # # #     foir_adj = foir_to_pd_adjustment(foir)
# # # #     deliq_multiplier = delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count)
# # # #     employment_adj = employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage)
# # # #     inquiry_adj = inquiry_pattern_to_pd_adjustment(recent_inquiries)
# # # #     ml_adj = ml_confidence_to_pd_adjustment(confidence, ml_decision)
# # # #     adjusted_base_pd = base_pd * deliq_multiplier
# # # #     final_pd = adjusted_base_pd + foir_adj + employment_adj + inquiry_adj + ml_adj
# # # #     final_pd = max(0.5, min(final_pd, 25.0))
# # # #     return round(final_pd, 2)

# # # # # =============================================================================
# # # # # RISK SCORE CALCULATION
# # # # # =============================================================================
# # # # def calculate_final_risk_score(bureau_score, ml_confidence, foir,
# # # #                                 dpd_90, dpd_30, net_surplus,
# # # #                                 bounces=0, missing_months=0, active_loans=0):
# # # #     bureau_points = (bureau_score / 900) * 400
# # # #     ml_points = (ml_confidence / 100) * 300
# # # #     foir_points = max(0, (1 - foir / 50) * 150)
# # # #     dpd_penalty = min((dpd_90 * 50) + (dpd_30 * 20), 150)
# # # #     behavioral_penalty = min((bounces * 10) + (missing_months * 10), 100)
# # # #     if net_surplus > 50000:
# # # #         surplus_points = 50
# # # #     elif net_surplus > 0:
# # # #         surplus_points = 20
# # # #     elif net_surplus < -50000:
# # # #         surplus_points = -50
# # # #     else:
# # # #         surplus_points = -20
# # # #     total = (bureau_points + ml_points + foir_points
# # # #              + surplus_points - dpd_penalty - behavioral_penalty)
# # # #     return max(0, min(int(total), 1000))

# # # # # =============================================================================
# # # # # CIBIL PDF EXTRACTION ENGINE (OCR + PATTERN MATCHING) – OPTIONAL
# # # # # =============================================================================
# # # # def extract_cibil_from_pdf(uploaded_file):
# # # #     if not OCR_AVAILABLE:
# # # #         return {'success': False, 'error': OCR_ERROR_MSG or 'OCR libraries not installed. Check packages.txt and requirements.txt.'}

# # # #     try:
# # # #         pdf_bytes = uploaded_file.read()
# # # #         images = convert_from_bytes(pdf_bytes, dpi=300)
# # # #         full_text = ""
# # # #         for image in images:
# # # #             gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
# # # #             _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # # #             full_text += pytesseract.image_to_string(binary) + "\n"

# # # #         credit_score = 720
# # # #         score_match = re.search(
# # # #             r'\b(\d{3})\s*(?:EXCELLENT|VERY\s*GOOD|GOOD|FAIR|SUBPRIME|POOR|NH|NA)\b',
# # # #             full_text, re.IGNORECASE
# # # #         )
# # # #         if score_match:
# # # #             val = int(score_match.group(1))
# # # #             if 300 <= val <= 900:
# # # #                 credit_score = val
# # # #         if credit_score == 720:
# # # #             score_match2 = re.search(
# # # #                 r'(?:cibil|credit)\s*score\s*[:\-\(]?\s*(\d{3})',
# # # #                 full_text, re.IGNORECASE
# # # #             )
# # # #             if score_match2:
# # # #                 val = int(score_match2.group(1))
# # # #                 if 300 <= val <= 900:
# # # #                     credit_score = val
# # # #         if credit_score == 720:
# # # #             score_match3 = re.search(r'score.*?\((\d{3})\)', full_text, re.IGNORECASE)
# # # #             if score_match3:
# # # #                 val = int(score_match3.group(1))
# # # #                 if 300 <= val <= 900:
# # # #                     credit_score = val

# # # #         monthly_income = 50000
# # # #         income_match = re.search(
# # # #             r'(?:net\s+monthly\s+income|monthly\s+income|net\s+income|salary)[^\n\r]{0,30}?'
# # # #             r'(?:rs\.?\s*|inr\s*|₹\s*)([\d,]+)',
# # # #             full_text, re.IGNORECASE
# # # #         )
# # # #         if income_match:
# # # #             val = int(income_match.group(1).replace(',', ''))
# # # #             if val > 1000:
# # # #                 monthly_income = val
# # # #         if monthly_income == 50000:
# # # #             income_match2 = re.search(r'(?:rs\.?\s*|₹\s*)([\d,]{4,})', full_text, re.IGNORECASE)
# # # #             if income_match2:
# # # #                 val = int(income_match2.group(1).replace(',', ''))
# # # #                 if 5000 <= val <= 1000000:
# # # #                     monthly_income = val

# # # #         cc_util_pct = 35
# # # #         util_match = re.search(r'utilization\s*[\(:\-]?\s*(\d{1,3})\s*%', full_text, re.IGNORECASE)
# # # #         if util_match:
# # # #             cc_util_pct = int(util_match.group(1))
# # # #         cc_util = cc_util_pct / 100.0
# # # #         high_util = 1 if cc_util_pct > 75 else 0

# # # #         age_extracted = 35
# # # #         dob_match = re.search(
# # # #             r'(?:date\s+of\s+birth|dob)[:\s]+(\d{2}[-/]\w{3,9}[-/]\d{2,4})',
# # # #             full_text, re.IGNORECASE
# # # #         )
# # # #         if dob_match:
# # # #             try:
# # # #                 from datetime import datetime as _dt
# # # #                 dob_str = dob_match.group(1)
# # # #                 for fmt in ('%d-%b-%Y', '%d/%b/%Y', '%d-%m-%Y', '%d/%m/%Y'):
# # # #                     try:
# # # #                         dob = _dt.strptime(dob_str, fmt)
# # # #                         age_extracted = int((datetime.now() - dob).days / 365.25)
# # # #                         break
# # # #                     except Exception:
# # # #                         continue
# # # #             except Exception:
# # # #                 pass

# # # #         biz_vintage = 3
# # # #         biz_match = re.search(r'business\s+vintage.*?(\d+)', full_text, re.IGNORECASE)
# # # #         if biz_match:
# # # #             biz_vintage = int(biz_match.group(1))

# # # #         lines = full_text.split('\n')
# # # #         in_accounts = False
# # # #         in_enquiry = False
# # # #         accounts = []
# # # #         enquiry_dates = []

# # # #         for line in lines:
# # # #             line_up = line.upper()
# # # #             if 'ACCOUNT DETAILS' in line_up:
# # # #                 in_accounts = True
# # # #                 in_enquiry = False
# # # #                 continue
# # # #             if 'ENQUIRY DETAILS' in line_up:
# # # #                 in_accounts = False
# # # #                 in_enquiry = True
# # # #                 continue

# # # #             if in_accounts:
# # # #                 if re.search(r'SUMMARY|SCORE|PERSONAL\s+INFO', line_up):
# # # #                     break
# # # #                 if re.search(r'\bLender\b|\bAccount\s*No\b|\bOpen\s*Date\b|\bDPD\b|\bStatus\b', line, re.IGNORECASE):
# # # #                     continue
# # # #                 stripped = line.strip()
# # # #                 if not stripped:
# # # #                     continue
# # # #                 dpd_match = re.search(r'\b(0\d0|0\d\d|\d{3})\b', stripped)
# # # #                 status_match = re.search(
# # # #                     r'\b(Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss)\s*$',
# # # #                     stripped, re.IGNORECASE
# # # #                 )
# # # #                 if (re.search(r'\bINR\b', stripped, re.IGNORECASE) or
# # # #                         re.match(r'^[A-Z][a-zA-Z\s]+(?:Bank|Finance|Capital|Fincorp|SBI|ICICI|HDFC|Axis|Bajaj|Tata|Kotak)', stripped)):
# # # #                     dpd_val = int(dpd_match.group(1)) if dpd_match else 0
# # # #                     status_str = status_match.group(1) if status_match else 'Active'
# # # #                     accounts.append({'dpd': dpd_val, 'status': status_str.lower()})

# # # #             if in_enquiry:
# # # #                 enq_date = re.match(r'^\s*(\d{2}-[A-Za-z]{3}-\d{4})', line)
# # # #                 if enq_date:
# # # #                     enquiry_dates.append(enq_date.group(1))

# # # #         written_off_count = 0
# # # #         settled_count = 0
# # # #         dpd_90_count = 0
# # # #         dpd_60_count = 0
# # # #         dpd_30_count = 0
# # # #         active_count = 0
# # # #         sub_standard_count = 0

# # # #         if accounts:
# # # #             for acc in accounts:
# # # #                 dpd = acc.get('dpd', 0)
# # # #                 status = acc.get('status', '')
# # # #                 if dpd >= 90:
# # # #                     dpd_90_count += 1
# # # #                 elif dpd >= 60:
# # # #                     dpd_60_count += 1
# # # #                 elif dpd >= 30:
# # # #                     dpd_30_count += 1
# # # #                 if 'written' in status:
# # # #                     written_off_count += 1
# # # #                 elif 'settled' in status:
# # # #                     settled_count += 1
# # # #                 elif 'active' in status:
# # # #                     active_count += 1
# # # #                 if dpd >= 30:
# # # #                     sub_standard_count += 1
# # # #         else:
# # # #             written_off_count = len(re.findall(r'\bwritten[-\s]?off\b', full_text, re.IGNORECASE))
# # # #             settled_count     = len(re.findall(r'\bsettled\b', full_text, re.IGNORECASE))
# # # #             dpd_90_count      = len(re.findall(r'\b090\b|90\+?\s*dpd', full_text, re.IGNORECASE))
# # # #             dpd_60_count      = len(re.findall(r'\b060\b|60\+?\s*dpd', full_text, re.IGNORECASE))
# # # #             dpd_30_count      = len(re.findall(r'\b030\b|30\+?\s*dpd', full_text, re.IGNORECASE))
# # # #             active_sum = re.search(r'Total\s+Accounts\s+Active.*?(\d+)\s+(\d+)', full_text, re.IGNORECASE)
# # # #             if active_sum:
# # # #                 active_count = int(active_sum.group(2))

# # # #         if active_count == 0:
# # # #             summary_match = re.search(
# # # #                 r'Total\s+Accounts\s+Active[^\n]*\n\s*(\d+)\s+(\d+)',
# # # #                 full_text, re.IGNORECASE
# # # #             )
# # # #             if summary_match:
# # # #                 active_count = int(summary_match.group(2))
# # # #             else:
# # # #                 inline = re.search(
# # # #                     r'(?:Total\s+Accounts.*?Active.*?Closed.*?\n|(\d+)\s+(\d+)\s+(\d+)\s+[\d,]+\s+\d+)',
# # # #                     full_text, re.IGNORECASE
# # # #                 )
# # # #                 if inline and inline.group(2):
# # # #                     active_count = int(inline.group(2))

# # # #         enq_12m_total = len(enquiry_dates)
# # # #         enq_sum_match = re.search(r'Enquiries?\s*\(?12M\)?\s*[:\s]+(\d+)', full_text, re.IGNORECASE)
# # # #         if enq_sum_match:
# # # #             enq_12m_total = max(enq_12m_total, int(enq_sum_match.group(1)))

# # # #         enq_L3m = min(len(enquiry_dates), enq_12m_total)
# # # #         enq_L6m = enq_12m_total
# # # #         enq_L12m = enq_12m_total

# # # #         if credit_score >= 750 and (written_off_count > 0 or dpd_90_count > 0):
# # # #             credit_score = 550

# # # #         total_accounts = max(len(accounts), active_count + settled_count + written_off_count)
# # # #         pct_active = active_count / total_accounts if total_accounts > 0 else 0.6

# # # #         extracted_data = {
# # # #             'Credit_Score': credit_score,
# # # #             'max_delinquency_level': max(dpd_90_count * 90, dpd_60_count * 60, dpd_30_count * 30),
# # # #             'num_times_30p_dpd': dpd_30_count,
# # # #             'num_times_60p_dpd': dpd_60_count,
# # # #             'num_times_delinquent': dpd_30_count + dpd_60_count + dpd_90_count,
# # # #             'num_deliq_6mts': dpd_30_count + dpd_60_count + dpd_90_count,
# # # #             'num_deliq_12mts': dpd_30_count + dpd_60_count + dpd_90_count,
# # # #             'max_deliq_6mts': dpd_90_count,
# # # #             'max_deliq_12mts': dpd_90_count,
# # # #             'enq_L3m': enq_L3m,
# # # #             'enq_L6m': enq_L6m,
# # # #             'enq_L12m': enq_L12m,
# # # #             'num_std': active_count,
# # # #             'num_std_6mts': active_count,
# # # #             'num_std_12mts': active_count,
# # # #             'num_sub': sub_standard_count,
# # # #             'num_sub_6mts': sub_standard_count,
# # # #             'num_dbt': dpd_90_count,
# # # #             'num_lss': written_off_count,
# # # #             'pct_of_active_TLs_ever': round(pct_active, 2),
# # # #             'pct_currentBal_all_TL': 0.3,
# # # #             'CC_utilization': round(cc_util, 2),
# # # #             'PL_utilization': 0.25,
# # # #             'max_unsec_exposure_inPct': cc_util_pct,
# # # #             'AGE': age_extracted,
# # # #             'NETMONTHLYINCOME': monthly_income,
# # # #             'Time_With_Curr_Empr': biz_vintage * 12,
# # # #             'CC_Flag': 1 if re.search(r'credit card', full_text, re.IGNORECASE) else 0,
# # # #             'PL_Flag': 1 if re.search(r'personal loan', full_text, re.IGNORECASE) else 0,
# # # #             'HL_Flag': 1 if re.search(r'home loan', full_text, re.IGNORECASE) else 0,
# # # #             'GL_Flag': 1 if re.search(r'gold loan', full_text, re.IGNORECASE) else 0,
# # # #             'raw_text': full_text,
# # # #             'success': True,
# # # #             'extraction_method': 'OCR+robust',
# # # #             'written_off_count': written_off_count,
# # # #             'settled_count': settled_count,
# # # #             'high_util_flag': high_util,
# # # #             'dpd_90_count_6m': dpd_90_count,
# # # #             'recent_deliq_flag': 1 if (dpd_90_count > 0 or dpd_60_count > 0) else 0,
# # # #             'account_quality_score': max(0, 100 - (written_off_count * 20) - (settled_count * 10) - (dpd_90_count * 15) - (dpd_30_count * 5))
# # # #         }
# # # #         return extracted_data
# # # #     except Exception as e:
# # # #         return {'error': str(e), 'message': f'Error extracting CIBIL data: {str(e)}', 'success': False}

# # # # # =============================================================================
# # # # # HYBRID DECISION ENGINE
# # # # # =============================================================================
# # # # def make_hybrid_decision_enhanced(customer_dict):
# # # #     policy_checks = {}
# # # #     age = customer_dict.get('age', 0)
# # # #     employment_type = customer_dict.get('employment_type', 'Salaried')
# # # #     kyc_verified = customer_dict.get('kyc_verified', True)
# # # #     bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
# # # #     fraud_flag = customer_dict.get('fraud_flag', False)
# # # #     age_min, age_max = 24, 70
# # # #     if age < age_min or age > age_max:
# # # #         policy_checks['age'] = f"❌ Age {age} (Required: {age_min}-{age_max})"
# # # #         return {'decision': "REJECT", 'reason': "Policy Gate: Age outside allowed range", 'confidence': 0,
# # # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # # #     policy_checks['age'] = f"✅ Age {age} (Valid)"
# # # #     if not kyc_verified:
# # # #         policy_checks['kyc'] = "❌ KYC Not Verified"
# # # #         return {'decision': "REJECT", 'reason': "Policy Gate: KYC verification required", 'confidence': 0,
# # # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # # #     policy_checks['kyc'] = "✅ KYC Verified"
# # # #     if bankruptcy_flag:
# # # #         policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
# # # #         return {'decision': "REJECT", 'reason': "Policy Gate: Active bankruptcy", 'confidence': 0,
# # # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # # #     policy_checks['bankruptcy'] = "✅ No Bankruptcy"
# # # #     if fraud_flag:
# # # #         policy_checks['fraud'] = "❌ Fraud Flag"
# # # #         return {'decision': "REJECT", 'reason': "Policy Gate: Fraud detected", 'confidence': 0,
# # # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # # #     policy_checks['fraud'] = "✅ No Fraud History"

# # # #     dependents = customer_dict.get('dependents', 0)
# # # #     dependents_flag_review = False
# # # #     if dependents > 5:
# # # #         policy_checks['dependents'] = f"⚠️ Dependents {dependents} (>5: Review Required)"
# # # #         dependents_flag_review = True
# # # #     else:
# # # #         policy_checks['dependents'] = f"✅ Dependents {dependents} (Acceptable)"

# # # #     monthly_income = customer_dict.get('avg_salary_6m', 0)
# # # #     employment_tenure = customer_dict.get('employment_tenure_months', 0)
# # # #     business_vintage = customer_dict.get('business_vintage_years', 0)
# # # #     if monthly_income < 15000:
# # # #         policy_checks['income'] = f"❌ Income ₹{monthly_income:,.0f} (Min: ₹15,000)"
# # # #         return {'decision': "REJECT", 'reason': "Policy Gate: Income below minimum", 'confidence': 0,
# # # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # # #     policy_checks['income'] = f"✅ Income ₹{monthly_income:,.0f}"
# # # #     if employment_type == 'Salaried' and employment_tenure < 6:
# # # #         policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
# # # #         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient tenure", 'confidence': 0,
# # # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # # #     elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
# # # #         policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
# # # #         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient business vintage", 'confidence': 0,
# # # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # # #     if employment_type == 'Salaried':
# # # #         policy_checks['tenure'] = f"✅ Tenure {employment_tenure} months"
# # # #     else:
# # # #         policy_checks['tenure'] = f"✅ Business Vintage {business_vintage} years"

# # # #     bureau_score = customer_dict.get('bureau_score', 0)
# # # #     dpd_90 = customer_dict.get('dpd_90_count_6m', 0)
# # # #     credit_utilization = customer_dict.get('credit_utilization_pct', 0)
# # # #     recent_inquiries = customer_dict.get('recent_inquiries_3m', 0)
# # # #     if bureau_score < 550:
# # # #         policy_checks['bureau'] = f"❌ Bureau Score {bureau_score} (Min: 550)"
# # # #         return {'decision': "REJECT", 'reason': "Policy Gate: Bureau score too low", 'confidence': 0,
# # # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # # #     policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"
# # # #     if dpd_90 > 0:
# # # #         policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD"
# # # #         return {'decision': "REJECT", 'reason': "Policy Gate: Severe delinquency", 'confidence': 0,
# # # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # # #     policy_checks['dpd'] = "✅ No 90+ DPD"
# # # #     if credit_utilization > 80:
# # # #         policy_checks['utilization'] = f"⚠️ High utilization {credit_utilization}%"
# # # #     else:
# # # #         policy_checks['utilization'] = f"✅ Utilization {credit_utilization}%"
# # # #     if recent_inquiries > 5:
# # # #         policy_checks['inquiries'] = f"⚠️ {recent_inquiries} recent inquiries"
# # # #     else:
# # # #         policy_checks['inquiries'] = f"✅ {recent_inquiries} inquiries"

# # # #     input_df = pd.DataFrame([customer_dict])
# # # #     for col in TOP_FEATURES:
# # # #         if col not in input_df.columns:
# # # #             if col in LE_MAP:
# # # #                 input_df[col] = "Unknown"
# # # #             else:
# # # #                 input_df[col] = 0
# # # #     for col, le in LE_MAP.items():
# # # #         if col in input_df.columns:
# # # #             val = str(input_df[col].values[0])
# # # #             try:
# # # #                 input_df[col] = le.transform([val])[0]
# # # #             except ValueError:
# # # #                 input_df[col] = 0
# # # #     final_input = input_df[TOP_FEATURES]
# # # #     pred_idx = MODEL.predict(final_input)[0]
# # # #     ml_decision = TARGET_LE.inverse_transform([pred_idx])[0]
# # # #     try:
# # # #         pred_proba = MODEL.predict_proba(final_input)[0]
# # # #         confidence = max(pred_proba) * 100
# # # #         class_probs = {cls: prob * 100 for cls, prob in zip(TARGET_LE.classes_, pred_proba)}
# # # #     except Exception:
# # # #         confidence = 75.0
# # # #         class_probs = {ml_decision: 100.0}

# # # #     loan_amount = customer_dict.get('loan_amount', 0)
# # # #     loan_tenure = customer_dict.get('loan_tenure_months', 12)
# # # #     interest_rate = customer_dict.get('interest_rate', 10.5)
# # # #     existing_emi = customer_dict.get('existing_emi', 0)
# # # #     affordability_data = calculate_affordability(monthly_income, loan_amount, interest_rate, loan_tenure, existing_emi)
# # # #     foir = affordability_data['foir_percentage']
# # # #     if ml_decision == "APPROVE" and foir > 50:
# # # #         ml_decision = "REVIEW"
# # # #     if dependents_flag_review and ml_decision == "APPROVE":
# # # #         ml_decision = "REVIEW"

# # # #     risk_score = calculate_final_risk_score(
# # # #         bureau_score=bureau_score,
# # # #         ml_confidence=confidence,
# # # #         foir=foir,
# # # #         dpd_90=dpd_90,
# # # #         dpd_30=customer_dict.get('dpd_30_count_6m', 0),
# # # #         net_surplus=customer_dict.get('net_cash_surplus_6m', 0),
# # # #         active_loans=customer_dict.get('active_loans_count', 0)
# # # #     )

# # # #     pd_percentage = calculate_final_pd(
# # # #         bureau_score=bureau_score,
# # # #         foir=foir,
# # # #         confidence=confidence,
# # # #         dpd_90_count=dpd_90,
# # # #         dpd_30_count=customer_dict.get('dpd_30_count_6m', 0),
# # # #         employment_type=employment_type,
# # # #         employment_tenure=employment_tenure,
# # # #         business_vintage=business_vintage,
# # # #         recent_inquiries=recent_inquiries,
# # # #         ml_decision=ml_decision
# # # #     )

# # # #     return {
# # # #         'decision': ml_decision,
# # # #         'reason': "Decision based on comprehensive assessment",
# # # #         'confidence': confidence,
# # # #         'class_probs': class_probs,
# # # #         'policy_checks': policy_checks,
# # # #         'risk_score': risk_score,
# # # #         'pd_percentage': round(pd_percentage, 2),
# # # #         'affordability_data': affordability_data
# # # #     }

# # # # # =============================================================================
# # # # # BATCH PREDICTION ENGINE
# # # # # =============================================================================
# # # # def process_batch_predictions(df):
# # # #     results = []
# # # #     for idx, row in df.iterrows():
# # # #         customer_dict = row.to_dict()
# # # #         for key, value in customer_dict.items():
# # # #             if isinstance(value, str):
# # # #                 if value.lower() in ['yes', 'true', '1']:
# # # #                     customer_dict[key] = True
# # # #                 elif value.lower() in ['no', 'false', '0']:
# # # #                     customer_dict[key] = False
# # # #         required_fields = {
# # # #             'age': 35, 'employment_type': 'Salaried', 'kyc_verified': True,
# # # #             'bankruptcy_flag': False, 'fraud_flag': False, 'employment_tenure_months': 24,
# # # #             'business_vintage_years': 0, 'bureau_score': 700, 'dpd_90_count_6m': 0,
# # # #             'dpd_30_count_6m': 0, 'credit_utilization_pct': 30, 'recent_inquiries_3m': 0,
# # # #             'active_loans_count': 0, 'existing_emi': 0, 'avg_salary_6m': 50000,
# # # #             'AMT_INCOME_TOTAL': 600000, 'net_cash_surplus_6m': 20000, 'salary_stability_flag': 'STABLE',
# # # #             'loan_amount': 180000, 'loan_tenure_months': 24, 'interest_rate': 10.5, 'AMT_ANNUITY': 8500,
# # # #             'dependents': 2,
# # # #         }
# # # #         for field, default in required_fields.items():
# # # #             if field not in customer_dict or pd.isna(customer_dict[field]):
# # # #                 customer_dict[field] = default
# # # #         try:
# # # #             decision_data = make_hybrid_decision_enhanced(customer_dict)
# # # #             reasons = generate_reason_codes(
# # # #                 decision=decision_data.get('decision', 'ERROR'),
# # # #                 customer_data=customer_dict,
# # # #                 affordability_data=decision_data.get('affordability_data', {}),
# # # #                 policy_checks=decision_data.get('policy_checks', {})
# # # #             )
# # # #             app_id = f"BATCH_{idx+1:04d}"
# # # #             affordability = decision_data.get('affordability_data', {})
# # # #             result = {
# # # #                 'application_id': app_id,
# # # #                 'decision': decision_data.get('decision', 'ERROR'),
# # # #                 'risk_score': decision_data.get('risk_score', 0),
# # # #                 'pd_percentage': decision_data.get('pd_percentage', 0),
# # # #                 'confidence': round(decision_data.get('confidence', 0), 2),
# # # #                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# # # #                 'reason_1': reasons[0] if len(reasons) > 0 else '',
# # # #                 'reason_2': reasons[1] if len(reasons) > 1 else '',
# # # #                 'reason_3': reasons[2] if len(reasons) > 2 else '',
# # # #                 'age': customer_dict.get('age', ''),
# # # #                 'employment_type': customer_dict.get('employment_type', ''),
# # # #                 'bureau_score': customer_dict.get('bureau_score', ''),
# # # #                 'monthly_income': customer_dict.get('avg_salary_6m', ''),
# # # #                 'loan_amount': customer_dict.get('loan_amount', ''),
# # # #                 'loan_tenure_months': customer_dict.get('loan_tenure_months', ''),
# # # #                 'interest_rate': customer_dict.get('interest_rate', ''),
# # # #                 'new_emi': affordability.get('new_emi', 0),
# # # #                 'existing_emi': affordability.get('existing_emi', 0),
# # # #                 'total_emi': affordability.get('total_emi', 0),
# # # #                 'foir_percentage': round(affordability.get('foir_percentage', 0), 2),
# # # #                 'net_disposable': affordability.get('net_disposable', 0),
# # # #                 'affordability_status': affordability.get('status', 'N/A'),
# # # #                 'dpd_90_count': customer_dict.get('dpd_90_count_6m', 0),
# # # #                 'dpd_30_count': customer_dict.get('dpd_30_count_6m', 0),
# # # #                 'credit_utilization': customer_dict.get('credit_utilization_pct', 0),
# # # #                 'recent_inquiries': customer_dict.get('recent_inquiries_3m', 0),
# # # #                 'active_loans': customer_dict.get('active_loans_count', 0),
# # # #                 'employment_tenure': customer_dict.get('employment_tenure_months', 0),
# # # #                 'business_vintage': customer_dict.get('business_vintage_years', 0),
# # # #                 'salary_stability': customer_dict.get('salary_stability_flag', ''),
# # # #                 'kyc_status': 'Verified' if customer_dict.get('kyc_verified', True) else 'Not Verified',
# # # #                 'bankruptcy': 'Yes' if customer_dict.get('bankruptcy_flag', False) else 'No',
# # # #                 'fraud': 'Yes' if customer_dict.get('fraud_flag', False) else 'No',
# # # #                 'prob_approve': round(decision_data.get('class_probs', {}).get('APPROVE', 0), 2),
# # # #                 'prob_review': round(decision_data.get('class_probs', {}).get('REVIEW', 0), 2),
# # # #                 'prob_reject': round(decision_data.get('class_probs', {}).get('REJECT', 0), 2),
# # # #             }
# # # #         except Exception as e:
# # # #             result = {
# # # #                 'application_id': f"BATCH_{idx+1:04d}",
# # # #                 'decision': 'ERROR',
# # # #                 'risk_score': 0, 'pd_percentage': 0, 'confidence': 0,
# # # #                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# # # #                 'reason_1': '', 'reason_2': '', 'reason_3': '',
# # # #                 'age': customer_dict.get('age', ''),
# # # #                 'employment_type': customer_dict.get('employment_type', ''),
# # # #                 'bureau_score': customer_dict.get('bureau_score', ''),
# # # #                 'monthly_income': customer_dict.get('avg_salary_6m', ''),
# # # #                 'loan_amount': customer_dict.get('loan_amount', ''),
# # # #                 'error_message': str(e)
# # # #             }
# # # #         results.append(result)
# # # #     return pd.DataFrame(results)

# # # # def create_download_link(df, filename="batch_results.csv"):
# # # #     csv = df.to_csv(index=False)
# # # #     b64 = base64.b64encode(csv.encode()).decode()
# # # #     return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;">📥 Download CSV</a>'

# # # # # =============================================================================
# # # # # MODERN UI COMPONENTS
# # # # # =============================================================================
# # # # def render_decision_header(decision_data, customer_data):
# # # #     decision = decision_data.get('decision', 'ERROR')
# # # #     risk_score = decision_data.get('risk_score', 0)
# # # #     pd_score = decision_data.get('pd_percentage', 0)
# # # #     approved_amount = customer_data.get('loan_amount', 0)
# # # #     tenure = customer_data.get('loan_tenure_months', 24)
# # # #     app_id = customer_data.get('application_id', 'N/A')
# # # #     timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# # # #     if decision == "APPROVE":
# # # #         card_class = "decision-card decision-card-approved"
# # # #         icon = "✓"
# # # #         subtitle = "Application Approved Successfully"
# # # #     elif decision == "REJECT":
# # # #         card_class = "decision-card decision-card-rejected"
# # # #         icon = "✗"
# # # #         subtitle = "Application Not Approved"
# # # #     else:
# # # #         card_class = "decision-card decision-card-review"
# # # #         icon = "⚠"
# # # #         subtitle = "Requires Manual Review"
# # # #     st.markdown(f"""
# # # #         <div class="{card_class}">
# # # #             <div class="decision-title"><span>{icon}</span><span>{decision}</span></div>
# # # #             <div class="decision-subtitle">{subtitle}</div>
# # # #         </div>
# # # #     """, unsafe_allow_html=True)
# # # #     col1, col2, col3, col4, col5 = st.columns(5)
# # # #     with col1:
# # # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{risk_score}</div><div class="stat-label">Risk Score</div></div>', unsafe_allow_html=True)
# # # #     with col2:
# # # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{pd_score}%</div><div class="stat-label">PD Score</div></div>', unsafe_allow_html=True)
# # # #     with col3:
# # # #         st.markdown(f'<div class="stat-card"><div class="stat-number">₹{approved_amount:,.0f}</div><div class="stat-label">Loan Amount</div></div>', unsafe_allow_html=True)
# # # #     with col4:
# # # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{tenure}</div><div class="stat-label">Tenure (Months)</div></div>', unsafe_allow_html=True)
# # # #     with col5:
# # # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{decision_data.get("confidence", 0):.0f}%</div><div class="stat-label">Confidence</div></div>', unsafe_allow_html=True)
# # # #     st.markdown("<br>", unsafe_allow_html=True)
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         st.markdown(f'<div class="info-box"><strong>📋 Application ID:</strong> {app_id}</div>', unsafe_allow_html=True)
# # # #     with col2:
# # # #         st.markdown(f'<div class="info-box"><strong>🕐 Decision Timestamp:</strong> {timestamp}</div>', unsafe_allow_html=True)

# # # # def render_info_card(title, icon, data_dict, status_dict=None):
# # # #     st.markdown(f'<div class="info-card"><div class="info-card-title"><span class="icon">{icon}</span><span>{title}</span></div><div class="info-card-content">', unsafe_allow_html=True)
# # # #     for label, value in data_dict.items():
# # # #         status = ""
# # # #         if status_dict and label in status_dict:
# # # #             if status_dict[label] == "pass":
# # # #                 status = '<span class="status-badge badge-pass">✓ Passed</span>'
# # # #             elif status_dict[label] == "fail":
# # # #                 status = '<span class="status-badge badge-fail">✗ Failed</span>'
# # # #             elif status_dict[label] == "warning":
# # # #                 status = '<span class="status-badge badge-warning">⚠ Warning</span>'
# # # #         st.markdown(f'<div class="data-row"><span class="data-label">{label}</span><span class="data-value">{value} {status}</span></div>', unsafe_allow_html=True)
# # # #     st.markdown('</div></div>', unsafe_allow_html=True)

# # # # def render_reason_codes(reasons):
# # # #     st.markdown('<div class="info-card"><div class="info-card-title"><span class="icon">📝</span><span>Decision Reasons</span></div><div class="info-card-content">', unsafe_allow_html=True)
# # # #     for i, reason in enumerate(reasons, 1):
# # # #         st.markdown(f'<div class="reason-item"><span class="reason-icon">{i}.</span><span>{reason}</span></div>', unsafe_allow_html=True)
# # # #     st.markdown('</div></div>', unsafe_allow_html=True)

# # # # def create_modern_gauge(value, title, max_value=100):
# # # #     if value <= 50:
# # # #         color = "#f56565"
# # # #     elif value <= 75:
# # # #         color = "#ed8936"
# # # #     else:
# # # #         color = "#48bb78"
# # # #     fig = go.Figure(go.Indicator(
# # # #         mode="gauge+number",
# # # #         value=value,
# # # #         title={'text': title, 'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}},
# # # #         number={'suffix': "%", 'font': {'size': 36, 'color': '#2d3748', 'family': 'Inter'}},
# # # #         gauge={
# # # #             'axis': {'range': [0, max_value], 'tickfont': {'size': 12, 'color': '#718096'}},
# # # #             'bar': {'color': color, 'thickness': 0.75},
# # # #             'bgcolor': 'white', 'borderwidth': 0,
# # # #             'steps': [
# # # #                 {'range': [0, 50], 'color': '#fed7d7'},
# # # #                 {'range': [50, 75], 'color': '#feebc8'},
# # # #                 {'range': [75, 100], 'color': '#c6f6d5'}
# # # #             ]
# # # #         }
# # # #     ))
# # # #     fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white',
# # # #                       font={'family': 'Inter', 'color': '#2d3748'})
# # # #     return fig

# # # # def create_modern_bar_chart(class_probs):
# # # #     df = pd.DataFrame({'Decision': list(class_probs.keys()), 'Probability': list(class_probs.values())})
# # # #     colors = {'REVIEW': '#ed8936', 'APPROVE': '#48bb78', 'REJECT': '#f56565'}
# # # #     fig = px.bar(df, x='Decision', y='Probability', title='Decision Probabilities', color='Decision',
# # # #                  color_discrete_map=colors, text='Probability')
# # # #     fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
# # # #     fig.update_layout(
# # # #         showlegend=False, yaxis_title='Probability (%)', xaxis_title='', height=300,
# # # #         margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white', plot_bgcolor='white',
# # # #         font={'family': 'Inter', 'color': '#2d3748'},
# # # #         yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]},
# # # #         xaxis={'tickfont': {'size': 14, 'color': '#2d3748'}}
# # # #     )
# # # #     return fig

# # # # # =============================================================================
# # # # # STAGE 2 RESULTS DISPLAY FUNCTION
# # # # # =============================================================================
# # # # def display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data):
# # # #     st.markdown("---")
# # # #     st.markdown('<p class="main-header">🎯 Stage 2 Final Results</p>', unsafe_allow_html=True)

# # # #     final_decision = stage2_result.get('final_decision', 'ERROR')
# # # #     interest_range = stage2_result.get('interest_rate_range', 'N/A')
# # # #     stage2_tier = stage2_result.get('stage2_tier', 'N/A')
# # # #     stage2_confidence = stage2_result.get('stage2_confidence', 0)
# # # #     combined_risk_score = stage2_result.get('combined_risk_score', 0)

# # # #     if final_decision == "APPROVE":
# # # #         card_class = "decision-card decision-card-approved"
# # # #         icon = "✓"
# # # #         subtitle = "Application Approved - Proceed to Disbursement"
# # # #     elif final_decision in ["REVIEW", "MANUAL_REVIEW"]:
# # # #         card_class = "decision-card decision-card-review"
# # # #         icon = "⚠"
# # # #         subtitle = "Requires Manual Review"
# # # #     else:
# # # #         card_class = "decision-card decision-card-rejected"
# # # #         icon = "✗"
# # # #         subtitle = "Application Rejected"

# # # #     st.markdown(f"""
# # # #         <div class="{card_class}">
# # # #             <div class="decision-title"><span>{icon}</span><span>{final_decision}</span></div>
# # # #             <div class="decision-subtitle">{subtitle}</div>
# # # #         </div>
# # # #     """, unsafe_allow_html=True)

# # # #     col1, col2, col3, col4 = st.columns(4)
# # # #     with col1:
# # # #         st.metric("Risk Tier", stage2_tier)
# # # #     with col2:
# # # #         st.metric("Interest Rate", interest_range)
# # # #     with col3:
# # # #         st.metric("Combined Risk Score", combined_risk_score)
        
# # # #     with col4:
# # # #         confidence_display = f"{stage2_confidence:.1f}%" if stage2_confidence is not None else "N/A"
# # # #         st.metric("Stage 2 Confidence", confidence_display)

# # # #     st.markdown("<br>", unsafe_allow_html=True)

# # # #     tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔬 Analysis", "📋 Data", "📥 Download"])

# # # #     with tab1:
# # # #         st.markdown("### 📊 Decision Comparison")
# # # #         comparison_df = pd.DataFrame([
# # # #         {'Stage': 'Stage 1 (Basic)', 'Decision': st.session_state.get('stage1_decision'),
# # # #         'Risk Score': stage1_data.get('risk_score', 'N/A'), 'Tier': 'N/A'},
# # # #         {'Stage': 'Stage 2 (CIBIL Deep)', 'Decision': final_decision,
# # # #         'Risk Score': combined_risk_score, 'Tier': f"{stage2_tier} | {interest_range}"}
# # # #         ])
# # # #         st.dataframe(comparison_df, width='stretch', hide_index=True)

# # # #         st.markdown("### 🎯 Risk Tier Details")
# # # #         tier_info = {
# # # #             'P1': {'name': 'Premium', 'color': '#10B981', 'desc': 'Excellent credit profile'},
# # # #             'P2': {'name': 'Standard', 'color': '#3B82F6', 'desc': 'Good credit profile'},
# # # #             'P3': {'name': 'Subprime', 'color': '#F59E0B', 'desc': 'Fair credit with concerns'},
# # # #             'P4': {'name': 'High Risk', 'color': '#EF4444', 'desc': 'High risk profile'},
# # # #         }
# # # #         if stage2_tier in tier_info:
# # # #             tier_data = tier_info[stage2_tier]
# # # #             st.markdown(f"""
# # # #                 <div style="background: {tier_data['color']}; color: white; padding: 1rem; border-radius: 0.5rem;">
# # # #                     <h3 style="margin: 0; color: white;">{stage2_tier}: {tier_data['name']}</h3>
# # # #                     <p style="margin: 0.5rem 0;">Interest Rate: {interest_range}</p>
# # # #                     <p style="margin: 0;">{tier_data['desc']}</p>
# # # #                 </div>
# # # #             """, unsafe_allow_html=True)
# # # #         st.markdown("### 📝 Decision Reasoning")
# # # #         st.info(stage2_result.get('reason', 'N/A'))

# # # #     with tab2:
# # # #         st.markdown("### 🔬 Detailed Analysis")
# # # #         col1, col2 = st.columns(2)
# # # #         with col1:
# # # #             st.markdown("**Tier Probabilities**")
# # # #             if 'tier_probabilities' in stage2_result:
# # # #                 for tier, prob in stage2_result['tier_probabilities'].items():
# # # #                     st.metric(tier, f"{prob:.1f}%")
# # # #         with col2:
# # # #             st.markdown("**Stage Scores**")
# # # #             st.metric("Stage 1 Risk Score", stage1_data.get('risk_score', 'N/A'))
# # # #             st.metric("Stage 2 Risk Score", stage2_result.get('stage2_risk_score', 'N/A'))
# # # #             st.metric("Combined Score", combined_risk_score)
# # # #         with st.expander("📋 Complete Stage 2 Result"):
# # # #             st.json(stage2_result)

# # # #     with tab3:
# # # #         st.markdown("### 📋 Input Data")
# # # #         col1, col2 = st.columns(2)
# # # #         with col1:
# # # #             with st.expander("Stage 1 Customer Data"):
# # # #                 st.json(stage1_customer)
# # # #         with col2:
# # # #             with st.expander("Enhanced CIBIL Data"):
# # # #                 st.json(enhanced_customer_data)

# # # #     with tab4:
# # # #         st.markdown("### 📥 Download Reports")
# # # #         bureau_score = stage1_customer.get('bureau_score', 0)
# # # #         dpd_90 = stage1_customer.get('dpd_90_count_6m', 0)
# # # #         dpd_30 = stage1_customer.get('dpd_30_count_6m', 0)
# # # #         foir = stage1_data.get('affordability_data', {}).get('foir_percentage', 0)
# # # #         employment_type = stage1_customer.get('employment_type', 'Salaried')
# # # #         employment_tenure = stage1_customer.get('employment_tenure_months', 0)
# # # #         business_vintage = stage1_customer.get('business_vintage_years', 0)
# # # #         ml_decision = stage1_data.get('decision', 'ERROR')
# # # #         confidence = stage1_data.get('confidence', 0)

# # # #         report_data = {
# # # #             'application_id': stage1_customer.get('application_id'),
# # # #             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# # # #             'decision': stage1_data.get('decision'),
# # # #             'risk_score': stage1_data.get('risk_score'),
# # # #             'pd_percentage': stage1_data.get('pd_percentage'),
# # # #             'confidence': stage1_data.get('confidence'),
# # # #             'policy_checks': stage1_data.get('policy_checks', {}),
# # # #             'affordability_data': stage1_data.get('affordability_data', {}),
# # # #             'customer_data': stage1_customer,
# # # #             'reason_codes': stage1_customer.get('reason_codes', []),
# # # #             'pd_calculation_factors': {
# # # #                 'bureau_score': bureau_score,
# # # #                 'base_pd': bureau_score_to_pd(bureau_score),
# # # #                 'dpd_90': dpd_90, 'dpd_30': dpd_30,
# # # #                 'delinquency_multiplier': delinquency_to_pd_multiplier(dpd_90, dpd_30),
# # # #                 'foir': foir,
# # # #                 'foir_adjustment': foir_to_pd_adjustment(foir),
# # # #                 'employment_adjustment': employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage),
# # # #                 'ml_adjustment': ml_confidence_to_pd_adjustment(confidence, ml_decision),
# # # #                 'final_pd': stage1_data.get('pd_percentage', 0)
# # # #             },
# # # #             'stage2_final_decision': final_decision,
# # # #             'stage2_tier': stage2_tier,
# # # #             'stage2_interest_range': interest_range,
# # # #             'stage2_combined_risk_score': combined_risk_score,
# # # #             'stage2_confidence': stage2_confidence,
# # # #             'stage2_reason': stage2_result.get('reason'),
# # # #             'stage2_tier_probabilities': stage2_result.get('tier_probabilities'),
# # # #             'stage2_complete_analysis': stage2_result,
# # # #             'stage1_data': stage1_data,
# # # #             'enhanced_customer_data': enhanced_customer_data
# # # #         }

# # # #         if PDF_AVAILABLE and generate_audit_pdf is not None:
# # # #             try:
# # # #                 pdf_buffer = generate_audit_pdf(report_data)
# # # #                 st.download_button(
# # # #                     "📥 Download PDF Report",
# # # #                     data=pdf_buffer,
# # # #                     file_name=f"stage2_report_{stage1_customer.get('application_id', 'unknown')}.pdf",
# # # #                     mime="application/pdf",
# # # #                     use_container_width=True
# # # #                 )
# # # #             except Exception as e:
# # # #                 st.error(f"PDF generation failed: {str(e)}")
# # # #         else:
# # # #             st.warning("PDF generation is not available. Please install the required PDF generator module.")

# # # #     st.markdown("---")
# # # #     col1, col2, col3 = st.columns(3)
# # # #     with col1:
# # # #         if st.button("🔄 New Assessment", key="new_assessment_stage2", use_container_width=True):
# # # #             st.session_state.stage1_complete = False
# # # #             st.session_state.stage1_decision = None
# # # #             st.session_state.stage1_data = None
# # # #             st.session_state.current_customer_data = None
# # # #             st.session_state.page_navigation = "👤 Assessment"
# # # #             st.rerun()
# # # #     with col2:
# # # #         if st.button("← Back to Stage 1", key="back_to_stage1", use_container_width=True):
# # # #             st.session_state.page_navigation = "👤 Assessment"
# # # #             st.rerun()
# # # #     with col3:
# # # #         if st.button("🏠 Home", key="home_stage2", use_container_width=True):
# # # #             st.session_state.page_navigation = "🏠 Home"
# # # #             st.rerun()

# # # # # =============================================================================
# # # # # SIDEBAR
# # # # # =============================================================================
# # # # with st.sidebar:
# # # #     st.markdown("# 🏦 Credit Risk Engine")
# # # #     st.markdown("---")

# # # #     navigation_options = ["🏠 Home", "👤 Assessment", "📊 Batch Process", "📈 Model Info", "ℹ️ About"]

# # # #     if (st.session_state.stage1_complete and
# # # #             st.session_state.stage1_decision in ['APPROVE', 'REVIEW']):
# # # #         navigation_options.insert(2, "🔬 Stage 2 Analysis")
# # # #         st.success(f"✅ Stage 1: {st.session_state.stage1_decision}")
# # # #         st.info("🔬 Stage 2 Analysis unlocked!")
# # # #     elif st.session_state.stage1_complete:
# # # #         st.warning(f"⚠️ Stage 1: {st.session_state.stage1_decision}")
# # # #         st.caption("Stage 2 only for APPROVE/REVIEW")

# # # #     page = st.radio(
# # # #         "**Navigation**",
# # # #         navigation_options,
# # # #         label_visibility="collapsed",
# # # #         key="page_navigation"
# # # #     )

# # # #     st.markdown("---")

# # # #     stage2_indicator = '✅ Active' if STAGE2_AVAILABLE and is_stage2_available() else '❌ Inactive'
# # # #     ocr_indicator = '✅ Ready' if OCR_AVAILABLE else '❌ Not Installed'
# # # #     if not OCR_AVAILABLE and OCR_ERROR_MSG:
# # # #         ocr_indicator += ' ⚠️'
# # # #     pdf_indicator = '✅ Ready' if PDF_AVAILABLE else '❌ Not Installed'

# # # #     st.markdown(f"""
# # # #     <div class="info-card">
# # # #         <div class="info-card-title">System Status</div>
# # # #         <div class="info-card-content">
# # # #             <div class="data-row"><span class="data-label">Model</span><span class="data-value">✅ Loaded</span></div>
# # # #             <div class="data-row"><span class="data-label">Version</span><span class="data-value">8.3</span></div>
# # # #             <div class="data-row"><span class="data-label">Stage 2</span><span class="data-value">{stage2_indicator}</span></div>
# # # #             <div class="data-row"><span class="data-label">OCR</span><span class="data-value">{ocr_indicator}</span></div>
# # # #             <div class="data-row"><span class="data-label">PDF Gen</span><span class="data-value">{pdf_indicator}</span></div>
# # # #             <div class="data-row"><span class="data-label">Features</span><span class="data-value">{len(TOP_FEATURES)}</span></div>
# # # #         </div>
# # # #     </div>
# # # #     """, unsafe_allow_html=True)

# # # #     with st.expander("🎯 **Top Features**"):
# # # #         for i, feat in enumerate(TOP_FEATURES[:5], 1):
# # # #             st.markdown(f"`{i}.` {feat}")

# # # #     if st.session_state.stage1_complete:
# # # #         st.markdown("---")
# # # #         st.markdown("### 🚀 Quick Actions")
# # # #         if st.button("🔄 New Assessment", key="new_assessment_sidebar", use_container_width=True):
# # # #             st.session_state.stage1_complete = False
# # # #             st.session_state.stage1_decision = None
# # # #             st.session_state.stage1_data = None
# # # #             st.session_state.current_customer_data = None
# # # #             st.session_state.extracted_cibil_data = None
# # # #             st.rerun()

# # # # # =============================================================================
# # # # # PAGE ROUTING
# # # # # =============================================================================

# # # # if page == "🏠 Home":
# # # #     st.markdown('<p class="main-header">Credit Risk Assessment Platform</p>', unsafe_allow_html=True)
# # # #     st.markdown("""
# # # #         <div class="info-box">
# # # #             <h3 style="margin-top: 0;">🎯 AI-Powered Lending Decisions</h3>
# # # #             <p style="margin-bottom: 0;">Comprehensive credit risk evaluation combining hard policy rules,
# # # #             machine learning models, and affordability analysis for accurate lending decisions.</p>
# # # #         </div>
# # # #     """, unsafe_allow_html=True)
# # # #     st.markdown("<br>", unsafe_allow_html=True)
# # # #     col1, col2, col3 = st.columns(3)
# # # #     with col1:
# # # #         st.markdown("""
# # # #             <div class="info-card"><div class="info-card-title"><span class="icon">🛡️</span><span>Policy Gates</span></div>
# # # #             <div class="info-card-content"><ul><li>Age & KYC verification</li><li>Employment stability</li>
# # # #             <li>Minimum income checks</li><li>Credit bureau thresholds</li><li>Bankruptcy & fraud detection</li></ul></div></div>
# # # #         """, unsafe_allow_html=True)
# # # #     with col2:
# # # #         st.markdown("""
# # # #             <div class="info-card"><div class="info-card-title"><span class="icon">🤖</span><span>ML Assessment</span></div>
# # # #             <div class="info-card-content"><ul><li>Random Forest classifier</li><li>60K+ training samples</li>
# # # #             <li>Confidence scoring</li><li>Multi-class prediction</li><li>Feature importance</li></ul></div></div>
# # # #         """, unsafe_allow_html=True)
# # # #     with col3:
# # # #         st.markdown("""
# # # #             <div class="info-card"><div class="info-card-title"><span class="icon">💰</span><span>Affordability</span></div>
# # # #             <div class="info-card-content"><ul><li>EMI calculation</li><li>FOIR analysis (max 50%)</li>
# # # #             <li>Net disposable income</li><li>Debt burden assessment</li><li>Affordability scoring</li></ul></div></div>
# # # #         """, unsafe_allow_html=True)
# # # #     st.markdown("<br>", unsafe_allow_html=True)
# # # #     col1, col2, col3, col4 = st.columns(4)
# # # #     with col1: st.metric("🎯 Accuracy", "85%", "+2%")
# # # #     with col2: st.metric("⚡ Avg Response", "1.2s", "-0.3s")
# # # #     with col3: st.metric("📊 Features", len(TOP_FEATURES))
# # # #     with col4: st.metric("🔄 Version", "8.3", "Latest")
# # # #     st.markdown("<br>", unsafe_allow_html=True)
# # # #     st.markdown("""
# # # #         <div class="warning-box">
# # # #             <strong>🆕 New in Version 8.3:</strong><br>
# # # #             • Fixed Mixed Numeric Types Error<br>
# # # #             • Fixed Missing Submit Button<br>
# # # #             • Dependents field properly integrated<br>
# # # #             • PDF auto-fill from CIBIL report<br>
# # # #             • Industry-Standard PD Methodology<br>
# # # #             • Professional UI/UX Enhancements
# # # #         </div>
# # # #     """, unsafe_allow_html=True)

# # # # elif page == "👤 Assessment":
# # # #     st.markdown('<p class="main-header">Credit Assessment</p>', unsafe_allow_html=True)

# # # #     pdf_just_extracted = st.session_state.get('pdf_just_extracted', False)

# # # #     with st.expander("📄 Upload CIBIL PDF to auto‑fill bureau fields",
# # # #                      expanded=pdf_just_extracted or not st.session_state.get('pdf_bureau_score')):

# # # #         if pdf_just_extracted:
# # # #             ex = st.session_state.get('_last_extraction', {})
# # # #             st.success("✅ CIBIL data extracted — form fields below have been updated automatically.")
# # # #             c1, c2, c3, c4 = st.columns(4)
# # # #             c1.metric("Credit Score",    ex.get('Credit_Score', '—'))
# # # #             c2.metric("Monthly Income",  f"₹{ex.get('NETMONTHLYINCOME', 0):,}")
# # # #             c3.metric("DPD 90+ Count",   ex.get('dpd_90_count_6m', 0))
# # # #             c4.metric("CC Utilization",  f"{ex.get('CC_utilization', 0)*100:.0f}%")
# # # #             c1, c2, c3, c4 = st.columns(4)
# # # #             c1.metric("DPD 30+ Count",  ex.get('num_times_30p_dpd', 0))
# # # #             c2.metric("Inquiries (3M)", ex.get('enq_L3m', 0))
# # # #             c3.metric("Active Accounts", ex.get('num_std', 0))
# # # #             c4.metric("Written-Off",    ex.get('written_off_count', 0))
# # # #             if ex.get('written_off_count', 0) > 0 or ex.get('settled_count', 0) > 0:
# # # #                 st.warning(f"⚠️ Severe negatives detected: "
# # # #                            f"{ex.get('written_off_count', 0)} written-off, "
# # # #                            f"{ex.get('settled_count', 0)} settled accounts. "
# # # #                            f"Score overridden to {ex.get('Credit_Score', '?')}.")
# # # #             if st.toggle("📋 Show full extracted JSON"):
# # # #                 st.json({k: v for k, v in ex.items() if k != 'raw_text'})
# # # #             st.markdown("---")
# # # #             if st.button("🔄 Upload a different PDF", key="reset_pdf"):
# # # #                 st.session_state.pdf_just_extracted = False
# # # #                 st.session_state.pop('_last_extraction', None)
# # # #                 st.rerun()
# # # #         else:
# # # #             st.markdown('<div class="info-box">💡 Complete the form below or upload a CIBIL PDF to auto‑fill bureau data.</div>', unsafe_allow_html=True)
# # # #             uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="assessment_pdf")
# # # #             if uploaded_pdf is not None:
# # # #                 st.info(f"📄 File ready: **{uploaded_pdf.name}** ({uploaded_pdf.size/1024:.1f} KB)")
# # # #                 if st.button("🔍 Extract & Auto-fill Form", key="extract_assessment", type="primary", use_container_width=True):
# # # #                     with st.spinner("🔄 Running OCR on CIBIL PDF — this takes 10-30 seconds..."):
# # # #                         extraction_result = extract_cibil_from_pdf(uploaded_pdf)
# # # #                     if extraction_result.get('success', False):
# # # #                         st.session_state.pdf_age               = int(extraction_result.get('AGE', 35))
# # # #                         st.session_state.pdf_employment_type   = 'Salaried'
# # # #                         st.session_state.pdf_kyc               = True
# # # #                         st.session_state.pdf_bankruptcy        = False
# # # #                         st.session_state.pdf_fraud             = False
# # # #                         st.session_state.pdf_bureau_score      = int(extraction_result.get('Credit_Score', 720))
# # # #                         st.session_state.pdf_dpd_90            = int(extraction_result.get('dpd_90_count_6m', 0))
# # # #                         st.session_state.pdf_dpd_30            = int(extraction_result.get('num_times_30p_dpd', 0))
# # # #                         st.session_state.pdf_credit_util       = int(float(extraction_result.get('CC_utilization', 0.35)) * 100)
# # # #                         st.session_state.pdf_inquiries         = int(extraction_result.get('enq_L3m', 2))
# # # #                         st.session_state.pdf_active_loans      = int(extraction_result.get('num_std', 1))
# # # #                         st.session_state.pdf_existing_emi      = int(extraction_result.get('existing_emi', 15000))
# # # #                         st.session_state.pdf_monthly_income    = int(extraction_result.get('NETMONTHLYINCOME', 50000))
# # # #                         st.session_state.pdf_annual_income     = int(extraction_result.get('NETMONTHLYINCOME', 50000)) * 12
# # # #                         st.session_state.pdf_net_surplus       = int(extraction_result.get('net_surplus', 20000))
# # # #                         st.session_state.pdf_salary_stability  = 'STABLE'
# # # #                         st.session_state.pdf_loan_amount       = int(extraction_result.get('loan_amount', 180000))
# # # #                         st.session_state.pdf_loan_tenure       = int(extraction_result.get('loan_tenure', 24))
# # # #                         st.session_state.pdf_interest_rate     = float(extraction_result.get('interest_rate', 10.5))
# # # #                         st.session_state.pdf_amt_annuity       = int(extraction_result.get('amt_annuity', 8500))
# # # #                         st.session_state.pdf_employment_tenure = int(extraction_result.get('Time_With_Curr_Empr', 24))
# # # #                         st.session_state.pdf_business_vintage  = int(extraction_result.get('business_vintage', 3))
# # # #                         st.session_state.pdf_dependents        = int(extraction_result.get('dependents', 2))
# # # #                         st.session_state.pdf_just_extracted    = True
# # # #                         st.session_state._last_extraction      = extraction_result
# # # #                         st.rerun()
# # # #                     else:
# # # #                         st.error(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")
# # # #                         st.info("Tip: Make sure Tesseract and Poppler are installed and paths are set correctly.")

# # # #     with st.form("assessment_form"):
# # # #         st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
# # # #         col1, col2, col3 = st.columns(3)
# # # #         with col1:
# # # #             age = st.number_input(
# # # #                 "Age", 24, 70,
# # # #                 value=int(st.session_state.get('pdf_age', 35)),
# # # #                 help="Customer's age in years (Minimum: 24, Maximum: 70)"
# # # #             )
# # # #             employment_type = st.selectbox(
# # # #                 "Employment Type",
# # # #                 ['Salaried', 'Self-Employed', 'Business'],
# # # #                 index=['Salaried', 'Self-Employed', 'Business'].index(
# # # #                     st.session_state.get('pdf_employment_type', 'Salaried')
# # # #                 )
# # # #             )
# # # #         with col2:
# # # #             dependents = st.number_input(
# # # #                 "Number of Dependents", 0, 20,
# # # #                 value=int(st.session_state.get('pdf_dependents', 2)),
# # # #                 help="1-5: Approve eligible | >5: Review required"
# # # #             )
# # # #             kyc_verified = st.selectbox(
# # # #                 "KYC Verified", ['Yes', 'No'],
# # # #                 index=0 if st.session_state.get('pdf_kyc', True) else 1
# # # #             ) == 'Yes'
# # # #         with col3:
# # # #             bankruptcy_flag = st.selectbox(
# # # #                 "Bankruptcy Flag", ['No', 'Yes'],
# # # #                 index=0 if not st.session_state.get('pdf_bankruptcy', False) else 1
# # # #             ) == 'Yes'
# # # #             fraud_flag = st.selectbox(
# # # #                 "Fraud Flag", ['No', 'Yes'],
# # # #                 index=0 if not st.session_state.get('pdf_fraud', False) else 1
# # # #             ) == 'Yes'
# # # #             if employment_type == 'Salaried':
# # # #                 employment_tenure = st.number_input(
# # # #                     "Employment Tenure (months)", 0, 600,
# # # #                     value=int(st.session_state.get('pdf_employment_tenure', 24))
# # # #                 )
# # # #                 business_vintage = 0
# # # #             else:
# # # #                 business_vintage = st.number_input(
# # # #                     "Business Vintage (years)", 0, 50,
# # # #                     value=int(st.session_state.get('pdf_business_vintage', 3))
# # # #                 )
# # # #                 employment_tenure = 0

# # # #         st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
# # # #         col1, col2, col3 = st.columns(3)
# # # #         with col1:
# # # #             bureau_score = st.number_input(
# # # #                 "Bureau Score", 300, 900,
# # # #                 value=int(st.session_state.get('pdf_bureau_score', 720)), step=10
# # # #             )
# # # #             dpd_90_6m = st.number_input(
# # # #                 "DPD 90+ (Last 6M)", 0, 20,
# # # #                 value=int(st.session_state.get('pdf_dpd_90', 0))
# # # #             )
# # # #             dpd_30_6m = st.number_input(
# # # #                 "DPD 30+ (Last 6M)", 0, 20,
# # # #                 value=int(st.session_state.get('pdf_dpd_30', 0))
# # # #             )
# # # #         with col2:
# # # #             credit_utilization = st.number_input(
# # # #                 "Credit Utilization (%)", 0, 100,
# # # #                 value=int(st.session_state.get('pdf_credit_util', 30))
# # # #             )
# # # #             recent_inquiries = st.number_input(
# # # #                 "Recent Inquiries (3M)", 0, 20,
# # # #                 value=int(st.session_state.get('pdf_inquiries', 2))
# # # #             )
# # # #         with col3:
# # # #             active_loans = st.number_input(
# # # #                 "Active Loans", 0, 10,
# # # #                 value=int(st.session_state.get('pdf_active_loans', 1))
# # # #             )
# # # #             existing_emi = st.number_input(
# # # #                 "Existing Total EMI (₹)", 0, 200000,
# # # #                 value=int(st.session_state.get('pdf_existing_emi', 15000)), step=1000
# # # #             )

# # # #         st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
# # # #         col1, col2, col3, col4 = st.columns(4)
# # # #         with col1:
# # # #             avg_salary = st.number_input(
# # # #                 "Monthly Income (₹)", 0, 1000000,
# # # #                 value=int(st.session_state.get('pdf_monthly_income', 50000)), step=5000
# # # #             )
# # # #             amt_income = st.number_input(
# # # #                 "Annual Income (₹)", 0, 10000000,
# # # #                 value=int(st.session_state.get('pdf_annual_income', 600000)), step=10000
# # # #             )
# # # #         with col2:
# # # #             net_surplus = st.number_input(
# # # #                 "Net Cash Surplus (₹)", -100000, 500000,
# # # #                 value=int(st.session_state.get('pdf_net_surplus', 20000)), step=5000
# # # #             )
# # # #             salary_stability = st.selectbox(
# # # #                 "Salary Stability",
# # # #                 ['STABLE', 'MODERATE', 'UNSTABLE'],
# # # #                 index=['STABLE', 'MODERATE', 'UNSTABLE'].index(
# # # #                     st.session_state.get('pdf_salary_stability', 'STABLE')
# # # #                 )
# # # #             )
# # # #         with col3:
# # # #             loan_amount = st.number_input(
# # # #                 "Loan Amount (₹)", 0, 5000000,
# # # #                 value=int(st.session_state.get('pdf_loan_amount', 180000)), step=10000
# # # #             )
# # # #             loan_tenure = st.number_input(
# # # #                 "Tenure (months)", 3, 360,
# # # #                 value=int(st.session_state.get('pdf_loan_tenure', 24))
# # # #             )
# # # #         with col4:
# # # #             interest_rate = st.number_input(
# # # #                 "Interest Rate (%)", 8.0, 20.0,
# # # #                 value=float(st.session_state.get('pdf_interest_rate', 10.5)), step=0.5
# # # #             )
# # # #             amt_annuity = st.number_input(
# # # #                 "Requested EMI (₹)", 0, 200000,
# # # #                 value=int(st.session_state.get('pdf_amt_annuity', 8500)), step=500
# # # #             )

# # # #         st.markdown("<br>", unsafe_allow_html=True)
# # # #         submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)

# # # #     if submitted:
# # # #         timestamp = datetime.now()
# # # #         app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
# # # #         customer_data = {
# # # #             'age': age,
# # # #             'employment_type': employment_type,
# # # #             'dependents': dependents,
# # # #             'kyc_verified': kyc_verified,
# # # #             'bankruptcy_flag': bankruptcy_flag,
# # # #             'fraud_flag': fraud_flag,
# # # #             'employment_tenure_months': employment_tenure,
# # # #             'business_vintage_years': business_vintage,
# # # #             'bureau_score': bureau_score,
# # # #             'dpd_90_count_6m': dpd_90_6m,
# # # #             'dpd_30_count_6m': dpd_30_6m,
# # # #             'credit_utilization_pct': credit_utilization,
# # # #             'max_utilization': credit_utilization,
# # # #             'recent_inquiries_3m': recent_inquiries,
# # # #             'active_loans_count': active_loans,
# # # #             'avg_salary_6m': avg_salary,
# # # #             'AMT_INCOME_TOTAL': amt_income,
# # # #             'net_cash_surplus_6m': net_surplus,
# # # #             'salary_stability_flag': salary_stability,
# # # #             'loan_amount': loan_amount,
# # # #             'loan_tenure_months': loan_tenure,
# # # #             'interest_rate': interest_rate,
# # # #             'existing_emi': existing_emi,
# # # #             'AMT_ANNUITY': amt_annuity,
# # # #             'application_id': app_id,
# # # #             'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
# # # #             'inward_bounce_count_3m': 0,
# # # #             'salary_missing_months': 0,
# # # #             'payment_discipline_flag': 'GOOD',
# # # #             'liquidity_flag': 'LOW',
# # # #             'cashflow_health': 'MODERATE',
# # # #             'bureau_risk_flag': 'LOW',
# # # #             'hard_reject_flag': 0,
# # # #             'total_dpd_count': dpd_90_6m + dpd_30_6m,
# # # #             'max_dpd_6m': 90 if dpd_90_6m > 0 else (30 if dpd_30_6m > 0 else 0),
# # # #             'salary_amount_cv': 0.1,
# # # #             'salary_creditor_consistent': 1.0,
# # # #             'salary_txn_count_6m': 6,
# # # #             'total_late_15_6m': 0, 'total_late_30_6m': dpd_30_6m, 'total_late_90_6m': dpd_90_6m,
# # # #             'recent_payment_stress': 1 if dpd_90_6m > 0 else 0,
# # # #             'total_emi_monthly': existing_emi,
# # # #         }

# # # #         with st.spinner("🔄 Processing Stage 1 assessment..."):
# # # #             decision_data = make_hybrid_decision_enhanced(customer_data)

# # # #         reasons = generate_reason_codes(
# # # #             decision=decision_data.get('decision', 'ERROR'),
# # # #             customer_data=customer_data,
# # # #             affordability_data=decision_data.get('affordability_data', {}),
# # # #             policy_checks=decision_data.get('policy_checks', {})
# # # #         )
# # # #         customer_data['reason_codes'] = reasons

# # # #         st.session_state.stage1_complete = True
# # # #         st.session_state.stage1_decision = decision_data.get('decision', 'ERROR')
# # # #         st.session_state.stage1_data = decision_data
# # # #         st.session_state.current_customer_data = customer_data

# # # #         for key in list(st.session_state.keys()):
# # # #             if key.startswith('pdf_') or key in ('_last_extraction',):
# # # #                 del st.session_state[key]

# # # #         tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])

# # # #         with tab1:
# # # #             st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
# # # #             col1, col2 = st.columns(2)
# # # #             with col1:
# # # #                 render_info_card("👤 Identity", "👤",
# # # #                                  {"Age": age, "Employment": employment_type, "Dependents": dependents,
# # # #                                   "KYC Status": "Verified" if kyc_verified else "Not Verified",
# # # #                                   "Tenure": f"{employment_tenure} months" if employment_type == 'Salaried' else f"{business_vintage} years"})
# # # #                 render_info_card("💰 Financial", "💰",
# # # #                                  {"Monthly Income": f"₹{avg_salary:,}", "Annual Income": f"₹{amt_income:,}",
# # # #                                   "Net Surplus": f"₹{net_surplus:,}", "Stability": salary_stability})
# # # #             with col2:
# # # #                 render_info_card("🏦 Credit Bureau", "🏦",
# # # #                                  {"Bureau Score": bureau_score, "DPD 90+": dpd_90_6m, "DPD 30+": dpd_30_6m,
# # # #                                   "Utilization": f"{credit_utilization}%", "Recent Inquiries": recent_inquiries,
# # # #                                   "Existing EMI": f"₹{existing_emi:,}"})
# # # #                 render_info_card("📋 Loan Request", "📋",
# # # #                                  {"Amount": f"₹{loan_amount:,}", "Tenure": f"{loan_tenure} months",
# # # #                                   "Interest Rate": f"{interest_rate}%", "Requested EMI": f"₹{amt_annuity:,}"})

# # # #         with tab2:
# # # #             st.markdown('<p class="section-header">Decision Summary</p>', unsafe_allow_html=True)
# # # #             render_decision_header(decision_data, customer_data)
# # # #             st.markdown("<br>", unsafe_allow_html=True)

# # # #             final_decision = decision_data.get('decision', 'ERROR')

# # # #             if final_decision in ['APPROVE', 'REVIEW']:
# # # #                 st.markdown("---")
# # # #                 st.markdown("""
# # # #                     <div class="info-box" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white; text-align: center;">
# # # #                         <h3 style="margin: 0; color: white;">✅ Eligible for Stage 2 Deep Dive</h3>
# # # #                         <p style="margin: 0.5rem 0 0 0;">Choose an input method to proceed:</p>
# # # #                     </div>
# # # #                 """, unsafe_allow_html=True)
# # # #                 col1, col2, col3 = st.columns(3)
# # # #                 with col1:
# # # #                     if st.button("📝 Manual Entry", key="stage2_manual_btn", use_container_width=True, type="primary"):
# # # #                         st.session_state.stage2_selected_tab = "Manual Entry"
# # # #                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
# # # #                         st.rerun()
# # # #                 with col2:
# # # #                     if st.button("📄 PDF Upload", key="stage2_pdf_btn", use_container_width=True, type="primary"):
# # # #                         st.session_state.stage2_selected_tab = "PDF Upload"
# # # #                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
# # # #                         st.rerun()
# # # #                 with col3:
# # # #                     if st.button("📊 Batch Analysis", key="stage2_batch_btn", use_container_width=True, type="primary"):
# # # #                         st.session_state.stage2_selected_tab = "Batch Analysis"
# # # #                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
# # # #                         st.rerun()
# # # #             elif final_decision == 'REJECT':
# # # #                 st.markdown("---")
# # # #                 st.markdown("""
# # # #                     <div class="warning-box" style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; text-align: center;">
# # # #                         <h3 style="margin: 0; color: white;">❌ Stage 2 Not Available</h3>
# # # #                         <p style="margin: 0.5rem 0 0 0;">Application rejected. Stage 2 only for APPROVE/REVIEW.</p>
# # # #                     </div>
# # # #                 """, unsafe_allow_html=True)

# # # #             st.markdown("<br>", unsafe_allow_html=True)
# # # #             affordability = decision_data.get('affordability_data', {})
# # # #             foir = affordability.get('foir_percentage', 0)
# # # #             total_emi = affordability.get('total_emi', 0)
# # # #             net_disp = affordability.get('net_disposable', 0)

# # # #             col1, col2, col3 = st.columns(3)
# # # #             with col1:
# # # #                 render_info_card("Identity & Eligibility", "👤",
# # # #                                  {f"Age: {age}": "", f"Employment: {employment_type}": "",
# # # #                                   f"Dependents: {dependents}": "",
# # # #                                   f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""},
# # # #                                  {f"Age: {age}": "pass" if 24 <= age <= 70 else "fail",
# # # #                                   f"Employment: {employment_type}": "pass",
# # # #                                   f"Dependents: {dependents}": "pass" if dependents <= 5 else "warning",
# # # #                                   f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_verified else "fail"})
# # # #             with col2:
# # # #                 bureau_pass = bureau_score >= 550
# # # #                 dpd_pass = dpd_90_6m == 0
# # # #                 render_info_card("Credit Bureau", "🏦",
# # # #                                  {f"Bureau Score: {bureau_score}": "", f"DPD 90+: {dpd_90_6m}": "",
# # # #                                   f"Utilization: {credit_utilization}%": ""},
# # # #                                  {f"Bureau Score: {bureau_score}": "pass" if bureau_pass else "fail",
# # # #                                   f"DPD 90+: {dpd_90_6m}": "pass" if dpd_pass else "fail",
# # # #                                   f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"})
# # # #             with col3:
# # # #                 render_info_card("Affordability", "💰",
# # # #                                  {f"Monthly Income: ₹{avg_salary:,}": "", f"FOIR: {foir:.1f}%": "",
# # # #                                   f"Total EMI: ₹{total_emi:,}": "", f"Net Disposable: ₹{net_disp:,}": ""},
# # # #                                  {f"Monthly Income: ₹{avg_salary:,}": "pass",
# # # #                                   f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
# # # #                                   f"Total EMI: ₹{total_emi:,}": "pass",
# # # #                                   f"Net Disposable: ₹{net_disp:,}": "pass" if net_disp >= 10000 else "warning"})

# # # #             st.markdown("<br>", unsafe_allow_html=True)
# # # #             render_reason_codes(reasons)
# # # #             st.markdown("<br>", unsafe_allow_html=True)

# # # #             col1, col2, col3 = st.columns([1, 1, 2])
# # # #             with col1:
# # # #                 if PDF_AVAILABLE and generate_decision_pdf is not None:
# # # #                     try:
# # # #                         pdf_buffer = generate_decision_pdf(
# # # #                             decision_data=decision_data, customer_data=customer_data,
# # # #                             affordability_data=decision_data.get('affordability_data', {}), reasons=reasons)
# # # #                         st.download_button("📥 Decision Report (PDF)", data=pdf_buffer,
# # # #                                            file_name=f"credit_decision_{app_id}.pdf", mime="application/pdf",
# # # #                                            use_container_width=True)
# # # #                     except Exception as e:
# # # #                         st.error(f"Error generating PDF: {str(e)}")
# # # #                 else:
# # # #                     st.warning("PDF generation not available.")
# # # #             with col2:
# # # #                 if st.button("🔄 Re-Evaluate", key="reevaluate_btn", use_container_width=True):
# # # #                     st.rerun()

# # # #         with tab3:
# # # #             st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
# # # #             col1, col2 = st.columns(2)
# # # #             with col1:
# # # #                 fig1 = create_modern_gauge(decision_data.get('confidence', 0), "Model Confidence")
# # # #                 st.plotly_chart(fig1, use_container_width=True)
# # # #             with col2:
# # # #                 final_decision_tab3 = decision_data.get('decision', 'ERROR')
# # # #                 if final_decision_tab3 == "REVIEW":
# # # #                     class_probs = {"APPROVE": 0, "REVIEW": 100, "REJECT": 0}
# # # #                 elif final_decision_tab3 == "REJECT":
# # # #                     class_probs = {"APPROVE": 0, "REVIEW": 0, "REJECT": 100}
# # # #                 else:
# # # #                     class_probs = decision_data.get('class_probs', {"APPROVE": 0, "REVIEW": 0, "REJECT": 0})
# # # #                 fig2 = create_modern_bar_chart(class_probs)
# # # #                 st.plotly_chart(fig2, use_container_width=True)

# # # #             st.markdown("<br>", unsafe_allow_html=True)
# # # #             st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
# # # #             policy_df = pd.DataFrame([{'Check': k, 'Result': v} for k, v in decision_data.get('policy_checks', {}).items()])
# # # #             st.dataframe(policy_df, width='stretch', hide_index=True)

# # # #             st.markdown('<p class="section-header">PD Calculation Breakdown</p>', unsafe_allow_html=True)
# # # #             pd_factors_display = {
# # # #                 'Bureau Score': f"{bureau_score} → Base PD: {bureau_score_to_pd(bureau_score):.1f}%",
# # # #                 'Delinquency': f"DPD 90+: {dpd_90_6m}, DPD 30+: {dpd_30_6m} → Multiplier: {delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x",
# # # #                 'FOIR Impact': f"{foir:.1f}% → Adjustment: {foir_to_pd_adjustment(foir):.1f}%",
# # # #                 'Employment Stability': f"{employment_type}, {employment_tenure if employment_type == 'Salaried' else business_vintage}{' months' if employment_type == 'Salaried' else ' years'} → Adjustment: {employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage):.1f}%",
# # # #                 'ML Confidence': f"{decision_data.get('confidence', 0):.1f}% → Adjustment: {ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')):.1f}%",
# # # #                 'Final PD': f"{decision_data.get('pd_percentage', 0)}%"
# # # #             }
# # # #             for factor, value in pd_factors_display.items():
# # # #                 st.markdown(f"**{factor}:** {value}")

# # # #         with tab4:
# # # #             st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
# # # #             audit_log_raw = {
# # # #                 'application_id': app_id,
# # # #                 'timestamp': timestamp.isoformat(),
# # # #                 'decision': decision_data.get('decision', 'ERROR'),
# # # #                 'risk_score': decision_data.get('risk_score', 0),
# # # #                 'pd_percentage': decision_data.get('pd_percentage', 0),
# # # #                 'confidence': round(decision_data.get('confidence', 0), 2),
# # # #                 'model_version': '8.3',
# # # #                 'reason_codes': reasons,
# # # #                 'policy_checks': decision_data.get('policy_checks', {}),
# # # #                 'affordability': decision_data.get('affordability_data', {}),
# # # #                 'customer_data': {k: v for k, v in customer_data.items() if k not in ['application_id', 'timestamp', 'reason_codes']},
# # # #                 'pd_calculation_factors': {
# # # #                     'bureau_score': bureau_score,
# # # #                     'base_pd': bureau_score_to_pd(bureau_score),
# # # #                     'dpd_90': dpd_90_6m, 'dpd_30': dpd_30_6m,
# # # #                     'delinquency_multiplier': delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m),
# # # #                     'foir': foir,
# # # #                     'foir_adjustment': foir_to_pd_adjustment(foir),
# # # #                     'employment_adjustment': employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage),
# # # #                     'ml_adjustment': ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')),
# # # #                     'final_pd': decision_data.get('pd_percentage', 0)
# # # #                 }
# # # #             }
# # # #             audit_log = sanitize_for_json(audit_log_raw)

# # # #             with st.expander("📋 View Audit Log (JSON)"):
# # # #                 st.json(audit_log)

# # # #             col1, col2 = st.columns(2)
# # # #             with col1:
# # # #                 if PDF_AVAILABLE and generate_audit_pdf is not None:
# # # #                     try:
# # # #                         audit_pdf_buffer = generate_audit_pdf(audit_log)
# # # #                         st.download_button("📥 Download Audit Trail (PDF)",
# # # #                                            data=audit_pdf_buffer,
# # # #                                            file_name=f"audit_trail_{app_id}.pdf",
# # # #                                            mime="application/pdf",
# # # #                                            use_container_width=True)
# # # #                     except Exception as e:
# # # #                         st.error(f"Error generating audit PDF: {str(e)}")
# # # #                 else:
# # # #                     st.warning("Audit PDF generation is not available.")
# # # #             with col2:
# # # #                 audit_json = json.dumps(audit_log, indent=2)
# # # #                 st.download_button("📥 Download Audit Log (JSON)",
# # # #                                    data=audit_json,
# # # #                                    file_name=f"audit_{app_id}.json",
# # # #                                    mime="application/json",
# # # #                                    use_container_width=True)

# # # #             st.markdown('<p class="section-header">PD Calculation Summary</p>', unsafe_allow_html=True)
# # # #             pd_table = pd.DataFrame([
# # # #                 {"Factor": "Bureau Score", "Value": f"{bureau_score}", "Impact": f"{bureau_score_to_pd(bureau_score):.1f}% base PD"},
# # # #                 {"Factor": "Delinquency (DPD 90+)", "Value": f"{dpd_90_6m} times", "Impact": f"{delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x multiplier"},
# # # #                 {"Factor": "FOIR", "Value": f"{foir:.1f}%", "Impact": f"{foir_to_pd_adjustment(foir):.1f}% adjustment"},
# # # #                 {"Factor": "Employment Stability",
# # # #                  "Value": f"{employment_type} ({employment_tenure if employment_type == 'Salaried' else business_vintage}{' months' if employment_type == 'Salaried' else ' years'})",
# # # #                  "Impact": f"{employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage):.1f}% adjustment"},
# # # #                 {"Factor": "ML Decision Confidence",
# # # #                  "Value": f"{decision_data.get('confidence', 0):.1f}% ({decision_data.get('decision', 'ERROR')})",
# # # #                  "Impact": f"{ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')):.1f}% adjustment"},
# # # #                 {"Factor": "Final PD", "Value": f"{decision_data.get('pd_percentage', 0)}%", "Impact": "Industry-standard calculation"}
# # # #             ])
# # # #             st.dataframe(pd_table, width='stretch', hide_index=True)

# # # # elif page == "🔬 Stage 2 Analysis":
# # # #     st.markdown('<p class="main-header">Stage 2: CIBIL Deep Dive</p>', unsafe_allow_html=True)

# # # #     if not st.session_state.get('stage1_complete', False):
# # # #         st.error("❌ You must complete Stage 1 Assessment first!")
# # # #         st.info("Please go to the 👤 Assessment page and submit an application.")
# # # #         if st.button("← Go to Assessment", use_container_width=True):
# # # #             st.session_state.page_navigation = "👤 Assessment"
# # # #             st.rerun()
# # # #         st.stop()

# # # #     if st.session_state.get('stage1_decision') not in ['APPROVE', 'REVIEW']:
# # # #         st.error("❌ Stage 2 is only available for APPROVED or REVIEW applications!")
# # # #         st.warning(f"Your Stage 1 decision: {st.session_state.get('stage1_decision', 'Unknown')}")
# # # #         if st.button("← Go Back", use_container_width=True):
# # # #             st.session_state.page_navigation = "👤 Assessment"
# # # #             st.rerun()
# # # #         st.stop()

# # # #     if not (STAGE2_AVAILABLE and is_stage2_available()):
# # # #         st.error("❌ Stage 2 model not available!")
# # # #         st.info("Please ensure `stage2_cibil_model.pkl` is in the project directory.")
# # # #         if st.button("← Go Back", use_container_width=True):
# # # #             st.session_state.page_navigation = "👤 Assessment"
# # # #             st.rerun()
# # # #         st.stop()

# # # #     stage1_data = st.session_state.get('stage1_data', {})
# # # #     stage1_customer = st.session_state.get('current_customer_data', {})

# # # #     st.markdown(f"""
# # # #         <div class="info-box" style="background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); color: white;">
# # # #             <h3 style="margin: 0; color: white;">📊 Stage 1 Results</h3>
# # # #             <p style="margin: 0.5rem 0 0 0;">
# # # #                 <strong>Decision:</strong> {st.session_state.get('stage1_decision', 'N/A')} |
# # # #                 <strong>Risk Score:</strong> {stage1_data.get('risk_score', 'N/A')} |
# # # #                 <strong>Application ID:</strong> {stage1_customer.get('application_id', 'N/A')}
# # # #             </p>
# # # #         </div>
# # # #     """, unsafe_allow_html=True)

# # # #     st.markdown("<br>", unsafe_allow_html=True)

# # # #     tab_options = ["Manual Entry", "PDF Upload", "Batch Analysis"]
# # # #     default_tab = st.session_state.get('stage2_selected_tab', 'Manual Entry')
# # # #     if default_tab not in tab_options:
# # # #         default_tab = "Manual Entry"
# # # #     selected_tab = st.radio(
# # # #         "Select input method",
# # # #         tab_options,
# # # #         index=tab_options.index(default_tab),
# # # #         horizontal=True,
# # # #         label_visibility="collapsed"
# # # #     )

# # # #     if selected_tab == "Manual Entry":
# # # #         st.markdown('<p class="section-header">Manual CIBIL Data Entry</p>', unsafe_allow_html=True)
# # # #         st.markdown("""
# # # #             <div class="info-box">
# # # #                 📝 <strong>Manual Data Entry</strong><br>
# # # #                 Enter CIBIL bureau data to enhance Stage 1 customer profile.<br>
# # # #                 The Stage 2 model will use this data to predict risk tier (P1/P2/P3/P4).
# # # #             </div>
# # # #         """, unsafe_allow_html=True)

# # # #         with st.form("stage2_manual_form"):
# # # #             st.markdown("### 📋 Application Reference")
# # # #             col1, col2 = st.columns(2)
# # # #             with col1:
# # # #                 st.text_input("Application ID", value=stage1_customer.get('application_id', 'N/A'), disabled=True)
# # # #                 st.text_input("Stage 1 Decision", value=st.session_state.get('stage1_decision', 'N/A'), disabled=True)
# # # #             with col2:
# # # #                 st.text_input("Customer Name (Optional)", "")
# # # #                 st.number_input("Stage 1 Risk Score", value=int(stage1_data.get('risk_score', 750)), disabled=True)

# # # #             st.markdown("---")
# # # #             st.markdown("### 🏦 CIBIL Bureau Data")

# # # #             col1, col2, col3 = st.columns(3)
# # # #             with col1:
# # # #                 st.markdown("**Credit Score & History**")
# # # #                 cibil_score = st.number_input("Credit Score", 300, 900, 720, 10)
# # # #                 max_delinquency = st.number_input("Max Delinquency Level", 0, 100, 0)
# # # #                 num_times_30dpd = st.number_input("Times 30+ DPD", 0, 50, 0)
# # # #                 num_times_60dpd = st.number_input("Times 60+ DPD", 0, 50, 0)
# # # #                 num_times_delinquent = st.number_input("Total Delinquent", 0, 50, 0)
# # # #             with col2:
# # # #                 st.markdown("**Recent Behavior (6-12M)**")
# # # #                 num_deliq_6m = st.number_input("Delinquencies (6M)", 0, 20, 0)
# # # #                 num_deliq_12m = st.number_input("Delinquencies (12M)", 0, 20, 0)
# # # #                 max_deliq_6m = st.number_input("Max Delinq (6M)", 0, 100, 0)
# # # #                 max_deliq_12m = st.number_input("Max Delinq (12M)", 0, 100, 0)
# # # #                 enq_L3m = st.number_input("Inquiries (3M)", 0, 20, 2)
# # # #                 enq_L6m = st.number_input("Inquiries (6M)", 0, 30, 4)
# # # #                 enq_L12m = st.number_input("Inquiries (12M)", 0, 50, 6)
# # # #             with col3:
# # # #                 st.markdown("**Account Quality**")
# # # #                 num_std = st.number_input("Standard Accounts", 0, 50, 3)
# # # #                 num_std_6m = st.number_input("Standard (6M)", 0, 50, 3)
# # # #                 num_std_12m = st.number_input("Standard (12M)", 0, 50, 3)
# # # #                 num_sub = st.number_input("Sub-standard", 0, 20, 0)
# # # #                 num_sub_6m = st.number_input("Sub-standard (6M)", 0, 20, 0)
# # # #                 num_dbt = st.number_input("Doubtful", 0, 10, 0)
# # # #                 num_lss = st.number_input("Loss", 0, 10, 0)

# # # #             col1, col2, col3 = st.columns(3)
# # # #             with col1:
# # # #                 st.markdown("**Utilization**")
# # # #                 pct_active_tls = st.number_input("% Active TLs", 0.0, 1.0, 0.60, 0.01)
# # # #                 pct_current_bal = st.number_input("Current Balance %", 0.0, 1.0, 0.30, 0.01)
# # # #                 cc_utilization = st.number_input("CC Utilization", 0.0, 1.0, 0.35, 0.01)
# # # #                 pl_utilization = st.number_input("PL Utilization", 0.0, 1.0, 0.25, 0.01)
# # # #                 max_unsec_exposure = st.number_input("Max Unsec Exposure %", 0, 100, 30)
# # # #             with col2:
# # # #                 st.markdown("**Demographics**")
# # # #                 age_cibil = st.number_input("Age", 24, 70, int(stage1_customer.get('age', 35)))
# # # #                 net_monthly_income = st.number_input("Net Monthly Income", 0, 1000000,
# # # #                                                       int(stage1_customer.get('avg_salary_6m', 50000)), 5000)
# # # #                 time_curr_employer = st.number_input("Employment Tenure (months)", 0, 600,
# # # #                                                       int(stage1_customer.get('employment_tenure_months', 24)))
# # # #             with col3:
# # # #                 st.markdown("**Product Flags**")
# # # #                 cc_flag = st.selectbox("Credit Card", ["Yes", "No"]) == "Yes"
# # # #                 pl_flag = st.selectbox("Personal Loan", ["Yes", "No"]) == "No"
# # # #                 hl_flag = st.selectbox("Home Loan", ["Yes", "No"]) == "No"
# # # #                 gl_flag = st.selectbox("Gold Loan", ["Yes", "No"]) == "No"

# # # #             st.markdown("<br>", unsafe_allow_html=True)
# # # #             submitted_s2 = st.form_submit_button("🔬 Run Stage 2 Analysis", use_container_width=True, type="primary")

# # # #         if submitted_s2:
# # # #             with st.spinner("🔬 Running Stage 2 CIBIL Deep Analysis..."):
# # # #                 enhanced_customer_data = stage1_customer.copy()
# # # #                 _s1_inc = stage1_customer.get('avg_salary_6m', 50000)
# # # #                 _s2_inc = net_monthly_income or 0
# # # #                 _final_income = _s1_inc if (_s2_inc > 0 and _s2_inc < _s1_inc * 0.4) else (_s2_inc or _s1_inc)
# # # #                 if _s2_inc > 0 and _s2_inc < _s1_inc * 0.4:
# # # #                     st.warning(f'⚠️ CIBIL income ₹{_s2_inc:,} is much lower than application income ₹{_s1_inc:,}. Using application income for FOIR.')
# # # #                 enhanced_customer_data.update({
# # # #                     'bureau_score': cibil_score,
# # # #                     'age': age_cibil,
# # # #                     'avg_salary_6m': _final_income,
# # # #                     'employment_tenure_months': time_curr_employer,
# # # #                     'dpd_30_count_6m': num_times_30dpd,
# # # #                     'dpd_90_count_6m': num_times_60dpd,
# # # #                     'max_delinquency_level': max_delinquency,
# # # #                     'num_times_delinquent': num_times_delinquent,
# # # #                     'num_deliq_6mts': num_deliq_6m,
# # # #                     'num_deliq_12mts': num_deliq_12m,
# # # #                     'max_deliq_6mts': max_deliq_6m,
# # # #                     'max_deliq_12mts': max_deliq_12m,
# # # #                     'recent_inquiries_3m': enq_L3m,
# # # #                     'enq_L6m': enq_L6m,
# # # #                     'enq_L12m': enq_L12m,
# # # #                     'active_loans_count': num_std,
# # # #                     'num_std_6mts': num_std_6m,
# # # #                     'num_std_12mts': num_std_12m,
# # # #                     'num_sub': num_sub,
# # # #                     'num_sub_6mts': num_sub_6m,
# # # #                     'num_dbt': num_dbt,
# # # #                     'num_lss': num_lss,
# # # #                     'credit_utilization_pct': cc_utilization * 100,
# # # #                     'pct_of_active_TLs_ever': pct_active_tls,
# # # #                     'pct_currentBal_all_TL': pct_current_bal,
# # # #                     'CC_utilization': cc_utilization,
# # # #                     'PL_utilization': pl_utilization,
# # # #                     'max_unsec_exposure_inPct': max_unsec_exposure,
# # # #                     'CC_Flag': 1 if cc_flag else 0,
# # # #                     'PL_Flag': 1 if pl_flag else 0,
# # # #                     'HL_Flag': 1 if hl_flag else 0,
# # # #                     'GL_Flag': 1 if gl_flag else 0,
# # # #                 })
# # # #                 try:
# # # #                     stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
# # # #                     display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
# # # #                 except Exception as e:
# # # #                     st.error(f"❌ Stage 2 analysis failed: {str(e)}")
# # # #                     st.exception(e)

# # # #     elif selected_tab == "PDF Upload":
# # # #         st.markdown('<p class="section-header">📄 CIBIL PDF Upload</p>', unsafe_allow_html=True)
# # # #         if not OCR_AVAILABLE:
# # # #             st.error("❌ OCR not available. " + (OCR_ERROR_MSG or "Check packages.txt and requirements.txt."))
# # # #             st.warning("For now, please use the **Manual Entry** tab.")
# # # #         else:
# # # #             st.markdown("""
# # # #                 <div class="info-box">
# # # #                     📄 <strong>CIBIL PDF Extraction</strong><br>
# # # #                     Upload a CIBIL bureau report PDF for automatic extraction and analysis.
# # # #                 </div>
# # # #             """, unsafe_allow_html=True)
# # # #             uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="stage2_pdf")
# # # #             if uploaded_pdf is not None:
# # # #                 st.success(f"✅ File uploaded: {uploaded_pdf.name} ({uploaded_pdf.size / 1024:.1f} KB)")
# # # #                 if st.button("🔬 Extract & Analyze", key="extract_analyze_stage2", type="primary", use_container_width=True):
# # # #                     with st.spinner("🔄 Extracting data from PDF..."):
# # # #                         extraction_result = extract_cibil_from_pdf(uploaded_pdf)

# # # #                     if extraction_result.get('success', False):
# # # #                         st.success("✅ PDF extraction successful!")

# # # #                         # --- Display key metrics (summary) ---
# # # #                         st.markdown("### 📋 Extracted CIBIL Data (Summary)")
# # # #                         col1, col2, col3 = st.columns(3)
# # # #                         with col1:
# # # #                             st.metric("Credit Score", extraction_result.get('Credit_Score', 'N/A'))
# # # #                             st.metric("Max Delinquency Level", extraction_result.get('max_delinquency_level', 0))
# # # #                         with col2:
# # # #                             st.metric("Times 30+ DPD", extraction_result.get('num_times_30p_dpd', 0))
# # # #                             st.metric("Times 60+ DPD", extraction_result.get('num_times_60p_dpd', 0))
# # # #                         with col3:
# # # #                             st.metric("Total Delinquent", extraction_result.get('num_times_delinquent', 0))

# # # #                         # --- Show all extracted fields with names and IDs ---
# # # #                         with st.expander("🔍 View All Extracted Features (with internal IDs)"):
# # # #                             # Define a mapping for user-friendly names
# # # #                             friendly_names = {
# # # #                                 'Credit_Score': 'Credit Score',
# # # #                                 'AGE': 'Age',
# # # #                                 'max_delinquency_level': 'Max Delinquency Level',
# # # #                                 'num_times_30p_dpd': 'Times 30+ DPD',
# # # #                                 'num_times_60p_dpd': 'Times 60+ DPD',
# # # #                                 'num_times_delinquent': 'Total Delinquent',
# # # #                                 'dpd_90_count_6m': 'DPD 90+ (Last 6M)',
# # # #                                 'num_deliq_6mts': 'Delinquent Count (6M)',
# # # #                                 'num_deliq_12mts': 'Delinquent Count (12M)',
# # # #                                 'max_deliq_6mts': 'Max Delinquency (6M)',
# # # #                                 'max_deliq_12mts': 'Max Delinquency (12M)',
# # # #                                 'enq_L3m': 'Recent Inquiries (3M)',
# # # #                                 'enq_L6m': 'Inquiries (6M)',
# # # #                                 'enq_L12m': 'Inquiries (12M)',
# # # #                                 'num_std': 'Active Loans',
# # # #                                 'num_std_6mts': 'Standard Accounts (6M)',
# # # #                                 'num_std_12mts': 'Standard Accounts (12M)',
# # # #                                 'num_sub': 'Substandard Accounts',
# # # #                                 'num_sub_6mts': 'Substandard (6M)',
# # # #                                 'num_dbt': 'Doubtful Accounts',
# # # #                                 'num_lss': 'Loss Accounts',
# # # #                                 'CC_utilization': 'Credit Card Utilization',
# # # #                                 'PL_utilization': 'Personal Loan Utilization',
# # # #                                 'CC_Flag': 'Has Credit Card',
# # # #                                 'PL_Flag': 'Has Personal Loan',
# # # #                                 'HL_Flag': 'Has Home Loan',
# # # #                                 'GL_Flag': 'Has Gold Loan',
# # # #                                 'written_off_count': 'Written Off Count',
# # # #                                 'settled_count': 'Settled Count',
# # # #                                 'high_util_flag': 'High Utilization Flag',
# # # #                                 'recent_deliq_flag': 'Recent Delinquency Flag',
# # # #                                 'account_quality_score': 'Account Quality Score',
# # # #                                 'Time_With_Curr_Empr': 'Employment Tenure (months)',
# # # #                                 'NETMONTHLYINCOME': 'Net Monthly Income',
# # # #                                 'pct_of_active_TLs_ever': '% Active TLs Ever',
# # # #                                 'pct_currentBal_all_TL': '% Current Balance / All TL',
# # # #                                 'max_unsec_exposure_inPct': 'Max Unsecured Exposure %',
# # # #                             }

# # # #                             # Collect all items from extraction_result, exclude non‑data keys
# # # #                             exclude_keys = {'success', 'error', 'raw_text', 'extraction_method'}
# # # #                             data_items = []
# # # #                             for key, value in extraction_result.items():
# # # #                                 if key in exclude_keys:
# # # #                                     continue
# # # #                                 display_name = friendly_names.get(key, key.replace('_', ' ').title())
# # # #                                 data_items.append({
# # # #                                     "Feature Name": display_name,
# # # #                                     "Internal ID": key,
# # # #                                     "Value": value
# # # #                                 })

# # # #                             # Sort by feature name
# # # #                             data_items.sort(key=lambda x: x["Feature Name"])
# # # #                             df_all = pd.DataFrame(data_items)

# # # #                             # Display as a dataframe
# # # #                             st.dataframe(
# # # #                                 df_all,
# # # #                                 column_config={
# # # #                                     "Feature Name": "Feature Name",
# # # #                                     "Internal ID": "Internal ID",
# # # #                                     "Value": "Extracted Value"
# # # #                                 },
# # # #                                 hide_index=True,
# # # #                                 width='stretch'
# # # #                             )

# # # #                         # --- Continue with enhanced data and analysis ---
# # # #                         enhanced_customer_data = stage1_customer.copy()
# # # #                         _s1_income = stage1_customer.get('avg_salary_6m', 50000)
# # # #                         _s2_income = extraction_result.get('NETMONTHLYINCOME', 0)
# # # #                         _use_income = _s1_income if (_s2_income > 0 and _s2_income < _s1_income * 0.4) else (_s2_income or _s1_income)
# # # #                         if _s2_income > 0 and _s2_income < _s1_income * 0.4:
# # # #                             st.warning(f'⚠️ CIBIL income ₹{_s2_income:,} is much lower than application income ₹{_s1_income:,}. Using application income for FOIR.')

# # # #                         enhanced_customer_data.update({
# # # #                             'bureau_score': extraction_result.get('Credit_Score', 720),
# # # #                             'age': extraction_result.get('AGE', stage1_customer.get('age', 35)),
# # # #                             'avg_salary_6m': _use_income,
# # # #                             'employment_tenure_months': extraction_result.get('Time_With_Curr_Empr', stage1_customer.get('employment_tenure_months', 24)),
# # # #                             'dpd_30_count_6m': extraction_result.get('num_times_30p_dpd', 0),
# # # #                             'dpd_90_count_6m': extraction_result.get('dpd_90_count_6m', 0),
# # # #                             'max_delinquency_level': extraction_result.get('max_delinquency_level', 0),
# # # #                             'num_times_delinquent': extraction_result.get('num_times_delinquent', 0),
# # # #                             'num_deliq_6mts': extraction_result.get('num_deliq_6mts', 0),
# # # #                             'num_deliq_12mts': extraction_result.get('num_deliq_12mts', 0),
# # # #                             'max_deliq_6mts': extraction_result.get('max_deliq_6mts', 0),
# # # #                             'max_deliq_12mts': extraction_result.get('max_deliq_12mts', 0),
# # # #                             'recent_inquiries_3m': extraction_result.get('enq_L3m', 2),
# # # #                             'enq_L6m': extraction_result.get('enq_L6m', 4),
# # # #                             'enq_L12m': extraction_result.get('enq_L12m', 6),
# # # #                             'active_loans_count': extraction_result.get('num_std', 1),
# # # #                             'num_std_6mts': extraction_result.get('num_std_6mts', 0),
# # # #                             'num_std_12mts': extraction_result.get('num_std_12mts', 0),
# # # #                             'num_sub': extraction_result.get('num_sub', 0),
# # # #                             'num_sub_6mts': extraction_result.get('num_sub_6mts', 0),
# # # #                             'num_dbt': extraction_result.get('num_dbt', 0),
# # # #                             'num_lss': extraction_result.get('num_lss', 0),
# # # #                             'credit_utilization_pct': (0 if extraction_result.get('CC_utilization', 0) < 0 else extraction_result.get('CC_utilization', 0.35)) * 100,
# # # #                             'pct_of_active_TLs_ever': extraction_result.get('pct_of_active_TLs_ever', 0.6),
# # # #                             'pct_currentBal_all_TL': extraction_result.get('pct_currentBal_all_TL', 0.3),
# # # #                             'CC_utilization': 0 if extraction_result.get('CC_utilization', 0) < 0 else extraction_result.get('CC_utilization', 0.35),
# # # #                             'PL_utilization': 0 if extraction_result.get('PL_utilization', 0) < 0 else extraction_result.get('PL_utilization', 0.25),
# # # #                             'max_unsec_exposure_inPct': extraction_result.get('max_unsec_exposure_inPct', 30),
# # # #                             'CC_Flag': extraction_result.get('CC_Flag', 0),
# # # #                             'PL_Flag': extraction_result.get('PL_Flag', 0),
# # # #                             'HL_Flag': extraction_result.get('HL_Flag', 0),
# # # #                             'GL_Flag': extraction_result.get('GL_Flag', 0),
# # # #                             'written_off_count': extraction_result.get('written_off_count', 0),
# # # #                             'settled_count': extraction_result.get('settled_count', 0),
# # # #                             'high_util_flag': extraction_result.get('high_util_flag', 0),
# # # #                             'recent_deliq_flag': extraction_result.get('recent_deliq_flag', 0),
# # # #                             'account_quality_score': extraction_result.get('account_quality_score', 0)
# # # #                         })

# # # #                         with st.spinner("🔬 Running Stage 2 analysis..."):
# # # #                             try:
# # # #                                 stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
# # # #                                 display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
# # # #                             except Exception as e:
# # # #                                 st.error(f"❌ Analysis failed: {str(e)}")
# # # #                     else:
# # # #                         st.error("❌ PDF extraction failed! Error: " + extraction_result.get('error', 'Unknown'))

# # # #     elif selected_tab == "Batch Analysis":
# # # #         st.markdown('<p class="section-header">📊 Batch CIBIL Analysis</p>', unsafe_allow_html=True)
# # # #         st.info("📊 Batch analysis feature coming soon!")

# # # # elif page == "📊 Batch Process":
# # # #     st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
# # # #     st.markdown("""
# # # #         <div class="info-box">
# # # #             📤 Upload a CSV file with customer data for bulk credit assessment.
# # # #             The file should include all required fields for prediction.
# # # #         </div>
# # # #     """, unsafe_allow_html=True)
# # # #     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# # # #     if uploaded_file is not None:
# # # #         try:
# # # #             df = pd.read_csv(uploaded_file)
# # # #             st.success(f"✅ Successfully loaded {len(df)} records")
# # # #             with st.expander("📄 Preview Uploaded Data"):
# # # #                 st.dataframe(df.head(), width='stretch')
# # # #                 st.write(f"**Total Records:** {len(df)}")
# # # #                 st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
# # # #             required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
# # # #             missing_cols = [col for col in required_cols if col not in df.columns]
# # # #             if missing_cols:
# # # #                 st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
# # # #                 st.info("Please ensure your CSV includes at least these columns: age, employment_type, avg_salary_6m, bureau_score, loan_amount")
# # # #             else:
# # # #                 if st.button("🚀 Process Batch Predictions", key="process_batch_btn", type="primary", use_container_width=True):
# # # #                     with st.spinner(f"🔍 Processing {len(df)} records..."):
# # # #                         progress_bar = st.progress(0)
# # # #                         results_df = process_batch_predictions(df)
# # # #                         progress_bar.progress(100)
# # # #                         st.success(f"✅ Completed processing {len(results_df)} records!")
# # # #                         tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
# # # #                         with tab1:
# # # #                             st.dataframe(results_df, width='stretch')
# # # #                             col1, col2, col3, col4 = st.columns(4)
# # # #                             with col1:
# # # #                                 st.metric("✅ Approved", len(results_df[results_df['decision'] == 'APPROVE']))
# # # #                             with col2:
# # # #                                 st.metric("❌ Rejected", len(results_df[results_df['decision'] == 'REJECT']))
# # # #                             with col3:
# # # #                                 st.metric("⚠️ Review", len(results_df[results_df['decision'] == 'REVIEW']))
# # # #                             with col4:
# # # #                                 st.metric("📊 Avg Risk Score", f"{results_df['risk_score'].mean():.0f}")
# # # #                         with tab2:
# # # #                             col1, col2 = st.columns(2)
# # # #                             with col1:
# # # #                                 decision_counts = results_df['decision'].value_counts()
# # # #                                 fig1 = px.pie(values=decision_counts.values, names=decision_counts.index,
# # # #                                               title="Decision Distribution", color=decision_counts.index,
# # # #                                               color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
# # # #                                 st.plotly_chart(fig1, use_container_width=True)
# # # #                             with col2:
# # # #                                 fig2 = px.histogram(results_df, x='risk_score', title="Risk Score Distribution",
# # # #                                                     nbins=20, color_discrete_sequence=['#587042'])
# # # #                                 st.plotly_chart(fig2, use_container_width=True)
# # # #                             fig3 = px.scatter(results_df, x='monthly_income', y='loan_amount', color='decision',
# # # #                                               size='risk_score', title="Income vs Loan Amount (Colored by Decision)",
# # # #                                               hover_data=['application_id', 'foir_percentage'],
# # # #                                               color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
# # # #                             st.plotly_chart(fig3, use_container_width=True)
# # # #                             fig4 = px.box(results_df, x='decision', y='pd_percentage',
# # # #                                           title="PD Distribution by Decision", color='decision',
# # # #                                           color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
# # # #                             st.plotly_chart(fig4, use_container_width=True)
# # # #                         with tab3:
# # # #                             st.markdown("### Download Results")
# # # #                             col1, col2 = st.columns(2)
# # # #                             with col1:
# # # #                                 st.download_button(
# # # #                                     "📥 Download as CSV",
# # # #                                     data=results_df.to_csv(index=False),
# # # #                                     file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# # # #                                     mime="text/csv",
# # # #                                     use_container_width=True
# # # #                                 )
# # # #                             with col2:
# # # #                                 st.download_button(
# # # #                                     "📥 Download as JSON",
# # # #                                     data=results_df.to_json(orient='records', indent=2),
# # # #                                     file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
# # # #                                     mime="application/json",
# # # #                                     use_container_width=True
# # # #                                 )
# # # #                             st.markdown("---")
# # # #                             st.markdown("#### Filtered Downloads")
# # # #                             col1, col2, col3 = st.columns(3)
# # # #                             with col1:
# # # #                                 approved_df = results_df[results_df['decision'] == 'APPROVE']
# # # #                                 if len(approved_df) > 0:
# # # #                                     st.download_button(
# # # #                                         f"✅ Approved Only ({len(approved_df)})",
# # # #                                         data=approved_df.to_csv(index=False),
# # # #                                         file_name=f"approved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# # # #                                         mime="text/csv",
# # # #                                         use_container_width=True
# # # #                                     )
# # # #                             with col2:
# # # #                                 rejected_df = results_df[results_df['decision'] == 'REJECT']
# # # #                                 if len(rejected_df) > 0:
# # # #                                     st.download_button(
# # # #                                         f"❌ Rejected Only ({len(rejected_df)})",
# # # #                                         data=rejected_df.to_csv(index=False),
# # # #                                         file_name=f"rejected_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# # # #                                         mime="text/csv",
# # # #                                         use_container_width=True
# # # #                                     )
# # # #                             with col3:
# # # #                                 review_df = results_df[results_df['decision'] == 'REVIEW']
# # # #                                 if len(review_df) > 0:
# # # #                                     st.download_button(
# # # #                                         f"⚠️ Review Only ({len(review_df)})",
# # # #                                         data=review_df.to_csv(index=False),
# # # #                                         file_name=f"review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# # # #                                         mime="text/csv",
# # # #                                         use_container_width=True
# # # #                                     )
# # # #         except Exception as e:
# # # #             st.error(f"❌ Error processing file: {str(e)}")
# # # #             st.info("Please ensure the CSV file is properly formatted and contains the required columns.")
# # # #     else:
# # # #         st.markdown("---")
# # # #         st.markdown("### 📋 CSV Template")
# # # #         template_data = {
# # # #             'age': [35, 42, 28],
# # # #             'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
# # # #             'dependents': [2, 3, 6],
# # # #             'kyc_verified': ['Yes', 'Yes', 'No'],
# # # #             'bankruptcy_flag': ['No', 'No', 'No'],
# # # #             'fraud_flag': ['No', 'No', 'No'],
# # # #             'employment_tenure_months': [24, 0, 18],
# # # #             'business_vintage_years': [0, 5, 0],
# # # #             'bureau_score': [720, 680, 580],
# # # #             'dpd_90_count_6m': [0, 1, 2],
# # # #             'dpd_30_count_6m': [0, 2, 1],
# # # #             'credit_utilization_pct': [30, 45, 75],
# # # #             'recent_inquiries_3m': [2, 1, 5],
# # # #             'active_loans_count': [1, 2, 3],
# # # #             'avg_salary_6m': [50000, 75000, 35000],
# # # #             'AMT_INCOME_TOTAL': [600000, 900000, 420000],
# # # #             'net_cash_surplus_6m': [20000, 35000, 10000],
# # # #             'salary_stability_flag': ['STABLE', 'MODERATE', 'UNSTABLE'],
# # # #             'loan_amount': [180000, 250000, 100000],
# # # #             'loan_tenure_months': [24, 36, 12],
# # # #             'interest_rate': [10.5, 11.0, 12.0],
# # # #             'existing_emi': [15000, 20000, 8000],
# # # #             'AMT_ANNUITY': [8500, 9500, 4500]
# # # #         }
# # # #         template_df = pd.DataFrame(template_data)
# # # #         st.dataframe(template_df, width='stretch')
# # # #         st.download_button(
# # # #             "📥 Download CSV Template",
# # # #             data=template_df.to_csv(index=False),
# # # #             file_name="credit_assessment_template.csv",
# # # #             mime="text/csv",
# # # #             use_container_width=True
# # # #         )

# # # # elif page == "📈 Model Info":
# # # #     st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
# # # #     col1, col2, col3 = st.columns(3)
# # # #     with col1:
# # # #         st.markdown('<div class="stat-card"><div class="stat-number">RF</div><div class="stat-label">Model Type</div></div>', unsafe_allow_html=True)
# # # #     with col2:
# # # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TOP_FEATURES)}</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)
# # # #     with col3:
# # # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TARGET_LE.classes_)}</div><div class="stat-label">Classes</div></div>', unsafe_allow_html=True)
# # # #     st.markdown("<br>", unsafe_allow_html=True)
# # # #     st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
# # # #     feature_df = pd.DataFrame({'Rank': range(1, min(21, len(TOP_FEATURES) + 1)), 'Feature': TOP_FEATURES[:20]})
# # # #     st.dataframe(feature_df, width='stretch', hide_index=True)

# # # # elif page == "ℹ️ About":
# # # #     st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
# # # #     st.markdown("""
# # # #         <div class="info-card">
# # # #             <div class="info-card-title"><span class="icon">🏦</span><span>Credit Risk Assessment Platform</span></div>
# # # #             <div class="info-card-content">
# # # #                 <p><strong>Version:</strong> 8.3 - FIXED NUMERIC TYPES & SUBMIT BUTTON</p>
# # # #                 <p><strong>Developer:</strong> Zen Meraki</p>
# # # #                 <p><strong>Date:</strong> January 2026</p>
# # # #                 <br>
# # # #                 <p>A comprehensive credit risk evaluation system combining hard policy rules,
# # # #                 machine learning models, and affordability analysis for accurate and compliant lending decisions.</p>
# # # #             </div>
# # # #         </div>
# # # #     """, unsafe_allow_html=True)
# # # #     st.markdown("<br>", unsafe_allow_html=True)
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         st.markdown("""
# # # #             <div class="info-card">
# # # #                 <div class="info-card-title"><span class="icon">🎯</span><span>Key Features</span></div>
# # # #                 <div class="info-card-content">
# # # #                     <ul style="margin: 0; padding-left: 1.25rem;">
# # # #                         <li>Three-layer decision engine</li>
# # # #                         <li>Real-time risk assessment</li>
# # # #                         <li>Industry-standard PD calculation</li>
# # # #                         <li>FOIR calculation & validation</li>
# # # #                         <li>Automated reason generation</li>
# # # #                         <li>Complete audit trail (PDF)</li>
# # # #                         <li>Professional UI/UX</li>
# # # #                     </ul>
# # # #                 </div>
# # # #             </div>
# # # #         """, unsafe_allow_html=True)
# # # #     with col2:
# # # #         st.markdown("""
# # # #             <div class="info-card">
# # # #                 <div class="info-card-title"><span class="icon">🛠️</span><span>Technology Stack</span></div>
# # # #                 <div class="info-card-content">
# # # #                     <ul style="margin: 0; padding-left: 1.25rem;">
# # # #                         <li>Streamlit (UI Framework)</li>
# # # #                         <li>Scikit-learn (ML)</li>
# # # #                         <li>Plotly (Visualizations)</li>
# # # #                         <li>Pandas (Data Processing)</li>
# # # #                         <li>ReportLab (PDF Generation)</li>
# # # #                         <li>Python 3.8+</li>
# # # #                     </ul>
# # # #                 </div>
# # # #             </div>
# # # #         """, unsafe_allow_html=True) 






# # # """
# # # Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
# # # Enhanced with Modern UI/UX Design
# # # Run with: streamlit run test.py (from inside the notebooks folder)
# # # Author: Zen Meraki
# # # Date: January 2026
# # # VERSION: 8.3 - FULLY CORRECTED (all fixes applied)
# # # """

# # # import streamlit as st

# # # # =============================================================================
# # # # PAGE CONFIGURATION – MUST BE THE VERY FIRST STREAMLIT COMMAND
# # # # =============================================================================
# # # st.set_page_config(
# # #     page_title="Credit Risk Assessment",
# # #     page_icon="💳",
# # #     layout="wide",
# # #     initial_sidebar_state="expanded"
# # # )

# # # # =============================================================================
# # # # STANDARD LIBRARY / THIRD-PARTY IMPORTS
# # # # =============================================================================
# # # import pandas as pd
# # # import numpy as np
# # # import plotly.graph_objects as go
# # # import plotly.express as px
# # # import joblib
# # # import warnings
# # # from datetime import datetime
# # # import hashlib
# # # import io
# # # import base64
# # # from typing import Dict, List, Any, Union
# # # import json
# # # import sys
# # # import os
# # # from pathlib import Path
# # # import re

# # # # =============================================================================
# # # # SUPPRESS SCIKIT-LEARN VERSION WARNINGS (optional, but keeps logs clean)
# # # # =============================================================================
# # # warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# # # # =============================================================================
# # # # DYNAMIC PATH RESOLUTION – MAKE ALL PROJECT MODULES IMPORTABLE
# # # # =============================================================================
# # # CURRENT_DIR = Path(__file__).resolve().parent          # notebooks/
# # # PROJECT_ROOT = CURRENT_DIR.parent                      # credit_risk_engine/
# # # POSSIBLE_LOCATIONS = [
# # #     CURRENT_DIR,                           # notebooks/
# # #     PROJECT_ROOT,                           # credit_risk_engine/
# # #     PROJECT_ROOT / "loan",                   # credit_risk_engine/loan/
# # #     PROJECT_ROOT / "utils",                   # credit_risk_engine/utils/
# # #     PROJECT_ROOT / "notebooks",               # credit_risk_engine/notebooks/
# # # ]

# # # for loc in POSSIBLE_LOCATIONS:
# # #     if loc.exists() and str(loc) not in sys.path:
# # #         sys.path.insert(0, str(loc))

# # # # =============================================================================
# # # # OPTIONAL OCR DEPENDENCIES – GRACEFUL FALLBACK
# # # # Requires system packages (packages.txt):   tesseract-ocr  poppler-utils
# # # # Requires Python packages (requirements.txt): pytesseract pdf2image opencv-python-headless pillow
# # # # =============================================================================
# # # OCR_AVAILABLE = False
# # # OCR_ERROR_MSG = ""
# # # try:
# # #     import pytesseract
# # #     from pdf2image import convert_from_bytes
# # #     import cv2
# # #     from PIL import Image

# # #     # Auto-detect Tesseract binary (Streamlit Cloud / Linux / Mac / Windows)
# # #     import shutil as _shutil
# # #     _tess_cmd = (
# # #         _shutil.which("tesseract")
# # #         or r"C:\Program Files\Tesseract-OCR\tesseract.exe"   # Windows fallback
# # #     )
# # #     if _tess_cmd:
# # #         pytesseract.pytesseract.tesseract_cmd = _tess_cmd

# # #     # Verify tesseract binary is actually callable
# # #     pytesseract.get_tesseract_version()
# # #     OCR_AVAILABLE = True

# # # except ImportError as _e:
# # #     OCR_ERROR_MSG = (
# # #         f"Missing Python package: {_e}. "
# # #         "Add to requirements.txt: pytesseract  pdf2image  opencv-python-headless  pillow"
# # #     )
# # # except Exception as _e:
# # #     _name = type(_e).__name__
# # #     if "TesseractNotFound" in _name or "tesseract" in str(_e).lower():
# # #         OCR_ERROR_MSG = (
# # #             "Tesseract binary not found. "
# # #             "Streamlit Cloud → add 'tesseract-ocr' and 'poppler-utils' to packages.txt. "
# # #             "Linux → sudo apt install tesseract-ocr poppler-utils. "
# # #             "Mac → brew install tesseract poppler."
# # #         )
# # #     else:
# # #         OCR_ERROR_MSG = f"OCR init error ({_name}): {_e}"

# # # # =============================================================================
# # # # IMPORT CSS – WITH FALLBACK
# # # # =============================================================================
# # # try:
# # #     from css_styles import CSS
# # # except ImportError:
# # #     CSS = """
# # #     <style>
# # #         .main-header { font-size: 2rem; font-weight: bold; color: #2d3748; }
# # #         .section-header { font-size: 1.5rem; font-weight: 600; color: #2d3748; }
# # #         .info-box { background: #f7fafc; padding: 1rem; border-radius: 0.5rem; }
# # #         .decision-card { padding: 2rem; border-radius: 1rem; text-align: center; margin-bottom: 1rem; }
# # #         .decision-card-approved { background: #c6f6d5; border-left: 5px solid #48bb78; }
# # #         .decision-card-rejected { background: #fed7d7; border-left: 5px solid #f56565; }
# # #         .decision-card-review { background: #feebc8; border-left: 5px solid #ed8936; }
# # #         .decision-title { font-size: 2.5rem; font-weight: bold; }
# # #         .decision-subtitle { font-size: 1rem; opacity: 0.8; }
# # #         .stat-card { background: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
# # #         .stat-number { font-size: 1.8rem; font-weight: bold; color: #2d3748; }
# # #         .stat-label { font-size: 0.875rem; color: #718096; }
# # #         .info-card { background: white; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
# # #         .info-card-title { font-weight: 600; margin-bottom: 0.5rem; color: #2d3748; }
# # #         .info-card-content { font-size: 0.875rem; }
# # #         .data-row { display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #e2e8f0; }
# # #         .data-label { color: #4a5568; }
# # #         .data-value { font-weight: 500; }
# # #         .status-badge { padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; margin-left: 0.5rem; }
# # #         .badge-pass { background: #c6f6d5; color: #22543d; }
# # #         .badge-fail { background: #fed7d7; color: #742a2a; }
# # #         .badge-warning { background: #feebc8; color: #744210; }
# # #         .reason-item { padding: 0.25rem 0; }
# # #         .reason-icon { color: #587042; font-weight: bold; margin-right: 0.5rem; }
# # #     </style>
# # #     """

# # # # Apply CSS immediately after set_page_config
# # # st.markdown(CSS, unsafe_allow_html=True)

# # # # =============================================================================
# # # # SESSION STATE INITIALIZATION
# # # # =============================================================================
# # # def init_session_state():
# # #     if 'stage1_complete' not in st.session_state:
# # #         st.session_state.stage1_complete = False
# # #     if 'stage1_decision' not in st.session_state:
# # #         st.session_state.stage1_decision = None
# # #     if 'stage1_data' not in st.session_state:
# # #         st.session_state.stage1_data = None
# # #     if 'current_customer_data' not in st.session_state:
# # #         st.session_state.current_customer_data = None
# # #     if 'page_navigation' not in st.session_state:
# # #         st.session_state.page_navigation = "🏠 Home"
# # #     if 'use_two_stage' not in st.session_state:
# # #         st.session_state.use_two_stage = False
# # #     if 'stage2_selected_tab' not in st.session_state:
# # #         st.session_state.stage2_selected_tab = "Manual Entry"
# # #     if 'fairness_log' not in st.session_state:
# # #         st.session_state.fairness_log = []
# # # init_session_state()

# # # # =============================================================================
# # # # IMPORT BUSINESS LOGIC MODULES – WITH HELPFUL ERROR IF MISSING
# # # # =============================================================================
# # # try:
# # #     from affordability_engine import calculate_emi, calculate_affordability
# # #     from reason_codes import generate_reason_codes
# # #     from risk_engine import (
# # #         calculate_final_risk_score, fill_missing_ml_fields,
# # #         clean_sentinel_values, validate_cibil_identity
# # #     )
# # #     from affordability_engine import check_loan_to_income, check_net_disposable
# # # except ImportError as e:
# # #     st.error(f"❌ Failed to import required modules: {e}")
# # #     st.info("""
# # #     Please ensure the following files are placed in one of these directories:
# # #     - `notebooks/` (same folder as test.py)
# # #     - `loan/` (sibling of notebooks)
# # #     - `utils/` (containing pdf_generator.py and __init__.py)
# # #     - The project root (`credit_risk_engine/`)

# # #     Required files:
# # #     - affordability_engine.py
# # #     - reason_codes.py
# # #     - risk_engine.py
# # #     - utils/__init__.py
# # #     - utils/pdf_generator.py
# # #     """)
# # #     st.stop()

# # # # =============================================================================
# # # # STAGE 2 ENGINE – ROBUST FALLBACK
# # # # =============================================================================
# # # try:
# # #     import stage2_engine
# # #     from stage2_engine import make_two_stage_decision, is_stage2_available, get_stage2_status
# # #     STAGE2_AVAILABLE = is_stage2_available()
# # # except ImportError:
# # #     stage2_engine = None
# # #     STAGE2_AVAILABLE = False
# # #     def make_two_stage_decision(*args, **kwargs):
# # #         raise NotImplementedError("Stage 2 engine not available")
# # #     def is_stage2_available():
# # #         return False
# # #     def get_stage2_status():
# # #         return {"error": "Stage 2 engine module not found", "available": False}

# # # # =============================================================================
# # # # PDF GENERATION – SAFE FALLBACK
# # # # =============================================================================
# # # PDF_AVAILABLE = False
# # # generate_decision_pdf = None
# # # generate_audit_pdf = None
# # # try:
# # #     from utils.pdf_generator import generate_decision_pdf, generate_audit_pdf
# # #     PDF_AVAILABLE = True
# # # except ImportError:
# # #     pass

# # # # =============================================================================
# # # # JSON SANITIZER
# # # # =============================================================================
# # # def sanitize_for_json(obj: Any) -> Any:
# # #     if obj is None or isinstance(obj, (str, int, float, bool)):
# # #         return obj
# # #     if isinstance(obj, set):
# # #         return list(obj)
# # #     if isinstance(obj, datetime):
# # #         return obj.isoformat()
# # #     if isinstance(obj, np.integer):
# # #         return int(obj)
# # #     if isinstance(obj, np.floating):
# # #         return float(obj)
# # #     if isinstance(obj, np.ndarray):
# # #         return obj.tolist()
# # #     if isinstance(obj, dict):
# # #         return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
# # #     if isinstance(obj, (list, tuple)):
# # #         return [sanitize_for_json(item) for item in obj]
# # #     try:
# # #         json.dumps(obj)
# # #         return obj
# # #     except (TypeError, ValueError):
# # #         return str(obj)

# # # # =============================================================================
# # # # LOAD TRAINED MODEL ASSETS (Stage 1 Random Forest)
# # # # =============================================================================
# # # @st.cache_resource
# # # def load_model_assets():
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
# # #         return {'loaded': False, 'error': 'credit_risk_assets.pkl not found. Please run the training script first.'}
# # #     except Exception as e:
# # #         return {'loaded': False, 'error': f'Error loading model: {str(e)}'}

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
# # # # AFFORDABILITY CALCULATION ENGINE (embedded for safety)
# # # # =============================================================================
# # # def calculate_emi(principal, annual_rate, tenure_months):
# # #     if principal <= 0 or tenure_months <= 0:
# # #         return 0
# # #     monthly_rate = annual_rate / (12 * 100)
# # #     if monthly_rate == 0:
# # #         return principal / tenure_months
# # #     emi = (principal * monthly_rate * (1 + monthly_rate)**tenure_months) / \
# # #           ((1 + monthly_rate)**tenure_months - 1)
# # #     return round(emi, 2)

# # # def calculate_affordability(monthly_income, loan_amount, interest_rate, tenure_months, existing_emi):
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
# # # # REASON CODE GENERATION SYSTEM (embedded for safety)
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
# # #     'low_bureau':       'Credit score below minimum ({score} < 550)',
# # #     'high_foir':        'EMI burden too high (FOIR: {foir}% > 50%)',
# # #     'severe_dpd':       'Severe payment delays ({dpd} instances of 90+ DPD)',
# # #     'moderate_dpd':     'Frequent payment delays ({dpd} instances of 30+ DPD)',
# # #     'low_income':       'Income below minimum threshold (₹{income:,} < ₹15,000)',
# # #     'short_employment': 'Insufficient employment tenure ({tenure} months < 6)',
# # #     'short_vintage':    'Insufficient business vintage ({vintage} years < 2 years)',
# # #     'bankruptcy':       'Active bankruptcy detected',
# # #     'kyc_failed':       'KYC verification not completed',
# # #     'fraud_flag':       'Fraud flag present on application',
# # #     'high_utilization': 'High credit utilization ({util}% > 80%)',
# # #     'age_invalid':      'Age outside acceptable range ({age} years, must be 24–70)',
# # #     'high_dependents':  'High number of dependents ({deps}) reducing net disposable income',
# # # }
# # # REVIEW_REASONS = {
# # #     'borderline_bureau':  'Credit score in borderline range ({score})',
# # #     'moderate_foir':      'EMI burden moderate (FOIR: {foir}%)',
# # #     'mixed_signals':      'Mixed credit indicators requiring human review',
# # #     'recent_employment':  'Recent employment change requiring verification',
# # #     'high_loan_amount':   'Large loan amount requiring additional underwriting review',
# # #     'moderate_dpd':       'Recent 30-day payment delays requiring review ({dpd} instances)',
# # #     'moderate_dependents':'Moderate number of dependents ({deps}) may affect repayment',
# # # }

# # # def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
# # #     reasons = []
# # #     bureau_score      = customer_data.get('bureau_score', 0)
# # #     foir              = affordability_data.get('foir_percentage', 0)
# # #     dpd_90            = customer_data.get('dpd_90_count_6m', 0)
# # #     dpd_30            = customer_data.get('dpd_30_count_6m', 0)
# # #     income            = customer_data.get('avg_salary_6m', 0)
# # #     employment_tenure = customer_data.get('employment_tenure_months', 0)
# # #     business_vintage  = customer_data.get('business_vintage_years', 0)
# # #     employment_type   = customer_data.get('employment_type', 'Salaried')
# # #     credit_util       = customer_data.get('credit_utilization_pct', 0)
# # #     age               = customer_data.get('age', 0)
# # #     dependents        = customer_data.get('dependents', 0)

# # #     if decision == "APPROVE":
# # #         if bureau_score >= 750:
# # #             reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
# # #         if employment_tenure >= 24:
# # #             reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
# # #         if foir <= 40:
# # #             reasons.append(APPROVAL_REASONS['low_foir'].format(foir=round(foir, 1)))
# # #         if dpd_90 == 0 and dpd_30 == 0:
# # #             reasons.append(APPROVAL_REASONS['clean_payment'])
# # #         if income >= 75000:
# # #             reasons.append(APPROVAL_REASONS['strong_income'].format(income=income))
# # #         if credit_util <= 30:
# # #             reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))

# # #     elif decision == "REJECT":
# # #         for check_name, check_result in policy_checks.items():
# # #             if '❌' in str(check_result):
# # #                 cn = check_name.lower()
# # #                 if 'bureau' in cn:
# # #                     reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
# # #                 elif 'dpd' in cn:
# # #                     reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
# # #                 elif 'income' in cn:
# # #                     reasons.append(REJECTION_REASONS['low_income'].format(income=income))
# # #                 elif 'tenure' in cn:
# # #                     if employment_type == 'Salaried':
# # #                         reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
# # #                     else:
# # #                         reasons.append(REJECTION_REASONS['short_vintage'].format(vintage=business_vintage))
# # #                 elif 'kyc' in cn:
# # #                     reasons.append(REJECTION_REASONS['kyc_failed'])
# # #                 elif 'bankruptcy' in cn:
# # #                     reasons.append(REJECTION_REASONS['bankruptcy'])
# # #                 elif 'fraud' in cn:
# # #                     reasons.append(REJECTION_REASONS['fraud_flag'])
# # #                 elif 'age' in cn:
# # #                     reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))
# # #         if foir > 50:
# # #             reasons.append(REJECTION_REASONS['high_foir'].format(foir=round(foir, 1)))
# # #         if credit_util > 80:
# # #             reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
# # #         if dpd_30 >= 3 and dpd_90 == 0:
# # #             reasons.append(REJECTION_REASONS['moderate_dpd'].format(dpd=dpd_30))
# # #         if dependents >= 4:
# # #             reasons.append(REJECTION_REASONS['high_dependents'].format(deps=dependents))

# # #     elif decision == "REVIEW":
# # #         if 650 <= bureau_score < 700:
# # #             reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
# # #         if 40 < foir <= 50:
# # #             reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=round(foir, 1)))
# # #         if employment_tenure < 12:
# # #             reasons.append(REVIEW_REASONS['recent_employment'])
# # #         if dpd_30 >= 1 and dpd_90 == 0:
# # #             reasons.append(REVIEW_REASONS['moderate_dpd'].format(dpd=dpd_30))
# # #         if 2 <= dependents < 4:
# # #             reasons.append(REVIEW_REASONS['moderate_dependents'].format(deps=dependents))
# # #         if not reasons:
# # #             reasons.append(REVIEW_REASONS['mixed_signals'])

# # #     return reasons[:3] if reasons else ['Decision based on comprehensive model assessment']

# # # # =============================================================================
# # # # PD CALCULATION (embedded for safety)
# # # # =============================================================================
# # # def bureau_score_to_pd(bureau_score):
# # #     if bureau_score >= 800:
# # #         return 0.5 + (900 - bureau_score) / 200 * 0.5
# # #     elif bureau_score >= 750:
# # #         return 1.0 + (800 - bureau_score) / 50 * 1.0
# # #     elif bureau_score >= 700:
# # #         return 2.0 + (750 - bureau_score) / 50 * 1.5
# # #     elif bureau_score >= 650:
# # #         return 3.5 + (700 - bureau_score) / 50 * 2.5
# # #     elif bureau_score >= 600:
# # #         return 6.0 + (650 - bureau_score) / 50 * 4.0
# # #     elif bureau_score >= 550:
# # #         return 10.0 + (600 - bureau_score) / 50 * 5.0
# # #     else:
# # #         return min(25.0, 15.0 + (550 - bureau_score) / 50 * 10.0)

# # # def foir_to_pd_adjustment(foir_percentage):
# # #     if foir_percentage <= 30:
# # #         return -0.75
# # #     elif foir_percentage <= 40:
# # #         return 0.00
# # #     elif foir_percentage <= 45:
# # #         return 0.75
# # #     elif foir_percentage <= 50:
# # #         return 1.50
# # #     elif foir_percentage <= 55:
# # #         return 2.25
# # #     elif foir_percentage <= 60:
# # #         return 3.50
# # #     else:
# # #         return 6.00

# # # def delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count=0):
# # #     if dpd_90_count >= 3:
# # #         return 5.0
# # #     elif dpd_90_count == 2:
# # #         return 3.0
# # #     elif dpd_90_count == 1:
# # #         return 2.0
# # #     elif dpd_30_count >= 3:
# # #         return 1.6
# # #     elif dpd_30_count >= 1:
# # #         return 1.3
# # #     else:
# # #         return 1.0

# # # def employment_stability_to_pd_adjustment(employment_type, tenure_months, business_vintage_years=0):
# # #     if employment_type == 'Salaried':
# # #         if tenure_months >= 36:
# # #             return -0.5
# # #         elif tenure_months >= 12:
# # #             return 0.0
# # #         elif tenure_months >= 6:
# # #             return 0.5
# # #         else:
# # #             return 2.0
# # #     elif employment_type in ['Self-Employed', 'Business']:
# # #         if business_vintage_years >= 5:
# # #             return -0.5
# # #         elif business_vintage_years >= 2:
# # #             return 0.0
# # #         else:
# # #             return 1.5
# # #     else:
# # #         return 1.0

# # # def inquiry_pattern_to_pd_adjustment(recent_inquiries_3m):
# # #     if recent_inquiries_3m <= 1:
# # #         return -0.3
# # #     elif recent_inquiries_3m <= 3:
# # #         return 0.0
# # #     elif recent_inquiries_3m <= 5:
# # #         return 0.8
# # #     elif recent_inquiries_3m <= 8:
# # #         return 1.5
# # #     else:
# # #         return 3.0

# # # def ml_confidence_to_pd_adjustment(ml_confidence, ml_decision):
# # #     if ml_decision == "APPROVE":
# # #         if ml_confidence >= 90:
# # #             return -0.5
# # #         elif ml_confidence >= 70:
# # #             return 0.0
# # #         else:
# # #             return 0.5
# # #     elif ml_decision == "REVIEW":
# # #         return 1.0
# # #     else:
# # #         return 5.0

# # # def calculate_final_pd(bureau_score, foir, confidence, dpd_90_count=0, dpd_30_count=0,
# # #                        employment_type='Salaried', employment_tenure=24, business_vintage=0,
# # #                        recent_inquiries=2, ml_decision='APPROVE'):
# # #     base_pd = bureau_score_to_pd(bureau_score)
# # #     foir_adj = foir_to_pd_adjustment(foir)
# # #     deliq_multiplier = delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count)
# # #     employment_adj = employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage)
# # #     inquiry_adj = inquiry_pattern_to_pd_adjustment(recent_inquiries)
# # #     ml_adj = ml_confidence_to_pd_adjustment(confidence, ml_decision)
# # #     adjusted_base_pd = base_pd * deliq_multiplier
# # #     final_pd = adjusted_base_pd + foir_adj + employment_adj + inquiry_adj + ml_adj
# # #     final_pd = max(0.5, min(final_pd, 25.0))
# # #     return round(final_pd, 2)

# # # # =============================================================================
# # # # RISK SCORE CALCULATION (embedded for safety)
# # # # =============================================================================
# # # def calculate_final_risk_score(bureau_score, ml_confidence, foir,
# # #                                 dpd_90, dpd_30, net_surplus,
# # #                                 bounces=0, missing_months=0, active_loans=0):
# # #     bureau_points = (bureau_score / 900) * 400
# # #     ml_points = (ml_confidence / 100) * 300
# # #     foir_points = max(0, (1 - foir / 50) * 150)
# # #     dpd_penalty = min((dpd_90 * 50) + (dpd_30 * 20), 150)
# # #     behavioral_penalty = min((bounces * 10) + (missing_months * 10), 100)
# # #     if net_surplus > 50000:
# # #         surplus_points = 50
# # #     elif net_surplus > 0:
# # #         surplus_points = 20
# # #     elif net_surplus < -50000:
# # #         surplus_points = -50
# # #     else:
# # #         surplus_points = -20
# # #     total = (bureau_points + ml_points + foir_points
# # #              + surplus_points - dpd_penalty - behavioral_penalty)
# # #     return max(0, min(int(total), 1000))

# # # # =============================================================================
# # # # CIBIL PDF EXTRACTION ENGINE (OCR + PATTERN MATCHING) – OPTIONAL
# # # # =============================================================================
# # # def extract_cibil_from_pdf(uploaded_file):
# # #     if not OCR_AVAILABLE:
# # #         return {'success': False, 'error': OCR_ERROR_MSG or 'OCR libraries not installed. Check packages.txt and requirements.txt.'}

# # #     try:
# # #         pdf_bytes = uploaded_file.read()
# # #         images = convert_from_bytes(pdf_bytes, dpi=300)
# # #         full_text = ""
# # #         for image in images:
# # #             gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
# # #             _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # #             full_text += pytesseract.image_to_string(binary) + "\n"

# # #         credit_score = 720
# # #         score_match = re.search(
# # #             r'\b(\d{3})\s*(?:EXCELLENT|VERY\s*GOOD|GOOD|FAIR|SUBPRIME|POOR|NH|NA)\b',
# # #             full_text, re.IGNORECASE
# # #         )
# # #         if score_match:
# # #             val = int(score_match.group(1))
# # #             if 300 <= val <= 900:
# # #                 credit_score = val
# # #         if credit_score == 720:
# # #             score_match2 = re.search(
# # #                 r'(?:cibil|credit)\s*score\s*[:\-\(]?\s*(\d{3})',
# # #                 full_text, re.IGNORECASE
# # #             )
# # #             if score_match2:
# # #                 val = int(score_match2.group(1))
# # #                 if 300 <= val <= 900:
# # #                     credit_score = val
# # #         if credit_score == 720:
# # #             score_match3 = re.search(r'score.*?\((\d{3})\)', full_text, re.IGNORECASE)
# # #             if score_match3:
# # #                 val = int(score_match3.group(1))
# # #                 if 300 <= val <= 900:
# # #                     credit_score = val

# # #         monthly_income = 50000
# # #         income_match = re.search(
# # #             r'(?:net\s+monthly\s+income|monthly\s+income|net\s+income|salary)[^\n\r]{0,30}?'
# # #             r'(?:rs\.?\s*|inr\s*|₹\s*)([\d,]+)',
# # #             full_text, re.IGNORECASE
# # #         )
# # #         if income_match:
# # #             val = int(income_match.group(1).replace(',', ''))
# # #             if val > 1000:
# # #                 monthly_income = val
# # #         if monthly_income == 50000:
# # #             income_match2 = re.search(r'(?:rs\.?\s*|₹\s*)([\d,]{4,})', full_text, re.IGNORECASE)
# # #             if income_match2:
# # #                 val = int(income_match2.group(1).replace(',', ''))
# # #                 if 5000 <= val <= 1000000:
# # #                     monthly_income = val

# # #         cc_util_pct = 35
# # #         util_match = re.search(r'utilization\s*[\(:\-]?\s*(\d{1,3})\s*%', full_text, re.IGNORECASE)
# # #         if util_match:
# # #             cc_util_pct = int(util_match.group(1))
# # #         cc_util = cc_util_pct / 100.0
# # #         high_util = 1 if cc_util_pct > 75 else 0

# # #         age_extracted = 35
# # #         dob_match = re.search(
# # #             r'(?:date\s+of\s+birth|dob)[:\s]+(\d{2}[-/]\w{3,9}[-/]\d{2,4})',
# # #             full_text, re.IGNORECASE
# # #         )
# # #         if dob_match:
# # #             try:
# # #                 from datetime import datetime as _dt
# # #                 dob_str = dob_match.group(1)
# # #                 for fmt in ('%d-%b-%Y', '%d/%b/%Y', '%d-%m-%Y', '%d/%m/%Y'):
# # #                     try:
# # #                         dob = _dt.strptime(dob_str, fmt)
# # #                         age_extracted = int((datetime.now() - dob).days / 365.25)
# # #                         break
# # #                     except Exception:
# # #                         continue
# # #             except Exception:
# # #                 pass

# # #         biz_vintage = 3
# # #         biz_match = re.search(r'business\s+vintage.*?(\d+)', full_text, re.IGNORECASE)
# # #         if biz_match:
# # #             biz_vintage = int(biz_match.group(1))

# # #         lines = full_text.split('\n')
# # #         in_accounts = False
# # #         in_enquiry = False
# # #         accounts = []
# # #         enquiry_dates = []

# # #         for line in lines:
# # #             line_up = line.upper()
# # #             if 'ACCOUNT DETAILS' in line_up:
# # #                 in_accounts = True
# # #                 in_enquiry = False
# # #                 continue
# # #             if 'ENQUIRY DETAILS' in line_up:
# # #                 in_accounts = False
# # #                 in_enquiry = True
# # #                 continue

# # #             if in_accounts:
# # #                 if re.search(r'SUMMARY|SCORE|PERSONAL\s+INFO', line_up):
# # #                     break
# # #                 if re.search(r'\bLender\b|\bAccount\s*No\b|\bOpen\s*Date\b|\bDPD\b|\bStatus\b', line, re.IGNORECASE):
# # #                     continue
# # #                 stripped = line.strip()
# # #                 if not stripped:
# # #                     continue
# # #                 dpd_match = re.search(r'\b(0\d0|0\d\d|\d{3})\b', stripped)
# # #                 status_match = re.search(
# # #                     r'\b(Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss)\s*$',
# # #                     stripped, re.IGNORECASE
# # #                 )
# # #                 if (re.search(r'\bINR\b', stripped, re.IGNORECASE) or
# # #                         re.match(r'^[A-Z][a-zA-Z\s]+(?:Bank|Finance|Capital|Fincorp|SBI|ICICI|HDFC|Axis|Bajaj|Tata|Kotak)', stripped)):
# # #                     dpd_val = int(dpd_match.group(1)) if dpd_match else 0
# # #                     status_str = status_match.group(1) if status_match else 'Active'
# # #                     accounts.append({'dpd': dpd_val, 'status': status_str.lower()})

# # #             if in_enquiry:
# # #                 enq_date = re.match(r'^\s*(\d{2}-[A-Za-z]{3}-\d{4})', line)
# # #                 if enq_date:
# # #                     enquiry_dates.append(enq_date.group(1))

# # #         written_off_count = 0
# # #         settled_count = 0
# # #         dpd_90_count = 0
# # #         dpd_60_count = 0
# # #         dpd_30_count = 0
# # #         active_count = 0
# # #         sub_standard_count = 0

# # #         if accounts:
# # #             for acc in accounts:
# # #                 dpd = acc.get('dpd', 0)
# # #                 status = acc.get('status', '')
# # #                 if dpd >= 90:
# # #                     dpd_90_count += 1
# # #                 elif dpd >= 60:
# # #                     dpd_60_count += 1
# # #                 elif dpd >= 30:
# # #                     dpd_30_count += 1
# # #                 if 'written' in status:
# # #                     written_off_count += 1
# # #                 elif 'settled' in status:
# # #                     settled_count += 1
# # #                 elif 'active' in status:
# # #                     active_count += 1
# # #                 if dpd >= 30:
# # #                     sub_standard_count += 1
# # #         else:
# # #             written_off_count = len(re.findall(r'\bwritten[-\s]?off\b', full_text, re.IGNORECASE))
# # #             settled_count     = len(re.findall(r'\bsettled\b', full_text, re.IGNORECASE))
# # #             dpd_90_count      = len(re.findall(r'\b090\b|90\+?\s*dpd', full_text, re.IGNORECASE))
# # #             dpd_60_count      = len(re.findall(r'\b060\b|60\+?\s*dpd', full_text, re.IGNORECASE))
# # #             dpd_30_count      = len(re.findall(r'\b030\b|30\+?\s*dpd', full_text, re.IGNORECASE))
# # #             active_sum = re.search(r'Total\s+Accounts\s+Active.*?(\d+)\s+(\d+)', full_text, re.IGNORECASE)
# # #             if active_sum:
# # #                 active_count = int(active_sum.group(2))

# # #         if active_count == 0:
# # #             summary_match = re.search(
# # #                 r'Total\s+Accounts\s+Active[^\n]*\n\s*(\d+)\s+(\d+)',
# # #                 full_text, re.IGNORECASE
# # #             )
# # #             if summary_match:
# # #                 active_count = int(summary_match.group(2))
# # #             else:
# # #                 inline = re.search(
# # #                     r'(?:Total\s+Accounts.*?Active.*?Closed.*?\n|(\d+)\s+(\d+)\s+(\d+)\s+[\d,]+\s+\d+)',
# # #                     full_text, re.IGNORECASE
# # #                 )
# # #                 if inline and inline.group(2):
# # #                     active_count = int(inline.group(2))

# # #         enq_12m_total = len(enquiry_dates)
# # #         enq_sum_match = re.search(r'Enquiries?\s*\(?12M\)?\s*[:\s]+(\d+)', full_text, re.IGNORECASE)
# # #         if enq_sum_match:
# # #             enq_12m_total = max(enq_12m_total, int(enq_sum_match.group(1)))

# # #         enq_L3m = min(len(enquiry_dates), enq_12m_total)
# # #         enq_L6m = enq_12m_total
# # #         enq_L12m = enq_12m_total

# # #         if credit_score >= 750 and (written_off_count > 0 or dpd_90_count > 0):
# # #             credit_score = 550

# # #         total_accounts = max(len(accounts), active_count + settled_count + written_off_count)
# # #         pct_active = active_count / total_accounts if total_accounts > 0 else 0.6

# # #         extracted_data = {
# # #             'Credit_Score': credit_score,
# # #             'max_delinquency_level': max(dpd_90_count * 90, dpd_60_count * 60, dpd_30_count * 30),
# # #             'num_times_30p_dpd': dpd_30_count,
# # #             'num_times_60p_dpd': dpd_60_count,
# # #             'num_times_delinquent': dpd_30_count + dpd_60_count + dpd_90_count,
# # #             'num_deliq_6mts': dpd_30_count + dpd_60_count + dpd_90_count,
# # #             'num_deliq_12mts': dpd_30_count + dpd_60_count + dpd_90_count,
# # #             'max_deliq_6mts': dpd_90_count,
# # #             'max_deliq_12mts': dpd_90_count,
# # #             'enq_L3m': enq_L3m,
# # #             'enq_L6m': enq_L6m,
# # #             'enq_L12m': enq_L12m,
# # #             'num_std': active_count,
# # #             'num_std_6mts': active_count,
# # #             'num_std_12mts': active_count,
# # #             'num_sub': sub_standard_count,
# # #             'num_sub_6mts': sub_standard_count,
# # #             'num_dbt': dpd_90_count,
# # #             'num_lss': written_off_count,
# # #             'pct_of_active_TLs_ever': round(pct_active, 2),
# # #             'pct_currentBal_all_TL': 0.3,
# # #             'CC_utilization': round(cc_util, 2),
# # #             'PL_utilization': 0.25,
# # #             'max_unsec_exposure_inPct': cc_util_pct,
# # #             'AGE': age_extracted,
# # #             'NETMONTHLYINCOME': monthly_income,
# # #             'Time_With_Curr_Empr': biz_vintage * 12,
# # #             'CC_Flag': 1 if re.search(r'credit card', full_text, re.IGNORECASE) else 0,
# # #             'PL_Flag': 1 if re.search(r'personal loan', full_text, re.IGNORECASE) else 0,
# # #             'HL_Flag': 1 if re.search(r'home loan', full_text, re.IGNORECASE) else 0,
# # #             'GL_Flag': 1 if re.search(r'gold loan', full_text, re.IGNORECASE) else 0,
# # #             'raw_text': full_text,
# # #             'success': True,
# # #             'extraction_method': 'OCR+robust',
# # #             'written_off_count': written_off_count,
# # #             'settled_count': settled_count,
# # #             'high_util_flag': high_util,
# # #             'dpd_90_count_6m': dpd_90_count,
# # #             'recent_deliq_flag': 1 if (dpd_90_count > 0 or dpd_60_count > 0) else 0,
# # #             'account_quality_score': max(0, 100 - (written_off_count * 20) - (settled_count * 10) - (dpd_90_count * 15) - (dpd_30_count * 5))
# # #         }
# # #         return extracted_data
# # #     except Exception as e:
# # #         return {'error': str(e), 'message': f'Error extracting CIBIL data: {str(e)}', 'success': False}

# # # # =============================================================================
# # # # HYBRID DECISION ENGINE (PATCHED VERSION)
# # # # =============================================================================
# # # def make_hybrid_decision_enhanced(customer_dict):
# # #     # First, fill any missing ML fields (the 38 features not in the form)
# # #     fill_missing_ml_fields(customer_dict)

# # #     policy_checks = {}
# # #     age = customer_dict.get('age', 0)
# # #     employment_type = customer_dict.get('employment_type', 'Salaried')
# # #     kyc_verified = customer_dict.get('kyc_verified', True)
# # #     bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
# # #     fraud_flag = customer_dict.get('fraud_flag', False)
# # #     age_min, age_max = 24, 70
# # #     if age < age_min or age > age_max:
# # #         policy_checks['age'] = f"❌ Age {age} (Required: {age_min}-{age_max})"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: Age outside allowed range", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     policy_checks['age'] = f"✅ Age {age} (Valid)"
# # #     if not kyc_verified:
# # #         policy_checks['kyc'] = "❌ KYC Not Verified"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: KYC verification required", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     policy_checks['kyc'] = "✅ KYC Verified"
# # #     if bankruptcy_flag:
# # #         policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: Active bankruptcy", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     policy_checks['bankruptcy'] = "✅ No Bankruptcy"
# # #     if fraud_flag:
# # #         policy_checks['fraud'] = "❌ Fraud Flag"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: Fraud detected", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     policy_checks['fraud'] = "✅ No Fraud History"

# # #     dependents = customer_dict.get('dependents', 0)
# # #     dependents_flag_review = False
# # #     if dependents > 5:
# # #         policy_checks['dependents'] = f"⚠️ Dependents {dependents} (>5: Review Required)"
# # #         dependents_flag_review = True
# # #     else:
# # #         policy_checks['dependents'] = f"✅ Dependents {dependents} (Acceptable)"

# # #     monthly_income = customer_dict.get('avg_salary_6m', 0)
# # #     employment_tenure = customer_dict.get('employment_tenure_months', 0)
# # #     business_vintage = customer_dict.get('business_vintage_years', 0)
# # #     if monthly_income < 15000:
# # #         policy_checks['income'] = f"❌ Income ₹{monthly_income:,.0f} (Min: ₹15,000)"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: Income below minimum", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     policy_checks['income'] = f"✅ Income ₹{monthly_income:,.0f}"
# # #     if employment_type == 'Salaried' and employment_tenure < 6:
# # #         policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient tenure", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
# # #         policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient business vintage", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
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
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: Bureau score too low", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"
# # #     if dpd_90 > 0:
# # #         policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: Severe delinquency", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     policy_checks['dpd'] = "✅ No 90+ DPD"
# # #     if credit_utilization > 80:
# # #         policy_checks['utilization'] = f"⚠️ High utilization {credit_utilization}%"
# # #     else:
# # #         policy_checks['utilization'] = f"✅ Utilization {credit_utilization}%"
# # #     if recent_inquiries > 5:
# # #         policy_checks['inquiries'] = f"⚠️ {recent_inquiries} recent inquiries"
# # #     else:
# # #         policy_checks['inquiries'] = f"✅ {recent_inquiries} inquiries"

# # #     # Active loans check
# # #     active_loans = customer_dict.get('active_loans_count', 0)
# # #     if active_loans >= 5:
# # #         policy_checks['active_loans'] = f"⚠️ High active loans ({int(active_loans)}) — Review"
# # #         active_loans_flag = True
# # #     else:
# # #         policy_checks['active_loans'] = f"✅ Active loans: {int(active_loans)}"
# # #         active_loans_flag = False

# # #     # Salary stability check
# # #     salary_stability = customer_dict.get('salary_stability_flag', 'STABLE')
# # #     if salary_stability == 'UNSTABLE':
# # #         policy_checks['salary'] = "⚠️ Unstable salary — Review required"
# # #         salary_flag = True
# # #     elif salary_stability == 'MODERATE':
# # #         policy_checks['salary'] = "⚠️ Moderate salary stability"
# # #         salary_flag = False
# # #     else:
# # #         policy_checks['salary'] = "✅ Stable salary"
# # #         salary_flag = False

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
# # #     # Save raw ML decision before overrides
# # #     ml_raw_decision = ml_decision
# # #     try:
# # #         pred_proba = MODEL.predict_proba(final_input)[0]
# # #         confidence = max(pred_proba) * 100
# # #         class_probs = {cls: prob * 100 for cls, prob in zip(TARGET_LE.classes_, pred_proba)}
# # #     except Exception:
# # #         confidence = 75.0
# # #         class_probs = {ml_decision: 100.0}

# # #     loan_amount = customer_dict.get('loan_amount', 0)
# # #     loan_tenure = customer_dict.get('loan_tenure_months', 12)
# # #     interest_rate = customer_dict.get('interest_rate', 10.5)
# # #     existing_emi = customer_dict.get('existing_emi', 0)
# # #     affordability_data = calculate_affordability(monthly_income, loan_amount, interest_rate, loan_tenure, existing_emi)
# # #     foir = affordability_data['foir_percentage']
# # #         # --- FOIR > 50% forces REJECT immediately ---
# # #     if foir > 50:
# # #         ml_decision = "REJECT"
# # #         policy_checks['foir'] = f"❌ FOIR {foir:.1f}% exceeds maximum allowed (50%)"

# # #     # Other overrides only apply if still APPROVE
# # #     if dependents_flag_review and ml_decision == "APPROVE":
# # #         ml_decision = "REVIEW"
# # #     if active_loans_flag and ml_decision == "APPROVE":
# # #         ml_decision = "REVIEW"
# # #     if salary_flag and ml_decision == "APPROVE":
# # #         ml_decision = "REVIEW"


        
# # #     # if ml_decision == "APPROVE" and foir > 50:
# # #     #     ml_decision = "REVIEW"
# # #     # if dependents_flag_review and ml_decision == "APPROVE":
# # #     #     ml_decision = "REVIEW"
# # #     # if active_loans_flag and ml_decision == "APPROVE":
# # #     #     ml_decision = "REVIEW"
# # #     # if salary_flag and ml_decision == "APPROVE":
# # #     #     ml_decision = "REVIEW"



# # #     risk_score = calculate_final_risk_score(
# # #         bureau_score=bureau_score,
# # #         ml_confidence=confidence,
# # #         foir=foir,
# # #         dpd_90=dpd_90,
# # #         dpd_30=customer_dict.get('dpd_30_count_6m', 0),
# # #         net_surplus=customer_dict.get('net_cash_surplus_6m', 0),
# # #         bounces=customer_dict.get('inward_bounce_count_3m', 0),
# # #         missing_months=customer_dict.get('salary_missing_months', 0),
# # #         active_loans=active_loans
# # #     )

# # #     pd_percentage = calculate_final_pd(
# # #         bureau_score=bureau_score,
# # #         foir=foir,
# # #         confidence=confidence,
# # #         dpd_90_count=dpd_90,
# # #         dpd_30_count=customer_dict.get('dpd_30_count_6m', 0),
# # #         employment_type=employment_type,
# # #         employment_tenure=employment_tenure,
# # #         business_vintage=business_vintage,
# # #         recent_inquiries=recent_inquiries,
# # #         ml_decision=ml_decision
# # #     )

# # #     return {
# # #         'decision': ml_decision,
# # #         'ml_raw_decision': ml_raw_decision,
# # #         'reason': "Decision based on comprehensive assessment",
# # #         'confidence': confidence,
# # #         'class_probs': class_probs,
# # #         'policy_checks': policy_checks,
# # #         'risk_score': risk_score,
# # #         'pd_percentage': round(pd_percentage, 2),
# # #         'affordability_data': affordability_data
# # #     }

# # # # =============================================================================
# # # # BATCH PREDICTION ENGINE (updated defaults)
# # # # =============================================================================
# # # def process_batch_predictions(df):
# # #     results = []
# # #     for idx, row in df.iterrows():
# # #         customer_dict = row.to_dict()
# # #         for key, value in customer_dict.items():
# # #             if isinstance(value, str):
# # #                 if value.lower() in ['yes', 'true', '1']:
# # #                     customer_dict[key] = True
# # #                 elif value.lower() in ['no', 'false', '0']:
# # #                     customer_dict[key] = False
# # #         required_fields = {
# # #             'age': 35, 'employment_type': 'Salaried', 'kyc_verified': True,
# # #             'bankruptcy_flag': False, 'fraud_flag': False, 'employment_tenure_months': 24,
# # #             'business_vintage_years': 0, 'bureau_score': 700, 'dpd_90_count_6m': 0,
# # #             'dpd_30_count_6m': 0, 'credit_utilization_pct': 30, 'recent_inquiries_3m': 0,
# # #             'active_loans_count': 0, 'existing_emi': 0, 'avg_salary_6m': 50000,
# # #             'AMT_INCOME_TOTAL': 600000, 'net_cash_surplus_6m': 20000, 'salary_stability_flag': 'STABLE',
# # #             'loan_amount': 180000, 'loan_tenure_months': 24, 'interest_rate': 10.5, 'AMT_ANNUITY': 8500,
# # #             'dependents': 0,
# # #             'payment_discipline_flag': 'GOOD',
# # #             'liquidity_flag': 'LOW',
# # #             'cashflow_health': 'MODERATE',
# # #             'bureau_risk_flag': 'LOW',
# # #             'inward_bounce_count_3m': 0,
# # #             'salary_missing_months': 0,
# # #         }
# # #         for field, default in required_fields.items():
# # #             if field not in customer_dict or pd.isna(customer_dict[field]):
# # #                 customer_dict[field] = default
# # #         try:
# # #             decision_data = make_hybrid_decision_enhanced(customer_dict)
# # #             reasons = generate_reason_codes(
# # #                 decision=decision_data.get('decision', 'ERROR'),
# # #                 customer_data=customer_dict,
# # #                 affordability_data=decision_data.get('affordability_data', {}),
# # #                 policy_checks=decision_data.get('policy_checks', {})
# # #             )
# # #             app_id = f"BATCH_{idx+1:04d}"
# # #             affordability = decision_data.get('affordability_data', {})
# # #             result = {
# # #                 'application_id': app_id,
# # #                 'decision': decision_data.get('decision', 'ERROR'),
# # #                 'risk_score': decision_data.get('risk_score', 0),
# # #                 'pd_percentage': decision_data.get('pd_percentage', 0),
# # #                 'confidence': round(decision_data.get('confidence', 0), 2),
# # #                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# # #                 'reason_1': reasons[0] if len(reasons) > 0 else '',
# # #                 'reason_2': reasons[1] if len(reasons) > 1 else '',
# # #                 'reason_3': reasons[2] if len(reasons) > 2 else '',
# # #                 'age': customer_dict.get('age', ''),
# # #                 'employment_type': customer_dict.get('employment_type', ''),
# # #                 'bureau_score': customer_dict.get('bureau_score', ''),
# # #                 'monthly_income': customer_dict.get('avg_salary_6m', ''),
# # #                 'loan_amount': customer_dict.get('loan_amount', ''),
# # #                 'loan_tenure_months': customer_dict.get('loan_tenure_months', ''),
# # #                 'interest_rate': customer_dict.get('interest_rate', ''),
# # #                 'new_emi': affordability.get('new_emi', 0),
# # #                 'existing_emi': affordability.get('existing_emi', 0),
# # #                 'total_emi': affordability.get('total_emi', 0),
# # #                 'foir_percentage': round(affordability.get('foir_percentage', 0), 2),
# # #                 'net_disposable': affordability.get('net_disposable', 0),
# # #                 'affordability_status': affordability.get('status', 'N/A'),
# # #                 'dpd_90_count': customer_dict.get('dpd_90_count_6m', 0),
# # #                 'dpd_30_count': customer_dict.get('dpd_30_count_6m', 0),
# # #                 'credit_utilization': customer_dict.get('credit_utilization_pct', 0),
# # #                 'recent_inquiries': customer_dict.get('recent_inquiries_3m', 0),
# # #                 'active_loans': customer_dict.get('active_loans_count', 0),
# # #                 'employment_tenure': customer_dict.get('employment_tenure_months', 0),
# # #                 'business_vintage': customer_dict.get('business_vintage_years', 0),
# # #                 'salary_stability': customer_dict.get('salary_stability_flag', ''),
# # #                 'kyc_status': 'Verified' if customer_dict.get('kyc_verified', True) else 'Not Verified',
# # #                 'bankruptcy': 'Yes' if customer_dict.get('bankruptcy_flag', False) else 'No',
# # #                 'fraud': 'Yes' if customer_dict.get('fraud_flag', False) else 'No',
# # #                 'prob_approve': round(decision_data.get('class_probs', {}).get('APPROVE', 0), 2),
# # #                 'prob_review': round(decision_data.get('class_probs', {}).get('REVIEW', 0), 2),
# # #                 'prob_reject': round(decision_data.get('class_probs', {}).get('REJECT', 0), 2),
# # #             }
# # #         except Exception as e:
# # #             result = {
# # #                 'application_id': f"BATCH_{idx+1:04d}",
# # #                 'decision': 'ERROR',
# # #                 'risk_score': 0, 'pd_percentage': 0, 'confidence': 0,
# # #                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# # #                 'reason_1': '', 'reason_2': '', 'reason_3': '',
# # #                 'age': customer_dict.get('age', ''),
# # #                 'employment_type': customer_dict.get('employment_type', ''),
# # #                 'bureau_score': customer_dict.get('bureau_score', ''),
# # #                 'monthly_income': customer_dict.get('avg_salary_6m', ''),
# # #                 'loan_amount': customer_dict.get('loan_amount', ''),
# # #                 'error_message': str(e)
# # #             }
# # #         results.append(result)
# # #     return pd.DataFrame(results)

# # # def create_download_link(df, filename="batch_results.csv"):
# # #     csv = df.to_csv(index=False)
# # #     b64 = base64.b64encode(csv.encode()).decode()
# # #     return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;">📥 Download CSV</a>'

# # # # =============================================================================
# # # # MODERN UI COMPONENTS
# # # # =============================================================================
# # # def render_decision_header(decision_data, customer_data):
# # #     decision = decision_data.get('decision', 'ERROR')
# # #     risk_score = decision_data.get('risk_score', 0)
# # #     pd_score = decision_data.get('pd_percentage', 0)
# # #     approved_amount = customer_data.get('loan_amount', 0)
# # #     tenure = customer_data.get('loan_tenure_months', 24)
# # #     app_id = customer_data.get('application_id', 'N/A')
# # #     timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
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
# # #             <div class="decision-title"><span>{icon}</span><span>{decision}</span></div>
# # #             <div class="decision-subtitle">{subtitle}</div>
# # #         </div>
# # #     """, unsafe_allow_html=True)
# # #     col1, col2, col3, col4, col5 = st.columns(5)
# # #     with col1:
# # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{risk_score}</div><div class="stat-label">Risk Score</div></div>', unsafe_allow_html=True)
# # #     with col2:
# # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{pd_score}%</div><div class="stat-label">PD Score</div></div>', unsafe_allow_html=True)
# # #     with col3:
# # #         st.markdown(f'<div class="stat-card"><div class="stat-number">₹{approved_amount:,.0f}</div><div class="stat-label">Loan Amount</div></div>', unsafe_allow_html=True)
# # #     with col4:
# # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{tenure}</div><div class="stat-label">Tenure (Months)</div></div>', unsafe_allow_html=True)
# # #     with col5:
# # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{decision_data.get("confidence", 0):.0f}%</div><div class="stat-label">Confidence</div></div>', unsafe_allow_html=True)
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         st.markdown(f'<div class="info-box"><strong>📋 Application ID:</strong> {app_id}</div>', unsafe_allow_html=True)
# # #     with col2:
# # #         st.markdown(f'<div class="info-box"><strong>🕐 Decision Timestamp:</strong> {timestamp}</div>', unsafe_allow_html=True)

# # # def render_info_card(title, icon, data_dict, status_dict=None):
# # #     st.markdown(f'<div class="info-card"><div class="info-card-title"><span class="icon">{icon}</span><span>{title}</span></div><div class="info-card-content">', unsafe_allow_html=True)
# # #     for label, value in data_dict.items():
# # #         status = ""
# # #         if status_dict and label in status_dict:
# # #             if status_dict[label] == "pass":
# # #                 status = '<span class="status-badge badge-pass">✓ Passed</span>'
# # #             elif status_dict[label] == "fail":
# # #                 status = '<span class="status-badge badge-fail">✗ Failed</span>'
# # #             elif status_dict[label] == "warning":
# # #                 status = '<span class="status-badge badge-warning">⚠ Warning</span>'
# # #         st.markdown(f'<div class="data-row"><span class="data-label">{label}</span><span class="data-value">{value} {status}</span></div>', unsafe_allow_html=True)
# # #     st.markdown('</div></div>', unsafe_allow_html=True)

# # # def render_reason_codes(reasons):
# # #     st.markdown('<div class="info-card"><div class="info-card-title"><span class="icon">📝</span><span>Decision Reasons</span></div><div class="info-card-content">', unsafe_allow_html=True)
# # #     for i, reason in enumerate(reasons, 1):
# # #         st.markdown(f'<div class="reason-item"><span class="reason-icon">{i}.</span><span>{reason}</span></div>', unsafe_allow_html=True)
# # #     st.markdown('</div></div>', unsafe_allow_html=True)

# # # def create_modern_gauge(value, title, max_value=100):
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
# # #             'bgcolor': 'white', 'borderwidth': 0,
# # #             'steps': [
# # #                 {'range': [0, 50], 'color': '#fed7d7'},
# # #                 {'range': [50, 75], 'color': '#feebc8'},
# # #                 {'range': [75, 100], 'color': '#c6f6d5'}
# # #             ]
# # #         }
# # #     ))
# # #     fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white',
# # #                       font={'family': 'Inter', 'color': '#2d3748'})
# # #     return fig

# # # def create_modern_bar_chart(class_probs):
# # #     df = pd.DataFrame({'Decision': list(class_probs.keys()), 'Probability': list(class_probs.values())})
# # #     colors = {'REVIEW': '#ed8936', 'APPROVE': '#48bb78', 'REJECT': '#f56565'}
# # #     fig = px.bar(df, x='Decision', y='Probability', title='Decision Probabilities', color='Decision',
# # #                  color_discrete_map=colors, text='Probability')
# # #     fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
# # #     fig.update_layout(
# # #         showlegend=False, yaxis_title='Probability (%)', xaxis_title='', height=300,
# # #         margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white', plot_bgcolor='white',
# # #         font={'family': 'Inter', 'color': '#2d3748'},
# # #         yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]},
# # #         xaxis={'tickfont': {'size': 14, 'color': '#2d3748'}}
# # #     )
# # #     return fig

# # # # =============================================================================
# # # # STAGE 2 BINARY RESOLVER (embedded)
# # # # =============================================================================
# # # def resolve_stage2_to_binary(stage2_result: dict) -> dict:
# # #     result = stage2_result.copy()
# # #     tier   = result.get('stage2_tier', '')
# # #     raw    = result.get('final_decision', '')
# # #     score  = result.get('combined_risk_score', 0) or 0

# # #     TIER_TO_DECISION = {
# # #         'P1': 'APPROVE',
# # #         'P2': 'APPROVE',
# # #         'P3': 'REJECT',
# # #         'P4': 'REJECT',
# # #     }

# # #     if raw == 'REJECT':
# # #         result['final_decision'] = 'REJECT'
# # #     elif raw == 'APPROVE':
# # #         if tier in TIER_TO_DECISION:
# # #             result['final_decision'] = TIER_TO_DECISION[tier]
# # #         else:
# # #             result['final_decision'] = 'APPROVE'
# # #     else:
# # #         if tier in TIER_TO_DECISION:
# # #             result['final_decision'] = TIER_TO_DECISION[tier]
# # #             result['reason'] = (
# # #                 result.get('reason', '') +
# # #                 f" [REVIEW resolved to {TIER_TO_DECISION[tier]} via risk tier {tier}]"
# # #             )
# # #         else:
# # #             resolved = 'APPROVE' if score >= 600 else 'REJECT'
# # #             result['final_decision'] = resolved
# # #             result['reason'] = (
# # #                 result.get('reason', '') +
# # #                 f" [REVIEW resolved to {resolved} via combined risk score {score}]"
# # #             )

# # #     if result['final_decision'] == 'APPROVE':
# # #         result.setdefault('interest_rate_range',
# # #             {'P1': '9.5% – 11%', 'P2': '11% – 13%'}.get(tier, '11% – 14%'))
# # #     else:
# # #         result['interest_rate_range'] = 'N/A — Rejected'

# # #     return result

# # # # =============================================================================
# # # # STAGE 2 RESULTS DISPLAY (fixed version)
# # # # =============================================================================
# # # def display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data):
# # #     st.markdown("---")
# # #     st.markdown('<p class="main-header">🎯 Stage 2 Final Results</p>', unsafe_allow_html=True)

# # #     final_decision = stage2_result.get('final_decision', 'ERROR')
# # #     interest_range = stage2_result.get('interest_rate_range', 'N/A')
# # #     stage2_tier = stage2_result.get('stage2_tier', 'N/A')
# # #     stage2_confidence = stage2_result.get('stage2_confidence', 0)
# # #     combined_risk_score = stage2_result.get('combined_risk_score', 0)

# # #     if final_decision == "APPROVE":
# # #         card_class = "decision-card decision-card-approved"
# # #         icon = "✓"
# # #         subtitle = "✅ Final Decision: Approved — Proceed to Disbursement"
# # #     else:
# # #         card_class = "decision-card decision-card-rejected"
# # #         icon = "✗"
# # #         subtitle = "❌ Final Decision: Rejected — Application Declined"

# # #     st.markdown(f"""
# # #         <div class="{card_class}">
# # #             <div class="decision-title"><span>{icon}</span><span>{final_decision}</span></div>
# # #             <div class="decision-subtitle">{subtitle}</div>
# # #         </div>
# # #     """, unsafe_allow_html=True)

# # #     col1, col2, col3, col4 = st.columns(4)
# # #     with col1:
# # #         st.metric("Risk Tier", stage2_tier)
# # #     with col2:
# # #         st.metric("Interest Rate", interest_range)
# # #     with col3:
# # #         st.metric("Combined Risk Score", combined_risk_score)
# # #     with col4:
# # #         confidence_display = f"{stage2_confidence:.1f}%" if stage2_confidence is not None else "N/A"
# # #         st.metric("Stage 2 Confidence", confidence_display)

# # #     st.markdown("<br>", unsafe_allow_html=True)

# # #     tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔬 Analysis", "📋 Data", "📥 Download"])

# # #     with tab1:
# # #         st.markdown("### 📊 Decision Comparison")
# # #         s1_dec   = st.session_state.get('stage1_decision', 'N/A')
# # #         s2_label = "✅ APPROVE" if final_decision == "APPROVE" else "❌ REJECT"
# # #         comparison_df = pd.DataFrame([
# # #             {'Stage': 'Stage 1 (Screening)', 'Decision': s1_dec,
# # #              'Risk Score': stage1_data.get('risk_score', 'N/A'), 'Tier': 'N/A',
# # #              'Note': 'APPROVE / REVIEW → proceed to Stage 2'},
# # #             {'Stage': 'Stage 2 — FINAL (CIBIL Deep)', 'Decision': s2_label,
# # #              'Risk Score': combined_risk_score, 'Tier': f"{stage2_tier} | {interest_range}",
# # #              'Note': 'Binding final decision'}
# # #         ])
# # #         st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# # #         st.markdown("### 🎯 Risk Tier Details")
# # #         tier_info = {
# # #             'P1': {'name': 'Premium  → APPROVED',  'color': '#10B981',
# # #                    'desc': 'Excellent credit profile — lowest interest rate band'},
# # #             'P2': {'name': 'Standard → APPROVED',  'color': '#3B82F6',
# # #                    'desc': 'Good credit profile — standard interest rate band'},
# # #             'P3': {'name': 'Subprime → REJECTED',  'color': '#F59E0B',
# # #                    'desc': 'Fair credit with elevated risk — application declined'},
# # #             'P4': {'name': 'High Risk → REJECTED', 'color': '#EF4444',
# # #                    'desc': 'High risk profile — application declined'},
# # #         }
# # #         if stage2_tier in tier_info:
# # #             tier_data = tier_info[stage2_tier]
# # #             st.markdown(f"""
# # #                 <div style="background: {tier_data['color']}; color: white; padding: 1rem; border-radius: 0.5rem;">
# # #                     <h3 style="margin: 0; color: white;">{stage2_tier}: {tier_data['name']}</h3>
# # #                     <p style="margin: 0.5rem 0;">Interest Rate: {interest_range}</p>
# # #                     <p style="margin: 0;">{tier_data['desc']}</p>
# # #                 </div>
# # #             """, unsafe_allow_html=True)
# # #         st.markdown("### 📝 Decision Reasoning")
# # #         st.info(stage2_result.get('reason', 'N/A'))

# # #     with tab2:
# # #         st.markdown("### 🔬 Detailed Analysis")
# # #         col1, col2 = st.columns(2)
# # #         with col1:
# # #             st.markdown("**Tier Probabilities**")
# # #             if 'tier_probabilities' in stage2_result:
# # #                 for tier, prob in stage2_result['tier_probabilities'].items():
# # #                     st.metric(tier, f"{prob:.1f}%")
# # #         with col2:
# # #             st.markdown("**Stage Scores**")
# # #             st.metric("Stage 1 Risk Score", stage1_data.get('risk_score', 'N/A'))
# # #             st.metric("Stage 2 Risk Score", stage2_result.get('stage2_risk_score', 'N/A'))
# # #             st.metric("Combined Score", combined_risk_score)
# # #         with st.expander("📋 Complete Stage 2 Result"):
# # #             st.json(stage2_result)

# # #     with tab3:
# # #         st.markdown("### 📋 Input Data")
# # #         col1, col2 = st.columns(2)
# # #         with col1:
# # #             with st.expander("Stage 1 Customer Data"):
# # #                 st.json(stage1_customer)
# # #         with col2:
# # #             with st.expander("Enhanced CIBIL Data"):
# # #                 st.json(enhanced_customer_data)

# # #     with tab4:
# # #         st.markdown("### 📥 Download Reports")
# # #         bureau_score = stage1_customer.get('bureau_score', 0)
# # #         dpd_90 = stage1_customer.get('dpd_90_count_6m', 0)
# # #         dpd_30 = stage1_customer.get('dpd_30_count_6m', 0)
# # #         foir = stage1_data.get('affordability_data', {}).get('foir_percentage', 0)
# # #         employment_type = stage1_customer.get('employment_type', 'Salaried')
# # #         employment_tenure = stage1_customer.get('employment_tenure_months', 0)
# # #         business_vintage = stage1_customer.get('business_vintage_years', 0)
# # #         ml_decision = stage1_data.get('decision', 'ERROR')
# # #         confidence = stage1_data.get('confidence', 0)

# # #         def _safe(v, default='N/A'):
# # #             return v if v is not None else default

# # #         report_data = {
# # #             'application_id': _safe(stage1_customer.get('application_id'), 'N/A'),
# # #             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# # #             'decision': _safe(stage1_data.get('decision'), 'N/A'),
# # #             'risk_score': _safe(stage1_data.get('risk_score'), 0),
# # #             'pd_percentage': _safe(stage1_data.get('pd_percentage'), 0),
# # #             'confidence': _safe(stage1_data.get('confidence'), 0),
# # #             'policy_checks': stage1_data.get('policy_checks', {}),
# # #             'affordability_data': stage1_data.get('affordability_data', {}),
# # #             'customer_data': stage1_customer,
# # #             'reason_codes': stage1_customer.get('reason_codes', []),
# # #             'pd_calculation_factors': {
# # #                 'bureau_score': bureau_score,
# # #                 'base_pd': bureau_score_to_pd(bureau_score),
# # #                 'dpd_90': dpd_90, 'dpd_30': dpd_30,
# # #                 'delinquency_multiplier': delinquency_to_pd_multiplier(dpd_90, dpd_30),
# # #                 'foir': foir,
# # #                 'foir_adjustment': foir_to_pd_adjustment(foir),
# # #                 'employment_adjustment': employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage),
# # #                 'ml_adjustment': ml_confidence_to_pd_adjustment(confidence, ml_decision),
# # #                 'final_pd': stage1_data.get('pd_percentage', 0)
# # #             },
# # #             'stage2_final_decision': _safe(final_decision, 'N/A'),
# # #             'stage2_tier': _safe(stage2_tier, 'N/A'),
# # #             'stage2_interest_range': _safe(interest_range, 'N/A'),
# # #             'stage2_combined_risk_score': _safe(combined_risk_score, 0),
# # #             'stage2_confidence': _safe(stage2_confidence, 0),
# # #             'stage2_reason': _safe(stage2_result.get('reason'), 'N/A'),
# # #             'stage2_tier_probabilities': stage2_result.get('tier_probabilities') or {},
# # #             'stage2_complete_analysis': stage2_result,
# # #             'stage1_data': stage1_data,
# # #             'enhanced_customer_data': enhanced_customer_data
# # #         }

# # #         if PDF_AVAILABLE and generate_audit_pdf is not None:
# # #             try:
# # #                 pdf_buffer = generate_audit_pdf(report_data)
# # #                 st.download_button(
# # #                     "📥 Download PDF Report",
# # #                     data=pdf_buffer,
# # #                     file_name=f"stage2_report_{stage1_customer.get('application_id', 'unknown')}.pdf",
# # #                     mime="application/pdf",
# # #                     use_container_width=True
# # #                 )
# # #             except Exception as e:
# # #                 st.error(f"PDF generation failed: {str(e)}")
# # #         else:
# # #             st.warning("PDF generation is not available. Please install the required PDF generator module.")

# # #     st.markdown("---")
# # #     col1, col2, col3 = st.columns(3)
# # #     with col1:
# # #         if st.button("🔄 New Assessment", key="new_assessment_stage2", use_container_width=True):
# # #             st.session_state.stage1_complete = False
# # #             st.session_state.stage1_decision = None
# # #             st.session_state.stage1_data = None
# # #             st.session_state.current_customer_data = None
# # #             st.session_state.page_navigation = "👤 Assessment"
# # #             st.rerun()
# # #     with col2:
# # #         if st.button("← Back to Stage 1", key="back_to_stage1", use_container_width=True):
# # #             st.session_state.page_navigation = "👤 Assessment"
# # #             st.rerun()
# # #     with col3:
# # #         if st.button("🏠 Home", key="home_stage2", use_container_width=True):
# # #             st.session_state.page_navigation = "🏠 Home"
# # #             st.rerun()

# # # # =============================================================================
# # # # SIDEBAR
# # # # =============================================================================
# # # with st.sidebar:
# # #     st.markdown("# 🏦 Credit Risk Engine")
# # #     st.markdown("---")

# # #     navigation_options = ["🏠 Home", "👤 Assessment", "📊 Batch Process", "📈 Model Info", "ℹ️ About"]

# # #     if (st.session_state.stage1_complete and
# # #             st.session_state.stage1_decision in ['APPROVE', 'REVIEW']):
# # #         navigation_options.insert(2, "🔬 Stage 2 Analysis")
# # #         st.success(f"✅ Stage 1: {st.session_state.stage1_decision}")
# # #         st.info("🔬 Stage 2 Analysis unlocked!")
# # #     elif st.session_state.stage1_complete:
# # #         st.warning(f"⚠️ Stage 1: {st.session_state.stage1_decision}")
# # #         st.caption("Stage 2 only for APPROVE/REVIEW")

# # #     page = st.radio(
# # #         "**Navigation**",
# # #         navigation_options,
# # #         label_visibility="collapsed",
# # #         key="page_navigation"
# # #     )

# # #     st.markdown("---")

# # #     stage2_indicator = '✅ Active' if STAGE2_AVAILABLE and is_stage2_available() else '❌ Inactive'
# # #     ocr_indicator = '✅ Ready' if OCR_AVAILABLE else '❌ Not Installed'
# # #     if not OCR_AVAILABLE and OCR_ERROR_MSG:
# # #         ocr_indicator += ' ⚠️'
# # #     pdf_indicator = '✅ Ready' if PDF_AVAILABLE else '❌ Not Installed'

# # #     st.markdown(f"""
# # #     <div class="info-card">
# # #         <div class="info-card-title">System Status</div>
# # #         <div class="info-card-content">
# # #             <div class="data-row"><span class="data-label">Model</span><span class="data-value">✅ Loaded</span></div>
# # #             <div class="data-row"><span class="data-label">Version</span><span class="data-value">8.3</span></div>
# # #             <div class="data-row"><span class="data-label">Stage 2</span><span class="data-value">{stage2_indicator}</span></div>
# # #             <div class="data-row"><span class="data-label">OCR</span><span class="data-value">{ocr_indicator}</span></div>
# # #             <div class="data-row"><span class="data-label">PDF Gen</span><span class="data-value">{pdf_indicator}</span></div>
# # #             <div class="data-row"><span class="data-label">Features</span><span class="data-value">{len(TOP_FEATURES)}</span></div>
# # #         </div>
# # #     </div>
# # #     """, unsafe_allow_html=True)

# # #     with st.expander("🎯 **Top Features**"):
# # #         for i, feat in enumerate(TOP_FEATURES[:5], 1):
# # #             st.markdown(f"`{i}.` {feat}")

# # #     if st.session_state.stage1_complete:
# # #         st.markdown("---")
# # #         st.markdown("### 🚀 Quick Actions")
# # #         if st.button("🔄 New Assessment", key="new_assessment_sidebar", use_container_width=True):
# # #             st.session_state.stage1_complete = False
# # #             st.session_state.stage1_decision = None
# # #             st.session_state.stage1_data = None
# # #             st.session_state.current_customer_data = None
# # #             st.session_state.extracted_cibil_data = None
# # #             st.rerun()

# # # # =============================================================================
# # # # PAGE ROUTING
# # # # =============================================================================

# # # if page == "🏠 Home":
# # #     st.markdown('<p class="main-header">Credit Risk Engine</p>', unsafe_allow_html=True)
# # #     st.markdown("""
# # #         <div class="info-box">
# # #             <h3 style="margin-top: 0;">🎯 AI-Powered Lending Decisions</h3>
# # #             <p style="margin-bottom: 0;">Comprehensive credit risk evaluation combining hard policy rules,
# # #             machine learning models, and affordability analysis for accurate lending decisions.</p>
# # #         </div>
# # #     """, unsafe_allow_html=True)
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     col1, col2, col3 = st.columns(3)
# # #     with col1:
# # #         st.markdown("""
# # #             <div class="info-card"><div class="info-card-title"><span class="icon">🛡️</span><span>Policy Gates</span></div>
# # #             <div class="info-card-content"><ul><li>Age & KYC verification</li><li>Employment stability</li>
# # #             <li>Minimum income checks</li><li>Credit bureau thresholds</li><li>Bankruptcy & fraud detection</li></ul></div></div>
# # #         """, unsafe_allow_html=True)
# # #     with col2:
# # #         st.markdown("""
# # #             <div class="info-card"><div class="info-card-title"><span class="icon">🤖</span><span>ML Assessment</span></div>
# # #             <div class="info-card-content"><ul><li>Random Forest classifier</li><li>60K+ training samples</li>
# # #             <li>Confidence scoring</li><li>Multi-class prediction</li><li>Feature importance</li></ul></div></div>
# # #         """, unsafe_allow_html=True)
# # #     with col3:
# # #         st.markdown("""
# # #             <div class="info-card"><div class="info-card-title"><span class="icon">💰</span><span>Affordability</span></div>
# # #             <div class="info-card-content"><ul><li>EMI calculation</li><li>FOIR analysis (max 50%)</li>
# # #             <li>Net disposable income</li><li>Debt burden assessment</li><li>Affordability scoring</li></ul></div></div>
# # #         """, unsafe_allow_html=True)
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     col1, col2, col3, col4 = st.columns(4)
# # #     with col1: st.metric("🎯 Accuracy", "85%", "+2%")
# # #     with col2: st.metric("⚡ Avg Response", "1.2s", "-0.3s")
# # #     with col3: st.metric("📊 Features", len(TOP_FEATURES))
# # #     with col4: st.metric("🔄 Version", "8.3", "Latest")
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     st.markdown("""
# # #         <div class="warning-box">
# # #             <strong>🆕 New in Version 8.3:</strong><br>
# # #             • Fixed Mixed Numeric Types Error<br>
# # #             • Fixed Missing Submit Button<br>
# # #             • Dependents field properly integrated<br>
# # #             • PDF auto-fill from CIBIL report<br>
# # #             • Industry-Standard PD Methodology<br>
# # #             • Professional UI/UX Enhancements
# # #         </div>
# # #     """, unsafe_allow_html=True)

# # # elif page == "👤 Assessment":
# # #     st.markdown('<p class="main-header">Credit Assessment</p>', unsafe_allow_html=True)

# # #     pdf_just_extracted = st.session_state.get('pdf_just_extracted', False)

# # #     with st.expander("📄 Upload CIBIL PDF to auto‑fill bureau fields",
# # #                      expanded=pdf_just_extracted or not st.session_state.get('pdf_bureau_score')):

# # #         if pdf_just_extracted:
# # #             ex = st.session_state.get('_last_extraction', {})
# # #             st.success("✅ CIBIL data extracted — form fields below have been updated automatically.")
# # #             c1, c2, c3, c4 = st.columns(4)
# # #             c1.metric("Credit Score",    ex.get('Credit_Score', '—'))
# # #             c2.metric("Monthly Income",  f"₹{ex.get('NETMONTHLYINCOME', 0):,}")
# # #             c3.metric("DPD 90+ Count",   ex.get('dpd_90_count_6m', 0))
# # #             c4.metric("CC Utilization",  f"{ex.get('CC_utilization', 0)*100:.0f}%")
# # #             c1, c2, c3, c4 = st.columns(4)
# # #             c1.metric("DPD 30+ Count",  ex.get('num_times_30p_dpd', 0))
# # #             c2.metric("Inquiries (3M)", ex.get('enq_L3m', 0))
# # #             c3.metric("Active Accounts", ex.get('num_std', 0))
# # #             c4.metric("Written-Off",    ex.get('written_off_count', 0))
# # #             if ex.get('written_off_count', 0) > 0 or ex.get('settled_count', 0) > 0:
# # #                 st.warning(f"⚠️ Severe negatives detected: "
# # #                            f"{ex.get('written_off_count', 0)} written-off, "
# # #                            f"{ex.get('settled_count', 0)} settled accounts. "
# # #                            f"Score overridden to {ex.get('Credit_Score', '?')}.")
# # #             # Show application context – FIXED CRASH
# # #             if st.session_state.get('stage1_complete') and st.session_state.get('current_customer_data'):
# # #                 app_id_s1 = st.session_state.current_customer_data.get('application_id', 'Pending submission')
# # #                 st.markdown(f"""
# # #                     <div style="background:#1e3a5f;color:white;padding:0.5rem 1rem;border-radius:0.4rem;margin-bottom:0.5rem;font-size:0.9rem;">
# # #                         <strong>📋 Application ID:</strong> {app_id_s1}
# # #                     </div>
# # #                 """, unsafe_allow_html=True)
# # #             else:
# # #                 st.markdown("No active assessment. Please submit the form below.")
# # #             if st.toggle("📋 Show full extracted JSON"):
# # #                 st.json({k: v for k, v in ex.items() if k != 'raw_text'})
# # #             st.markdown("---")
# # #             if st.button("🔄 Upload a different PDF", key="reset_pdf"):
# # #                 st.session_state.pdf_just_extracted = False
# # #                 st.session_state.pop('_last_extraction', None)
# # #                 st.rerun()
# # #         else:
# # #             st.markdown('<div class="info-box">💡 Complete the form below or upload a CIBIL PDF to auto‑fill bureau data.</div>', unsafe_allow_html=True)
# # #             uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="assessment_pdf")
# # #             if uploaded_pdf is not None:
# # #                 st.info(f"📄 File ready: **{uploaded_pdf.name}** ({uploaded_pdf.size/1024:.1f} KB)")
# # #                 if st.button("🔍 Extract & Auto-fill Form", key="extract_assessment", type="primary", use_container_width=True):
# # #                     with st.spinner("🔄 Running OCR on CIBIL PDF — this takes 10-30 seconds..."):
# # #                         extraction_result = extract_cibil_from_pdf(uploaded_pdf)
# # #                     if extraction_result.get('success', False):
# # #                         st.session_state.pdf_age               = int(extraction_result.get('AGE', 35))
# # #                         st.session_state.pdf_employment_type   = 'Salaried'
# # #                         st.session_state.pdf_kyc               = True
# # #                         st.session_state.pdf_bankruptcy        = False
# # #                         st.session_state.pdf_fraud             = False
# # #                         st.session_state.pdf_bureau_score      = int(extraction_result.get('Credit_Score', 720))
# # #                         st.session_state.pdf_dpd_90            = int(extraction_result.get('dpd_90_count_6m', 0))
# # #                         st.session_state.pdf_dpd_30            = int(extraction_result.get('num_times_30p_dpd', 0))
# # #                         st.session_state.pdf_credit_util       = int(float(extraction_result.get('CC_utilization', 0.35)) * 100)
# # #                         st.session_state.pdf_inquiries         = int(extraction_result.get('enq_L3m', 2))
# # #                         st.session_state.pdf_active_loans      = int(extraction_result.get('num_std', 1))
# # #                         st.session_state.pdf_existing_emi      = int(extraction_result.get('existing_emi', 15000))
# # #                         st.session_state.pdf_monthly_income    = int(extraction_result.get('NETMONTHLYINCOME', 50000))
# # #                         st.session_state.pdf_annual_income     = int(extraction_result.get('NETMONTHLYINCOME', 50000)) * 12
# # #                         st.session_state.pdf_net_surplus       = int(extraction_result.get('net_surplus', 20000))
# # #                         st.session_state.pdf_salary_stability  = 'STABLE'
# # #                         st.session_state.pdf_loan_amount       = int(extraction_result.get('loan_amount', 180000))
# # #                         st.session_state.pdf_loan_tenure       = int(extraction_result.get('loan_tenure', 24))
# # #                         st.session_state.pdf_interest_rate     = float(extraction_result.get('interest_rate', 10.5))
# # #                         st.session_state.pdf_amt_annuity       = int(extraction_result.get('amt_annuity', 8500))
# # #                         st.session_state.pdf_employment_tenure = int(extraction_result.get('Time_With_Curr_Empr', 24))
# # #                         st.session_state.pdf_business_vintage  = int(extraction_result.get('business_vintage', 3))
# # #                         st.session_state.pdf_dependents        = int(extraction_result.get('dependents', 2))
# # #                         st.session_state.pdf_just_extracted    = True
# # #                         st.session_state._last_extraction      = extraction_result
# # #                         st.rerun()
# # #                     else:
# # #                         st.error(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")
# # #                         st.info("Tip: Make sure Tesseract and Poppler are installed and paths are set correctly.")

# # #     with st.form("assessment_form"):
# # #         st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
# # #         col1, col2, col3 = st.columns(3)
# # #         with col1:
# # #             age = st.number_input(
# # #                 "Age", 24, 70,
# # #                 value=int(st.session_state.get('pdf_age', 35)),
# # #                 help="Customer's age in years (Minimum: 24, Maximum: 70)"
# # #             )
# # #             employment_type = st.selectbox(
# # #                 "Employment Type",
# # #                 ['Salaried', 'Self-Employed', 'Business'],
# # #                 index=['Salaried', 'Self-Employed', 'Business'].index(
# # #                     st.session_state.get('pdf_employment_type', 'Salaried')
# # #                 )
# # #             )
# # #         with col2:
# # #             dependents = st.number_input(
# # #                 "Number of Dependents", 0, 20,
# # #                 value=int(st.session_state.get('pdf_dependents', 2)),
# # #                 help="1-5: Approve eligible | >5: Review required"
# # #             )
# # #             kyc_verified = st.selectbox(
# # #                 "KYC Verified", ['Yes', 'No'],
# # #                 index=0 if st.session_state.get('pdf_kyc', True) else 1
# # #             ) == 'Yes'
# # #         with col3:
# # #             bankruptcy_flag = st.selectbox(
# # #                 "Bankruptcy Flag", ['No', 'Yes'],
# # #                 index=0 if not st.session_state.get('pdf_bankruptcy', False) else 1
# # #             ) == 'Yes'
# # #             fraud_flag = st.selectbox(
# # #                 "Fraud Flag", ['No', 'Yes'],
# # #                 index=0 if not st.session_state.get('pdf_fraud', False) else 1
# # #             ) == 'Yes'
# # #             if employment_type == 'Salaried':
# # #                 employment_tenure = st.number_input(
# # #                     "Employment Tenure (months)", 0, 600,
# # #                     value=int(st.session_state.get('pdf_employment_tenure', 24))
# # #                 )
# # #                 business_vintage = 0
# # #             else:
# # #                 business_vintage = st.number_input(
# # #                     "Business Vintage (years)", 0, 50,
# # #                     value=int(st.session_state.get('pdf_business_vintage', 3))
# # #                 )
# # #                 employment_tenure = 0

# # #         st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
# # #         col1, col2, col3 = st.columns(3)
# # #         with col1:
# # #             bureau_score = st.number_input(
# # #                 "Bureau Score", 300, 900,
# # #                 value=int(st.session_state.get('pdf_bureau_score', 720)), step=10
# # #             )
# # #             dpd_90_6m = st.number_input(
# # #                 "DPD 90+ (Last 6M)", 0, 20,
# # #                 value=int(st.session_state.get('pdf_dpd_90', 0))
# # #             )
# # #             dpd_30_6m = st.number_input(
# # #                 "DPD 30+ (Last 6M)", 0, 20,
# # #                 value=int(st.session_state.get('pdf_dpd_30', 0))
# # #             )
# # #         with col2:
# # #             credit_utilization = st.number_input(
# # #                 "Credit Utilization (%)", 0, 100,
# # #                 value=int(st.session_state.get('pdf_credit_util', 30))
# # #             )
# # #             recent_inquiries = st.number_input(
# # #                 "Recent Inquiries (3M)", 0, 20,
# # #                 value=int(st.session_state.get('pdf_inquiries', 2))
# # #             )
# # #         with col3:
# # #             active_loans = st.number_input(
# # #                 "Active Loans", 0, 10,
# # #                 value=int(st.session_state.get('pdf_active_loans', 1))
# # #             )
# # #             existing_emi = st.number_input(
# # #                 "Existing Total EMI (₹)", 0, 200000,
# # #                 value=int(st.session_state.get('pdf_existing_emi', 15000)), step=1000
# # #             )

# # #         st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
# # #         col1, col2, col3, col4 = st.columns(4)
# # #         with col1:
# # #             avg_salary = st.number_input(
# # #                 "Monthly Income (₹)", 0, 1000000,
# # #                 value=int(st.session_state.get('pdf_monthly_income', 50000)), step=5000
# # #             )
# # #             amt_income = st.number_input(
# # #                 "Annual Income (₹)", 0, 10000000,
# # #                 value=int(st.session_state.get('pdf_annual_income', 600000)), step=10000
# # #             )
# # #         with col2:
# # #             net_surplus = st.number_input(
# # #                 "Net Cash Surplus (₹)", -100000, 500000,
# # #                 value=int(st.session_state.get('pdf_net_surplus', 20000)), step=5000
# # #             )
# # #             salary_stability = st.selectbox(
# # #                 "Salary Stability",
# # #                 ['STABLE', 'MODERATE', 'UNSTABLE'],
# # #                 index=['STABLE', 'MODERATE', 'UNSTABLE'].index(
# # #                     st.session_state.get('pdf_salary_stability', 'STABLE')
# # #                 )
# # #             )
# # #         with col3:
# # #             loan_amount = st.number_input(
# # #                 "Loan Amount (₹)", 0, 5000000,
# # #                 value=int(st.session_state.get('pdf_loan_amount', 180000)), step=10000
# # #             )
# # #             loan_tenure = st.number_input(
# # #                 "Tenure (months)", 3, 360,
# # #                 value=int(st.session_state.get('pdf_loan_tenure', 24))
# # #             )
# # #         with col4:
# # #             interest_rate = st.number_input(
# # #                 "Interest Rate (%)", 8.0, 20.0,
# # #                 value=float(st.session_state.get('pdf_interest_rate', 10.5)), step=0.5
# # #             )
# # #             amt_annuity = st.number_input(
# # #                 "Requested EMI (₹)", 0, 200000,
# # #                 value=int(st.session_state.get('pdf_amt_annuity', 8500)), step=500
# # #             )

# # #         # ===== New: Additional Credit Behaviour Fields =====
# # #         st.markdown('<p class="section-header">📋 Additional Credit Behaviour</p>', unsafe_allow_html=True)
# # #         col1, col2, col3 = st.columns(3)
# # #         with col1:
# # #             payment_discipline = st.selectbox(
# # #                 "Payment Discipline", ['GOOD', 'MODERATE', 'POOR'], index=0,
# # #                 help="Overall payment behavior pattern"
# # #             )
# # #             liquidity_flag = st.selectbox(
# # #                 "Liquidity", ['LOW', 'ADEQUATE', 'MODERATE'], index=0,
# # #                 help="Cash liquidity position"
# # #             )
# # #         with col2:
# # #             cashflow_health = st.selectbox(
# # #                 "Cashflow Health", ['MODERATE', 'HEALTHY', 'STRESSED'], index=0,
# # #                 help="Overall cashflow health assessment"
# # #             )
# # #             bureau_risk_flag = st.selectbox(
# # #                 "Bureau Risk", ['LOW', 'MEDIUM', 'HIGH'], index=0,
# # #                 help="External bureau risk rating"
# # #             )
# # #         with col3:
# # #             inward_bounce_count = st.number_input(
# # #                 "Inward Bounce Count (3M)", 0, 10, 0,
# # #                 help="Number of bounced inward cheques last 3 months"
# # #             )
# # #             salary_missing_months = st.number_input(
# # #                 "Missing Salary Months (6M)", 0, 6, 0,
# # #                 help="Months without salary credit"
# # #             )

# # #         st.markdown("<br>", unsafe_allow_html=True)
# # #         submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)

# # #     if submitted:
# # #         timestamp = datetime.now()
# # #         app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
# # #         customer_data = {
# # #             'age': age,
# # #             'employment_type': employment_type,
# # #             'dependents': dependents,
# # #             'kyc_verified': kyc_verified,
# # #             'bankruptcy_flag': bankruptcy_flag,
# # #             'fraud_flag': fraud_flag,
# # #             'employment_tenure_months': employment_tenure,
# # #             'business_vintage_years': business_vintage,
# # #             'bureau_score': bureau_score,
# # #             'dpd_90_count_6m': dpd_90_6m,
# # #             'dpd_30_count_6m': dpd_30_6m,
# # #             'credit_utilization_pct': credit_utilization,
# # #             'max_utilization': credit_utilization,
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
# # #             'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
# # #             # New fields:
# # #             'payment_discipline_flag': payment_discipline,
# # #             'liquidity_flag': liquidity_flag,
# # #             'cashflow_health': cashflow_health,
# # #             'bureau_risk_flag': bureau_risk_flag,
# # #             'inward_bounce_count_3m': inward_bounce_count,
# # #             'salary_missing_months': salary_missing_months,
# # #         }

# # #         with st.spinner("🔄 Processing Stage 1 assessment..."):
# # #             decision_data = make_hybrid_decision_enhanced(customer_data)

# # #         reasons = generate_reason_codes(
# # #             decision=decision_data.get('decision', 'ERROR'),
# # #             customer_data=customer_data,
# # #             affordability_data=decision_data.get('affordability_data', {}),
# # #             policy_checks=decision_data.get('policy_checks', {})
# # #         )
# # #         customer_data['reason_codes'] = reasons

# # #         st.session_state.stage1_complete = True
# # #         st.session_state.stage1_decision = decision_data.get('decision', 'ERROR')
# # #         st.session_state.stage1_data = decision_data
# # #         st.session_state.current_customer_data = customer_data

# # #         for key in list(st.session_state.keys()):
# # #             if key.startswith('pdf_') or key in ('_last_extraction',):
# # #                 del st.session_state[key]

# # #         tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])

# # #         with tab1:
# # #             st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
# # #             col1, col2 = st.columns(2)
# # #             with col1:
# # #                 render_info_card("👤 Identity", "👤",
# # #                                  {"Age": age, "Employment": employment_type, "Dependents": dependents,
# # #                                   "KYC Status": "Verified" if kyc_verified else "Not Verified",
# # #                                   "Tenure": f"{employment_tenure} months" if employment_type == 'Salaried' else f"{business_vintage} years"})
# # #                 render_info_card("💰 Financial", "💰",
# # #                                  {"Monthly Income": f"₹{avg_salary:,}", "Annual Income": f"₹{amt_income:,}",
# # #                                   "Net Surplus": f"₹{net_surplus:,}", "Stability": salary_stability})
# # #             with col2:
# # #                 render_info_card("🏦 Credit Bureau", "🏦",
# # #                                  {"Bureau Score": bureau_score, "DPD 90+": dpd_90_6m, "DPD 30+": dpd_30_6m,
# # #                                   "Utilization": f"{credit_utilization}%", "Recent Inquiries": recent_inquiries,
# # #                                   "Existing EMI": f"₹{existing_emi:,}"})
# # #                 render_info_card("📋 Loan Request", "📋",
# # #                                  {"Amount": f"₹{loan_amount:,}", "Tenure": f"{loan_tenure} months",
# # #                                   "Interest Rate": f"{interest_rate}%", "Requested EMI": f"₹{amt_annuity:,}"})

# # #         with tab2:
# # #             st.markdown('<p class="section-header">Decision Summary</p>', unsafe_allow_html=True)
# # #             render_decision_header(decision_data, customer_data)
# # #             st.markdown("<br>", unsafe_allow_html=True)

# # #             final_decision = decision_data.get('decision', 'ERROR')

# # #             if final_decision in ['APPROVE', 'REVIEW']:
# # #                 st.markdown("---")
# # #                 st.markdown("""
# # #                     <div class="info-box" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white; text-align: center;">
# # #                         <h3 style="margin: 0; color: white;">✅ Eligible for Stage 2 Deep Dive</h3>
# # #                         <p style="margin: 0.5rem 0 0 0;">Choose an input method to proceed:</p>
# # #                     </div>
# # #                 """, unsafe_allow_html=True)
# # #                 col1, col2, col3 = st.columns(3)
# # #                 with col1:
# # #                     if st.button("📝 Manual Entry", key="stage2_manual_btn", use_container_width=True, type="primary"):
# # #                         st.session_state.stage2_selected_tab = "Manual Entry"
# # #                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
# # #                         st.rerun()
# # #                 with col2:
# # #                     if st.button("📄 PDF Upload", key="stage2_pdf_btn", use_container_width=True, type="primary"):
# # #                         st.session_state.stage2_selected_tab = "PDF Upload"
# # #                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
# # #                         st.rerun()
# # #                 with col3:
# # #                     if st.button("📊 Batch Analysis", key="stage2_batch_btn", use_container_width=True, type="primary"):
# # #                         st.session_state.stage2_selected_tab = "Batch Analysis"
# # #                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
# # #                         st.rerun()
# # #             elif final_decision == 'REJECT':
# # #                 st.markdown("---")
# # #                 st.markdown("""
# # #                     <div class="warning-box" style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; text-align: center;">
# # #                         <h3 style="margin: 0; color: white;">❌ Stage 2 Not Available</h3>
# # #                         <p style="margin: 0.5rem 0 0 0;">Application rejected. Stage 2 only for APPROVE/REVIEW.</p>
# # #                     </div>
# # #                 """, unsafe_allow_html=True)

# # #             st.markdown("<br>", unsafe_allow_html=True)
# # #             affordability = decision_data.get('affordability_data', {})
# # #             foir = affordability.get('foir_percentage', 0)
# # #             total_emi = affordability.get('total_emi', 0)
# # #             net_disp = affordability.get('net_disposable', 0)

# # #             col1, col2, col3 = st.columns(3)
# # #             with col1:
# # #                 render_info_card("Identity & Eligibility", "👤",
# # #                                  {f"Age: {age}": "", f"Employment: {employment_type}": "",
# # #                                   f"Dependents: {dependents}": "",
# # #                                   f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""},
# # #                                  {f"Age: {age}": "pass" if 24 <= age <= 70 else "fail",
# # #                                   f"Employment: {employment_type}": "pass",
# # #                                   f"Dependents: {dependents}": "pass" if dependents <= 5 else "warning",
# # #                                   f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_verified else "fail"})
# # #             with col2:
# # #                 bureau_pass = bureau_score >= 550
# # #                 dpd_pass = dpd_90_6m == 0
# # #                 render_info_card("Credit Bureau", "🏦",
# # #                                  {f"Bureau Score: {bureau_score}": "", f"DPD 90+: {dpd_90_6m}": "",
# # #                                   f"Utilization: {credit_utilization}%": ""},
# # #                                  {f"Bureau Score: {bureau_score}": "pass" if bureau_pass else "fail",
# # #                                   f"DPD 90+: {dpd_90_6m}": "pass" if dpd_pass else "fail",
# # #                                   f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"})
# # #             with col3:
# # #                 render_info_card("Affordability", "💰",
# # #                                  {f"Monthly Income: ₹{avg_salary:,}": "", f"FOIR: {foir:.1f}%": "",
# # #                                   f"Total EMI: ₹{total_emi:,}": "", f"Net Disposable: ₹{net_disp:,}": ""},
# # #                                  {f"Monthly Income: ₹{avg_salary:,}": "pass",
# # #                                   f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
# # #                                   f"Total EMI: ₹{total_emi:,}": "pass",
# # #                                   f"Net Disposable: ₹{net_disp:,}": "pass" if net_disp >= 10000 else "warning"})

# # #             st.markdown("<br>", unsafe_allow_html=True)
# # #             render_reason_codes(reasons)
# # #             st.markdown("<br>", unsafe_allow_html=True)

# # #             col1, col2, col3 = st.columns([1, 1, 2])
# # #             with col1:
# # #                 if PDF_AVAILABLE and generate_decision_pdf is not None:
# # #                     try:
# # #                         pdf_buffer = generate_decision_pdf(
# # #                             decision_data=decision_data, customer_data=customer_data,
# # #                             affordability_data=decision_data.get('affordability_data', {}), reasons=reasons)
# # #                         st.download_button("📥 Decision Report (PDF)", data=pdf_buffer,
# # #                                            file_name=f"credit_decision_{app_id}.pdf", mime="application/pdf",
# # #                                            use_container_width=True)
# # #                     except Exception as e:
# # #                         st.error(f"Error generating PDF: {str(e)}")
# # #                 else:
# # #                     st.warning("PDF generation not available.")
# # #             with col2:
# # #                 if st.button("🔄 Re-Evaluate", key="reevaluate_btn", use_container_width=True):
# # #                     st.rerun()

# # #         with tab3:
# # #             st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
# # #             col1, col2 = st.columns(2)
# # #             with col1:
# # #                 fig1 = create_modern_gauge(decision_data.get('confidence', 0), "Model Confidence")
# # #                 st.plotly_chart(fig1, use_container_width=True)
# # #             with col2:
# # #                 final_decision_tab3 = decision_data.get('decision', 'ERROR')
# # #                 # Always show real probabilities, never hardcode
# # #                 class_probs = decision_data.get('class_probs', {"APPROVE": 0, "REVIEW": 0, "REJECT": 0})
# # #                 fig2 = create_modern_bar_chart(class_probs)
# # #                 st.plotly_chart(fig2, use_container_width=True)

# # #             st.markdown("<br>", unsafe_allow_html=True)
# # #             st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
# # #             policy_df = pd.DataFrame([{'Check': k, 'Result': v} for k, v in decision_data.get('policy_checks', {}).items()])
# # #             st.dataframe(policy_df, use_container_width=True, hide_index=True)

# # #             st.markdown('<p class="section-header">PD Calculation Breakdown</p>', unsafe_allow_html=True)
# # #             pd_factors_display = {
# # #                 'Bureau Score': f"{bureau_score} → Base PD: {bureau_score_to_pd(bureau_score):.1f}%",
# # #                 'Delinquency': f"DPD 90+: {dpd_90_6m}, DPD 30+: {dpd_30_6m} → Multiplier: {delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x",
# # #                 'FOIR Impact': f"{foir:.1f}% → Adjustment: {foir_to_pd_adjustment(foir):.1f}%",
# # #                 'Employment Stability': f"{employment_type}, {employment_tenure if employment_type == 'Salaried' else business_vintage}{' months' if employment_type == 'Salaried' else ' years'} → Adjustment: {employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage):.1f}%",
# # #                 'ML Confidence': f"{decision_data.get('confidence', 0):.1f}% → Adjustment: {ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')):.1f}%",
# # #                 'Final PD': f"{decision_data.get('pd_percentage', 0)}%"
# # #             }
# # #             for factor, value in pd_factors_display.items():
# # #                 st.markdown(f"**{factor}:** {value}")

# # #         with tab4:
# # #             st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
# # #             audit_log_raw = {
# # #                 'application_id': app_id,
# # #                 'timestamp': timestamp.isoformat(),
# # #                 'decision': decision_data.get('decision', 'ERROR'),
# # #                 'risk_score': decision_data.get('risk_score', 0),
# # #                 'pd_percentage': decision_data.get('pd_percentage', 0),
# # #                 'confidence': round(decision_data.get('confidence', 0), 2),
# # #                 'model_version': '8.3',
# # #                 'reason_codes': reasons,
# # #                 'policy_checks': decision_data.get('policy_checks', {}),
# # #                 'affordability': decision_data.get('affordability_data', {}),
# # #                 'customer_data': {k: v for k, v in customer_data.items() if k not in ['application_id', 'timestamp', 'reason_codes']},
# # #                 'pd_calculation_factors': {
# # #                     'bureau_score': bureau_score,
# # #                     'base_pd': bureau_score_to_pd(bureau_score),
# # #                     'dpd_90': dpd_90_6m, 'dpd_30': dpd_30_6m,
# # #                     'delinquency_multiplier': delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m),
# # #                     'foir': foir,
# # #                     'foir_adjustment': foir_to_pd_adjustment(foir),
# # #                     'employment_adjustment': employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage),
# # #                     'ml_adjustment': ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')),
# # #                     'final_pd': decision_data.get('pd_percentage', 0)
# # #                 }
# # #             }
# # #             audit_log = sanitize_for_json(audit_log_raw)

# # #             with st.expander("📋 View Audit Log (JSON)"):
# # #                 st.json(audit_log)

# # #             col1, col2 = st.columns(2)
# # #             with col1:
# # #                 if PDF_AVAILABLE and generate_audit_pdf is not None:
# # #                     try:
# # #                         audit_pdf_buffer = generate_audit_pdf(audit_log)
# # #                         st.download_button("📥 Download Audit Trail (PDF)",
# # #                                            data=audit_pdf_buffer,
# # #                                            file_name=f"audit_trail_{app_id}.pdf",
# # #                                            mime="application/pdf",
# # #                                            use_container_width=True)
# # #                     except Exception as e:
# # #                         st.error(f"Error generating audit PDF: {str(e)}")
# # #                 else:
# # #                     st.warning("Audit PDF generation is not available.")
# # #             with col2:
# # #                 audit_json = json.dumps(audit_log, indent=2)
# # #                 st.download_button("📥 Download Audit Log (JSON)",
# # #                                    data=audit_json,
# # #                                    file_name=f"audit_{app_id}.json",
# # #                                    mime="application/json",
# # #                                    use_container_width=True)

# # #             st.markdown('<p class="section-header">PD Calculation Summary</p>', unsafe_allow_html=True)
# # #             pd_table = pd.DataFrame([
# # #                 {"Factor": "Bureau Score", "Value": f"{bureau_score}", "Impact": f"{bureau_score_to_pd(bureau_score):.1f}% base PD"},
# # #                 {"Factor": "Delinquency (DPD 90+)", "Value": f"{dpd_90_6m} times", "Impact": f"{delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x multiplier"},
# # #                 {"Factor": "FOIR", "Value": f"{foir:.1f}%", "Impact": f"{foir_to_pd_adjustment(foir):.1f}% adjustment"},
# # #                 {"Factor": "Employment Stability",
# # #                  "Value": f"{employment_type} ({employment_tenure if employment_type == 'Salaried' else business_vintage}{' months' if employment_type == 'Salaried' else ' years'})",
# # #                  "Impact": f"{employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage):.1f}% adjustment"},
# # #                 {"Factor": "ML Decision Confidence",
# # #                  "Value": f"{decision_data.get('confidence', 0):.1f}% ({decision_data.get('decision', 'ERROR')})",
# # #                  "Impact": f"{ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')):.1f}% adjustment"},
# # #                 {"Factor": "Final PD", "Value": f"{decision_data.get('pd_percentage', 0)}%", "Impact": "Industry-standard calculation"}
# # #             ])
# # #             st.dataframe(pd_table, use_container_width=True, hide_index=True)

# # # elif page == "🔬 Stage 2 Analysis":
# # #     st.markdown('<p class="main-header">Stage 2: CIBIL Deep Dive</p>', unsafe_allow_html=True)

# # #     if not st.session_state.get('stage1_complete', False):
# # #         st.error("❌ You must complete Stage 1 Assessment first!")
# # #         st.info("Please go to the 👤 Assessment page and submit an application.")
# # #         if st.button("← Go to Assessment", use_container_width=True):
# # #             st.session_state.page_navigation = "👤 Assessment"
# # #             st.rerun()
# # #         st.stop()

# # #     if st.session_state.get('stage1_decision') not in ['APPROVE', 'REVIEW']:
# # #         st.error("❌ Stage 2 is only available for APPROVED or REVIEW applications!")
# # #         st.warning(f"Your Stage 1 decision: {st.session_state.get('stage1_decision', 'Unknown')}")
# # #         if st.button("← Go Back", use_container_width=True):
# # #             st.session_state.page_navigation = "👤 Assessment"
# # #             st.rerun()
# # #         st.stop()

# # #     if not (STAGE2_AVAILABLE and is_stage2_available()):
# # #         st.error("❌ Stage 2 model not available!")
# # #         st.info("Please ensure `stage2_cibil_model.pkl` is in the project directory.")
# # #         if st.button("← Go Back", use_container_width=True):
# # #             st.session_state.page_navigation = "👤 Assessment"
# # #             st.rerun()
# # #         st.stop()

# # #     stage1_data = st.session_state.get('stage1_data', {})
# # #     stage1_customer = st.session_state.get('current_customer_data', {})

# # #     st.markdown(f"""
# # #         <div class="info-box" style="background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); color: white;">
# # #             <h3 style="margin: 0; color: white;">📊 Stage 1 Results</h3>
# # #             <p style="margin: 0.5rem 0 0 0;">
# # #                 <strong>Decision:</strong> {st.session_state.get('stage1_decision', 'N/A')} |
# # #                 <strong>Risk Score:</strong> {stage1_data.get('risk_score', 'N/A')} |
# # #                 <strong>Application ID:</strong> {stage1_customer.get('application_id', 'N/A')}
# # #             </p>
# # #         </div>
# # #     """, unsafe_allow_html=True)

# # #     st.markdown("<br>", unsafe_allow_html=True)

# # #     tab_options = ["Manual Entry", "PDF Upload", "Batch Analysis"]
# # #     default_tab = st.session_state.get('stage2_selected_tab', 'Manual Entry')
# # #     if default_tab not in tab_options:
# # #         default_tab = "Manual Entry"
# # #     selected_tab = st.radio(
# # #         "Select input method",
# # #         tab_options,
# # #         index=tab_options.index(default_tab),
# # #         horizontal=True,
# # #         label_visibility="collapsed"
# # #     )

# # #     if selected_tab == "Manual Entry":
# # #         st.markdown('<p class="section-header">Manual CIBIL Data Entry</p>', unsafe_allow_html=True)
# # #         st.markdown("""
# # #             <div class="info-box">
# # #                 📝 <strong>Manual Data Entry</strong><br>
# # #                 Enter CIBIL bureau data to enhance Stage 1 customer profile.<br>
# # #                 The Stage 2 model will use this data to predict risk tier (P1/P2/P3/P4).
# # #             </div>
# # #         """, unsafe_allow_html=True)

# # #         with st.form("stage2_manual_form"):
# # #             st.markdown("### 📋 Application Reference")
# # #             col1, col2 = st.columns(2)
# # #             with col1:
# # #                 st.text_input("Application ID", value=stage1_customer.get('application_id', 'N/A'), disabled=True)
# # #                 st.text_input("Stage 1 Decision", value=st.session_state.get('stage1_decision', 'N/A'), disabled=True)
# # #             with col2:
# # #                 st.text_input("Customer Name (Optional)", "")
# # #                 st.number_input("Stage 1 Risk Score", value=int(stage1_data.get('risk_score', 750)), disabled=True)

# # #             st.markdown("---")
# # #             st.markdown("### 🏦 CIBIL Bureau Data")




# # #             st.markdown("---")
# # #             st.markdown("### 👤 Demographics & Product Enquiries")

# # #             col1, col2, col3 = st.columns(3)
# # #             with col1:
# # #                 gender = st.selectbox(
# # #                     "Gender",
# # #                     ["Male", "Female", "Others"],
# # #                     help="Select gender as per CIBIL report"
# # #                 )
# # #             with col2:
# # #                 marital_status = st.selectbox(
# # #                     "Marital Status",
# # #                     ["Married", "Single", "Divorced", "Widowed", "Others"],
# # #                     help="Marital status from bureau data"
# # #                 )
# # #             with col3:
# # #                 education = st.selectbox(
# # #                     "Education",
# # #                     ["Graduate", "Post Graduate", "Under Graduate", "Professional", "Others"],
# # #                     help="Highest education level"
# # #                 )


# # #             col1, col2, col3 = st.columns(3)
# # #             with col1:
# # #                 st.markdown("**Credit Score & History**")
# # #                 cibil_score = st.number_input("Credit Score", 300, 900, 720, 10)
# # #                 max_delinquency = st.number_input("Max Delinquency Level", 0, 100, 0)
# # #                 num_times_30dpd = st.number_input("Times 30+ DPD", 0, 50, 0)
# # #                 num_times_60dpd = st.number_input("Times 60+ DPD", 0, 50, 0)
# # #                 num_times_delinquent = st.number_input("Total Delinquent", 0, 50, 0)
# # #             with col2:
# # #                 st.markdown("**Recent Behavior (6-12M)**")
# # #                 num_deliq_6m = st.number_input("Delinquencies (6M)", 0, 20, 0)
# # #                 num_deliq_12m = st.number_input("Delinquencies (12M)", 0, 20, 0)
# # #                 max_deliq_6m = st.number_input("Max Delinq (6M)", 0, 100, 0)
# # #                 max_deliq_12m = st.number_input("Max Delinq (12M)", 0, 100, 0)
# # #                 enq_L3m = st.number_input("Inquiries (3M)", 0, 20, 2)
# # #                 enq_L6m = st.number_input("Inquiries (6M)", 0, 30, 4)
# # #                 enq_L12m = st.number_input("Inquiries (12M)", 0, 50, 6)
# # #             with col3:
# # #                 st.markdown("**Account Quality**")
# # #                 num_std = st.number_input("Standard Accounts", 0, 50, 3)
# # #                 num_std_6m = st.number_input("Standard (6M)", 0, 50, 3)
# # #                 num_std_12m = st.number_input("Standard (12M)", 0, 50, 3)
# # #                 num_sub = st.number_input("Sub-standard", 0, 20, 0)
# # #                 num_sub_6m = st.number_input("Sub-standard (6M)", 0, 20, 0)
# # #                 num_dbt = st.number_input("Doubtful", 0, 10, 0)
# # #                 num_lss = st.number_input("Loss", 0, 10, 0)

# # #             col1, col2, col3 = st.columns(3)
# # #             with col1:
# # #                 st.markdown("**Utilization**")
# # #                 pct_active_tls = st.number_input("% Active TLs", 0.0, 1.0, 0.60, 0.01)
# # #                 pct_current_bal = st.number_input("Current Balance %", 0.0, 1.0, 0.30, 0.01)
# # #                 cc_utilization = st.number_input("CC Utilization", 0.0, 1.0, 0.35, 0.01)
# # #                 pl_utilization = st.number_input("PL Utilization", 0.0, 1.0, 0.25, 0.01)
# # #                 max_unsec_exposure = st.number_input("Max Unsec Exposure %", 0, 100, 30)
# # #             with col2:
# # #                 st.markdown("**Demographics**")
# # #                 age_cibil = st.number_input("Age", 24, 70, int(stage1_customer.get('age', 35)))
# # #                 net_monthly_income = st.number_input("Net Monthly Income", 0, 1000000,
# # #                                                       int(stage1_customer.get('avg_salary_6m', 50000)), 5000)
# # #                 time_curr_employer = st.number_input("Employment Tenure (months)", 0, 600,
# # #                                                       int(stage1_customer.get('employment_tenure_months', 24)))
# # #             with col3:
# # #                 st.markdown("**Product Flags**")
# # #                 cc_flag = st.selectbox("Credit Card", ["Yes", "No"]) == "Yes"
# # #                 pl_flag = st.selectbox("Personal Loan", ["Yes", "No"]) == "No"
# # #                 hl_flag = st.selectbox("Home Loan", ["Yes", "No"]) == "No"
# # #                 gl_flag = st.selectbox("Gold Loan", ["Yes", "No"]) == "No"

# # #             st.markdown("<br>", unsafe_allow_html=True)
# # #             submitted_s2 = st.form_submit_button("🔬 Run Stage 2 Analysis", use_container_width=True, type="primary")

# # #         if submitted_s2:
# # #             with st.spinner("🔬 Running Stage 2 CIBIL Deep Analysis..."):
# # #                 enhanced_customer_data = stage1_customer.copy()
# # #                 _s1_inc = stage1_customer.get('avg_salary_6m', 50000)
# # #                 _s2_inc = net_monthly_income or 0
# # #                 _final_income = _s1_inc if (_s2_inc > 0 and _s2_inc < _s1_inc * 0.4) else (_s2_inc or _s1_inc)
# # #                 if _s2_inc > 0 and _s2_inc < _s1_inc * 0.4:
# # #                     st.warning(f'⚠️ CIBIL income ₹{_s2_inc:,} is much lower than application income ₹{_s1_inc:,}. Using application income for FOIR.')
# # #                 enhanced_customer_data.update({
# # #                     'bureau_score': cibil_score,
# # #                     'age': age_cibil,
# # #                     'avg_salary_6m': _final_income,
# # #                     'employment_tenure_months': time_curr_employer,
# # #                     'dpd_30_count_6m': num_times_30dpd,
# # #                     'dpd_90_count_6m': num_times_60dpd,
# # #                     'max_delinquency_level': max_delinquency,
# # #                     'num_times_delinquent': num_times_delinquent,
# # #                     'num_deliq_6mts': num_deliq_6m,
# # #                     'num_deliq_12mts': num_deliq_12m,
# # #                     'max_deliq_6mts': max_deliq_6m,
# # #                     'max_deliq_12mts': max_deliq_12m,
# # #                     'recent_inquiries_3m': enq_L3m,
# # #                     'enq_L6m': enq_L6m,
# # #                     'enq_L12m': enq_L12m,
# # #                     'active_loans_count': num_std,
# # #                     'num_std_6mts': num_std_6m,
# # #                     'num_std_12mts': num_std_12m,
# # #                     'num_sub': num_sub,
# # #                     'num_sub_6mts': num_sub_6m,
# # #                     'num_dbt': num_dbt,
# # #                     'num_lss': num_lss,
# # #                     'credit_utilization_pct': cc_utilization * 100,
# # #                     'pct_of_active_TLs_ever': pct_active_tls,
# # #                     'pct_currentBal_all_TL': pct_current_bal,
# # #                     'CC_utilization': cc_utilization,
# # #                     'PL_utilization': pl_utilization,
# # #                     'max_unsec_exposure_inPct': max_unsec_exposure,
# # #                     'CC_Flag': 1 if cc_flag else 0,
# # #                     'PL_Flag': 1 if pl_flag else 0,
# # #                     'HL_Flag': 1 if hl_flag else 0,
# # #                     'GL_Flag': 1 if gl_flag else 0,
# # #                      'GENDER': gender,
# # #                      'MARITALSTATUS': marital_status,
# # #                      'EDUCATION': education,
# # #                 })
# # #                 # Clean sentinel values before passing to stage2 engine
# # #                 enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)
# # #                 try:
# # #                     stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
# # #                     stage2_result = resolve_stage2_to_binary(stage2_result)
# # #                     display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
# # #                 except Exception as e:
# # #                     st.error(f"❌ Stage 2 analysis failed: {str(e)}")
# # #                     st.exception(e)

# # #     elif selected_tab == "PDF Upload":
# # #         st.markdown('<p class="section-header">📄 CIBIL PDF Upload</p>', unsafe_allow_html=True)
# # #         if not OCR_AVAILABLE:
# # #             st.error("❌ OCR not available. " + (OCR_ERROR_MSG or "Check packages.txt and requirements.txt."))
# # #             st.warning("For now, please use the **Manual Entry** tab.")
# # #         else:
# # #             st.markdown("""
# # #                 <div class="info-box">
# # #                     📄 <strong>CIBIL PDF Extraction</strong><br>
# # #                     Upload a CIBIL bureau report PDF for automatic extraction and analysis.
# # #                 </div>
# # #             """, unsafe_allow_html=True)
# # #             uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="stage2_pdf")
# # #             if uploaded_pdf is not None:
# # #                 st.success(f"✅ File uploaded: {uploaded_pdf.name} ({uploaded_pdf.size / 1024:.1f} KB)")
# # #                 if st.button("🔬 Extract & Analyze", key="extract_analyze_stage2", type="primary", use_container_width=True):
# # #                     with st.spinner("🔄 Extracting data from PDF..."):
# # #                         extraction_result = extract_cibil_from_pdf(uploaded_pdf)
# # #                     if extraction_result.get('success', False):
# # #                         st.success("✅ PDF extraction successful!")

# # #                         # Application context banner
# # #                         app_id_display  = stage1_customer.get('application_id', 'N/A')
# # #                         cust_name       = stage1_customer.get('customer_name', 'N/A')
# # #                         s1_decision     = st.session_state.get('stage1_decision', 'N/A')
# # #                         s1_risk         = stage1_data.get('risk_score', 'N/A')
# # #                         st.markdown(f"""
# # #                             <div style="background:#1e3a5f;color:white;padding:0.75rem 1rem;border-radius:0.5rem;margin-bottom:0.75rem;">
# # #                                 <strong>📋 Application ID:</strong> {app_id_display} &nbsp;|&nbsp;
# # #                                 <strong>Stage 1:</strong> {s1_decision} &nbsp;|&nbsp;
# # #                                 <strong>Risk Score:</strong> {s1_risk}
# # #                             </div>
# # #                         """, unsafe_allow_html=True)

# # #                         st.markdown("### 📋 Extracted CIBIL Data (Summary)")
# # #                         col1, col2, col3, col4 = st.columns(4)
# # #                         with col1:
# # #                             st.metric("Credit Score", extraction_result.get('Credit_Score', 'N/A'))
# # #                             st.metric("Max Delinquency Level", extraction_result.get('max_delinquency_level', 0))
# # #                         with col2:
# # #                             st.metric("Times 30+ DPD", extraction_result.get('num_times_30p_dpd', 0))
# # #                             st.metric("Times 60+ DPD", extraction_result.get('num_times_60p_dpd', 0))
# # #                         with col3:
# # #                             st.metric("Total Delinquent", extraction_result.get('num_times_delinquent', 0))
# # #                             st.metric("DPD 90+ (6M)", extraction_result.get('dpd_90_count_6m', 0))
# # #                         with col4:
# # #                             st.metric("Active Accounts", extraction_result.get('num_std', 0))
# # #                             st.metric("Written Off", extraction_result.get('written_off_count', 0))

# # #                         with st.expander("🔍 View All Extracted Features (with internal IDs)", expanded=False):
# # #                             friendly_names = {
# # #                                 'Credit_Score': 'Credit Score',
# # #                                 'AGE': 'Age',
# # #                                 'max_delinquency_level': 'Max Delinquency Level',
# # #                                 'num_times_30p_dpd': 'Times 30+ DPD',
# # #                                 'num_times_60p_dpd': 'Times 60+ DPD',
# # #                                 'num_times_delinquent': 'Total Times Delinquent',
# # #                                 'dpd_90_count_6m': 'DPD 90+ Count (6M)',
# # #                                 'num_deliq_6mts': 'Delinquent Count (6M)',
# # #                                 'num_deliq_12mts': 'Delinquent Count (12M)',
# # #                                 'max_deliq_6mts': 'Max Delinquency (6M)',
# # #                                 'max_deliq_12mts': 'Max Delinquency (12M)',
# # #                                 'enq_L3m': 'Recent Inquiries (3M)',
# # #                                 'enq_L6m': 'Inquiries (6M)',
# # #                                 'enq_L12m': 'Inquiries (12M)',
# # #                                 'num_std': 'Standard / Active Accounts',
# # #                                 'num_std_6mts': 'Standard Accounts (6M)',
# # #                                 'num_std_12mts': 'Standard Accounts (12M)',
# # #                                 'num_sub': 'Sub-standard Accounts',
# # #                                 'num_sub_6mts': 'Sub-standard (6M)',
# # #                                 'num_dbt': 'Doubtful Accounts',
# # #                                 'num_lss': 'Loss / Written-Off Accounts',
# # #                                 'CC_utilization': 'Credit Card Utilization (0–1)',
# # #                                 'PL_utilization': 'Personal Loan Utilization (0–1)',
# # #                                 'CC_Flag': 'Has Credit Card (1=Yes)',
# # #                                 'PL_Flag': 'Has Personal Loan (1=Yes)',
# # #                                 'HL_Flag': 'Has Home Loan (1=Yes)',
# # #                                 'GL_Flag': 'Has Gold Loan (1=Yes)',
# # #                                 'written_off_count': 'Written Off Count',
# # #                                 'settled_count': 'Settled Account Count',
# # #                                 'high_util_flag': 'High Utilization Flag (1=Yes)',
# # #                                 'recent_deliq_flag': 'Recent Delinquency Flag (1=Yes)',
# # #                                 'account_quality_score': 'Account Quality Score (0–100)',
# # #                                 'Time_With_Curr_Empr': 'Employment Tenure (months)',
# # #                                 'NETMONTHLYINCOME': 'Net Monthly Income (₹)',
# # #                                 'pct_of_active_TLs_ever': '% Active Trade Lines Ever',
# # #                                 'pct_currentBal_all_TL': '% Current Balance / All TL',
# # #                                 'max_unsec_exposure_inPct': 'Max Unsecured Exposure (%)',
# # #                                 'extraction_method': 'Extraction Method',
# # #                             }
# # #                             exclude_keys = {'success', 'error', 'raw_text'}
# # #                             data_items = []
# # #                             for key, val in extraction_result.items():
# # #                                 if key in exclude_keys:
# # #                                     continue
# # #                                 fname = friendly_names.get(key, key.replace('_', ' ').title())
# # #                                 data_items.append({"Feature Name": fname, "Internal ID": key, "Extracted Value": str(val)})
# # #                             data_items.sort(key=lambda x: x["Feature Name"])
# # #                             # Prepend Application context at top
# # #                             data_items = [
# # #                                 {"Feature Name": "── Application ID", "Internal ID": "application_id", "Extracted Value": app_id_display},
# # #                                 {"Feature Name": "── Customer Name", "Internal ID": "customer_name", "Extracted Value": cust_name},
# # #                                 {"Feature Name": "── Stage 1 Decision", "Internal ID": "stage1_decision", "Extracted Value": s1_decision},
# # #                                 {"Feature Name": "── Stage 1 Risk Score", "Internal ID": "stage1_risk_score", "Extracted Value": str(s1_risk)},
# # #                             ] + data_items
# # #                             import pandas as _pd
# # #                             df_all = _pd.DataFrame(data_items)
# # #                             st.dataframe(df_all, use_container_width=True, hide_index=True)

# # #                         enhanced_customer_data = stage1_customer.copy()
# # #                         _s1_income = stage1_customer.get('avg_salary_6m', 50000)
# # #                         _s2_income = extraction_result.get('NETMONTHLYINCOME', 0)
# # #                         _use_income = _s1_income if (_s2_income > 0 and _s2_income < _s1_income * 0.4) else (_s2_income or _s1_income)
# # #                         if _s2_income > 0 and _s2_income < _s1_income * 0.4:
# # #                             st.warning(f'⚠️ CIBIL income ₹{_s2_income:,} is much lower than application income ₹{_s1_income:,}. Using application income for FOIR.')

# # #                         enhanced_customer_data.update({
# # #                             'bureau_score': extraction_result.get('Credit_Score', 720),
# # #                             'age': extraction_result.get('AGE', stage1_customer.get('age', 35)),
# # #                             'avg_salary_6m': _use_income,
# # #                             'employment_tenure_months': extraction_result.get('Time_With_Curr_Empr', stage1_customer.get('employment_tenure_months', 24)),
# # #                             'dpd_30_count_6m': extraction_result.get('num_times_30p_dpd', 0),
# # #                             'dpd_90_count_6m': extraction_result.get('dpd_90_count_6m', 0),
# # #                             'max_delinquency_level': extraction_result.get('max_delinquency_level', 0),
# # #                             'num_times_delinquent': extraction_result.get('num_times_delinquent', 0),
# # #                             'num_deliq_6mts': extraction_result.get('num_deliq_6mts', 0),
# # #                             'num_deliq_12mts': extraction_result.get('num_deliq_12mts', 0),
# # #                             'max_deliq_6mts': extraction_result.get('max_deliq_6mts', 0),
# # #                             'max_deliq_12mts': extraction_result.get('max_deliq_12mts', 0),
# # #                             'recent_inquiries_3m': extraction_result.get('enq_L3m', 2),
# # #                             'enq_L6m': extraction_result.get('enq_L6m', 4),
# # #                             'enq_L12m': extraction_result.get('enq_L12m', 6),
# # #                             'active_loans_count': extraction_result.get('num_std', 1),
# # #                             'num_std_6mts': extraction_result.get('num_std_6mts', 0),
# # #                             'num_std_12mts': extraction_result.get('num_std_12mts', 0),
# # #                             'num_sub': extraction_result.get('num_sub', 0),
# # #                             'num_sub_6mts': extraction_result.get('num_sub_6mts', 0),
# # #                             'num_dbt': extraction_result.get('num_dbt', 0),
# # #                             'num_lss': extraction_result.get('num_lss', 0),
# # #                             'credit_utilization_pct': (0 if extraction_result.get('CC_utilization', 0) < 0 else extraction_result.get('CC_utilization', 0.35)) * 100,
# # #                             'pct_of_active_TLs_ever': extraction_result.get('pct_of_active_TLs_ever', 0.6),
# # #                             'pct_currentBal_all_TL': extraction_result.get('pct_currentBal_all_TL', 0.3),
# # #                             'CC_utilization': 0 if extraction_result.get('CC_utilization', 0) < 0 else extraction_result.get('CC_utilization', 0.35),
# # #                             'PL_utilization': 0 if extraction_result.get('PL_utilization', 0) < 0 else extraction_result.get('PL_utilization', 0.25),
# # #                             'max_unsec_exposure_inPct': extraction_result.get('max_unsec_exposure_inPct', 30),
# # #                             'CC_Flag': extraction_result.get('CC_Flag', 0),
# # #                             'PL_Flag': extraction_result.get('PL_Flag', 0),
# # #                             'HL_Flag': extraction_result.get('HL_Flag', 0),
# # #                             'GL_Flag': extraction_result.get('GL_Flag', 0),
# # #                             'written_off_count': extraction_result.get('written_off_count', 0),
# # #                             'settled_count': extraction_result.get('settled_count', 0),
# # #                             'high_util_flag': extraction_result.get('high_util_flag', 0),
# # #                             'recent_deliq_flag': extraction_result.get('recent_deliq_flag', 0),
# # #                             'account_quality_score': extraction_result.get('account_quality_score', 0)
# # #                         })

# # #                         # Clean sentinel values before passing to stage2 engine
# # #                         enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)

# # #                         with st.spinner("🔬 Running Stage 2 analysis..."):
# # #                             try:
# # #                                 stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
# # #                                 stage2_result = resolve_stage2_to_binary(stage2_result)
# # #                                 display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
# # #                             except Exception as e:
# # #                                 st.error(f"❌ Analysis failed: {str(e)}")
# # #                     else:
# # #                         st.error("❌ PDF extraction failed! Error: " + extraction_result.get('error', 'Unknown'))

# # #     elif selected_tab == "Batch Analysis":
# # #         st.markdown('<p class="section-header">📊 Batch CIBIL Analysis</p>', unsafe_allow_html=True)
# # #         st.info("📊 Batch analysis feature coming soon! (Upload a CSV with all required CIBIL fields)")

# # # elif page == "📊 Batch Process":
# # #     st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
# # #     st.markdown("""
# # #         <div class="info-box">
# # #             📤 Upload a CSV file with customer data for bulk credit assessment.
# # #             The file should include all required fields for prediction.
# # #         </div>
# # #     """, unsafe_allow_html=True)
# # #     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# # #     if uploaded_file is not None:
# # #         try:
# # #             df = pd.read_csv(uploaded_file)
# # #             st.success(f"✅ Successfully loaded {len(df)} records")
# # #             with st.expander("📄 Preview Uploaded Data"):
# # #                 st.dataframe(df.head(), use_container_width=True)
# # #                 st.write(f"**Total Records:** {len(df)}")
# # #                 st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
# # #             required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
# # #             missing_cols = [col for col in required_cols if col not in df.columns]
# # #             if missing_cols:
# # #                 st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
# # #                 st.info("Please ensure your CSV includes at least these columns: age, employment_type, avg_salary_6m, bureau_score, loan_amount")
# # #             else:
# # #                 if st.button("🚀 Process Batch Predictions", key="process_batch_btn", type="primary", use_container_width=True):
# # #                     with st.spinner(f"🔍 Processing {len(df)} records..."):
# # #                         progress_bar = st.progress(0)
# # #                         results_df = process_batch_predictions(df)
# # #                         progress_bar.progress(100)
# # #                         st.success(f"✅ Completed processing {len(results_df)} records!")
# # #                         tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
# # #                         with tab1:
# # #                             st.dataframe(results_df, use_container_width=True)
# # #                             col1, col2, col3, col4 = st.columns(4)
# # #                             with col1:
# # #                                 st.metric("✅ Approved", len(results_df[results_df['decision'] == 'APPROVE']))
# # #                             with col2:
# # #                                 st.metric("❌ Rejected", len(results_df[results_df['decision'] == 'REJECT']))
# # #                             with col3:
# # #                                 st.metric("⚠️ Review", len(results_df[results_df['decision'] == 'REVIEW']))
# # #                             with col4:
# # #                                 st.metric("📊 Avg Risk Score", f"{results_df['risk_score'].mean():.0f}")
# # #                         with tab2:
# # #                             col1, col2 = st.columns(2)
# # #                             with col1:
# # #                                 decision_counts = results_df['decision'].value_counts()
# # #                                 fig1 = px.pie(values=decision_counts.values, names=decision_counts.index,
# # #                                               title="Decision Distribution", color=decision_counts.index,
# # #                                               color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
# # #                                 st.plotly_chart(fig1, use_container_width=True)
# # #                             with col2:
# # #                                 fig2 = px.histogram(results_df, x='risk_score', title="Risk Score Distribution",
# # #                                                     nbins=20, color_discrete_sequence=['#587042'])
# # #                                 st.plotly_chart(fig2, use_container_width=True)
# # #                             fig3 = px.scatter(results_df, x='monthly_income', y='loan_amount', color='decision',
# # #                                               size='risk_score', title="Income vs Loan Amount (Colored by Decision)",
# # #                                               hover_data=['application_id', 'foir_percentage'],
# # #                                               color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
# # #                             st.plotly_chart(fig3, use_container_width=True)
# # #                             fig4 = px.box(results_df, x='decision', y='pd_percentage',
# # #                                           title="PD Distribution by Decision", color='decision',
# # #                                           color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
# # #                             st.plotly_chart(fig4, use_container_width=True)
# # #                         with tab3:
# # #                             st.markdown("### Download Results")
# # #                             col1, col2 = st.columns(2)
# # #                             with col1:
# # #                                 st.download_button(
# # #                                     "📥 Download as CSV",
# # #                                     data=results_df.to_csv(index=False),
# # #                                     file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# # #                                     mime="text/csv",
# # #                                     use_container_width=True
# # #                                 )
# # #                             with col2:
# # #                                 st.download_button(
# # #                                     "📥 Download as JSON",
# # #                                     data=results_df.to_json(orient='records', indent=2),
# # #                                     file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
# # #                                     mime="application/json",
# # #                                     use_container_width=True
# # #                                 )
# # #                             st.markdown("---")
# # #                             st.markdown("#### Filtered Downloads")
# # #                             col1, col2, col3 = st.columns(3)
# # #                             with col1:
# # #                                 approved_df = results_df[results_df['decision'] == 'APPROVE']
# # #                                 if len(approved_df) > 0:
# # #                                     st.download_button(
# # #                                         f"✅ Approved Only ({len(approved_df)})",
# # #                                         data=approved_df.to_csv(index=False),
# # #                                         file_name=f"approved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# # #                                         mime="text/csv",
# # #                                         use_container_width=True
# # #                                     )
# # #                             with col2:
# # #                                 rejected_df = results_df[results_df['decision'] == 'REJECT']
# # #                                 if len(rejected_df) > 0:
# # #                                     st.download_button(
# # #                                         f"❌ Rejected Only ({len(rejected_df)})",
# # #                                         data=rejected_df.to_csv(index=False),
# # #                                         file_name=f"rejected_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# # #                                         mime="text/csv",
# # #                                         use_container_width=True
# # #                                     )
# # #                             with col3:
# # #                                 review_df = results_df[results_df['decision'] == 'REVIEW']
# # #                                 if len(review_df) > 0:
# # #                                     st.download_button(
# # #                                         f"⚠️ Review Only ({len(review_df)})",
# # #                                         data=review_df.to_csv(index=False),
# # #                                         file_name=f"review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# # #                                         mime="text/csv",
# # #                                         use_container_width=True
# # #                                     )
# # #         except Exception as e:
# # #             st.error(f"❌ Error processing file: {str(e)}")
# # #             st.info("Please ensure the CSV file is properly formatted and contains the required columns.")
# # #     else:
# # #         st.markdown("---")
# # #         st.markdown("### 📋 CSV Template")
# # #         template_data = {
# # #             'age': [35, 42, 28],
# # #             'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
# # #             'dependents': [2, 3, 6],
# # #             'kyc_verified': ['Yes', 'Yes', 'No'],
# # #             'bankruptcy_flag': ['No', 'No', 'No'],
# # #             'fraud_flag': ['No', 'No', 'No'],
# # #             'employment_tenure_months': [24, 0, 18],
# # #             'business_vintage_years': [0, 5, 0],
# # #             'bureau_score': [720, 680, 580],
# # #             'dpd_90_count_6m': [0, 1, 2],
# # #             'dpd_30_count_6m': [0, 2, 1],
# # #             'credit_utilization_pct': [30, 45, 75],
# # #             'recent_inquiries_3m': [2, 1, 5],
# # #             'active_loans_count': [1, 2, 3],
# # #             'avg_salary_6m': [50000, 75000, 35000],
# # #             'AMT_INCOME_TOTAL': [600000, 900000, 420000],
# # #             'net_cash_surplus_6m': [20000, 35000, 10000],
# # #             'salary_stability_flag': ['STABLE', 'MODERATE', 'UNSTABLE'],
# # #             'loan_amount': [180000, 250000, 100000],
# # #             'loan_tenure_months': [24, 36, 12],
# # #             'interest_rate': [10.5, 11.0, 12.0],
# # #             'existing_emi': [15000, 20000, 8000],
# # #             'AMT_ANNUITY': [8500, 9500, 4500],
# # #             # New fields:
# # #             'payment_discipline_flag': ['GOOD', 'MODERATE', 'POOR'],
# # #             'liquidity_flag': ['LOW', 'ADEQUATE', 'LOW'],
# # #             'cashflow_health': ['HEALTHY', 'MODERATE', 'STRESSED'],
# # #             'bureau_risk_flag': ['LOW', 'MEDIUM', 'HIGH'],
# # #             'inward_bounce_count_3m': [0, 1, 3],
# # #             'salary_missing_months': [0, 0, 2],
# # #         }
# # #         template_df = pd.DataFrame(template_data)
# # #         st.dataframe(template_df, use_container_width=True)
# # #         st.caption("📝 Note: `dependents > 5` will automatically trigger REVIEW regardless of other factors.")
# # #         st.download_button(
# # #             "📥 Download CSV Template",
# # #             data=template_df.to_csv(index=False),
# # #             file_name="credit_assessment_template.csv",
# # #             mime="text/csv",
# # #             use_container_width=True
# # #         )

# # # elif page == "📈 Model Info":
# # #     st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
# # #     col1, col2, col3 = st.columns(3)
# # #     with col1:
# # #         st.markdown('<div class="stat-card"><div class="stat-number">RF</div><div class="stat-label">Model Type</div></div>', unsafe_allow_html=True)
# # #     with col2:
# # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TOP_FEATURES)}</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)
# # #     with col3:
# # #         st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TARGET_LE.classes_)}</div><div class="stat-label">Classes</div></div>', unsafe_allow_html=True)
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
# # #     feature_df = pd.DataFrame({'Rank': range(1, min(21, len(TOP_FEATURES) + 1)), 'Feature': TOP_FEATURES[:20]})
# # #     st.dataframe(feature_df, use_container_width=True, hide_index=True)

# # # elif page == "ℹ️ About":
# # #     st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
# # #     st.markdown("""
# # #         <div class="info-card">
# # #             <div class="info-card-title"><span class="icon">🏦</span><span>Credit Risk Assessment Platform</span></div>
# # #             <div class="info-card-content">
# # #                 <p><strong>Version:</strong> 8.3 - FIXED NUMERIC TYPES & SUBMIT BUTTON</p>
# # #                 <p><strong>Developer:</strong> Zen Meraki</p>
# # #                 <p><strong>Date:</strong> January 2026</p>
# # #                 <br>
# # #                 <p>A comprehensive credit risk evaluation system combining hard policy rules,
# # #                 machine learning models, and affordability analysis for accurate and compliant lending decisions.</p>
# # #             </div>
# # #         </div>
# # #     """, unsafe_allow_html=True)
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         st.markdown("""
# # #             <div class="info-card">
# # #                 <div class="info-card-title"><span class="icon">🎯</span><span>Key Features</span></div>
# # #                 <div class="info-card-content">
# # #                     <ul style="margin: 0; padding-left: 1.25rem;">
# # #                         <li>Three-layer decision engine</li>
# # #                         <li>Real-time risk assessment</li>
# # #                         <li>Industry-standard PD calculation</li>
# # #                         <li>FOIR calculation & validation</li>
# # #                         <li>Automated reason generation</li>
# # #                         <li>Complete audit trail (PDF)</li>
# # #                         <li>Professional UI/UX</li>
# # #                     </ul>
# # #                 </div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
# # #     with col2:
# # #         st.markdown("""
# # #             <div class="info-card">
# # #                 <div class="info-card-title"><span class="icon">🛠️</span><span>Technology Stack</span></div>
# # #                 <div class="info-card-content">
# # #                     <ul style="margin: 0; padding-left: 1.25rem;">
# # #                         <li>Streamlit (UI Framework)</li>
# # #                         <li>Scikit-learn (ML)</li>
# # #                         <li>Plotly (Visualizations)</li>
# # #                         <li>Pandas (Data Processing)</li>
# # #                         <li>ReportLab (PDF Generation)</li>
# # #                         <li>Python 3.8+</li>
# # #                     </ul>
# # #                 </div>
# # #             </div>
# # #         """, unsafe_allow_html=True)















# # """
# # Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
# # Enhanced with Modern UI/UX Design
# # Run with: streamlit run test.py (from inside the notebooks folder)
# # Author: Zen Meraki
# # Date: January 2026
# # VERSION: 8.4 - OCR AUTO-FILL FIX (all categorical dropdowns now update from PDF)
# # """

# # import streamlit as st

# # # =============================================================================
# # # PAGE CONFIGURATION – MUST BE THE VERY FIRST STREAMLIT COMMAND
# # # =============================================================================
# # st.set_page_config(
# #     page_title="Credit Risk Assessment",
# #     page_icon="💳",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # =============================================================================
# # # STANDARD LIBRARY / THIRD-PARTY IMPORTS
# # # =============================================================================
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
# # from typing import Dict, List, Any, Union
# # import json
# # import sys
# # import os
# # from pathlib import Path
# # import re

# # # =============================================================================
# # # SUPPRESS SCIKIT-LEARN VERSION WARNINGS
# # # =============================================================================
# # warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# # # =============================================================================
# # # DYNAMIC PATH RESOLUTION
# # # =============================================================================
# # CURRENT_DIR = Path(__file__).resolve().parent
# # PROJECT_ROOT = CURRENT_DIR.parent
# # POSSIBLE_LOCATIONS = [
# #     CURRENT_DIR,
# #     PROJECT_ROOT,
# #     PROJECT_ROOT / "loan",
# #     PROJECT_ROOT / "utils",
# #     PROJECT_ROOT / "notebooks",
# # ]

# # for loc in POSSIBLE_LOCATIONS:
# #     if loc.exists() and str(loc) not in sys.path:
# #         sys.path.insert(0, str(loc))

# # # =============================================================================
# # # OPTIONAL OCR DEPENDENCIES – GRACEFUL FALLBACK
# # # =============================================================================
# # OCR_AVAILABLE = False
# # OCR_ERROR_MSG = ""
# # try:
# #     import pytesseract
# #     from pdf2image import convert_from_bytes
# #     import cv2
# #     from PIL import Image

# #     import shutil as _shutil
# #     _tess_cmd = (
# #         _shutil.which("tesseract")
# #         or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# #     )
# #     if _tess_cmd:
# #         pytesseract.pytesseract.tesseract_cmd = _tess_cmd

# #     pytesseract.get_tesseract_version()
# #     OCR_AVAILABLE = True

# # except ImportError as _e:
# #     OCR_ERROR_MSG = (
# #         f"Missing Python package: {_e}. "
# #         "Add to requirements.txt: pytesseract  pdf2image  opencv-python-headless  pillow"
# #     )
# # except Exception as _e:
# #     _name = type(_e).__name__
# #     if "TesseractNotFound" in _name or "tesseract" in str(_e).lower():
# #         OCR_ERROR_MSG = (
# #             "Tesseract binary not found. "
# #             "Streamlit Cloud → add 'tesseract-ocr' and 'poppler-utils' to packages.txt. "
# #             "Linux → sudo apt install tesseract-ocr poppler-utils. "
# #             "Mac → brew install tesseract poppler."
# #         )
# #     else:
# #         OCR_ERROR_MSG = f"OCR init error ({_name}): {_e}"

# # # =============================================================================
# # # IMPORT CSS – WITH FALLBACK
# # # =============================================================================
# # try:
# #     from css_styles import CSS
# # except ImportError:
# #     CSS = """
# #     <style>
# #         .main-header { font-size: 2rem; font-weight: bold; color: #2d3748; }
# #         .section-header { font-size: 1.5rem; font-weight: 600; color: #2d3748; }
# #         .info-box { background: #f7fafc; padding: 1rem; border-radius: 0.5rem; }
# #         .decision-card { padding: 2rem; border-radius: 1rem; text-align: center; margin-bottom: 1rem; }
# #         .decision-card-approved { background: #c6f6d5; border-left: 5px solid #48bb78; }
# #         .decision-card-rejected { background: #fed7d7; border-left: 5px solid #f56565; }
# #         .decision-card-review { background: #feebc8; border-left: 5px solid #ed8936; }
# #         .decision-title { font-size: 2.5rem; font-weight: bold; }
# #         .decision-subtitle { font-size: 1rem; opacity: 0.8; }
# #         .stat-card { background: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
# #         .stat-number { font-size: 1.8rem; font-weight: bold; color: #2d3748; }
# #         .stat-label { font-size: 0.875rem; color: #718096; }
# #         .info-card { background: white; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
# #         .info-card-title { font-weight: 600; margin-bottom: 0.5rem; color: #2d3748; }
# #         .info-card-content { font-size: 0.875rem; }
# #         .data-row { display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #e2e8f0; }
# #         .data-label { color: #4a5568; }
# #         .data-value { font-weight: 500; }
# #         .status-badge { padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; margin-left: 0.5rem; }
# #         .badge-pass { background: #c6f6d5; color: #22543d; }
# #         .badge-fail { background: #fed7d7; color: #742a2a; }
# #         .badge-warning { background: #feebc8; color: #744210; }
# #         .reason-item { padding: 0.25rem 0; }
# #         .reason-icon { color: #587042; font-weight: bold; margin-right: 0.5rem; }
# #     </style>
# #     """

# # st.markdown(CSS, unsafe_allow_html=True)

# # # =============================================================================
# # # SESSION STATE INITIALIZATION
# # # =============================================================================
# # def init_session_state():
# #     if 'stage1_complete' not in st.session_state:
# #         st.session_state.stage1_complete = False
# #     if 'stage1_decision' not in st.session_state:
# #         st.session_state.stage1_decision = None
# #     if 'stage1_data' not in st.session_state:
# #         st.session_state.stage1_data = None
# #     if 'current_customer_data' not in st.session_state:
# #         st.session_state.current_customer_data = None
# #     if 'page_navigation' not in st.session_state:
# #         st.session_state.page_navigation = "🏠 Home"
# #     if 'use_two_stage' not in st.session_state:
# #         st.session_state.use_two_stage = False
# #     if 'stage2_selected_tab' not in st.session_state:
# #         st.session_state.stage2_selected_tab = "Manual Entry"

# # init_session_state()

# # # =============================================================================
# # # IMPORT BUSINESS LOGIC MODULES
# # # =============================================================================
# # try:
# #     from affordability_engine import calculate_emi, calculate_affordability
# #     from reason_codes import generate_reason_codes
# #     from risk_engine import (
# #         calculate_final_risk_score, fill_missing_ml_fields,
# #         clean_sentinel_values, validate_cibil_identity
# #     )
# #     from affordability_engine import check_loan_to_income, check_net_disposable
# # except ImportError as e:
# #     st.error(f"❌ Failed to import required modules: {e}")
# #     st.info("""
# #     Please ensure the following files are placed in one of these directories:
# #     - `notebooks/` (same folder as test.py)
# #     - `loan/` (sibling of notebooks)
# #     - `utils/` (containing pdf_generator.py and __init__.py)
# #     - The project root (`credit_risk_engine/`)

# #     Required files:
# #     - affordability_engine.py
# #     - reason_codes.py
# #     - risk_engine.py
# #     - utils/__init__.py
# #     - utils/pdf_generator.py
# #     """)
# #     st.stop()

# # # =============================================================================
# # # STAGE 2 ENGINE – ROBUST FALLBACK
# # # =============================================================================
# # try:
# #     import stage2_engine
# #     from stage2_engine import make_two_stage_decision, is_stage2_available, get_stage2_status
# #     STAGE2_AVAILABLE = is_stage2_available()
# # except ImportError:
# #     stage2_engine = None
# #     STAGE2_AVAILABLE = False
# #     def make_two_stage_decision(*args, **kwargs):
# #         raise NotImplementedError("Stage 2 engine not available")
# #     def is_stage2_available():
# #         return False
# #     def get_stage2_status():
# #         return {"error": "Stage 2 engine module not found", "available": False}

# # # =============================================================================
# # # PDF GENERATION – SAFE FALLBACK
# # # =============================================================================
# # PDF_AVAILABLE = False
# # generate_decision_pdf = None
# # generate_audit_pdf = None
# # try:
# #     from utils.pdf_generator import generate_decision_pdf, generate_audit_pdf
# #     PDF_AVAILABLE = True
# # except ImportError:
# #     pass

# # # =============================================================================
# # # JSON SANITIZER
# # # =============================================================================
# # def sanitize_for_json(obj: Any) -> Any:
# #     if obj is None or isinstance(obj, (str, int, float, bool)):
# #         return obj
# #     if isinstance(obj, set):
# #         return list(obj)
# #     if isinstance(obj, datetime):
# #         return obj.isoformat()
# #     if isinstance(obj, np.integer):
# #         return int(obj)
# #     if isinstance(obj, np.floating):
# #         return float(obj)
# #     if isinstance(obj, np.ndarray):
# #         return obj.tolist()
# #     if isinstance(obj, dict):
# #         return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
# #     if isinstance(obj, (list, tuple)):
# #         return [sanitize_for_json(item) for item in obj]
# #     try:
# #         json.dumps(obj)
# #         return obj
# #     except (TypeError, ValueError):
# #         return str(obj)

# # # =============================================================================
# # # LOAD TRAINED MODEL ASSETS (Stage 1 Random Forest)
# # # =============================================================================
# # @st.cache_resource
# # def load_model_assets():
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
# #         return {'loaded': False, 'error': 'credit_risk_assets.pkl not found. Please run the training script first.'}
# #     except Exception as e:
# #         return {'loaded': False, 'error': f'Error loading model: {str(e)}'}

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
# #     if principal <= 0 or tenure_months <= 0:
# #         return 0
# #     monthly_rate = annual_rate / (12 * 100)
# #     if monthly_rate == 0:
# #         return principal / tenure_months
# #     emi = (principal * monthly_rate * (1 + monthly_rate)**tenure_months) / \
# #           ((1 + monthly_rate)**tenure_months - 1)
# #     return round(emi, 2)

# # def calculate_affordability(monthly_income, loan_amount, interest_rate, tenure_months, existing_emi):
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
# #     'low_bureau':       'Credit score below minimum ({score} < 550)',
# #     'high_foir':        'EMI burden too high (FOIR: {foir}% > 50%)',
# #     'severe_dpd':       'Severe payment delays ({dpd} instances of 90+ DPD)',
# #     'moderate_dpd':     'Frequent payment delays ({dpd} instances of 30+ DPD)',
# #     'low_income':       'Income below minimum threshold (₹{income:,} < ₹15,000)',
# #     'short_employment': 'Insufficient employment tenure ({tenure} months < 6)',
# #     'short_vintage':    'Insufficient business vintage ({vintage} years < 2 years)',
# #     'bankruptcy':       'Active bankruptcy detected',
# #     'kyc_failed':       'KYC verification not completed',
# #     'fraud_flag':       'Fraud flag present on application',
# #     'high_utilization': 'High credit utilization ({util}% > 80%)',
# #     'age_invalid':      'Age outside acceptable range ({age} years, must be 24–70)',
# #     'high_dependents':  'High number of dependents ({deps}) reducing net disposable income',
# # }
# # REVIEW_REASONS = {
# #     'borderline_bureau':  'Credit score in borderline range ({score})',
# #     'moderate_foir':      'EMI burden moderate (FOIR: {foir}%)',
# #     'mixed_signals':      'Mixed credit indicators requiring human review',
# #     'recent_employment':  'Recent employment change requiring verification',
# #     'high_loan_amount':   'Large loan amount requiring additional underwriting review',
# #     'moderate_dpd':       'Recent 30-day payment delays requiring review ({dpd} instances)',
# #     'moderate_dependents':'Moderate number of dependents ({deps}) may affect repayment',
# # }

# # def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
# #     reasons = []
# #     bureau_score      = customer_data.get('bureau_score', 0)
# #     foir              = affordability_data.get('foir_percentage', 0)
# #     dpd_90            = customer_data.get('dpd_90_count_6m', 0)
# #     dpd_30            = customer_data.get('dpd_30_count_6m', 0)
# #     income            = customer_data.get('avg_salary_6m', 0)
# #     employment_tenure = customer_data.get('employment_tenure_months', 0)
# #     business_vintage  = customer_data.get('business_vintage_years', 0)
# #     employment_type   = customer_data.get('employment_type', 'Salaried')
# #     credit_util       = customer_data.get('credit_utilization_pct', 0)
# #     age               = customer_data.get('age', 0)
# #     dependents        = customer_data.get('dependents', 0)

# #     if decision == "APPROVE":
# #         if bureau_score >= 750:
# #             reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
# #         if employment_tenure >= 24:
# #             reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
# #         if foir <= 40:
# #             reasons.append(APPROVAL_REASONS['low_foir'].format(foir=round(foir, 1)))
# #         if dpd_90 == 0 and dpd_30 == 0:
# #             reasons.append(APPROVAL_REASONS['clean_payment'])
# #         if income >= 75000:
# #             reasons.append(APPROVAL_REASONS['strong_income'].format(income=income))
# #         if credit_util <= 30:
# #             reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))

# #     elif decision == "REJECT":
# #         for check_name, check_result in policy_checks.items():
# #             if '❌' in str(check_result):
# #                 cn = check_name.lower()
# #                 if 'bureau' in cn:
# #                     reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
# #                 elif 'dpd' in cn:
# #                     reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
# #                 elif 'income' in cn:
# #                     reasons.append(REJECTION_REASONS['low_income'].format(income=income))
# #                 elif 'tenure' in cn:
# #                     if employment_type == 'Salaried':
# #                         reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
# #                     else:
# #                         reasons.append(REJECTION_REASONS['short_vintage'].format(vintage=business_vintage))
# #                 elif 'kyc' in cn:
# #                     reasons.append(REJECTION_REASONS['kyc_failed'])
# #                 elif 'bankruptcy' in cn:
# #                     reasons.append(REJECTION_REASONS['bankruptcy'])
# #                 elif 'fraud' in cn:
# #                     reasons.append(REJECTION_REASONS['fraud_flag'])
# #                 elif 'age' in cn:
# #                     reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))
# #         if foir > 50:
# #             reasons.append(REJECTION_REASONS['high_foir'].format(foir=round(foir, 1)))
# #         if credit_util > 80:
# #             reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
# #         if dpd_30 >= 3 and dpd_90 == 0:
# #             reasons.append(REJECTION_REASONS['moderate_dpd'].format(dpd=dpd_30))
# #         if dependents >= 4:
# #             reasons.append(REJECTION_REASONS['high_dependents'].format(deps=dependents))

# #     elif decision == "REVIEW":
# #         if 650 <= bureau_score < 700:
# #             reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
# #         if 40 < foir <= 50:
# #             reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=round(foir, 1)))
# #         if employment_tenure < 12:
# #             reasons.append(REVIEW_REASONS['recent_employment'])
# #         if dpd_30 >= 1 and dpd_90 == 0:
# #             reasons.append(REVIEW_REASONS['moderate_dpd'].format(dpd=dpd_30))
# #         if 2 <= dependents < 4:
# #             reasons.append(REVIEW_REASONS['moderate_dependents'].format(deps=dependents))
# #         if not reasons:
# #             reasons.append(REVIEW_REASONS['mixed_signals'])

# #     return reasons[:3] if reasons else ['Decision based on comprehensive model assessment']

# # # =============================================================================
# # # PD CALCULATION
# # # =============================================================================
# # def bureau_score_to_pd(bureau_score):
# #     if bureau_score >= 800:
# #         return 0.5 + (900 - bureau_score) / 200 * 0.5
# #     elif bureau_score >= 750:
# #         return 1.0 + (800 - bureau_score) / 50 * 1.0
# #     elif bureau_score >= 700:
# #         return 2.0 + (750 - bureau_score) / 50 * 1.5
# #     elif bureau_score >= 650:
# #         return 3.5 + (700 - bureau_score) / 50 * 2.5
# #     elif bureau_score >= 600:
# #         return 6.0 + (650 - bureau_score) / 50 * 4.0
# #     elif bureau_score >= 550:
# #         return 10.0 + (600 - bureau_score) / 50 * 5.0
# #     else:
# #         return min(25.0, 15.0 + (550 - bureau_score) / 50 * 10.0)

# # def foir_to_pd_adjustment(foir_percentage):
# #     if foir_percentage <= 30:
# #         return -0.75
# #     elif foir_percentage <= 40:
# #         return 0.00
# #     elif foir_percentage <= 45:
# #         return 0.75
# #     elif foir_percentage <= 50:
# #         return 1.50
# #     elif foir_percentage <= 55:
# #         return 2.25
# #     elif foir_percentage <= 60:
# #         return 3.50
# #     else:
# #         return 6.00

# # def delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count=0):
# #     if dpd_90_count >= 3:
# #         return 5.0
# #     elif dpd_90_count == 2:
# #         return 3.0
# #     elif dpd_90_count == 1:
# #         return 2.0
# #     elif dpd_30_count >= 3:
# #         return 1.6
# #     elif dpd_30_count >= 1:
# #         return 1.3
# #     else:
# #         return 1.0

# # def employment_stability_to_pd_adjustment(employment_type, tenure_months, business_vintage_years=0):
# #     if employment_type == 'Salaried':
# #         if tenure_months >= 36:
# #             return -0.5
# #         elif tenure_months >= 12:
# #             return 0.0
# #         elif tenure_months >= 6:
# #             return 0.5
# #         else:
# #             return 2.0
# #     elif employment_type in ['Self-Employed', 'Business']:
# #         if business_vintage_years >= 5:
# #             return -0.5
# #         elif business_vintage_years >= 2:
# #             return 0.0
# #         else:
# #             return 1.5
# #     else:
# #         return 1.0

# # def inquiry_pattern_to_pd_adjustment(recent_inquiries_3m):
# #     if recent_inquiries_3m <= 1:
# #         return -0.3
# #     elif recent_inquiries_3m <= 3:
# #         return 0.0
# #     elif recent_inquiries_3m <= 5:
# #         return 0.8
# #     elif recent_inquiries_3m <= 8:
# #         return 1.5
# #     else:
# #         return 3.0

# # def ml_confidence_to_pd_adjustment(ml_confidence, ml_decision):
# #     if ml_decision == "APPROVE":
# #         if ml_confidence >= 90:
# #             return -0.5
# #         elif ml_confidence >= 70:
# #             return 0.0
# #         else:
# #             return 0.5
# #     elif ml_decision == "REVIEW":
# #         return 1.0
# #     else:
# #         return 5.0

# # def calculate_final_pd(bureau_score, foir, confidence, dpd_90_count=0, dpd_30_count=0,
# #                        employment_type='Salaried', employment_tenure=24, business_vintage=0,
# #                        recent_inquiries=2, ml_decision='APPROVE'):
# #     base_pd = bureau_score_to_pd(bureau_score)
# #     foir_adj = foir_to_pd_adjustment(foir)
# #     deliq_multiplier = delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count)
# #     employment_adj = employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage)
# #     inquiry_adj = inquiry_pattern_to_pd_adjustment(recent_inquiries)
# #     ml_adj = ml_confidence_to_pd_adjustment(confidence, ml_decision)
# #     adjusted_base_pd = base_pd * deliq_multiplier
# #     final_pd = adjusted_base_pd + foir_adj + employment_adj + inquiry_adj + ml_adj
# #     final_pd = max(0.5, min(final_pd, 25.0))
# #     return round(final_pd, 2)

# # # =============================================================================
# # # RISK SCORE CALCULATION
# # # =============================================================================
# # def calculate_final_risk_score(bureau_score, ml_confidence, foir,
# #                                 dpd_90, dpd_30, net_surplus,
# #                                 bounces=0, missing_months=0, active_loans=0):
# #     bureau_points = (bureau_score / 900) * 400
# #     ml_points = (ml_confidence / 100) * 300
# #     foir_points = max(0, (1 - foir / 50) * 150)
# #     dpd_penalty = min((dpd_90 * 50) + (dpd_30 * 20), 150)
# #     behavioral_penalty = min((bounces * 10) + (missing_months * 10), 100)
# #     if net_surplus > 50000:
# #         surplus_points = 50
# #     elif net_surplus > 0:
# #         surplus_points = 20
# #     elif net_surplus < -50000:
# #         surplus_points = -50
# #     else:
# #         surplus_points = -20
# #     total = (bureau_points + ml_points + foir_points
# #              + surplus_points - dpd_penalty - behavioral_penalty)
# #     return max(0, min(int(total), 1000))

# # # =============================================================================
# # # CATEGORICAL FLAG INFERENCE FROM CIBIL DATA
# # # v8.5: Dual-dataset calibration.
# # #
# # # Dataset A — train_60k_rule_accepted.csv (bank-statement enriched):
# # #   Has: net_cash_surplus_6m, inward_bounce_count_3m, salary_missing_months
# # #   payment_discipline: GOOD 99.9%,  MODERATE 0.02%, POOR 0.04%
# # #   cashflow_health   : MODERATE 90%, HEALTHY 8.8%, STRESSED 0.8%, STABLE 0.4%
# # #   liquidity_flag    : LOW 87.7%,   ADEQUATE 11.9%, MODERATE 0.4%
# # #   bureau_risk_flag  : LOW 97.9%,   HIGH 1.3%,      MEDIUM 0.75%
# # #   salary_stability  : MODERATE 85.8%, STABLE 12.1%, UNSTABLE 2.1%
# # #
# # # Dataset B — External_Cibil_Dataset.xlsx (bureau-only, 51,336 rows):
# # #   Has: num_times_30p_dpd, num_times_60p_dpd, num_lss, num_dbt,
# # #        NETMONTHLYINCOME, Time_With_Curr_Empr, Credit_Score
# # #   NO:  net_cash_surplus_6m, inward_bounce_count_3m, salary_missing_months
# # #   Income median: ₹23,000 (vs ₹50,000 in Dataset A — very different scale)
# # #   payment_discipline: POOR 10.5%, MODERATE 5.1%, GOOD 84.4%
# # #   bureau_risk_flag  : HIGH 5.0%,  MEDIUM 10.3%, LOW 84.7%
# # #   salary_stability  : UNSTABLE 0.04%, STABLE 11.2%, MODERATE 88.8%
# # #   Tier mapping      : P1(score 701+), P2(669-700), P3(subprime), P4(high risk)
# # #
# # # Auto-detection: if 'NETMONTHLYINCOME' key present → Dataset B path (bureau-only).
# # #                 Otherwise → Dataset A path (bank-statement enriched).
# # # =============================================================================
# # def _infer_surplus_from_cibil(score: int, dpd_60: int, dpd_30: int, income: float) -> float:
# #     """
# #     Estimate net cash surplus when no bank statement data is available.
# #     Used for External_Cibil_Dataset (bureau-only) OCR path.

# #     Calibrated against External_Cibil_Dataset tier distributions:
# #       - Score >= 700, clean DPD  -> income likely covers expenses  -> +30% income
# #       - Score 650-699, clean DPD -> borderline                      -> +10% income
# #       - Score < 650 OR 60+ DPD   -> stressed                        -> -20% income
# #       - 60+ DPD >= 3             -> severe stress                   -> -50% income
# #     """
# #     if dpd_60 >= 3:
# #         return income * -0.5
# #     elif score < 650 or dpd_60 >= 1:
# #         return income * -0.2
# #     elif score < 700:
# #         return income * 0.1
# #     else:
# #         return income * 0.3


# # def infer_categorical_flags(extraction_result: dict) -> dict:
# #     """
# #     Convert numeric CIBIL fields into the 5 categorical flags used by the
# #     Stage 1 assessment form.

# #     Automatically detects whether this is a bank-statement-enriched result
# #     (Dataset A / train_60k) or a bureau-only result (Dataset B / External CIBIL)
# #     and applies the appropriate calibrated thresholds for each.

# #     Args:
# #         extraction_result: dict returned by extract_cibil_from_pdf()

# #     Returns:
# #         dict with keys: payment_discipline_flag, cashflow_health,
# #                         liquidity_flag, bureau_risk_flag, salary_stability_flag
# #     """
# #     # ── Common fields (present in both datasets) ─────────────────────
# #     score       = int(extraction_result.get('Credit_Score', 700) or 700)
# #     dpd_30      = int(extraction_result.get('num_times_30p_dpd', 0) or 0)
# #     dpd_60      = int(extraction_result.get('num_times_60p_dpd', 0) or 0)
# #     written_off = int(extraction_result.get('num_lss', 0) or
# #                       extraction_result.get('written_off_count', 0) or 0)
# #     doubtful    = int(extraction_result.get('num_dbt', 0) or 0)
# #     cc_util_raw = extraction_result.get('CC_utilization', 0) or 0
# #     # Sentinel -99999 → 0 (no credit card on file)
# #     cc_util     = float(cc_util_raw) if cc_util_raw > 0 else 0.0
# #     income      = float(extraction_result.get('NETMONTHLYINCOME', 0) or
# #                         extraction_result.get('avg_salary_6m', 50_000) or 50_000)
# #     tenure      = int(extraction_result.get('Time_With_Curr_Empr', 24) or 24)

# #     # ── Detect dataset type ──────────────────────────────────────────
# #     # Dataset B (External CIBIL) uses NETMONTHLYINCOME key and lacks bank-stmt fields.
# #     # Dataset A (train_60k) uses avg_salary_6m and HAS surplus/bounce/missing.
# #     is_bureau_only = (
# #         'NETMONTHLYINCOME' in extraction_result
# #         and 'net_cash_surplus_6m' not in extraction_result
# #         and 'net_surplus' not in extraction_result
# #     )

# #     if is_bureau_only:
# #         # ── DATASET B PATH (External_Cibil_Dataset) ─────────────────
# #         # Income median ₹23k, score range 469-811, bureau fields only.
# #         # num_times_60p_dpd used as dpd_90 proxy (60+ includes 90+ DPD).

# #         dpd_90_proxy = dpd_60   # 60+ is the closest to 90+ in this dataset

# #         # Estimate surplus since no bank statement
# #         surplus = _infer_surplus_from_cibil(score, dpd_60, dpd_30, income)

# #         # 1. payment_discipline_flag
# #         # External CIBIL: POOR=10.5% (60+dpd>=1 OR 30+dpd>=3), MODERATE=5.1%, GOOD=84.4%
# #         if dpd_60 >= 1 or dpd_30 >= 3:
# #             payment_discipline = 'POOR'
# #         elif dpd_30 >= 1:
# #             payment_discipline = 'MODERATE'
# #         else:
# #             payment_discipline = 'GOOD'

# #         # 2. cashflow_health (derived from surplus proxy + DPD)
# #         # External CIBIL distribution via proxy: STABLE 84%, STRESSED 14%, HEALTHY 1.2%
# #         if surplus >= 14_000:
# #             cashflow_health = 'HEALTHY'
# #         elif surplus >= 600:
# #             cashflow_health = 'STABLE'
# #         elif surplus < -1_000:
# #             cashflow_health = 'STRESSED'
# #         else:
# #             cashflow_health = 'MODERATE'

# #         # 3. liquidity_flag (derived from surplus proxy)
# #         # External CIBIL proxy: ADEQUATE 1.2%, MODERATE 98.6%, LOW 0.1%
# #         # Note: income-based surplus rarely reaches extremes → mostly MODERATE
# #         if surplus > 14_000:
# #             liquidity_flag = 'ADEQUATE'
# #         elif surplus > -32_000:
# #             liquidity_flag = 'MODERATE'
# #         else:
# #             liquidity_flag = 'LOW'

# #         # 4. bureau_risk_flag
# #         # External CIBIL: HIGH=5.0%, MEDIUM=10.3%, LOW=84.7%
# #         # num_lss (written-off) and num_dbt (doubtful) are strong HIGH signals.
# #         if written_off >= 1 or doubtful >= 1 or dpd_60 >= 3 or score < 580:
# #             bureau_risk = 'HIGH'
# #         elif score < 650 or (dpd_30 >= 2 and cc_util > 0.60):
# #             bureau_risk = 'MEDIUM'
# #         else:
# #             bureau_risk = 'LOW'

# #         # 5. salary_stability_flag
# #         # External CIBIL: UNSTABLE=0.04%(tenure<6m), STABLE=11.2%(tenure>=24,score>=700)
# #         # No salary_missing_months → use employment tenure + score + DPD
# #         if tenure < 6:
# #             salary_stability = 'UNSTABLE'
# #         elif tenure >= 24 and score >= 700 and dpd_30 == 0:
# #             salary_stability = 'STABLE'
# #         else:
# #             salary_stability = 'MODERATE'

# #     else:
# #         # ── DATASET A PATH (train_60k / bank-statement enriched) ────
# #         # Has actual surplus, bounce count, and missing salary months.
# #         dpd_90      = int(extraction_result.get('dpd_90_count_6m', 0) or 0)
# #         bounces     = int(extraction_result.get('inward_bounce_count_3m', 0) or 0)
# #         missing     = int(extraction_result.get('salary_missing_months', 0) or 0)
# #         hard_reject = int(extraction_result.get('hard_reject_flag', 0) or 0)
# #         surplus     = float(
# #             extraction_result.get('net_cash_surplus_6m')
# #             or extraction_result.get('net_surplus')
# #             or -50_000
# #         )

# #         # 1. payment_discipline_flag
# #         # train_60k: POOR/MODERATE rows bounce mean ~1.0, GOOD mean=0.008.
# #         if dpd_90 >= 1 or bounces >= 2:
# #             payment_discipline = 'POOR'
# #         elif bounces == 1 or dpd_30 >= 3:
# #             payment_discipline = 'MODERATE'
# #         else:
# #             payment_discipline = 'GOOD'

# #         # 2. cashflow_health
# #         # train_60k: HEALTHY min surplus=14k, STABLE min=600, STRESSED max=-1k.
# #         if surplus >= 14_000:
# #             cashflow_health = 'HEALTHY'
# #         elif 600 <= surplus < 14_000:
# #             cashflow_health = 'STABLE'
# #         elif surplus < -1_000:
# #             cashflow_health = 'STRESSED'
# #         else:
# #             cashflow_health = 'MODERATE'

# #         # 3. liquidity_flag
# #         # train_60k: ADEQUATE median=+83k, MODERATE median=-32k, LOW median=-109k.
# #         if surplus > 14_000:
# #             liquidity_flag = 'ADEQUATE'
# #         elif surplus > -32_000:
# #             liquidity_flag = 'MODERATE'
# #         else:
# #             liquidity_flag = 'LOW'

# #         # 4. bureau_risk_flag
# #         # train_60k: HIGH ~99% hard_rejected, dpd_90 mean=6.1; MEDIUM score median=539.
# #         if hard_reject or dpd_90 >= 3 or written_off >= 1 or (dpd_90 >= 1 and dpd_30 >= 2):
# #             bureau_risk = 'HIGH'
# #         elif score < 580 or (dpd_30 >= 2 and cc_util > 0.60):
# #             bureau_risk = 'MEDIUM'
# #         else:
# #             bureau_risk = 'LOW'

# #         # 5. salary_stability_flag
# #         # train_60k: UNSTABLE missing>=1, STABLE cv~0.05+zero missing, MODERATE rest.
# #         if missing >= 1:
# #             salary_stability = 'UNSTABLE'
# #         elif missing == 0 and score >= 700 and dpd_30 == 0 and bounces == 0:
# #             salary_stability = 'STABLE'
# #         else:
# #             salary_stability = 'MODERATE'

# #     return {
# #         'payment_discipline_flag': payment_discipline,
# #         'cashflow_health':         cashflow_health,
# #         'liquidity_flag':          liquidity_flag,
# #         'bureau_risk_flag':        bureau_risk,
# #         'salary_stability_flag':   salary_stability,
# #         '_inference_path':         'bureau_only' if is_bureau_only else 'bank_statement',
# #         '_surplus_used':           surplus if is_bureau_only else locals().get('surplus', 0),
# #     }

# # # =============================================================================
# # # CIBIL PDF EXTRACTION ENGINE (OCR + PATTERN MATCHING) – OPTIONAL
# # # =============================================================================
# # def extract_cibil_from_pdf(uploaded_file):
# #     if not OCR_AVAILABLE:
# #         return {'success': False, 'error': OCR_ERROR_MSG or 'OCR libraries not installed. Check packages.txt and requirements.txt.'}

# #     try:
# #         pdf_bytes = uploaded_file.read()
# #         images = convert_from_bytes(pdf_bytes, dpi=300)
# #         full_text = ""
# #         for image in images:
# #             gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
# #             _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# #             full_text += pytesseract.image_to_string(binary) + "\n"

# #         credit_score = 720
# #         score_match = re.search(
# #             r'\b(\d{3})\s*(?:EXCELLENT|VERY\s*GOOD|GOOD|FAIR|SUBPRIME|POOR|NH|NA)\b',
# #             full_text, re.IGNORECASE
# #         )
# #         if score_match:
# #             val = int(score_match.group(1))
# #             if 300 <= val <= 900:
# #                 credit_score = val
# #         if credit_score == 720:
# #             score_match2 = re.search(
# #                 r'(?:cibil|credit)\s*score\s*[:\-\(]?\s*(\d{3})',
# #                 full_text, re.IGNORECASE
# #             )
# #             if score_match2:
# #                 val = int(score_match2.group(1))
# #                 if 300 <= val <= 900:
# #                     credit_score = val
# #         if credit_score == 720:
# #             score_match3 = re.search(r'score.*?\((\d{3})\)', full_text, re.IGNORECASE)
# #             if score_match3:
# #                 val = int(score_match3.group(1))
# #                 if 300 <= val <= 900:
# #                     credit_score = val

# #         monthly_income = 50000
# #         income_match = re.search(
# #             r'(?:net\s+monthly\s+income|monthly\s+income|net\s+income|salary)[^\n\r]{0,30}?'
# #             r'(?:rs\.?\s*|inr\s*|₹\s*)([\d,]+)',
# #             full_text, re.IGNORECASE
# #         )
# #         if income_match:
# #             val = int(income_match.group(1).replace(',', ''))
# #             if val > 1000:
# #                 monthly_income = val
# #         if monthly_income == 50000:
# #             income_match2 = re.search(r'(?:rs\.?\s*|₹\s*)([\d,]{4,})', full_text, re.IGNORECASE)
# #             if income_match2:
# #                 val = int(income_match2.group(1).replace(',', ''))
# #                 if 5000 <= val <= 1000000:
# #                     monthly_income = val

# #         cc_util_pct = 35
# #         util_match = re.search(r'utilization\s*[\(:\-]?\s*(\d{1,3})\s*%', full_text, re.IGNORECASE)
# #         if util_match:
# #             cc_util_pct = int(util_match.group(1))
# #         cc_util = cc_util_pct / 100.0
# #         high_util = 1 if cc_util_pct > 75 else 0

# #         age_extracted = 35
# #         dob_match = re.search(
# #             r'(?:date\s+of\s+birth|dob)[:\s]+(\d{2}[-/]\w{3,9}[-/]\d{2,4})',
# #             full_text, re.IGNORECASE
# #         )
# #         if dob_match:
# #             try:
# #                 from datetime import datetime as _dt
# #                 dob_str = dob_match.group(1)
# #                 for fmt in ('%d-%b-%Y', '%d/%b/%Y', '%d-%m-%Y', '%d/%m/%Y'):
# #                     try:
# #                         dob = _dt.strptime(dob_str, fmt)
# #                         age_extracted = int((datetime.now() - dob).days / 365.25)
# #                         break
# #                     except Exception:
# #                         continue
# #             except Exception:
# #                 pass

# #         biz_vintage = 3
# #         biz_match = re.search(r'business\s+vintage.*?(\d+)', full_text, re.IGNORECASE)
# #         if biz_match:
# #             biz_vintage = int(biz_match.group(1))

# #         lines = full_text.split('\n')
# #         in_accounts = False
# #         in_enquiry = False
# #         accounts = []
# #         enquiry_dates = []

# #         for line in lines:
# #             line_up = line.upper()
# #             if 'ACCOUNT DETAILS' in line_up:
# #                 in_accounts = True
# #                 in_enquiry = False
# #                 continue
# #             if 'ENQUIRY DETAILS' in line_up:
# #                 in_accounts = False
# #                 in_enquiry = True
# #                 continue

# #             if in_accounts:
# #                 if re.search(r'SUMMARY|SCORE|PERSONAL\s+INFO', line_up):
# #                     break
# #                 if re.search(r'\bLender\b|\bAccount\s*No\b|\bOpen\s*Date\b|\bDPD\b|\bStatus\b', line, re.IGNORECASE):
# #                     continue
# #                 stripped = line.strip()
# #                 if not stripped:
# #                     continue
# #                 dpd_match = re.search(r'\b(0\d0|0\d\d|\d{3})\b', stripped)
# #                 status_match = re.search(
# #                     r'\b(Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss)\s*$',
# #                     stripped, re.IGNORECASE
# #                 )
# #                 if (re.search(r'\bINR\b', stripped, re.IGNORECASE) or
# #                         re.match(r'^[A-Z][a-zA-Z\s]+(?:Bank|Finance|Capital|Fincorp|SBI|ICICI|HDFC|Axis|Bajaj|Tata|Kotak)', stripped)):
# #                     dpd_val = int(dpd_match.group(1)) if dpd_match else 0
# #                     status_str = status_match.group(1) if status_match else 'Active'
# #                     accounts.append({'dpd': dpd_val, 'status': status_str.lower()})

# #             if in_enquiry:
# #                 enq_date = re.match(r'^\s*(\d{2}-[A-Za-z]{3}-\d{4})', line)
# #                 if enq_date:
# #                     enquiry_dates.append(enq_date.group(1))

# #         written_off_count = 0
# #         settled_count = 0
# #         dpd_90_count = 0
# #         dpd_60_count = 0
# #         dpd_30_count = 0
# #         active_count = 0
# #         sub_standard_count = 0

# #         if accounts:
# #             for acc in accounts:
# #                 dpd = acc.get('dpd', 0)
# #                 status = acc.get('status', '')
# #                 if dpd >= 90:
# #                     dpd_90_count += 1
# #                 elif dpd >= 60:
# #                     dpd_60_count += 1
# #                 elif dpd >= 30:
# #                     dpd_30_count += 1
# #                 if 'written' in status:
# #                     written_off_count += 1
# #                 elif 'settled' in status:
# #                     settled_count += 1
# #                 elif 'active' in status:
# #                     active_count += 1
# #                 if dpd >= 30:
# #                     sub_standard_count += 1
# #         else:
# #             written_off_count = len(re.findall(r'\bwritten[-\s]?off\b', full_text, re.IGNORECASE))
# #             settled_count     = len(re.findall(r'\bsettled\b', full_text, re.IGNORECASE))
# #             dpd_90_count      = len(re.findall(r'\b090\b|90\+?\s*dpd', full_text, re.IGNORECASE))
# #             dpd_60_count      = len(re.findall(r'\b060\b|60\+?\s*dpd', full_text, re.IGNORECASE))
# #             dpd_30_count      = len(re.findall(r'\b030\b|30\+?\s*dpd', full_text, re.IGNORECASE))
# #             active_sum = re.search(r'Total\s+Accounts\s+Active.*?(\d+)\s+(\d+)', full_text, re.IGNORECASE)
# #             if active_sum:
# #                 active_count = int(active_sum.group(2))

# #         if active_count == 0:
# #             summary_match = re.search(
# #                 r'Total\s+Accounts\s+Active[^\n]*\n\s*(\d+)\s+(\d+)',
# #                 full_text, re.IGNORECASE
# #             )
# #             if summary_match:
# #                 active_count = int(summary_match.group(2))
# #             else:
# #                 inline = re.search(
# #                     r'(?:Total\s+Accounts.*?Active.*?Closed.*?\n|(\d+)\s+(\d+)\s+(\d+)\s+[\d,]+\s+\d+)',
# #                     full_text, re.IGNORECASE
# #                 )
# #                 if inline and inline.group(2):
# #                     active_count = int(inline.group(2))

# #         enq_12m_total = len(enquiry_dates)
# #         enq_sum_match = re.search(r'Enquiries?\s*\(?12M\)?\s*[:\s]+(\d+)', full_text, re.IGNORECASE)
# #         if enq_sum_match:
# #             enq_12m_total = max(enq_12m_total, int(enq_sum_match.group(1)))

# #         enq_L3m = min(len(enquiry_dates), enq_12m_total)
# #         enq_L6m = enq_12m_total
# #         enq_L12m = enq_12m_total

# #         if credit_score >= 750 and (written_off_count > 0 or dpd_90_count > 0):
# #             credit_score = 550

# #         total_accounts = max(len(accounts), active_count + settled_count + written_off_count)
# #         pct_active = active_count / total_accounts if total_accounts > 0 else 0.6

# #         # ── Employment tenure extraction ──────────────────────────────
# #         # Try: "X years Y months at <employer>", "employed since <date>",
# #         # "total employment X months", or fallback to biz_vintage * 12
# #         employment_tenure_months = biz_vintage * 12
# #         tenure_match = re.search(
# #             r'(?:employed\s+for|employment\s+tenure|with\s+current\s+employer)[^\d]*(\d+)\s*(?:year|yr)',
# #             full_text, re.IGNORECASE
# #         )
# #         if tenure_match:
# #             employment_tenure_months = int(tenure_match.group(1)) * 12
# #         else:
# #             tenure_m = re.search(
# #                 r'(?:employed\s+for|employment\s+tenure)[^\d]*(\d+)\s*month',
# #                 full_text, re.IGNORECASE
# #             )
# #             if tenure_m:
# #                 employment_tenure_months = int(tenure_m.group(1))

# #         # ── Gender / Marital / Education extraction ───────────────────
# #         gender = 'M'
# #         if re.search(r'\bfemale\b|\bF\b', full_text, re.IGNORECASE):
# #             gender = 'F'

# #         marital_status = 'Married'
# #         if re.search(r'\bsingle\b|\bunmarried\b', full_text, re.IGNORECASE):
# #             marital_status = 'Single'

# #         education = 'GRADUATE'
# #         for edu_pat, edu_val in [
# #             (r'post.?grad', 'POST-GRADUATE'),
# #             (r'professional', 'PROFESSIONAL'),
# #             (r'under.?grad', 'UNDER GRADUATE'),
# #             (r'\b12th\b|\bhsc\b', '12TH'),
# #             (r'\bssc\b|\b10th\b', 'SSC'),
# #         ]:
# #             if re.search(edu_pat, full_text, re.IGNORECASE):
# #                 education = edu_val
# #                 break

# #         # ── Last / first product enquiry ─────────────────────────────
# #         prod_enq_map = {
# #             r'personal\s+loan': 'PL',
# #             r'credit\s+card':   'CC',
# #             r'home\s+loan':     'HL',
# #             r'auto\s+loan|car\s+loan': 'AL',
# #             r'consumer\s+loan': 'ConsumerLoan',
# #         }
# #         last_prod_enq = 'others'
# #         first_prod_enq = 'others'
# #         for pat, label in prod_enq_map.items():
# #             if re.search(pat, full_text, re.IGNORECASE):
# #                 last_prod_enq = label
# #                 first_prod_enq = label
# #                 break

# #         # ── Compute net surplus proxy (since no bank statement in PDF) ─
# #         # Uses calibrated income-based formula from External_Cibil_Dataset analysis.
# #         # Available if income was extracted; used by infer_categorical_flags().
# #         # dpd_60_count is the 60+ DPD proxy for this dataset.
# #         surplus_proxy = _infer_surplus_from_cibil(
# #             score=credit_score,
# #             dpd_60=dpd_60_count,
# #             dpd_30=dpd_30_count,
# #             income=float(monthly_income)
# #         )

# #         extracted_data = {
# #             # ── Core Credit Score ──────────────────────────────────────
# #             'Credit_Score': credit_score,

# #             # ── Delinquency (External CIBIL naming convention) ─────────
# #             'max_delinquency_level':    max(dpd_90_count * 90, dpd_60_count * 60, dpd_30_count * 30),
# #             'max_recent_level_of_deliq': max(dpd_60_count * 60, dpd_30_count * 30),
# #             'recent_level_of_deliq':    max(dpd_60_count * 60, dpd_30_count * 30),
# #             'num_times_30p_dpd':        dpd_30_count,
# #             'num_times_60p_dpd':        dpd_60_count,   # 60+ used as dpd_90 proxy
# #             'num_times_delinquent':     dpd_30_count + dpd_60_count + dpd_90_count,
# #             'num_deliq_6mts':           dpd_30_count + dpd_60_count + dpd_90_count,
# #             'num_deliq_12mts':          dpd_30_count + dpd_60_count + dpd_90_count,
# #             'num_deliq_6_12mts':        0,
# #             'max_deliq_6mts':           dpd_90_count if dpd_90_count > 0 else dpd_60_count,
# #             'max_deliq_12mts':          dpd_90_count if dpd_90_count > 0 else dpd_60_count,

# #             # ── Account Quality (External CIBIL naming) ────────────────
# #             'num_std':      active_count,
# #             'num_std_6mts': active_count,
# #             'num_std_12mts': active_count,
# #             'num_sub':      sub_standard_count,
# #             'num_sub_6mts': sub_standard_count,
# #             'num_sub_12mts': sub_standard_count,
# #             'num_dbt':      dpd_90_count,       # doubtful ≈ 90+ DPD proxy
# #             'num_dbt_6mts': 0,
# #             'num_dbt_12mts': 0,
# #             'num_lss':      written_off_count,  # loss/written-off
# #             'num_lss_6mts': 0,
# #             'num_lss_12mts': 0,

# #             # ── Enquiry fields ─────────────────────────────────────────
# #             'enq_L3m':  enq_L3m,
# #             'enq_L6m':  enq_L6m,
# #             'enq_L12m': enq_L12m,
# #             'tot_enq':  enq_L12m,
# #             'CC_enq':   0,  'CC_enq_L6m': 0,  'CC_enq_L12m': 0,
# #             'PL_enq':   0,  'PL_enq_L6m': 0,  'PL_enq_L12m': 0,
# #             'time_since_recent_enq': 30,

# #             # ── Utilization ────────────────────────────────────────────
# #             'pct_of_active_TLs_ever':      round(pct_active, 2),
# #             'pct_opened_TLs_L6m_of_L12m':  0.3,
# #             'pct_currentBal_all_TL':        0.3,
# #             'CC_utilization':               round(cc_util, 2) if cc_util > 0 else -99999,
# #             'PL_utilization':               0.25,
# #             'max_unsec_exposure_inPct':     cc_util_pct if cc_util_pct > 0 else 0,
# #             'pct_PL_enq_L6m_of_L12m':      0.0,
# #             'pct_CC_enq_L6m_of_L12m':      0.0,
# #             'pct_PL_enq_L6m_of_ever':      0.0,
# #             'pct_CC_enq_L6m_of_ever':      0.0,

# #             # ── Demographics (External CIBIL fields) ───────────────────
# #             'AGE':                  age_extracted,
# #             'NETMONTHLYINCOME':     monthly_income,    # ← External CIBIL key (not avg_salary_6m)
# #             'Time_With_Curr_Empr':  employment_tenure_months,
# #             'GENDER':               gender,
# #             'MARITALSTATUS':        marital_status,
# #             'EDUCATION':            education,

# #             # ── Product flags ──────────────────────────────────────────
# #             'CC_Flag': 1 if re.search(r'credit card', full_text, re.IGNORECASE) else 0,
# #             'PL_Flag': 1 if re.search(r'personal loan', full_text, re.IGNORECASE) else 0,
# #             'HL_Flag': 1 if re.search(r'home loan', full_text, re.IGNORECASE) else 0,
# #             'GL_Flag': 1 if re.search(r'gold loan', full_text, re.IGNORECASE) else 0,
# #             'last_prod_enq2':  last_prod_enq,
# #             'first_prod_enq2': first_prod_enq,

# #             # ── Time-since fields (sentinel if no event found) ─────────
# #             'time_since_recent_payment':     70,
# #             'time_since_first_deliquency':   -99999 if dpd_30_count == 0 else 180,
# #             'time_since_recent_deliquency':  -99999 if dpd_30_count == 0 else 90,

# #             # ── Surplus proxy (for infer_categorical_flags auto-path) ──
# #             # NOTE: NETMONTHLYINCOME key is set above, which triggers bureau_only path.
# #             # surplus_proxy stored here for session display only.
# #             '_surplus_proxy': int(surplus_proxy),

# #             # ── Legacy / internal fields ───────────────────────────────
# #             'written_off_count':    written_off_count,   # legacy alias
# #             'settled_count':        settled_count,
# #             'high_util_flag':       high_util,
# #             'dpd_90_count_6m':      dpd_90_count,        # Stage 1 form field name
# #             'recent_deliq_flag':    1 if (dpd_90_count > 0 or dpd_60_count > 0) else 0,
# #             'account_quality_score': max(0, 100
# #                 - written_off_count * 20
# #                 - settled_count * 10
# #                 - dpd_90_count * 15
# #                 - dpd_30_count * 5),

# #             # ── Metadata ──────────────────────────────────────────────
# #             'raw_text':          full_text,
# #             'success':           True,
# #             'extraction_method': 'OCR+ExternalCIBIL',
# #         }
# #         return extracted_data
# #     except Exception as e:
# #         return {'error': str(e), 'message': f'Error extracting CIBIL data: {str(e)}', 'success': False}

# # # =============================================================================
# # # HYBRID DECISION ENGINE
# # # =============================================================================
# # def make_hybrid_decision_enhanced(customer_dict):
# #     fill_missing_ml_fields(customer_dict)

# #     policy_checks = {}
# #     age = customer_dict.get('age', 0)
# #     employment_type = customer_dict.get('employment_type', 'Salaried')
# #     kyc_verified = customer_dict.get('kyc_verified', True)
# #     bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
# #     fraud_flag = customer_dict.get('fraud_flag', False)
# #     age_min, age_max = 24, 70
# #     if age < age_min or age > age_max:
# #         policy_checks['age'] = f"❌ Age {age} (Required: {age_min}-{age_max})"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Age outside allowed range", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 100.0, 'affordability_data': {}}
# #     policy_checks['age'] = f"✅ Age {age} (Valid)"
# #     if not kyc_verified:
# #         policy_checks['kyc'] = "❌ KYC Not Verified"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: KYC verification required", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 100.0, 'affordability_data': {}}
# #     policy_checks['kyc'] = "✅ KYC Verified"
# #     if bankruptcy_flag:
# #         policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Active bankruptcy", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 100.0, 'affordability_data': {}}
# #     policy_checks['bankruptcy'] = "✅ No Bankruptcy"
# #     if fraud_flag:
# #         policy_checks['fraud'] = "❌ Fraud Flag"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Fraud detected", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 100.0, 'affordability_data': {}}
# #     policy_checks['fraud'] = "✅ No Fraud History"

# #     dependents = customer_dict.get('dependents', 0)
# #     dependents_flag_review = False
# #     if dependents > 5:
# #         policy_checks['dependents'] = f"⚠️ Dependents {dependents} (>5: Review Required)"
# #         dependents_flag_review = True
# #     else:
# #         policy_checks['dependents'] = f"✅ Dependents {dependents} (Acceptable)"

# #     monthly_income = customer_dict.get('avg_salary_6m', 0)
# #     employment_tenure = customer_dict.get('employment_tenure_months', 0)
# #     business_vintage = customer_dict.get('business_vintage_years', 0)
# #     if monthly_income < 15000:
# #         policy_checks['income'] = f"❌ Income ₹{monthly_income:,.0f} (Min: ₹15,000)"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Income below minimum", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 100.0, 'affordability_data': {}}
# #     policy_checks['income'] = f"✅ Income ₹{monthly_income:,.0f}"
# #     if employment_type == 'Salaried' and employment_tenure < 6:
# #         policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient tenure", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 100.0, 'affordability_data': {}}
# #     elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
# #         policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient business vintage", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 100.0, 'affordability_data': {}}
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
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Bureau score too low", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 100.0, 'affordability_data': {}}
# #     policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"
# #     if dpd_90 > 0:
# #         policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Severe delinquency", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 100.0, 'affordability_data': {}}
# #     policy_checks['dpd'] = "✅ No 90+ DPD"
# #     if credit_utilization > 80:
# #         policy_checks['utilization'] = f"⚠️ High utilization {credit_utilization}%"
# #     else:
# #         policy_checks['utilization'] = f"✅ Utilization {credit_utilization}%"
# #     if recent_inquiries > 5:
# #         policy_checks['inquiries'] = f"⚠️ {recent_inquiries} recent inquiries"
# #     else:
# #         policy_checks['inquiries'] = f"✅ {recent_inquiries} inquiries"

# #     active_loans = customer_dict.get('active_loans_count', 0)
# #     if active_loans >= 5:
# #         policy_checks['active_loans'] = f"⚠️ High active loans ({int(active_loans)}) — Review"
# #         active_loans_flag = True
# #     else:
# #         policy_checks['active_loans'] = f"✅ Active loans: {int(active_loans)}"
# #         active_loans_flag = False

# #     salary_stability = customer_dict.get('salary_stability_flag', 'STABLE')
# #     if salary_stability == 'UNSTABLE':
# #         policy_checks['salary'] = "⚠️ Unstable salary — Review required"
# #         salary_flag = True
# #     elif salary_stability == 'MODERATE':
# #         policy_checks['salary'] = "⚠️ Moderate salary stability"
# #         salary_flag = False
# #     else:
# #         policy_checks['salary'] = "✅ Stable salary"
# #         salary_flag = False

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
# #     ml_raw_decision = ml_decision
# #     try:
# #         pred_proba = MODEL.predict_proba(final_input)[0]
# #         confidence = max(pred_proba) * 100
# #         class_probs = {cls: prob * 100 for cls, prob in zip(TARGET_LE.classes_, pred_proba)}
# #     except Exception:
# #         confidence = 75.0
# #         class_probs = {ml_decision: 100.0}

# #     loan_amount = customer_dict.get('loan_amount', 0)
# #     loan_tenure = customer_dict.get('loan_tenure_months', 12)
# #     interest_rate = customer_dict.get('interest_rate', 10.5)
# #     existing_emi = customer_dict.get('existing_emi', 0)
# #     affordability_data = calculate_affordability(monthly_income, loan_amount, interest_rate, loan_tenure, existing_emi)
# #     foir = affordability_data['foir_percentage']

# #     if foir > 50:
# #         ml_decision = "REJECT"
# #         policy_checks['foir'] = f"❌ FOIR {foir:.1f}% exceeds maximum allowed (50%)"

# #     if dependents_flag_review and ml_decision == "APPROVE":
# #         ml_decision = "REVIEW"
# #     if active_loans_flag and ml_decision == "APPROVE":
# #         ml_decision = "REVIEW"
# #     if salary_flag and ml_decision == "APPROVE":
# #         ml_decision = "REVIEW"

# #     risk_score = calculate_final_risk_score(
# #         bureau_score=bureau_score,
# #         ml_confidence=confidence,
# #         foir=foir,
# #         dpd_90=dpd_90,
# #         dpd_30=customer_dict.get('dpd_30_count_6m', 0),
# #         net_surplus=customer_dict.get('net_cash_surplus_6m', 0),
# #         bounces=customer_dict.get('inward_bounce_count_3m', 0),
# #         missing_months=customer_dict.get('salary_missing_months', 0),
# #         active_loans=active_loans
# #     )

# #     pd_percentage = calculate_final_pd(
# #         bureau_score=bureau_score,
# #         foir=foir,
# #         confidence=confidence,
# #         dpd_90_count=dpd_90,
# #         dpd_30_count=customer_dict.get('dpd_30_count_6m', 0),
# #         employment_type=employment_type,
# #         employment_tenure=employment_tenure,
# #         business_vintage=business_vintage,
# #         recent_inquiries=recent_inquiries,
# #         ml_decision=ml_decision
# #     )

# #     return {
# #         'decision': ml_decision,
# #         'ml_raw_decision': ml_raw_decision,
# #         'reason': "Decision based on comprehensive assessment",
# #         'confidence': confidence,
# #         'class_probs': class_probs,
# #         'policy_checks': policy_checks,
# #         'risk_score': risk_score,
# #         'pd_percentage': round(pd_percentage, 2),
# #         'affordability_data': affordability_data
# #     }

# # # =============================================================================
# # # BATCH PREDICTION ENGINE
# # # =============================================================================
# # def process_batch_predictions(df):
# #     results = []
# #     for idx, row in df.iterrows():
# #         customer_dict = row.to_dict()
# #         for key, value in customer_dict.items():
# #             if isinstance(value, str):
# #                 if value.lower() in ['yes', 'true', '1']:
# #                     customer_dict[key] = True
# #                 elif value.lower() in ['no', 'false', '0']:
# #                     customer_dict[key] = False
# #         required_fields = {
# #             'age': 35, 'employment_type': 'Salaried', 'kyc_verified': True,
# #             'bankruptcy_flag': False, 'fraud_flag': False, 'employment_tenure_months': 24,
# #             'business_vintage_years': 0, 'bureau_score': 700, 'dpd_90_count_6m': 0,
# #             'dpd_30_count_6m': 0, 'credit_utilization_pct': 30, 'recent_inquiries_3m': 0,
# #             'active_loans_count': 0, 'existing_emi': 0, 'avg_salary_6m': 50000,
# #             'AMT_INCOME_TOTAL': 600000, 'net_cash_surplus_6m': 20000, 'salary_stability_flag': 'STABLE',
# #             'loan_amount': 180000, 'loan_tenure_months': 24, 'interest_rate': 10.5, 'AMT_ANNUITY': 8500,
# #             'dependents': 0,
# #             'payment_discipline_flag': 'GOOD',
# #             'liquidity_flag': 'LOW',
# #             'cashflow_health': 'MODERATE',
# #             'bureau_risk_flag': 'LOW',
# #             'inward_bounce_count_3m': 0,
# #             'salary_missing_months': 0,
# #         }
# #         for field, default in required_fields.items():
# #             if field not in customer_dict or pd.isna(customer_dict[field]):
# #                 customer_dict[field] = default
# #         try:
# #             decision_data = make_hybrid_decision_enhanced(customer_dict)
# #             reasons = generate_reason_codes(
# #                 decision=decision_data.get('decision', 'ERROR'),
# #                 customer_data=customer_dict,
# #                 affordability_data=decision_data.get('affordability_data', {}),
# #                 policy_checks=decision_data.get('policy_checks', {})
# #             )
# #             app_id = f"BATCH_{idx+1:04d}"
# #             affordability = decision_data.get('affordability_data', {})
# #             result = {
# #                 'application_id': app_id,
# #                 'decision': decision_data.get('decision', 'ERROR'),
# #                 'risk_score': decision_data.get('risk_score', 0),
# #                 'pd_percentage': decision_data.get('pd_percentage', 0),
# #                 'confidence': round(decision_data.get('confidence', 0), 2),
# #                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# #                 'reason_1': reasons[0] if len(reasons) > 0 else '',
# #                 'reason_2': reasons[1] if len(reasons) > 1 else '',
# #                 'reason_3': reasons[2] if len(reasons) > 2 else '',
# #                 'age': customer_dict.get('age', ''),
# #                 'employment_type': customer_dict.get('employment_type', ''),
# #                 'bureau_score': customer_dict.get('bureau_score', ''),
# #                 'monthly_income': customer_dict.get('avg_salary_6m', ''),
# #                 'loan_amount': customer_dict.get('loan_amount', ''),
# #                 'loan_tenure_months': customer_dict.get('loan_tenure_months', ''),
# #                 'interest_rate': customer_dict.get('interest_rate', ''),
# #                 'new_emi': affordability.get('new_emi', 0),
# #                 'existing_emi': affordability.get('existing_emi', 0),
# #                 'total_emi': affordability.get('total_emi', 0),
# #                 'foir_percentage': round(affordability.get('foir_percentage', 0), 2),
# #                 'net_disposable': affordability.get('net_disposable', 0),
# #                 'affordability_status': affordability.get('status', 'N/A'),
# #                 'dpd_90_count': customer_dict.get('dpd_90_count_6m', 0),
# #                 'dpd_30_count': customer_dict.get('dpd_30_count_6m', 0),
# #                 'credit_utilization': customer_dict.get('credit_utilization_pct', 0),
# #                 'recent_inquiries': customer_dict.get('recent_inquiries_3m', 0),
# #                 'active_loans': customer_dict.get('active_loans_count', 0),
# #                 'employment_tenure': customer_dict.get('employment_tenure_months', 0),
# #                 'business_vintage': customer_dict.get('business_vintage_years', 0),
# #                 'salary_stability': customer_dict.get('salary_stability_flag', ''),
# #                 'kyc_status': 'Verified' if customer_dict.get('kyc_verified', True) else 'Not Verified',
# #                 'bankruptcy': 'Yes' if customer_dict.get('bankruptcy_flag', False) else 'No',
# #                 'fraud': 'Yes' if customer_dict.get('fraud_flag', False) else 'No',
# #                 'prob_approve': round(decision_data.get('class_probs', {}).get('APPROVE', 0), 2),
# #                 'prob_review': round(decision_data.get('class_probs', {}).get('REVIEW', 0), 2),
# #                 'prob_reject': round(decision_data.get('class_probs', {}).get('REJECT', 0), 2),
# #             }
# #         except Exception as e:
# #             result = {
# #                 'application_id': f"BATCH_{idx+1:04d}",
# #                 'decision': 'ERROR',
# #                 'risk_score': 0, 'pd_percentage': 0, 'confidence': 0,
# #                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# #                 'reason_1': '', 'reason_2': '', 'reason_3': '',
# #                 'age': customer_dict.get('age', ''),
# #                 'employment_type': customer_dict.get('employment_type', ''),
# #                 'bureau_score': customer_dict.get('bureau_score', ''),
# #                 'monthly_income': customer_dict.get('avg_salary_6m', ''),
# #                 'loan_amount': customer_dict.get('loan_amount', ''),
# #                 'error_message': str(e)
# #             }
# #         results.append(result)
# #     return pd.DataFrame(results)

# # def create_download_link(df, filename="batch_results.csv"):
# #     csv = df.to_csv(index=False)
# #     b64 = base64.b64encode(csv.encode()).decode()
# #     return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;">📥 Download CSV</a>'

# # # =============================================================================
# # # MODERN UI COMPONENTS
# # # =============================================================================
# # def render_decision_header(decision_data, customer_data):
# #     decision = decision_data.get('decision', 'ERROR')
# #     risk_score = decision_data.get('risk_score', 0)
# #     pd_score = decision_data.get('pd_percentage', 0)
# #     approved_amount = customer_data.get('loan_amount', 0)
# #     tenure = customer_data.get('loan_tenure_months', 24)
# #     app_id = customer_data.get('application_id', 'N/A')
# #     timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
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
# #             <div class="decision-title"><span>{icon}</span><span>{decision}</span></div>
# #             <div class="decision-subtitle">{subtitle}</div>
# #         </div>
# #     """, unsafe_allow_html=True)
# #     col1, col2, col3, col4, col5 = st.columns(5)
# #     with col1:
# #         st.markdown(f'<div class="stat-card"><div class="stat-number">{risk_score}</div><div class="stat-label">Risk Score</div></div>', unsafe_allow_html=True)
# #     with col2:
# #         st.markdown(f'<div class="stat-card"><div class="stat-number">{pd_score}%</div><div class="stat-label">PD Score</div></div>', unsafe_allow_html=True)
# #     with col3:
# #         st.markdown(f'<div class="stat-card"><div class="stat-number">₹{approved_amount:,.0f}</div><div class="stat-label">Loan Amount</div></div>', unsafe_allow_html=True)
# #     with col4:
# #         st.markdown(f'<div class="stat-card"><div class="stat-number">{tenure}</div><div class="stat-label">Tenure (Months)</div></div>', unsafe_allow_html=True)
# #     with col5:
# #         st.markdown(f'<div class="stat-card"><div class="stat-number">{decision_data.get("confidence", 0):.0f}%</div><div class="stat-label">Confidence</div></div>', unsafe_allow_html=True)
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.markdown(f'<div class="info-box"><strong>📋 Application ID:</strong> {app_id}</div>', unsafe_allow_html=True)
# #     with col2:
# #         st.markdown(f'<div class="info-box"><strong>🕐 Decision Timestamp:</strong> {timestamp}</div>', unsafe_allow_html=True)

# # def render_info_card(title, icon, data_dict, status_dict=None):
# #     st.markdown(f'<div class="info-card"><div class="info-card-title"><span class="icon">{icon}</span><span>{title}</span></div><div class="info-card-content">', unsafe_allow_html=True)
# #     for label, value in data_dict.items():
# #         status = ""
# #         if status_dict and label in status_dict:
# #             if status_dict[label] == "pass":
# #                 status = '<span class="status-badge badge-pass">✓ Passed</span>'
# #             elif status_dict[label] == "fail":
# #                 status = '<span class="status-badge badge-fail">✗ Failed</span>'
# #             elif status_dict[label] == "warning":
# #                 status = '<span class="status-badge badge-warning">⚠ Warning</span>'
# #         st.markdown(f'<div class="data-row"><span class="data-label">{label}</span><span class="data-value">{value} {status}</span></div>', unsafe_allow_html=True)
# #     st.markdown('</div></div>', unsafe_allow_html=True)

# # def render_reason_codes(reasons):
# #     st.markdown('<div class="info-card"><div class="info-card-title"><span class="icon">📝</span><span>Decision Reasons</span></div><div class="info-card-content">', unsafe_allow_html=True)
# #     for i, reason in enumerate(reasons, 1):
# #         st.markdown(f'<div class="reason-item"><span class="reason-icon">{i}.</span><span>{reason}</span></div>', unsafe_allow_html=True)
# #     st.markdown('</div></div>', unsafe_allow_html=True)

# # def create_modern_gauge(value, title, max_value=100):
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
# #             'bgcolor': 'white', 'borderwidth': 0,
# #             'steps': [
# #                 {'range': [0, 50], 'color': '#fed7d7'},
# #                 {'range': [50, 75], 'color': '#feebc8'},
# #                 {'range': [75, 100], 'color': '#c6f6d5'}
# #             ]
# #         }
# #     ))
# #     fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white',
# #                       font={'family': 'Inter', 'color': '#2d3748'})
# #     return fig

# # def create_modern_bar_chart(class_probs):
# #     df = pd.DataFrame({'Decision': list(class_probs.keys()), 'Probability': list(class_probs.values())})
# #     colors = {'REVIEW': '#ed8936', 'APPROVE': '#48bb78', 'REJECT': '#f56565'}
# #     fig = px.bar(df, x='Decision', y='Probability', title='Decision Probabilities', color='Decision',
# #                  color_discrete_map=colors, text='Probability')
# #     fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
# #     fig.update_layout(
# #         showlegend=False, yaxis_title='Probability (%)', xaxis_title='', height=300,
# #         margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white', plot_bgcolor='white',
# #         font={'family': 'Inter', 'color': '#2d3748'},
# #         yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]},
# #         xaxis={'tickfont': {'size': 14, 'color': '#2d3748'}}
# #     )
# #     return fig

# # # =============================================================================
# # # STAGE 2 BINARY RESOLVER
# # # =============================================================================
# # def resolve_stage2_to_binary(stage2_result: dict) -> dict:
# #     result = stage2_result.copy()
# #     tier   = result.get('stage2_tier', '')
# #     raw    = result.get('final_decision', '')
# #     score  = result.get('combined_risk_score', 0) or 0

# #     TIER_TO_DECISION = {
# #         'P1': 'APPROVE',
# #         'P2': 'APPROVE',
# #         'P3': 'REJECT',
# #         'P4': 'REJECT',
# #     }

# #     if raw == 'REJECT':
# #         result['final_decision'] = 'REJECT'
# #     elif raw == 'APPROVE':
# #         if tier in TIER_TO_DECISION:
# #             result['final_decision'] = TIER_TO_DECISION[tier]
# #         else:
# #             result['final_decision'] = 'APPROVE'
# #     else:
# #         if tier in TIER_TO_DECISION:
# #             result['final_decision'] = TIER_TO_DECISION[tier]
# #             result['reason'] = (
# #                 result.get('reason', '') +
# #                 f" [REVIEW resolved to {TIER_TO_DECISION[tier]} via risk tier {tier}]"
# #             )
# #         else:
# #             resolved = 'APPROVE' if score >= 600 else 'REJECT'
# #             result['final_decision'] = resolved
# #             result['reason'] = (
# #                 result.get('reason', '') +
# #                 f" [REVIEW resolved to {resolved} via combined risk score {score}]"
# #             )

# #     if result['final_decision'] == 'APPROVE':
# #         result.setdefault('interest_rate_range',
# #             {'P1': '9.5% – 11%', 'P2': '11% – 13%'}.get(tier, '11% – 14%'))
# #     else:
# #         result['interest_rate_range'] = 'N/A — Rejected'

# #     return result

# # # =============================================================================
# # # STAGE 2 RESULTS DISPLAY
# # # =============================================================================
# # def display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data):
# #     st.markdown("---")
# #     st.markdown('<p class="main-header">🎯 Stage 2 Final Results</p>', unsafe_allow_html=True)

# #     final_decision = stage2_result.get('final_decision', 'ERROR')
# #     interest_range = stage2_result.get('interest_rate_range', 'N/A')
# #     stage2_tier = stage2_result.get('stage2_tier', 'N/A')
# #     stage2_confidence = stage2_result.get('stage2_confidence', 0)
# #     combined_risk_score = stage2_result.get('combined_risk_score', 0)

# #     if final_decision == "APPROVE":
# #         card_class = "decision-card decision-card-approved"
# #         icon = "✓"
# #         subtitle = "✅ Final Decision: Approved — Proceed to Disbursement"
# #     else:
# #         card_class = "decision-card decision-card-rejected"
# #         icon = "✗"
# #         subtitle = "❌ Final Decision: Rejected — Application Declined"

# #     st.markdown(f"""
# #         <div class="{card_class}">
# #             <div class="decision-title"><span>{icon}</span><span>{final_decision}</span></div>
# #             <div class="decision-subtitle">{subtitle}</div>
# #         </div>
# #     """, unsafe_allow_html=True)

# #     col1, col2, col3, col4 = st.columns(4)
# #     with col1:
# #         st.metric("Risk Tier", stage2_tier)
# #     with col2:
# #         st.metric("Interest Rate", interest_range)
# #     with col3:
# #         st.metric("Combined Risk Score", combined_risk_score)
# #     with col4:
# #         confidence_display = f"{stage2_confidence:.1f}%" if stage2_confidence is not None else "N/A"
# #         st.metric("Stage 2 Confidence", confidence_display)

# #     st.markdown("<br>", unsafe_allow_html=True)

# #     tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔬 Analysis", "📋 Data", "📥 Download"])

# #     with tab1:
# #         st.markdown("### 📊 Decision Comparison")
# #         s1_dec   = st.session_state.get('stage1_decision', 'N/A')
# #         s2_label = "✅ APPROVE" if final_decision == "APPROVE" else "❌ REJECT"
# #         comparison_df = pd.DataFrame([
# #             {'Stage': 'Stage 1 (Screening)', 'Decision': s1_dec,
# #              'Risk Score': stage1_data.get('risk_score', 'N/A'), 'Tier': 'N/A',
# #              'Note': 'APPROVE / REVIEW → proceed to Stage 2'},
# #             {'Stage': 'Stage 2 — FINAL (CIBIL Deep)', 'Decision': s2_label,
# #              'Risk Score': combined_risk_score, 'Tier': f"{stage2_tier} | {interest_range}",
# #              'Note': 'Binding final decision'}
# #         ])
# #         st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# #         st.markdown("### 🎯 Risk Tier Details")
# #         tier_info = {
# #             'P1': {'name': 'Premium  → APPROVED',  'color': '#10B981',
# #                    'desc': 'Excellent credit profile — lowest interest rate band'},
# #             'P2': {'name': 'Standard → APPROVED',  'color': '#3B82F6',
# #                    'desc': 'Good credit profile — standard interest rate band'},
# #             'P3': {'name': 'Subprime → REJECTED',  'color': '#F59E0B',
# #                    'desc': 'Fair credit with elevated risk — application declined'},
# #             'P4': {'name': 'High Risk → REJECTED', 'color': '#EF4444',
# #                    'desc': 'High risk profile — application declined'},
# #         }
# #         if stage2_tier in tier_info:
# #             tier_data = tier_info[stage2_tier]
# #             st.markdown(f"""
# #                 <div style="background: {tier_data['color']}; color: white; padding: 1rem; border-radius: 0.5rem;">
# #                     <h3 style="margin: 0; color: white;">{stage2_tier}: {tier_data['name']}</h3>
# #                     <p style="margin: 0.5rem 0;">Interest Rate: {interest_range}</p>
# #                     <p style="margin: 0;">{tier_data['desc']}</p>
# #                 </div>
# #             """, unsafe_allow_html=True)
# #         st.markdown("### 📝 Decision Reasoning")
# #         st.info(stage2_result.get('reason', 'N/A'))

# #     with tab2:
# #         st.markdown("### 🔬 Detailed Analysis")
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             st.markdown("**Tier Probabilities**")
# #             if 'tier_probabilities' in stage2_result:
# #                 for tier, prob in stage2_result['tier_probabilities'].items():
# #                     st.metric(tier, f"{prob:.1f}%")
# #         with col2:
# #             st.markdown("**Stage Scores**")
# #             st.metric("Stage 1 Risk Score", stage1_data.get('risk_score', 'N/A'))
# #             st.metric("Stage 2 Risk Score", stage2_result.get('stage2_risk_score', 'N/A'))
# #             st.metric("Combined Score", combined_risk_score)
# #         with st.expander("📋 Complete Stage 2 Result"):
# #             st.json(stage2_result)

# #     with tab3:
# #         st.markdown("### 📋 Input Data")
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             with st.expander("Stage 1 Customer Data"):
# #                 st.json(stage1_customer)
# #         with col2:
# #             with st.expander("Enhanced CIBIL Data"):
# #                 st.json(enhanced_customer_data)

# #     with tab4:
# #         st.markdown("### 📥 Download Reports")
# #         bureau_score = stage1_customer.get('bureau_score', 0)
# #         dpd_90 = stage1_customer.get('dpd_90_count_6m', 0)
# #         dpd_30 = stage1_customer.get('dpd_30_count_6m', 0)
# #         foir = stage1_data.get('affordability_data', {}).get('foir_percentage', 0)
# #         employment_type = stage1_customer.get('employment_type', 'Salaried')
# #         employment_tenure = stage1_customer.get('employment_tenure_months', 0)
# #         business_vintage = stage1_customer.get('business_vintage_years', 0)
# #         ml_decision = stage1_data.get('decision', 'ERROR')
# #         confidence = stage1_data.get('confidence', 0)

# #         def _safe(v, default='N/A'):
# #             return v if v is not None else default

# #         report_data = {
# #             'application_id': _safe(stage1_customer.get('application_id'), 'N/A'),
# #             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# #             'decision': _safe(stage1_data.get('decision'), 'N/A'),
# #             'risk_score': _safe(stage1_data.get('risk_score'), 0),
# #             'pd_percentage': _safe(stage1_data.get('pd_percentage'), 0),
# #             'confidence': _safe(stage1_data.get('confidence'), 0),
# #             'policy_checks': stage1_data.get('policy_checks', {}),
# #             'affordability_data': stage1_data.get('affordability_data', {}),
# #             'customer_data': stage1_customer,
# #             'reason_codes': stage1_customer.get('reason_codes', []),
# #             'pd_calculation_factors': {
# #                 'bureau_score': bureau_score,
# #                 'base_pd': bureau_score_to_pd(bureau_score),
# #                 'dpd_90': dpd_90, 'dpd_30': dpd_30,
# #                 'delinquency_multiplier': delinquency_to_pd_multiplier(dpd_90, dpd_30),
# #                 'foir': foir,
# #                 'foir_adjustment': foir_to_pd_adjustment(foir),
# #                 'employment_adjustment': employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage),
# #                 'ml_adjustment': ml_confidence_to_pd_adjustment(confidence, ml_decision),
# #                 'final_pd': stage1_data.get('pd_percentage', 0)
# #             },
# #             'stage2_final_decision': _safe(final_decision, 'N/A'),
# #             'stage2_tier': _safe(stage2_tier, 'N/A'),
# #             'stage2_interest_range': _safe(interest_range, 'N/A'),
# #             'stage2_combined_risk_score': _safe(combined_risk_score, 0),
# #             'stage2_confidence': _safe(stage2_confidence, 0),
# #             'stage2_reason': _safe(stage2_result.get('reason'), 'N/A'),
# #             'stage2_tier_probabilities': stage2_result.get('tier_probabilities') or {},
# #             'stage2_complete_analysis': stage2_result,
# #             'stage1_data': stage1_data,
# #             'enhanced_customer_data': enhanced_customer_data
# #         }

# #         if PDF_AVAILABLE and generate_audit_pdf is not None:
# #             try:
# #                 pdf_buffer = generate_audit_pdf(report_data)
# #                 st.download_button(
# #                     "📥 Download PDF Report",
# #                     data=pdf_buffer,
# #                     file_name=f"stage2_report_{stage1_customer.get('application_id', 'unknown')}.pdf",
# #                     mime="application/pdf",
# #                     use_container_width=True
# #                 )
# #             except Exception as e:
# #                 st.error(f"PDF generation failed: {str(e)}")
# #         else:
# #             st.warning("PDF generation is not available. Please install the required PDF generator module.")

# #     st.markdown("---")
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         if st.button("🔄 New Assessment", key="new_assessment_stage2", use_container_width=True):
# #             st.session_state.stage1_complete = False
# #             st.session_state.stage1_decision = None
# #             st.session_state.stage1_data = None
# #             st.session_state.current_customer_data = None
# #             st.session_state.page_navigation = "👤 Assessment"
# #             st.rerun()
# #     with col2:
# #         if st.button("← Back to Stage 1", key="back_to_stage1", use_container_width=True):
# #             st.session_state.page_navigation = "👤 Assessment"
# #             st.rerun()
# #     with col3:
# #         if st.button("🏠 Home", key="home_stage2", use_container_width=True):
# #             st.session_state.page_navigation = "🏠 Home"
# #             st.rerun()

# # # =============================================================================
# # # SIDEBAR
# # # =============================================================================
# # with st.sidebar:
# #     st.markdown("# 🏦 Credit Risk Engine")
# #     st.markdown("---")

# #     navigation_options = ["🏠 Home", "👤 Assessment", "📊 Batch Process", "📈 Model Info", "ℹ️ About"]

# #     if (st.session_state.stage1_complete and
# #             st.session_state.stage1_decision in ['APPROVE', 'REVIEW']):
# #         navigation_options.insert(2, "🔬 Stage 2 Analysis")
# #         st.success(f"✅ Stage 1: {st.session_state.stage1_decision}")
# #         st.info("🔬 Stage 2 Analysis unlocked!")
# #     elif st.session_state.stage1_complete:
# #         st.warning(f"⚠️ Stage 1: {st.session_state.stage1_decision}")
# #         st.caption("Stage 2 only for APPROVE/REVIEW")

# #     page = st.radio(
# #         "**Navigation**",
# #         navigation_options,
# #         label_visibility="collapsed",
# #         key="page_navigation"
# #     )

# #     st.markdown("---")

# #     stage2_indicator = '✅ Active' if STAGE2_AVAILABLE and is_stage2_available() else '❌ Inactive'
# #     ocr_indicator = '✅ Ready' if OCR_AVAILABLE else '❌ Not Installed'
# #     if not OCR_AVAILABLE and OCR_ERROR_MSG:
# #         ocr_indicator += ' ⚠️'
# #     pdf_indicator = '✅ Ready' if PDF_AVAILABLE else '❌ Not Installed'

# #     st.markdown(f"""
# #     <div class="info-card">
# #         <div class="info-card-title">System Status</div>
# #         <div class="info-card-content">
# #             <div class="data-row"><span class="data-label">Model</span><span class="data-value">✅ Loaded</span></div>
# #             <div class="data-row"><span class="data-label">Version</span><span class="data-value">8.4</span></div>
# #             <div class="data-row"><span class="data-label">Stage 2</span><span class="data-value">{stage2_indicator}</span></div>
# #             <div class="data-row"><span class="data-label">OCR</span><span class="data-value">{ocr_indicator}</span></div>
# #             <div class="data-row"><span class="data-label">PDF Gen</span><span class="data-value">{pdf_indicator}</span></div>
# #             <div class="data-row"><span class="data-label">Features</span><span class="data-value">{len(TOP_FEATURES)}</span></div>
# #         </div>
# #     </div>
# #     """, unsafe_allow_html=True)

# #     with st.expander("🎯 **Top Features**"):
# #         for i, feat in enumerate(TOP_FEATURES[:5], 1):
# #             st.markdown(f"`{i}.` {feat}")

# #     if st.session_state.stage1_complete:
# #         st.markdown("---")
# #         st.markdown("### 🚀 Quick Actions")
# #         if st.button("🔄 New Assessment", key="new_assessment_sidebar", use_container_width=True):
# #             st.session_state.stage1_complete = False
# #             st.session_state.stage1_decision = None
# #             st.session_state.stage1_data = None
# #             st.session_state.current_customer_data = None
# #             st.session_state.extracted_cibil_data = None
# #             st.rerun()

# # # =============================================================================
# # # PAGE ROUTING
# # # =============================================================================

# # if page == "🏠 Home":
# #     st.markdown('<p class="main-header">Credit Risk Engine</p>', unsafe_allow_html=True)
# #     st.markdown("""
# #         <div class="info-box">
# #             <h3 style="margin-top: 0;">🎯 AI-Powered Lending Decisions</h3>
# #             <p style="margin-bottom: 0;">Comprehensive credit risk evaluation combining hard policy rules,
# #             machine learning models, and affordability analysis for accurate lending decisions.</p>
# #         </div>
# #     """, unsafe_allow_html=True)
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         st.markdown("""
# #             <div class="info-card"><div class="info-card-title"><span class="icon">🛡️</span><span>Policy Gates</span></div>
# #             <div class="info-card-content"><ul><li>Age & KYC verification</li><li>Employment stability</li>
# #             <li>Minimum income checks</li><li>Credit bureau thresholds</li><li>Bankruptcy & fraud detection</li></ul></div></div>
# #         """, unsafe_allow_html=True)
# #     with col2:
# #         st.markdown("""
# #             <div class="info-card"><div class="info-card-title"><span class="icon">🤖</span><span>ML Assessment</span></div>
# #             <div class="info-card-content"><ul><li>Random Forest classifier</li><li>60K+ training samples</li>
# #             <li>Confidence scoring</li><li>Multi-class prediction</li><li>Feature importance</li></ul></div></div>
# #         """, unsafe_allow_html=True)
# #     with col3:
# #         st.markdown("""
# #             <div class="info-card"><div class="info-card-title"><span class="icon">💰</span><span>Affordability</span></div>
# #             <div class="info-card-content"><ul><li>EMI calculation</li><li>FOIR analysis (max 50%)</li>
# #             <li>Net disposable income</li><li>Debt burden assessment</li><li>Affordability scoring</li></ul></div></div>
# #         """, unsafe_allow_html=True)
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     col1, col2, col3, col4 = st.columns(4)
# #     with col1: st.metric("🎯 Accuracy", "85%", "+2%")
# #     with col2: st.metric("⚡ Avg Response", "1.2s", "-0.3s")
# #     with col3: st.metric("📊 Features", len(TOP_FEATURES))
# #     with col4: st.metric("🔄 Version", "8.4", "Latest")
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     st.markdown("""
# #         <div class="warning-box">
# #             <strong>🆕 New in Version 8.4:</strong><br>
# #             • OCR Auto-fill Fix: All 5 categorical dropdowns now update correctly from PDF<br>
# #             • Payment Discipline inferred from DPD + bounce data (60K dataset calibrated)<br>
# #             • Cashflow Health inferred from net cash surplus thresholds<br>
# #             • Liquidity Flag inferred from net cash surplus<br>
# #             • Bureau Risk Flag inferred from score + DPD + hard-reject signals<br>
# #             • Salary Stability now uses data-driven inference (not hardcoded STABLE)
# #         </div>
# #     """, unsafe_allow_html=True)

# # elif page == "👤 Assessment":
# #     st.markdown('<p class="main-header">Credit Assessment</p>', unsafe_allow_html=True)

# #     pdf_just_extracted = st.session_state.get('pdf_just_extracted', False)

# #     with st.expander("📄 Upload CIBIL PDF to auto‑fill bureau fields",
# #                      expanded=pdf_just_extracted or not st.session_state.get('pdf_bureau_score')):

# #         if pdf_just_extracted:
# #             ex = st.session_state.get('_last_extraction', {})
# #             st.success("✅ CIBIL data extracted — form fields below have been updated automatically.")
# #             c1, c2, c3, c4 = st.columns(4)
# #             c1.metric("Credit Score",    ex.get('Credit_Score', '—'))
# #             c2.metric("Monthly Income",  f"₹{ex.get('NETMONTHLYINCOME') or ex.get('avg_salary_6m', 0):,}")
# #             c3.metric("DPD 60+ Count",   ex.get('num_times_60p_dpd', 0))
# #             c4.metric("CC Utilization",  f"{max(0, float(ex.get('CC_utilization', 0) or 0))*100:.0f}%")
# #             c1, c2, c3, c4 = st.columns(4)
# #             c1.metric("DPD 30+ Count",  ex.get('num_times_30p_dpd', 0))
# #             c2.metric("Inquiries (3M)", ex.get('enq_L3m', 0))
# #             c3.metric("Active Accounts", ex.get('num_std', 0))
# #             c4.metric("Written-Off",    ex.get('num_lss', ex.get('written_off_count', 0)))
# #             # Show surplus proxy and inference path
# #             _inf_path = ex.get('_surplus_proxy', None)
# #             if _inf_path is not None:
# #                 surplus_val = ex.get('_surplus_proxy', 0)
# #                 st.info(f"💡 **Bureau-only PDF** — net surplus estimated from income: ₹{surplus_val:,} "
# #                         f"(no bank statement in CIBIL report). Used for cashflow/liquidity inference.")
# #             if ex.get('written_off_count', 0) > 0 or ex.get('settled_count', 0) > 0:
# #                 st.warning(f"⚠️ Severe negatives detected: "
# #                            f"{ex.get('written_off_count', 0)} written-off, "
# #                            f"{ex.get('settled_count', 0)} settled accounts. "
# #                            f"Score overridden to {ex.get('Credit_Score', '?')}.")

# #             # ── FIX v8.4: Show inferred categorical flags in summary ──
# #             _inf = st.session_state.get('_last_inferred_flags', {})
# #             if _inf:
# #                 st.markdown("**📊 Inferred Categorical Flags (from CIBIL data):**")
# #                 fc1, fc2, fc3, fc4, fc5 = st.columns(5)
# #                 fc1.metric("Payment Discipline", _inf.get('payment_discipline_flag', '—'))
# #                 fc2.metric("Cashflow Health",    _inf.get('cashflow_health', '—'))
# #                 fc3.metric("Liquidity",          _inf.get('liquidity_flag', '—'))
# #                 fc4.metric("Bureau Risk",         _inf.get('bureau_risk_flag', '—'))
# #                 fc5.metric("Salary Stability",   _inf.get('salary_stability_flag', '—'))

# #             if st.session_state.get('stage1_complete') and st.session_state.get('current_customer_data'):
# #                 app_id_s1 = st.session_state.current_customer_data.get('application_id', 'Pending submission')
# #                 st.markdown(f"""
# #                     <div style="background:#1e3a5f;color:white;padding:0.5rem 1rem;border-radius:0.4rem;margin-bottom:0.5rem;font-size:0.9rem;">
# #                         <strong>📋 Application ID:</strong> {app_id_s1}
# #                     </div>
# #                 """, unsafe_allow_html=True)
# #             else:
# #                 st.markdown("No active assessment. Please submit the form below.")
# #             if st.toggle("📋 Show full extracted JSON"):
# #                 st.json({k: v for k, v in ex.items() if k != 'raw_text'})
# #             st.markdown("---")
# #             if st.button("🔄 Upload a different PDF", key="reset_pdf"):
# #                 st.session_state.pdf_just_extracted = False
# #                 st.session_state.pop('_last_extraction', None)
# #                 st.session_state.pop('_last_inferred_flags', None)
# #                 st.rerun()
# #         else:
# #             st.markdown('<div class="info-box">💡 Complete the form below or upload a CIBIL PDF to auto‑fill bureau data.</div>', unsafe_allow_html=True)
# #             uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="assessment_pdf")
# #             if uploaded_pdf is not None:
# #                 st.info(f"📄 File ready: **{uploaded_pdf.name}** ({uploaded_pdf.size/1024:.1f} KB)")
# #                 if st.button("🔍 Extract & Auto-fill Form", key="extract_assessment", type="primary", use_container_width=True):
# #                     with st.spinner("🔄 Running OCR on CIBIL PDF — this takes 10-30 seconds..."):
# #                         extraction_result = extract_cibil_from_pdf(uploaded_pdf)
# #                     if extraction_result.get('success', False):
# #                         st.session_state.pdf_age               = int(extraction_result.get('AGE', 35))
# #                         st.session_state.pdf_employment_type   = 'Salaried'
# #                         st.session_state.pdf_kyc               = True
# #                         st.session_state.pdf_bankruptcy        = False
# #                         st.session_state.pdf_fraud             = False
# #                         st.session_state.pdf_bureau_score      = int(extraction_result.get('Credit_Score', 720))
# #                         st.session_state.pdf_dpd_90            = int(extraction_result.get('dpd_90_count_6m', 0))
# #                         st.session_state.pdf_dpd_30            = int(extraction_result.get('num_times_30p_dpd', 0))
# #                         st.session_state.pdf_credit_util       = int(max(0, float(extraction_result.get('CC_utilization', 0) or 0)) * 100)
# #                         st.session_state.pdf_inquiries         = int(extraction_result.get('enq_L3m', 2))
# #                         st.session_state.pdf_active_loans      = int(extraction_result.get('num_std', 1))
# #                         st.session_state.pdf_existing_emi      = int(extraction_result.get('existing_emi', 15000))
# #                         # ── Income: External CIBIL uses NETMONTHLYINCOME (median ₹23k) ──
# #                         # train_60k uses avg_salary_6m (median ₹50k). Use whichever is present.
# #                         _income = int(
# #                             extraction_result.get('NETMONTHLYINCOME')
# #                             or extraction_result.get('avg_salary_6m')
# #                             or 50000
# #                         )
# #                         st.session_state.pdf_monthly_income    = _income
# #                         st.session_state.pdf_annual_income     = _income * 12
# #                         # ── Net surplus: use proxy if bureau-only, else actual ─────────
# #                         _surplus = int(
# #                             extraction_result.get('net_cash_surplus_6m')
# #                             or extraction_result.get('net_surplus')
# #                             or extraction_result.get('_surplus_proxy')
# #                             or 20000
# #                         )
# #                         st.session_state.pdf_net_surplus       = _surplus
# #                         st.session_state.pdf_loan_amount       = int(extraction_result.get('loan_amount', 180000))
# #                         st.session_state.pdf_loan_tenure       = int(extraction_result.get('loan_tenure', 24))
# #                         st.session_state.pdf_interest_rate     = float(extraction_result.get('interest_rate', 10.5))
# #                         st.session_state.pdf_amt_annuity       = int(extraction_result.get('amt_annuity', 8500))
# #                         st.session_state.pdf_employment_tenure = int(extraction_result.get('Time_With_Curr_Empr', 24))
# #                         st.session_state.pdf_business_vintage  = int(extraction_result.get('business_vintage', 3))
# #                         st.session_state.pdf_dependents        = int(extraction_result.get('dependents', 2))

# #                         # ── FIX v8.4: Infer all 5 categorical flags from CIBIL data ──
# #                         _inferred = infer_categorical_flags(extraction_result)
# #                         st.session_state.pdf_salary_stability   = _inferred['salary_stability_flag']
# #                         st.session_state.pdf_payment_discipline = _inferred['payment_discipline_flag']
# #                         st.session_state.pdf_cashflow_health    = _inferred['cashflow_health']
# #                         st.session_state.pdf_liquidity_flag     = _inferred['liquidity_flag']
# #                         st.session_state.pdf_bureau_risk_flag   = _inferred['bureau_risk_flag']
# #                         # Store inferred flags for display in the summary banner
# #                         st.session_state._last_inferred_flags   = _inferred

# #                         st.session_state.pdf_just_extracted    = True
# #                         st.session_state._last_extraction      = extraction_result
# #                         st.rerun()
# #                     else:
# #                         st.error(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")
# #                         st.info("Tip: Make sure Tesseract and Poppler are installed and paths are set correctly.")

# #     with st.form("assessment_form"):
# #         st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
# #         col1, col2, col3 = st.columns(3)
# #         with col1:
# #             age = st.number_input(
# #                 "Age", 24, 70,
# #                 value=int(st.session_state.get('pdf_age', 35)),
# #                 help="Customer's age in years (Minimum: 24, Maximum: 70)"
# #             )
# #             employment_type = st.selectbox(
# #                 "Employment Type",
# #                 ['Salaried', 'Self-Employed', 'Business'],
# #                 index=['Salaried', 'Self-Employed', 'Business'].index(
# #                     st.session_state.get('pdf_employment_type', 'Salaried')
# #                 )
# #             )
# #         with col2:
# #             dependents = st.number_input(
# #                 "Number of Dependents", 0, 20,
# #                 value=int(st.session_state.get('pdf_dependents', 2)),
# #                 help="1-5: Approve eligible | >5: Review required"
# #             )
# #             kyc_verified = st.selectbox(
# #                 "KYC Verified", ['Yes', 'No'],
# #                 index=0 if st.session_state.get('pdf_kyc', True) else 1
# #             ) == 'Yes'
# #         with col3:
# #             bankruptcy_flag = st.selectbox(
# #                 "Bankruptcy Flag", ['No', 'Yes'],
# #                 index=0 if not st.session_state.get('pdf_bankruptcy', False) else 1
# #             ) == 'Yes'
# #             fraud_flag = st.selectbox(
# #                 "Fraud Flag", ['No', 'Yes'],
# #                 index=0 if not st.session_state.get('pdf_fraud', False) else 1
# #             ) == 'Yes'
# #             if employment_type == 'Salaried':
# #                 employment_tenure = st.number_input(
# #                     "Employment Tenure (months)", 0, 600,
# #                     value=int(st.session_state.get('pdf_employment_tenure', 24))
# #                 )
# #                 business_vintage = 0
# #             else:
# #                 business_vintage = st.number_input(
# #                     "Business Vintage (years)", 0, 50,
# #                     value=int(st.session_state.get('pdf_business_vintage', 3))
# #                 )
# #                 employment_tenure = 0

# #         st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
# #         col1, col2, col3 = st.columns(3)
# #         with col1:
# #             bureau_score = st.number_input(
# #                 "Bureau Score", 300, 900,
# #                 value=int(st.session_state.get('pdf_bureau_score', 720)), step=10
# #             )
# #             dpd_90_6m = st.number_input(
# #                 "DPD 90+ (Last 6M)", 0, 20,
# #                 value=int(st.session_state.get('pdf_dpd_90', 0))
# #             )
# #             dpd_30_6m = st.number_input(
# #                 "DPD 30+ (Last 6M)", 0, 20,
# #                 value=int(st.session_state.get('pdf_dpd_30', 0))
# #             )
# #         with col2:
# #             credit_utilization = st.number_input(
# #                 "Credit Utilization (%)", 0, 100,
# #                 value=int(st.session_state.get('pdf_credit_util', 30))
# #             )
# #             recent_inquiries = st.number_input(
# #                 "Recent Inquiries (3M)", 0, 20,
# #                 value=int(st.session_state.get('pdf_inquiries', 2))
# #             )
# #         with col3:
# #             active_loans = st.number_input(
# #                 "Active Loans", 0, 10,
# #                 value=int(st.session_state.get('pdf_active_loans', 1))
# #             )
# #             existing_emi = st.number_input(
# #                 "Existing Total EMI (₹)", 0, 200000,
# #                 value=int(st.session_state.get('pdf_existing_emi', 15000)), step=1000
# #             )

# #         st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
# #         col1, col2, col3, col4 = st.columns(4)
# #         with col1:
# #             avg_salary = st.number_input(
# #                 "Monthly Income (₹)", 0, 1000000,
# #                 value=int(st.session_state.get('pdf_monthly_income', 50000)), step=5000
# #             )
# #             amt_income = st.number_input(
# #                 "Annual Income (₹)", 0, 10000000,
# #                 value=int(st.session_state.get('pdf_annual_income', 600000)), step=10000
# #             )
# #         with col2:
# #             net_surplus = st.number_input(
# #                 "Net Cash Surplus (₹)", -100000, 500000,
# #                 value=int(st.session_state.get('pdf_net_surplus', 20000)), step=5000
# #             )
# #             # ── FIX v8.4: Salary Stability reads inferred session_state key ──
# #             _ss_opts = ['STABLE', 'MODERATE', 'UNSTABLE']
# #             salary_stability = st.selectbox(
# #                 "Salary Stability",
# #                 _ss_opts,
# #                 index=_ss_opts.index(st.session_state.get('pdf_salary_stability', 'STABLE'))
# #             )
# #         with col3:
# #             loan_amount = st.number_input(
# #                 "Loan Amount (₹)", 0, 5000000,
# #                 value=int(st.session_state.get('pdf_loan_amount', 180000)), step=10000
# #             )
# #             loan_tenure = st.number_input(
# #                 "Tenure (months)", 3, 360,
# #                 value=int(st.session_state.get('pdf_loan_tenure', 24))
# #             )
# #         with col4:
# #             interest_rate = st.number_input(
# #                 "Interest Rate (%)", 8.0, 20.0,
# #                 value=float(st.session_state.get('pdf_interest_rate', 10.5)), step=0.5
# #             )
# #             amt_annuity = st.number_input(
# #                 "Requested EMI (₹)", 0, 200000,
# #                 value=int(st.session_state.get('pdf_amt_annuity', 8500)), step=500
# #             )

# #         # ── Additional Credit Behaviour ──────────────────────────────────────
# #         st.markdown('<p class="section-header">📋 Additional Credit Behaviour</p>', unsafe_allow_html=True)
# #         col1, col2, col3 = st.columns(3)
# #         with col1:
# #             # ── FIX v8.4: payment_discipline reads inferred session_state key ──
# #             _pd_opts = ['GOOD', 'MODERATE', 'POOR']
# #             payment_discipline = st.selectbox(
# #                 "Payment Discipline", _pd_opts,
# #                 index=_pd_opts.index(st.session_state.get('pdf_payment_discipline', 'GOOD')),
# #                 help="Overall payment behavior pattern"
# #             )
# #             # ── FIX v8.4: liquidity_flag reads inferred session_state key ──
# #             _lq_opts = ['LOW', 'ADEQUATE', 'MODERATE']
# #             liquidity_flag = st.selectbox(
# #                 "Liquidity", _lq_opts,
# #                 index=_lq_opts.index(st.session_state.get('pdf_liquidity_flag', 'LOW')),
# #                 help="Cash liquidity position"
# #             )
# #         with col2:
# #             # ── FIX v8.4: cashflow_health reads inferred session_state key ──
# #             _cf_opts = ['MODERATE', 'HEALTHY', 'STRESSED', 'STABLE']
# #             cashflow_health = st.selectbox(
# #                 "Cashflow Health", _cf_opts,
# #                 index=_cf_opts.index(st.session_state.get('pdf_cashflow_health', 'MODERATE')),
# #                 help="Overall cashflow health assessment"
# #             )
# #             # ── FIX v8.4: bureau_risk_flag reads inferred session_state key ──
# #             _br_opts = ['LOW', 'MEDIUM', 'HIGH']
# #             bureau_risk_flag = st.selectbox(
# #                 "Bureau Risk", _br_opts,
# #                 index=_br_opts.index(st.session_state.get('pdf_bureau_risk_flag', 'LOW')),
# #                 help="External bureau risk rating"
# #             )
# #         with col3:
# #             inward_bounce_count = st.number_input(
# #                 "Inward Bounce Count (3M)", 0, 10, 0,
# #                 help="Number of bounced inward cheques last 3 months"
# #             )
# #             salary_missing_months = st.number_input(
# #                 "Missing Salary Months (6M)", 0, 6, 0,
# #                 help="Months without salary credit"
# #             )

# #         st.markdown("<br>", unsafe_allow_html=True)
# #         submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)

# #     if submitted:
# #         timestamp = datetime.now()
# #         app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
# #         customer_data = {
# #             'age': age,
# #             'employment_type': employment_type,
# #             'dependents': dependents,
# #             'kyc_verified': kyc_verified,
# #             'bankruptcy_flag': bankruptcy_flag,
# #             'fraud_flag': fraud_flag,
# #             'employment_tenure_months': employment_tenure,
# #             'business_vintage_years': business_vintage,
# #             'bureau_score': bureau_score,
# #             'dpd_90_count_6m': dpd_90_6m,
# #             'dpd_30_count_6m': dpd_30_6m,
# #             'credit_utilization_pct': credit_utilization,
# #             'max_utilization': credit_utilization,
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
# #             'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
# #             'payment_discipline_flag': payment_discipline,
# #             'liquidity_flag': liquidity_flag,
# #             'cashflow_health': cashflow_health,
# #             'bureau_risk_flag': bureau_risk_flag,
# #             'inward_bounce_count_3m': inward_bounce_count,
# #             'salary_missing_months': salary_missing_months,
# #         }

# #         with st.spinner("🔄 Processing Stage 1 assessment..."):
# #             decision_data = make_hybrid_decision_enhanced(customer_data)

# #         reasons = generate_reason_codes(
# #             decision=decision_data.get('decision', 'ERROR'),
# #             customer_data=customer_data,
# #             affordability_data=decision_data.get('affordability_data', {}),
# #             policy_checks=decision_data.get('policy_checks', {})
# #         )
# #         customer_data['reason_codes'] = reasons

# #         st.session_state.stage1_complete = True
# #         st.session_state.stage1_decision = decision_data.get('decision', 'ERROR')
# #         st.session_state.stage1_data = decision_data
# #         st.session_state.current_customer_data = customer_data

# #         for key in list(st.session_state.keys()):
# #             if key.startswith('pdf_') or key in ('_last_extraction', '_last_inferred_flags'):
# #                 del st.session_state[key]

# #         tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])

# #         with tab1:
# #             st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
# #             col1, col2 = st.columns(2)
# #             with col1:
# #                 render_info_card("👤 Identity", "👤",
# #                                  {"Age": age, "Employment": employment_type, "Dependents": dependents,
# #                                   "KYC Status": "Verified" if kyc_verified else "Not Verified",
# #                                   "Tenure": f"{employment_tenure} months" if employment_type == 'Salaried' else f"{business_vintage} years"})
# #                 render_info_card("💰 Financial", "💰",
# #                                  {"Monthly Income": f"₹{avg_salary:,}", "Annual Income": f"₹{amt_income:,}",
# #                                   "Net Surplus": f"₹{net_surplus:,}", "Stability": salary_stability})
# #             with col2:
# #                 render_info_card("🏦 Credit Bureau", "🏦",
# #                                  {"Bureau Score": bureau_score, "DPD 90+": dpd_90_6m, "DPD 30+": dpd_30_6m,
# #                                   "Utilization": f"{credit_utilization}%", "Recent Inquiries": recent_inquiries,
# #                                   "Existing EMI": f"₹{existing_emi:,}"})
# #                 render_info_card("📋 Loan Request", "📋",
# #                                  {"Amount": f"₹{loan_amount:,}", "Tenure": f"{loan_tenure} months",
# #                                   "Interest Rate": f"{interest_rate}%", "Requested EMI": f"₹{amt_annuity:,}"})

# #         with tab2:
# #             st.markdown('<p class="section-header">Decision Summary</p>', unsafe_allow_html=True)
# #             render_decision_header(decision_data, customer_data)
# #             st.markdown("<br>", unsafe_allow_html=True)

# #             final_decision = decision_data.get('decision', 'ERROR')

# #             if final_decision in ['APPROVE', 'REVIEW']:
# #                 st.markdown("---")
# #                 st.markdown("""
# #                     <div class="info-box" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white; text-align: center;">
# #                         <h3 style="margin: 0; color: white;">✅ Eligible for Stage 2 Deep Dive</h3>
# #                         <p style="margin: 0.5rem 0 0 0;">Choose an input method to proceed:</p>
# #                     </div>
# #                 """, unsafe_allow_html=True)
# #                 col1, col2, col3 = st.columns(3)
# #                 with col1:
# #                     if st.button("📝 Manual Entry", key="stage2_manual_btn", use_container_width=True, type="primary"):
# #                         st.session_state.stage2_selected_tab = "Manual Entry"
# #                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
# #                         st.rerun()
# #                 with col2:
# #                     if st.button("📄 PDF Upload", key="stage2_pdf_btn", use_container_width=True, type="primary"):
# #                         st.session_state.stage2_selected_tab = "PDF Upload"
# #                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
# #                         st.rerun()
# #                 with col3:
# #                     if st.button("📊 Batch Analysis", key="stage2_batch_btn", use_container_width=True, type="primary"):
# #                         st.session_state.stage2_selected_tab = "Batch Analysis"
# #                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
# #                         st.rerun()
# #             elif final_decision == 'REJECT':
# #                 st.markdown("---")
# #                 st.markdown("""
# #                     <div class="warning-box" style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; text-align: center;">
# #                         <h3 style="margin: 0; color: white;">❌ Stage 2 Not Available</h3>
# #                         <p style="margin: 0.5rem 0 0 0;">Application rejected. Stage 2 only for APPROVE/REVIEW.</p>
# #                     </div>
# #                 """, unsafe_allow_html=True)

# #             st.markdown("<br>", unsafe_allow_html=True)
# #             affordability = decision_data.get('affordability_data', {})
# #             foir = affordability.get('foir_percentage', 0)
# #             total_emi = affordability.get('total_emi', 0)
# #             net_disp = affordability.get('net_disposable', 0)

# #             col1, col2, col3 = st.columns(3)
# #             with col1:
# #                 render_info_card("Identity & Eligibility", "👤",
# #                                  {f"Age: {age}": "", f"Employment: {employment_type}": "",
# #                                   f"Dependents: {dependents}": "",
# #                                   f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""},
# #                                  {f"Age: {age}": "pass" if 24 <= age <= 70 else "fail",
# #                                   f"Employment: {employment_type}": "pass",
# #                                   f"Dependents: {dependents}": "pass" if dependents <= 5 else "warning",
# #                                   f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_verified else "fail"})
# #             with col2:
# #                 bureau_pass = bureau_score >= 550
# #                 dpd_pass = dpd_90_6m == 0
# #                 render_info_card("Credit Bureau", "🏦",
# #                                  {f"Bureau Score: {bureau_score}": "", f"DPD 90+: {dpd_90_6m}": "",
# #                                   f"Utilization: {credit_utilization}%": ""},
# #                                  {f"Bureau Score: {bureau_score}": "pass" if bureau_pass else "fail",
# #                                   f"DPD 90+: {dpd_90_6m}": "pass" if dpd_pass else "fail",
# #                                   f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"})
# #             with col3:
# #                 render_info_card("Affordability", "💰",
# #                                  {f"Monthly Income: ₹{avg_salary:,}": "", f"FOIR: {foir:.1f}%": "",
# #                                   f"Total EMI: ₹{total_emi:,}": "", f"Net Disposable: ₹{net_disp:,}": ""},
# #                                  {f"Monthly Income: ₹{avg_salary:,}": "pass",
# #                                   f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
# #                                   f"Total EMI: ₹{total_emi:,}": "pass",
# #                                   f"Net Disposable: ₹{net_disp:,}": "pass" if net_disp >= 10000 else "warning"})

# #             st.markdown("<br>", unsafe_allow_html=True)
# #             render_reason_codes(reasons)
# #             st.markdown("<br>", unsafe_allow_html=True)

# #             col1, col2 = st.columns([1, 1])
# #             with col1:
# #                 if PDF_AVAILABLE and generate_decision_pdf is not None:
# #                     try:
# #                         pdf_buffer = generate_decision_pdf(
# #                             decision_data=decision_data, customer_data=customer_data,
# #                             affordability_data=decision_data.get('affordability_data', {}), reasons=reasons)
# #                         st.download_button("📥 Decision Report (PDF)", data=pdf_buffer,
# #                                            file_name=f"credit_decision_{app_id}.pdf", mime="application/pdf",
# #                                            use_container_width=True)
# #                     except Exception as e:
# #                         st.error(f"Error generating PDF: {str(e)}")
# #                 else:
# #                     st.warning("PDF generation not available.")
# #             with col2:
# #                 if st.button("🔄 Re-Evaluate", key="reevaluate_btn", use_container_width=True):
# #                     st.rerun()

# #         with tab3:
# #             st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
# #             col1, col2 = st.columns(2)
# #             with col1:
# #                 fig1 = create_modern_gauge(decision_data.get('confidence', 0), "Model Confidence")
# #                 st.plotly_chart(fig1, use_container_width=True)
# #             with col2:
# #                 class_probs = decision_data.get('class_probs', {"APPROVE": 0, "REVIEW": 0, "REJECT": 0})
# #                 fig2 = create_modern_bar_chart(class_probs)
# #                 st.plotly_chart(fig2, use_container_width=True)

# #             st.markdown("<br>", unsafe_allow_html=True)
# #             st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
# #             policy_df = pd.DataFrame([{'Check': k, 'Result': v} for k, v in decision_data.get('policy_checks', {}).items()])
# #             st.dataframe(policy_df, use_container_width=True, hide_index=True)

# #             st.markdown('<p class="section-header">PD Calculation Breakdown</p>', unsafe_allow_html=True)
# #             pd_factors_display = {
# #                 'Bureau Score': f"{bureau_score} → Base PD: {bureau_score_to_pd(bureau_score):.1f}%",
# #                 'Delinquency': f"DPD 90+: {dpd_90_6m}, DPD 30+: {dpd_30_6m} → Multiplier: {delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x",
# #                 'FOIR Impact': f"{foir:.1f}% → Adjustment: {foir_to_pd_adjustment(foir):.1f}%",
# #                 'Employment Stability': f"{employment_type}, {employment_tenure if employment_type == 'Salaried' else business_vintage}{' months' if employment_type == 'Salaried' else ' years'} → Adjustment: {employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage):.1f}%",
# #                 'ML Confidence': f"{decision_data.get('confidence', 0):.1f}% → Adjustment: {ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')):.1f}%",
# #                 'Final PD': f"{decision_data.get('pd_percentage', 0)}%"
# #             }
# #             for factor, value in pd_factors_display.items():
# #                 st.markdown(f"**{factor}:** {value}")

# #         with tab4:
# #             st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
# #             audit_log_raw = {
# #                 'application_id': app_id,
# #                 'timestamp': timestamp.isoformat(),
# #                 'decision': decision_data.get('decision', 'ERROR'),
# #                 'risk_score': decision_data.get('risk_score', 0),
# #                 'pd_percentage': decision_data.get('pd_percentage', 0),
# #                 'confidence': round(decision_data.get('confidence', 0), 2),
# #                 'model_version': '8.4',
# #                 'reason_codes': reasons,
# #                 'policy_checks': decision_data.get('policy_checks', {}),
# #                 'affordability': decision_data.get('affordability_data', {}),
# #                 'customer_data': {k: v for k, v in customer_data.items() if k not in ['application_id', 'timestamp', 'reason_codes']},
# #                 'pd_calculation_factors': {
# #                     'bureau_score': bureau_score,
# #                     'base_pd': bureau_score_to_pd(bureau_score),
# #                     'dpd_90': dpd_90_6m, 'dpd_30': dpd_30_6m,
# #                     'delinquency_multiplier': delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m),
# #                     'foir': foir,
# #                     'foir_adjustment': foir_to_pd_adjustment(foir),
# #                     'employment_adjustment': employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage),
# #                     'ml_adjustment': ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')),
# #                     'final_pd': decision_data.get('pd_percentage', 0)
# #                 }
# #             }
# #             audit_log = sanitize_for_json(audit_log_raw)

# #             with st.expander("📋 View Audit Log (JSON)"):
# #                 st.json(audit_log)

# #             col1, col2 = st.columns(2)
# #             with col1:
# #                 if PDF_AVAILABLE and generate_audit_pdf is not None:
# #                     try:
# #                         audit_pdf_buffer = generate_audit_pdf(audit_log)
# #                         st.download_button("📥 Download Audit Trail (PDF)",
# #                                            data=audit_pdf_buffer,
# #                                            file_name=f"audit_trail_{app_id}.pdf",
# #                                            mime="application/pdf",
# #                                            use_container_width=True)
# #                     except Exception as e:
# #                         st.error(f"Error generating audit PDF: {str(e)}")
# #                 else:
# #                     st.warning("Audit PDF generation is not available.")
# #             with col2:
# #                 audit_json = json.dumps(audit_log, indent=2)
# #                 st.download_button("📥 Download Audit Log (JSON)",
# #                                    data=audit_json,
# #                                    file_name=f"audit_{app_id}.json",
# #                                    mime="application/json",
# #                                    use_container_width=True)

# #             st.markdown('<p class="section-header">PD Calculation Summary</p>', unsafe_allow_html=True)
# #             pd_table = pd.DataFrame([
# #                 {"Factor": "Bureau Score", "Value": f"{bureau_score}", "Impact": f"{bureau_score_to_pd(bureau_score):.1f}% base PD"},
# #                 {"Factor": "Delinquency (DPD 90+)", "Value": f"{dpd_90_6m} times", "Impact": f"{delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x multiplier"},
# #                 {"Factor": "FOIR", "Value": f"{foir:.1f}%", "Impact": f"{foir_to_pd_adjustment(foir):.1f}% adjustment"},
# #                 {"Factor": "Employment Stability",
# #                  "Value": f"{employment_type} ({employment_tenure if employment_type == 'Salaried' else business_vintage}{' months' if employment_type == 'Salaried' else ' years'})",
# #                  "Impact": f"{employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage):.1f}% adjustment"},
# #                 {"Factor": "ML Decision Confidence",
# #                  "Value": f"{decision_data.get('confidence', 0):.1f}% ({decision_data.get('decision', 'ERROR')})",
# #                  "Impact": f"{ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')):.1f}% adjustment"},
# #                 {"Factor": "Final PD", "Value": f"{decision_data.get('pd_percentage', 0)}%", "Impact": "Industry-standard calculation"}
# #             ])
# #             st.dataframe(pd_table, use_container_width=True, hide_index=True)

# # elif page == "🔬 Stage 2 Analysis":
# #     st.markdown('<p class="main-header">Stage 2: CIBIL Deep Dive</p>', unsafe_allow_html=True)

# #     if not st.session_state.get('stage1_complete', False):
# #         st.error("❌ You must complete Stage 1 Assessment first!")
# #         st.info("Please go to the 👤 Assessment page and submit an application.")
# #         if st.button("← Go to Assessment", use_container_width=True):
# #             st.session_state.page_navigation = "👤 Assessment"
# #             st.rerun()
# #         st.stop()

# #     if st.session_state.get('stage1_decision') not in ['APPROVE', 'REVIEW']:
# #         st.error("❌ Stage 2 is only available for APPROVED or REVIEW applications!")
# #         st.warning(f"Your Stage 1 decision: {st.session_state.get('stage1_decision', 'Unknown')}")
# #         if st.button("← Go Back", use_container_width=True):
# #             st.session_state.page_navigation = "👤 Assessment"
# #             st.rerun()
# #         st.stop()

# #     if not (STAGE2_AVAILABLE and is_stage2_available()):
# #         st.error("❌ Stage 2 model not available!")
# #         st.info("Please ensure `stage2_cibil_model.pkl` is in the project directory.")
# #         if st.button("← Go Back", use_container_width=True):
# #             st.session_state.page_navigation = "👤 Assessment"
# #             st.rerun()
# #         st.stop()

# #     stage1_data = st.session_state.get('stage1_data', {})
# #     stage1_customer = st.session_state.get('current_customer_data', {})

# #     st.markdown(f"""
# #         <div class="info-box" style="background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); color: white;">
# #             <h3 style="margin: 0; color: white;">📊 Stage 1 Results</h3>
# #             <p style="margin: 0.5rem 0 0 0;">
# #                 <strong>Decision:</strong> {st.session_state.get('stage1_decision', 'N/A')} |
# #                 <strong>Risk Score:</strong> {stage1_data.get('risk_score', 'N/A')} |
# #                 <strong>Application ID:</strong> {stage1_customer.get('application_id', 'N/A')}
# #             </p>
# #         </div>
# #     """, unsafe_allow_html=True)

# #     st.markdown("<br>", unsafe_allow_html=True)

# #     tab_options = ["Manual Entry", "PDF Upload", "Batch Analysis"]
# #     default_tab = st.session_state.get('stage2_selected_tab', 'Manual Entry')
# #     if default_tab not in tab_options:
# #         default_tab = "Manual Entry"
# #     selected_tab = st.radio(
# #         "Select input method",
# #         tab_options,
# #         index=tab_options.index(default_tab),
# #         horizontal=True,
# #         label_visibility="collapsed"
# #     )

# #     if selected_tab == "Manual Entry":
# #         st.markdown('<p class="section-header">Manual CIBIL Data Entry</p>', unsafe_allow_html=True)
# #         st.markdown("""
# #             <div class="info-box">
# #                 📝 <strong>Manual Data Entry</strong><br>
# #                 Enter CIBIL bureau data to enhance Stage 1 customer profile.<br>
# #                 The Stage 2 model will use this data to predict risk tier (P1/P2/P3/P4).
# #             </div>
# #         """, unsafe_allow_html=True)

# #         with st.form("stage2_manual_form"):
# #             st.markdown("### 📋 Application Reference")
# #             col1, col2 = st.columns(2)
# #             with col1:
# #                 st.text_input("Application ID", value=stage1_customer.get('application_id', 'N/A'), disabled=True)
# #                 st.text_input("Stage 1 Decision", value=st.session_state.get('stage1_decision', 'N/A'), disabled=True)
# #             with col2:
# #                 st.text_input("Customer Name (Optional)", "")
# #                 st.number_input("Stage 1 Risk Score", value=int(stage1_data.get('risk_score', 750)), disabled=True)

# #             st.markdown("---")
# #             st.markdown("### 🏦 CIBIL Bureau Data")

# #             st.markdown("---")
# #             st.markdown("### 👤 Demographics & Product Enquiries")

# #             col1, col2, col3 = st.columns(3)
# #             with col1:
# #                 gender = st.selectbox(
# #                     "Gender",
# #                     ["Male", "Female", "Others"],
# #                     help="Select gender as per CIBIL report"
# #                 )
# #             with col2:
# #                 marital_status = st.selectbox(
# #                     "Marital Status",
# #                     ["Married", "Single", "Divorced", "Widowed", "Others"],
# #                     help="Marital status from bureau data"
# #                 )
# #             with col3:
# #                 education = st.selectbox(
# #                     "Education",
# #                     ["Graduate", "Post Graduate", "Under Graduate", "Professional", "Others"],
# #                     help="Highest education level"
# #                 )

# #             col1, col2, col3 = st.columns(3)
# #             with col1:
# #                 st.markdown("**Credit Score & History**")
# #                 cibil_score = st.number_input("Credit Score", 300, 900, 720, 10)
# #                 max_delinquency = st.number_input("Max Delinquency Level", 0, 100, 0)
# #                 num_times_30dpd = st.number_input("Times 30+ DPD", 0, 50, 0)
# #                 num_times_60dpd = st.number_input("Times 60+ DPD", 0, 50, 0)
# #                 num_times_delinquent = st.number_input("Total Delinquent", 0, 50, 0)
# #             with col2:
# #                 st.markdown("**Recent Behavior (6-12M)**")
# #                 num_deliq_6m = st.number_input("Delinquencies (6M)", 0, 20, 0)
# #                 num_deliq_12m = st.number_input("Delinquencies (12M)", 0, 20, 0)
# #                 max_deliq_6m = st.number_input("Max Delinq (6M)", 0, 100, 0)
# #                 max_deliq_12m = st.number_input("Max Delinq (12M)", 0, 100, 0)
# #                 enq_L3m = st.number_input("Inquiries (3M)", 0, 20, 2)
# #                 enq_L6m = st.number_input("Inquiries (6M)", 0, 30, 4)
# #                 enq_L12m = st.number_input("Inquiries (12M)", 0, 50, 6)
# #             with col3:
# #                 st.markdown("**Account Quality**")
# #                 num_std = st.number_input("Standard Accounts", 0, 50, 3)
# #                 num_std_6m = st.number_input("Standard (6M)", 0, 50, 3)
# #                 num_std_12m = st.number_input("Standard (12M)", 0, 50, 3)
# #                 num_sub = st.number_input("Sub-standard", 0, 20, 0)
# #                 num_sub_6m = st.number_input("Sub-standard (6M)", 0, 20, 0)
# #                 num_dbt = st.number_input("Doubtful", 0, 10, 0)
# #                 num_lss = st.number_input("Loss", 0, 10, 0)

# #             col1, col2, col3 = st.columns(3)
# #             with col1:
# #                 st.markdown("**Utilization**")
# #                 pct_active_tls = st.number_input("% Active TLs", 0.0, 1.0, 0.60, 0.01)
# #                 pct_current_bal = st.number_input("Current Balance %", 0.0, 1.0, 0.30, 0.01)
# #                 cc_utilization = st.number_input("CC Utilization", 0.0, 1.0, 0.35, 0.01)
# #                 pl_utilization = st.number_input("PL Utilization", 0.0, 1.0, 0.25, 0.01)
# #                 max_unsec_exposure = st.number_input("Max Unsec Exposure %", 0, 100, 30)
# #             with col2:
# #                 st.markdown("**Demographics**")
# #                 age_cibil = st.number_input("Age", 24, 70, int(stage1_customer.get('age', 35)))
# #                 net_monthly_income = st.number_input("Net Monthly Income", 0, 1000000,
# #                                                       int(stage1_customer.get('avg_salary_6m', 50000)), 5000)
# #                 time_curr_employer = st.number_input("Employment Tenure (months)", 0, 600,
# #                                                       int(stage1_customer.get('employment_tenure_months', 24)))
# #             with col3:
# #                 st.markdown("**Product Flags**")
# #                 cc_flag = st.selectbox("Credit Card", ["Yes", "No"]) == "Yes"
# #                 pl_flag = st.selectbox("Personal Loan", ["Yes", "No"]) == "No"
# #                 hl_flag = st.selectbox("Home Loan", ["Yes", "No"]) == "No"
# #                 gl_flag = st.selectbox("Gold Loan", ["Yes", "No"]) == "No"

# #             st.markdown("<br>", unsafe_allow_html=True)
# #             submitted_s2 = st.form_submit_button("🔬 Run Stage 2 Analysis", use_container_width=True, type="primary")

# #         if submitted_s2:
# #             with st.spinner("🔬 Running Stage 2 CIBIL Deep Analysis..."):
# #                 enhanced_customer_data = stage1_customer.copy()
# #                 _s1_inc = stage1_customer.get('avg_salary_6m', 50000)
# #                 _s2_inc = net_monthly_income or 0
# #                 _final_income = _s1_inc if (_s2_inc > 0 and _s2_inc < _s1_inc * 0.4) else (_s2_inc or _s1_inc)
# #                 if _s2_inc > 0 and _s2_inc < _s1_inc * 0.4:
# #                     st.warning(f'⚠️ CIBIL income ₹{_s2_inc:,} is much lower than application income ₹{_s1_inc:,}. Using application income for FOIR.')
# #                 enhanced_customer_data.update({
# #                     'bureau_score': cibil_score,
# #                     'age': age_cibil,
# #                     'avg_salary_6m': _final_income,
# #                     'employment_tenure_months': time_curr_employer,
# #                     'dpd_30_count_6m': num_times_30dpd,
# #                     'dpd_90_count_6m': num_times_60dpd,
# #                     'max_delinquency_level': max_delinquency,
# #                     'num_times_delinquent': num_times_delinquent,
# #                     'num_deliq_6mts': num_deliq_6m,
# #                     'num_deliq_12mts': num_deliq_12m,
# #                     'max_deliq_6mts': max_deliq_6m,
# #                     'max_deliq_12mts': max_deliq_12m,
# #                     'recent_inquiries_3m': enq_L3m,
# #                     'enq_L6m': enq_L6m,
# #                     'enq_L12m': enq_L12m,
# #                     'active_loans_count': num_std,
# #                     'num_std_6mts': num_std_6m,
# #                     'num_std_12mts': num_std_12m,
# #                     'num_sub': num_sub,
# #                     'num_sub_6mts': num_sub_6m,
# #                     'num_dbt': num_dbt,
# #                     'num_lss': num_lss,
# #                     'credit_utilization_pct': cc_utilization * 100,
# #                     'pct_of_active_TLs_ever': pct_active_tls,
# #                     'pct_currentBal_all_TL': pct_current_bal,
# #                     'CC_utilization': cc_utilization,
# #                     'PL_utilization': pl_utilization,
# #                     'max_unsec_exposure_inPct': max_unsec_exposure,
# #                     'CC_Flag': 1 if cc_flag else 0,
# #                     'PL_Flag': 1 if pl_flag else 0,
# #                     'HL_Flag': 1 if hl_flag else 0,
# #                     'GL_Flag': 1 if gl_flag else 0,
# #                     'GENDER': gender,
# #                     'MARITALSTATUS': marital_status,
# #                     'EDUCATION': education,
# #                 })
# #                 enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)
# #                 try:
# #                     stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
# #                     stage2_result = resolve_stage2_to_binary(stage2_result)
# #                     display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
# #                 except Exception as e:
# #                     st.error(f"❌ Stage 2 analysis failed: {str(e)}")
# #                     st.exception(e)

# #     elif selected_tab == "PDF Upload":
# #         st.markdown('<p class="section-header">📄 CIBIL PDF Upload</p>', unsafe_allow_html=True)
# #         if not OCR_AVAILABLE:
# #             st.error("❌ OCR not available. " + (OCR_ERROR_MSG or "Check packages.txt and requirements.txt."))
# #             st.warning("For now, please use the **Manual Entry** tab.")
# #         else:
# #             st.markdown("""
# #                 <div class="info-box">
# #                     📄 <strong>CIBIL PDF Extraction</strong><br>
# #                     Upload a CIBIL bureau report PDF for automatic extraction and analysis.
# #                 </div>
# #             """, unsafe_allow_html=True)
# #             uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="stage2_pdf")
# #             if uploaded_pdf is not None:
# #                 st.success(f"✅ File uploaded: {uploaded_pdf.name} ({uploaded_pdf.size / 1024:.1f} KB)")
# #                 if st.button("🔬 Extract & Analyze", key="extract_analyze_stage2", type="primary", use_container_width=True):
# #                     with st.spinner("🔄 Extracting data from PDF..."):
# #                         extraction_result = extract_cibil_from_pdf(uploaded_pdf)
# #                     if extraction_result.get('success', False):
# #                         st.success("✅ PDF extraction successful!")

# #                         app_id_display  = stage1_customer.get('application_id', 'N/A')
# #                         cust_name       = stage1_customer.get('customer_name', 'N/A')
# #                         s1_decision     = st.session_state.get('stage1_decision', 'N/A')
# #                         s1_risk         = stage1_data.get('risk_score', 'N/A')
# #                         st.markdown(f"""
# #                             <div style="background:#1e3a5f;color:white;padding:0.75rem 1rem;border-radius:0.5rem;margin-bottom:0.75rem;">
# #                                 <strong>📋 Application ID:</strong> {app_id_display} &nbsp;|&nbsp;
# #                                 <strong>Stage 1:</strong> {s1_decision} &nbsp;|&nbsp;
# #                                 <strong>Risk Score:</strong> {s1_risk}
# #                             </div>
# #                         """, unsafe_allow_html=True)

# #                         st.markdown("### 📋 Extracted CIBIL Data (Summary)")
# #                         col1, col2, col3, col4 = st.columns(4)
# #                         with col1:
# #                             st.metric("Credit Score", extraction_result.get('Credit_Score', 'N/A'))
# #                             st.metric("Max Delinquency Level", extraction_result.get('max_delinquency_level', 0))
# #                         with col2:
# #                             st.metric("Times 30+ DPD", extraction_result.get('num_times_30p_dpd', 0))
# #                             st.metric("Times 60+ DPD", extraction_result.get('num_times_60p_dpd', 0))
# #                         with col3:
# #                             st.metric("Total Delinquent", extraction_result.get('num_times_delinquent', 0))
# #                             st.metric("DPD 90+ (6M)", extraction_result.get('dpd_90_count_6m', 0))
# #                         with col4:
# #                             st.metric("Active Accounts", extraction_result.get('num_std', 0))
# #                             st.metric("Written Off", extraction_result.get('written_off_count', 0))

# #                         with st.expander("🔍 View All Extracted Features (with internal IDs)", expanded=False):
# #                             friendly_names = {
# #                                 'Credit_Score': 'Credit Score',
# #                                 'AGE': 'Age',
# #                                 'max_delinquency_level': 'Max Delinquency Level',
# #                                 'num_times_30p_dpd': 'Times 30+ DPD',
# #                                 'num_times_60p_dpd': 'Times 60+ DPD',
# #                                 'num_times_delinquent': 'Total Times Delinquent',
# #                                 'dpd_90_count_6m': 'DPD 90+ Count (6M)',
# #                                 'num_deliq_6mts': 'Delinquent Count (6M)',
# #                                 'num_deliq_12mts': 'Delinquent Count (12M)',
# #                                 'max_deliq_6mts': 'Max Delinquency (6M)',
# #                                 'max_deliq_12mts': 'Max Delinquency (12M)',
# #                                 'enq_L3m': 'Recent Inquiries (3M)',
# #                                 'enq_L6m': 'Inquiries (6M)',
# #                                 'enq_L12m': 'Inquiries (12M)',
# #                                 'num_std': 'Standard / Active Accounts',
# #                                 'num_std_6mts': 'Standard Accounts (6M)',
# #                                 'num_std_12mts': 'Standard Accounts (12M)',
# #                                 'num_sub': 'Sub-standard Accounts',
# #                                 'num_sub_6mts': 'Sub-standard (6M)',
# #                                 'num_dbt': 'Doubtful Accounts',
# #                                 'num_lss': 'Loss / Written-Off Accounts',
# #                                 'CC_utilization': 'Credit Card Utilization (0–1)',
# #                                 'PL_utilization': 'Personal Loan Utilization (0–1)',
# #                                 'CC_Flag': 'Has Credit Card (1=Yes)',
# #                                 'PL_Flag': 'Has Personal Loan (1=Yes)',
# #                                 'HL_Flag': 'Has Home Loan (1=Yes)',
# #                                 'GL_Flag': 'Has Gold Loan (1=Yes)',
# #                                 'written_off_count': 'Written Off Count',
# #                                 'settled_count': 'Settled Account Count',
# #                                 'high_util_flag': 'High Utilization Flag (1=Yes)',
# #                                 'recent_deliq_flag': 'Recent Delinquency Flag (1=Yes)',
# #                                 'account_quality_score': 'Account Quality Score (0–100)',
# #                                 'Time_With_Curr_Empr': 'Employment Tenure (months)',
# #                                 'NETMONTHLYINCOME': 'Net Monthly Income (₹)',
# #                                 'pct_of_active_TLs_ever': '% Active Trade Lines Ever',
# #                                 'pct_currentBal_all_TL': '% Current Balance / All TL',
# #                                 'max_unsec_exposure_inPct': 'Max Unsecured Exposure (%)',
# #                                 'extraction_method': 'Extraction Method',
# #                             }
# #                             exclude_keys = {'success', 'error', 'raw_text'}
# #                             data_items = []
# #                             for key, val in extraction_result.items():
# #                                 if key in exclude_keys:
# #                                     continue
# #                                 fname = friendly_names.get(key, key.replace('_', ' ').title())
# #                                 data_items.append({"Feature Name": fname, "Internal ID": key, "Extracted Value": str(val)})
# #                             data_items.sort(key=lambda x: x["Feature Name"])
# #                             data_items = [
# #                                 {"Feature Name": "── Application ID", "Internal ID": "application_id", "Extracted Value": app_id_display},
# #                                 {"Feature Name": "── Customer Name", "Internal ID": "customer_name", "Extracted Value": cust_name},
# #                                 {"Feature Name": "── Stage 1 Decision", "Internal ID": "stage1_decision", "Extracted Value": s1_decision},
# #                                 {"Feature Name": "── Stage 1 Risk Score", "Internal ID": "stage1_risk_score", "Extracted Value": str(s1_risk)},
# #                             ] + data_items
# #                             import pandas as _pd
# #                             df_all = _pd.DataFrame(data_items)
# #                             st.dataframe(df_all, use_container_width=True, hide_index=True)

# #                         enhanced_customer_data = stage1_customer.copy()
# #                         _s1_income = stage1_customer.get('avg_salary_6m', 50000)
# #                         _s2_income = extraction_result.get('NETMONTHLYINCOME', 0)
# #                         _use_income = _s1_income if (_s2_income > 0 and _s2_income < _s1_income * 0.4) else (_s2_income or _s1_income)
# #                         if _s2_income > 0 and _s2_income < _s1_income * 0.4:
# #                             st.warning(f'⚠️ CIBIL income ₹{_s2_income:,} is much lower than application income ₹{_s1_income:,}. Using application income for FOIR.')

# #                         enhanced_customer_data.update({
# #                             'bureau_score': extraction_result.get('Credit_Score', 720),
# #                             'age': extraction_result.get('AGE', stage1_customer.get('age', 35)),
# #                             'avg_salary_6m': _use_income,
# #                             'employment_tenure_months': extraction_result.get('Time_With_Curr_Empr', stage1_customer.get('employment_tenure_months', 24)),
# #                             'dpd_30_count_6m': extraction_result.get('num_times_30p_dpd', 0),
# #                             'dpd_90_count_6m': extraction_result.get('dpd_90_count_6m', 0),
# #                             'max_delinquency_level': extraction_result.get('max_delinquency_level', 0),
# #                             'num_times_delinquent': extraction_result.get('num_times_delinquent', 0),
# #                             'num_deliq_6mts': extraction_result.get('num_deliq_6mts', 0),
# #                             'num_deliq_12mts': extraction_result.get('num_deliq_12mts', 0),
# #                             'max_deliq_6mts': extraction_result.get('max_deliq_6mts', 0),
# #                             'max_deliq_12mts': extraction_result.get('max_deliq_12mts', 0),
# #                             'recent_inquiries_3m': extraction_result.get('enq_L3m', 2),
# #                             'enq_L6m': extraction_result.get('enq_L6m', 4),
# #                             'enq_L12m': extraction_result.get('enq_L12m', 6),
# #                             'active_loans_count': extraction_result.get('num_std', 1),
# #                             'num_std_6mts': extraction_result.get('num_std_6mts', 0),
# #                             'num_std_12mts': extraction_result.get('num_std_12mts', 0),
# #                             'num_sub': extraction_result.get('num_sub', 0),
# #                             'num_sub_6mts': extraction_result.get('num_sub_6mts', 0),
# #                             'num_dbt': extraction_result.get('num_dbt', 0),
# #                             'num_lss': extraction_result.get('num_lss', 0),
# #                             'credit_utilization_pct': (0 if extraction_result.get('CC_utilization', 0) < 0 else extraction_result.get('CC_utilization', 0.35)) * 100,
# #                             'pct_of_active_TLs_ever': extraction_result.get('pct_of_active_TLs_ever', 0.6),
# #                             'pct_currentBal_all_TL': extraction_result.get('pct_currentBal_all_TL', 0.3),
# #                             'CC_utilization': 0 if extraction_result.get('CC_utilization', 0) < 0 else extraction_result.get('CC_utilization', 0.35),
# #                             'PL_utilization': 0 if extraction_result.get('PL_utilization', 0) < 0 else extraction_result.get('PL_utilization', 0.25),
# #                             'max_unsec_exposure_inPct': extraction_result.get('max_unsec_exposure_inPct', 30),
# #                             'CC_Flag': extraction_result.get('CC_Flag', 0),
# #                             'PL_Flag': extraction_result.get('PL_Flag', 0),
# #                             'HL_Flag': extraction_result.get('HL_Flag', 0),
# #                             'GL_Flag': extraction_result.get('GL_Flag', 0),
# #                             'written_off_count': extraction_result.get('written_off_count', 0),
# #                             'settled_count': extraction_result.get('settled_count', 0),
# #                             'high_util_flag': extraction_result.get('high_util_flag', 0),
# #                             'recent_deliq_flag': extraction_result.get('recent_deliq_flag', 0),
# #                             'account_quality_score': extraction_result.get('account_quality_score', 0)
# #                         })

# #                         enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)

# #                         with st.spinner("🔬 Running Stage 2 analysis..."):
# #                             try:
# #                                 stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
# #                                 stage2_result = resolve_stage2_to_binary(stage2_result)
# #                                 display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
# #                             except Exception as e:
# #                                 st.error(f"❌ Analysis failed: {str(e)}")
# #                     else:
# #                         st.error("❌ PDF extraction failed! Error: " + extraction_result.get('error', 'Unknown'))

# #     elif selected_tab == "Batch Analysis":
# #         st.markdown('<p class="section-header">📊 Batch CIBIL Analysis</p>', unsafe_allow_html=True)
# #         st.info("📊 Batch analysis feature coming soon! (Upload a CSV with all required CIBIL fields)")

# # elif page == "📊 Batch Process":
# #     st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
# #     st.markdown("""
# #         <div class="info-box">
# #             📤 Upload a CSV file with customer data for bulk credit assessment.
# #             The file should include all required fields for prediction.
# #         </div>
# #     """, unsafe_allow_html=True)
# #     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# #     if uploaded_file is not None:
# #         try:
# #             df = pd.read_csv(uploaded_file)
# #             st.success(f"✅ Successfully loaded {len(df)} records")
# #             with st.expander("📄 Preview Uploaded Data"):
# #                 st.dataframe(df.head(), use_container_width=True)
# #                 st.write(f"**Total Records:** {len(df)}")
# #                 st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
# #             required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
# #             missing_cols = [col for col in required_cols if col not in df.columns]
# #             if missing_cols:
# #                 st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
# #                 st.info("Please ensure your CSV includes at least these columns: age, employment_type, avg_salary_6m, bureau_score, loan_amount")
# #             else:
# #                 if st.button("🚀 Process Batch Predictions", key="process_batch_btn", type="primary", use_container_width=True):
# #                     with st.spinner(f"🔍 Processing {len(df)} records..."):
# #                         progress_bar = st.progress(0)
# #                         results_df = process_batch_predictions(df)
# #                         progress_bar.progress(100)
# #                         st.success(f"✅ Completed processing {len(results_df)} records!")
# #                         tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
# #                         with tab1:
# #                             st.dataframe(results_df, use_container_width=True)
# #                             col1, col2, col3, col4 = st.columns(4)
# #                             with col1:
# #                                 st.metric("✅ Approved", len(results_df[results_df['decision'] == 'APPROVE']))
# #                             with col2:
# #                                 st.metric("❌ Rejected", len(results_df[results_df['decision'] == 'REJECT']))
# #                             with col3:
# #                                 st.metric("⚠️ Review", len(results_df[results_df['decision'] == 'REVIEW']))
# #                             with col4:
# #                                 st.metric("📊 Avg Risk Score", f"{results_df['risk_score'].mean():.0f}")
# #                         with tab2:
# #                             col1, col2 = st.columns(2)
# #                             with col1:
# #                                 decision_counts = results_df['decision'].value_counts()
# #                                 fig1 = px.pie(values=decision_counts.values, names=decision_counts.index,
# #                                               title="Decision Distribution", color=decision_counts.index,
# #                                               color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
# #                                 st.plotly_chart(fig1, use_container_width=True)
# #                             with col2:
# #                                 fig2 = px.histogram(results_df, x='risk_score', title="Risk Score Distribution",
# #                                                     nbins=20, color_discrete_sequence=['#587042'])
# #                                 st.plotly_chart(fig2, use_container_width=True)
# #                             fig3 = px.scatter(results_df, x='monthly_income', y='loan_amount', color='decision',
# #                                               size='risk_score', title="Income vs Loan Amount (Colored by Decision)",
# #                                               hover_data=['application_id', 'foir_percentage'],
# #                                               color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
# #                             st.plotly_chart(fig3, use_container_width=True)
# #                             fig4 = px.box(results_df, x='decision', y='pd_percentage',
# #                                           title="PD Distribution by Decision", color='decision',
# #                                           color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
# #                             st.plotly_chart(fig4, use_container_width=True)
# #                         with tab3:
# #                             st.markdown("### Download Results")
# #                             col1, col2 = st.columns(2)
# #                             with col1:
# #                                 st.download_button(
# #                                     "📥 Download as CSV",
# #                                     data=results_df.to_csv(index=False),
# #                                     file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# #                                     mime="text/csv",
# #                                     use_container_width=True
# #                                 )
# #                             with col2:
# #                                 st.download_button(
# #                                     "📥 Download as JSON",
# #                                     data=results_df.to_json(orient='records', indent=2),
# #                                     file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
# #                                     mime="application/json",
# #                                     use_container_width=True
# #                                 )
# #                             st.markdown("---")
# #                             st.markdown("#### Filtered Downloads")
# #                             col1, col2, col3 = st.columns(3)
# #                             with col1:
# #                                 approved_df = results_df[results_df['decision'] == 'APPROVE']
# #                                 if len(approved_df) > 0:
# #                                     st.download_button(
# #                                         f"✅ Approved Only ({len(approved_df)})",
# #                                         data=approved_df.to_csv(index=False),
# #                                         file_name=f"approved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# #                                         mime="text/csv",
# #                                         use_container_width=True
# #                                     )
# #                             with col2:
# #                                 rejected_df = results_df[results_df['decision'] == 'REJECT']
# #                                 if len(rejected_df) > 0:
# #                                     st.download_button(
# #                                         f"❌ Rejected Only ({len(rejected_df)})",
# #                                         data=rejected_df.to_csv(index=False),
# #                                         file_name=f"rejected_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# #                                         mime="text/csv",
# #                                         use_container_width=True
# #                                     )
# #                             with col3:
# #                                 review_df = results_df[results_df['decision'] == 'REVIEW']
# #                                 if len(review_df) > 0:
# #                                     st.download_button(
# #                                         f"⚠️ Review Only ({len(review_df)})",
# #                                         data=review_df.to_csv(index=False),
# #                                         file_name=f"review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# #                                         mime="text/csv",
# #                                         use_container_width=True
# #                                     )
# #         except Exception as e:
# #             st.error(f"❌ Error processing file: {str(e)}")
# #             st.info("Please ensure the CSV file is properly formatted and contains the required columns.")
# #     else:
# #         st.markdown("---")
# #         st.markdown("### 📋 CSV Template")
# #         template_data = {
# #             'age': [35, 42, 28],
# #             'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
# #             'dependents': [2, 3, 6],
# #             'kyc_verified': ['Yes', 'Yes', 'No'],
# #             'bankruptcy_flag': ['No', 'No', 'No'],
# #             'fraud_flag': ['No', 'No', 'No'],
# #             'employment_tenure_months': [24, 0, 18],
# #             'business_vintage_years': [0, 5, 0],
# #             'bureau_score': [720, 680, 580],
# #             'dpd_90_count_6m': [0, 1, 2],
# #             'dpd_30_count_6m': [0, 2, 1],
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
# #             'AMT_ANNUITY': [8500, 9500, 4500],
# #             'payment_discipline_flag': ['GOOD', 'MODERATE', 'POOR'],
# #             'liquidity_flag': ['LOW', 'ADEQUATE', 'LOW'],
# #             'cashflow_health': ['HEALTHY', 'MODERATE', 'STRESSED'],
# #             'bureau_risk_flag': ['LOW', 'MEDIUM', 'HIGH'],
# #             'inward_bounce_count_3m': [0, 1, 3],
# #             'salary_missing_months': [0, 0, 2],
# #         }
# #         template_df = pd.DataFrame(template_data)
# #         st.dataframe(template_df, use_container_width=True)
# #         st.caption("📝 Note: `dependents > 5` will automatically trigger REVIEW regardless of other factors.")
# #         st.download_button(
# #             "📥 Download CSV Template",
# #             data=template_df.to_csv(index=False),
# #             file_name="credit_assessment_template.csv",
# #             mime="text/csv",
# #             use_container_width=True
# #         )

# # elif page == "📈 Model Info":
# #     st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         st.markdown('<div class="stat-card"><div class="stat-number">RF</div><div class="stat-label">Model Type</div></div>', unsafe_allow_html=True)
# #     with col2:
# #         st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TOP_FEATURES)}</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)
# #     with col3:
# #         st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TARGET_LE.classes_)}</div><div class="stat-label">Classes</div></div>', unsafe_allow_html=True)
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
# #     feature_df = pd.DataFrame({'Rank': range(1, min(21, len(TOP_FEATURES) + 1)), 'Feature': TOP_FEATURES[:20]})
# #     st.dataframe(feature_df, use_container_width=True, hide_index=True)

# # elif page == "ℹ️ About":
# #     st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
# #     st.markdown("""
# #         <div class="info-card">
# #             <div class="info-card-title"><span class="icon">🏦</span><span>Credit Risk Assessment Platform</span></div>
# #             <div class="info-card-content">
# #                 <p><strong>Version:</strong> 8.4 - OCR AUTO-FILL FIX (categorical dropdowns now update from PDF)</p>
# #                 <p><strong>Developer:</strong> Zen Meraki</p>
# #                 <p><strong>Date:</strong> January 2026</p>
# #                 <br>
# #                 <p>A comprehensive credit risk evaluation system combining hard policy rules,
# #                 machine learning models, and affordability analysis for accurate and compliant lending decisions.</p>
# #             </div>
# #         </div>
# #     """, unsafe_allow_html=True)
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.markdown("""
# #             <div class="info-card">
# #                 <div class="info-card-title"><span class="icon">🎯</span><span>Key Features</span></div>
# #                 <div class="info-card-content">
# #                     <ul style="margin: 0; padding-left: 1.25rem;">
# #                         <li>Three-layer decision engine</li>
# #                         <li>Real-time risk assessment</li>
# #                         <li>Industry-standard PD calculation</li>
# #                         <li>FOIR calculation & validation</li>
# #                         <li>Automated reason generation</li>
# #                         <li>Complete audit trail (PDF)</li>
# #                         <li>Professional UI/UX</li>
# #                         <li>OCR auto-fill with full categorical inference</li>
# #                     </ul>
# #                 </div>
# #             </div>
# #         """, unsafe_allow_html=True)
# #     with col2:
# #         st.markdown("""
# #             <div class="info-card">
# #                 <div class="info-card-title"><span class="icon">🛠️</span><span>Technology Stack</span></div>
# #                 <div class="info-card-content">
# #                     <ul style="margin: 0; padding-left: 1.25rem;">
# #                         <li>Streamlit (UI Framework)</li>
# #                         <li>Scikit-learn (ML)</li>
# #                         <li>Plotly (Visualizations)</li>
# #                         <li>Pandas (Data Processing)</li>
# #                         <li>ReportLab (PDF Generation)</li>
# #                         <li>Python 3.8+</li>
# #                     </ul>
# #                 </div>
# #             </div>
# #         """, unsafe_allow_html=True)







# """
# Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
# Enhanced with Modern UI/UX Design
# Run with: streamlit run test.py (from inside the notebooks folder)
# Author: Zen Meraki
# Date: January 2026
# VERSION: 8.6 - CLEANED (removed duplicates) + Fairness Monitoring + City Tier + RBI Consent
# """

# import streamlit as st

# # =============================================================================
# # PAGE CONFIGURATION – MUST BE THE VERY FIRST STREAMLIT COMMAND
# # =============================================================================
# st.set_page_config(
#     page_title="Credit Risk Assessment",
#     page_icon="💳",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # =============================================================================
# # STANDARD LIBRARY / THIRD-PARTY IMPORTS
# # =============================================================================
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import joblib
# import warnings
# from datetime import datetime
# import base64
# from typing import List, Any
# import json
# import sys
# import os
# from pathlib import Path
# import re

# # =============================================================================
# # SUPPRESS SCIKIT-LEARN VERSION WARNINGS
# # =============================================================================
# warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# # =============================================================================
# # DYNAMIC PATH RESOLUTION
# # =============================================================================
# CURRENT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = CURRENT_DIR.parent
# POSSIBLE_LOCATIONS = [
#     CURRENT_DIR, PROJECT_ROOT,
#     PROJECT_ROOT / "loan", PROJECT_ROOT / "utils", PROJECT_ROOT / "notebooks",
# ]
# for loc in POSSIBLE_LOCATIONS:
#     if loc.exists() and str(loc) not in sys.path:
#         sys.path.insert(0, str(loc))

# # =============================================================================
# # OPTIONAL OCR DEPENDENCIES – GRACEFUL FALLBACK
# # =============================================================================
# OCR_AVAILABLE = False
# OCR_ERROR_MSG = ""
# try:
#     import pytesseract
#     from pdf2image import convert_from_bytes
#     import cv2
#     from PIL import Image
#     import shutil as _shutil
#     _tess_cmd = (
#         _shutil.which("tesseract")
#         or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#     )
#     if _tess_cmd:
#         pytesseract.pytesseract.tesseract_cmd = _tess_cmd
#     pytesseract.get_tesseract_version()
#     OCR_AVAILABLE = True
# except ImportError as _e:
#     OCR_ERROR_MSG = (
#         f"Missing Python package: {_e}. "
#         "Add to requirements.txt: pytesseract  pdf2image  opencv-python-headless  pillow"
#     )
# except Exception as _e:
#     _name = type(_e).__name__
#     if "TesseractNotFound" in _name or "tesseract" in str(_e).lower():
#         OCR_ERROR_MSG = (
#             "Tesseract binary not found. "
#             "Streamlit Cloud → add 'tesseract-ocr' and 'poppler-utils' to packages.txt. "
#             "Linux → sudo apt install tesseract-ocr poppler-utils. "
#             "Mac → brew install tesseract poppler."
#         )
#     else:
#         OCR_ERROR_MSG = f"OCR init error ({_name}): {_e}"

# # =============================================================================
# # IMPORT CSS – WITH FALLBACK
# # =============================================================================
# try:
#     from css_styles import CSS
# except ImportError:
#     CSS = """
#     <style>
#         .main-header { font-size: 2rem; font-weight: bold; color: #2d3748; }
#         .section-header { font-size: 1.5rem; font-weight: 600; color: #2d3748; }
#         .info-box { background: #f7fafc; padding: 1rem; border-radius: 0.5rem; }
#         .decision-card { padding: 2rem; border-radius: 1rem; text-align: center; margin-bottom: 1rem; }
#         .decision-card-approved { background: #c6f6d5; border-left: 5px solid #48bb78; }
#         .decision-card-rejected { background: #fed7d7; border-left: 5px solid #f56565; }
#         .decision-card-review { background: #feebc8; border-left: 5px solid #ed8936; }
#         .decision-title { font-size: 2.5rem; font-weight: bold; }
#         .decision-subtitle { font-size: 1rem; opacity: 0.8; }
#         .stat-card { background: white; padding: 1rem; border-radius: 0.5rem;
#                      box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
#         .stat-number { font-size: 1.8rem; font-weight: bold; color: #2d3748; }
#         .stat-label { font-size: 0.875rem; color: #718096; }
#         .info-card { background: white; border-radius: 0.5rem; padding: 1rem;
#                      margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
#         .info-card-title { font-weight: 600; margin-bottom: 0.5rem; color: #2d3748; }
#         .info-card-content { font-size: 0.875rem; }
#         .data-row { display: flex; justify-content: space-between;
#                     padding: 0.25rem 0; border-bottom: 1px solid #e2e8f0; }
#         .data-label { color: #4a5568; }
#         .data-value { font-weight: 500; }
#         .status-badge { padding: 0.25rem 0.5rem; border-radius: 0.25rem;
#                         font-size: 0.75rem; margin-left: 0.5rem; }
#         .badge-pass { background: #c6f6d5; color: #22543d; }
#         .badge-fail { background: #fed7d7; color: #742a2a; }
#         .badge-warning { background: #feebc8; color: #744210; }
#         .reason-item { padding: 0.25rem 0; }
#         .reason-icon { color: #587042; font-weight: bold; margin-right: 0.5rem; }
#     </style>
#     """
# st.markdown(CSS, unsafe_allow_html=True)

# # =============================================================================
# # CITY TIER MAPPING
# # =============================================================================
# CITY_TIERS = {
#     "Tier 1 – Metro (Mumbai, Delhi, Bengaluru, Chennai, Hyderabad, Kolkata, Pune, Ahmedabad)": "Tier 1",
#     "Tier 2 – Large City (Jaipur, Lucknow, Kochi, Nagpur, Indore, Bhopal, Patna, Vadodara…)": "Tier 2",
#     "Tier 3 – Small City / Town": "Tier 3",
#     "Rural / Village": "Rural",
# }

# # =============================================================================
# # SESSION STATE INITIALIZATION
# # =============================================================================
# def init_session_state():
#     defaults = {
#         'stage1_complete':       False,
#         'stage1_decision':       None,
#         'stage1_data':           None,
#         'current_customer_data': None,
#         'page_navigation':       "🏠 Home",
#         'use_two_stage':         False,
#         'stage2_selected_tab':   "Manual Entry",
#         # Fairness log — persists across sessions in memory
#         'fairness_log':          [],
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v

# init_session_state()

# # =============================================================================
# # IMPORT BUSINESS LOGIC MODULES
# # =============================================================================
# try:
#     from affordability_engine import calculate_emi, calculate_affordability
#     from reason_codes import generate_reason_codes
#     from risk_engine import (
#         calculate_final_risk_score, fill_missing_ml_fields,
#         clean_sentinel_values
#     )
#     from affordability_engine import check_net_disposable
# except ImportError as e:
#     st.error(f"❌ Failed to import required modules: {e}")
#     st.info("""
#     Required files (place in notebooks/, loan/, utils/, or project root):
#     - affordability_engine.py  |  reason_codes.py  |  risk_engine.py
#     - utils/__init__.py  |  utils/pdf_generator.py
#     """)
#     st.stop()

# # =============================================================================
# # STAGE 2 ENGINE – ROBUST FALLBACK
# # =============================================================================
# try:
#     import stage2_engine
#     from stage2_engine import make_two_stage_decision, is_stage2_available, get_stage2_status
#     STAGE2_AVAILABLE = is_stage2_available()
# except ImportError:
#     stage2_engine = None
#     STAGE2_AVAILABLE = False
#     def make_two_stage_decision(*args, **kwargs):
#         raise NotImplementedError("Stage 2 engine not available")
#     def is_stage2_available(): return False
#     def get_stage2_status(): return {"error": "Stage 2 engine module not found", "available": False}

# # =============================================================================
# # PDF GENERATION – SAFE FALLBACK
# # =============================================================================
# PDF_AVAILABLE = False
# generate_decision_pdf = None
# generate_audit_pdf = None
# try:
#     from utils.pdf_generator import generate_decision_pdf, generate_audit_pdf
#     PDF_AVAILABLE = True
# except ImportError:
#     pass

# # =============================================================================
# # JSON SANITIZER
# # =============================================================================
# def sanitize_for_json(obj: Any) -> Any:
#     if obj is None or isinstance(obj, (str, int, float, bool)): return obj
#     if isinstance(obj, set): return list(obj)
#     if isinstance(obj, datetime): return obj.isoformat()
#     if isinstance(obj, np.integer): return int(obj)
#     if isinstance(obj, np.floating): return float(obj)
#     if isinstance(obj, np.ndarray): return obj.tolist()
#     if isinstance(obj, dict): return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)): return [sanitize_for_json(item) for item in obj]
#     try:
#         json.dumps(obj); return obj
#     except (TypeError, ValueError): return str(obj)

# # =============================================================================
# # LOAD TRAINED MODEL ASSETS (Stage 1 Random Forest)
# # =============================================================================
# @st.cache_resource
# def load_model_assets():
#     try:
#         possible_paths = [
#             'credit_risk_assets.pkl',
#             'notebooks/credit_risk_assets.pkl',
#             '../notebooks/credit_risk_assets.pkl'
#         ]
#         assets = None
#         for path in possible_paths:
#             try: assets = joblib.load(path); break
#             except FileNotFoundError: continue
#         if assets is None:
#             raise FileNotFoundError("Could not find credit_risk_assets.pkl")
#         return {
#             'model': assets['model'], 'features': assets['features'],
#             'le_map': assets['le_map'], 'target_le': assets['target_le'],
#             'loaded': True, 'error': None
#         }
#     except FileNotFoundError:
#         return {'loaded': False, 'error': 'credit_risk_assets.pkl not found. Please run the training script first.'}
#     except Exception as e:
#         return {'loaded': False, 'error': f'Error loading model: {str(e)}'}

# ASSETS = load_model_assets()
# if not ASSETS['loaded']:
#     st.error(f"❌ {ASSETS['error']}")
#     st.info("Please ensure 'credit_risk_assets.pkl' is in the same directory as this app.")
#     st.stop()

# MODEL      = ASSETS['model']
# TOP_FEATURES = ASSETS['features']
# LE_MAP     = ASSETS['le_map']
# TARGET_LE  = ASSETS['target_le']

# # =============================================================================
# # PD CALCULATION FUNCTIONS
# # NOTE: calculate_emi, calculate_affordability, generate_reason_codes,
# #       calculate_final_risk_score are imported from their respective modules.
# #       The PD functions below are NOT in any module so are kept here.
# # =============================================================================
# def bureau_score_to_pd(bureau_score):
#     if bureau_score >= 800: return 0.5 + (900 - bureau_score) / 200 * 0.5
#     elif bureau_score >= 750: return 1.0 + (800 - bureau_score) / 50 * 1.0
#     elif bureau_score >= 700: return 2.0 + (750 - bureau_score) / 50 * 1.5
#     elif bureau_score >= 650: return 3.5 + (700 - bureau_score) / 50 * 2.5
#     elif bureau_score >= 600: return 6.0 + (650 - bureau_score) / 50 * 4.0
#     elif bureau_score >= 550: return 10.0 + (600 - bureau_score) / 50 * 5.0
#     else: return min(25.0, 15.0 + (550 - bureau_score) / 50 * 10.0)

# def foir_to_pd_adjustment(foir_percentage):
#     if foir_percentage <= 30: return -0.75
#     elif foir_percentage <= 40: return 0.00
#     elif foir_percentage <= 45: return 0.75
#     elif foir_percentage <= 50: return 1.50
#     elif foir_percentage <= 55: return 2.25
#     elif foir_percentage <= 60: return 3.50
#     else: return 6.00

# def delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count=0):
#     if dpd_90_count >= 3: return 5.0
#     elif dpd_90_count == 2: return 3.0
#     elif dpd_90_count == 1: return 2.0
#     elif dpd_30_count >= 3: return 1.6
#     elif dpd_30_count >= 1: return 1.3
#     else: return 1.0

# def employment_stability_to_pd_adjustment(employment_type, tenure_months, business_vintage_years=0):
#     if employment_type == 'Salaried':
#         if tenure_months >= 36: return -0.5
#         elif tenure_months >= 12: return 0.0
#         elif tenure_months >= 6: return 0.5
#         else: return 2.0
#     elif employment_type in ['Self-Employed', 'Business']:
#         if business_vintage_years >= 5: return -0.5
#         elif business_vintage_years >= 2: return 0.0
#         else: return 1.5
#     else: return 1.0

# def inquiry_pattern_to_pd_adjustment(recent_inquiries_3m):
#     if recent_inquiries_3m <= 1: return -0.3
#     elif recent_inquiries_3m <= 3: return 0.0
#     elif recent_inquiries_3m <= 5: return 0.8
#     elif recent_inquiries_3m <= 8: return 1.5
#     else: return 3.0

# def ml_confidence_to_pd_adjustment(ml_confidence, ml_decision):
#     if ml_decision == "APPROVE":
#         if ml_confidence >= 90: return -0.5
#         elif ml_confidence >= 70: return 0.0
#         else: return 0.5
#     elif ml_decision == "REVIEW": return 1.0
#     else: return 5.0

# def calculate_final_pd(bureau_score, foir, confidence, dpd_90_count=0, dpd_30_count=0,
#                        employment_type='Salaried', employment_tenure=24, business_vintage=0,
#                        recent_inquiries=2, ml_decision='APPROVE'):
#     base_pd = bureau_score_to_pd(bureau_score)
#     foir_adj = foir_to_pd_adjustment(foir)
#     deliq_multiplier = delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count)
#     employment_adj = employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage)
#     inquiry_adj = inquiry_pattern_to_pd_adjustment(recent_inquiries)
#     ml_adj = ml_confidence_to_pd_adjustment(confidence, ml_decision)
#     adjusted_base_pd = base_pd * deliq_multiplier
#     final_pd = adjusted_base_pd + foir_adj + employment_adj + inquiry_adj + ml_adj
#     return round(max(0.5, min(final_pd, 25.0)), 2)

# # =============================================================================
# # CATEGORICAL FLAG INFERENCE (v8.5 dual-dataset)
# # =============================================================================
# def _infer_surplus_from_cibil(score: int, dpd_60: int, dpd_30: int, income: float) -> float:
#     if dpd_60 >= 3: return income * -0.5
#     elif score < 650 or dpd_60 >= 1: return income * -0.2
#     elif score < 700: return income * 0.1
#     else: return income * 0.3

# def infer_categorical_flags(extraction_result: dict) -> dict:
#     score       = int(extraction_result.get('Credit_Score', 700) or 700)
#     dpd_30      = int(extraction_result.get('num_times_30p_dpd', 0) or 0)
#     dpd_60      = int(extraction_result.get('num_times_60p_dpd', 0) or 0)
#     written_off = int(extraction_result.get('num_lss', 0) or extraction_result.get('written_off_count', 0) or 0)
#     doubtful    = int(extraction_result.get('num_dbt', 0) or 0)
#     cc_util_raw = extraction_result.get('CC_utilization', 0) or 0
#     cc_util     = float(cc_util_raw) if cc_util_raw > 0 else 0.0
#     income      = float(extraction_result.get('NETMONTHLYINCOME', 0) or
#                         extraction_result.get('avg_salary_6m', 50_000) or 50_000)
#     tenure      = int(extraction_result.get('Time_With_Curr_Empr', 24) or 24)

#     is_bureau_only = (
#         'NETMONTHLYINCOME' in extraction_result
#         and 'net_cash_surplus_6m' not in extraction_result
#         and 'net_surplus' not in extraction_result
#     )

#     if is_bureau_only:
#         dpd_90_proxy = dpd_60
#         surplus = _infer_surplus_from_cibil(score, dpd_60, dpd_30, income)
#         payment_discipline = 'POOR' if (dpd_60 >= 1 or dpd_30 >= 3) else ('MODERATE' if dpd_30 >= 1 else 'GOOD')
#         cashflow_health = ('HEALTHY' if surplus >= 14_000 else 'STABLE' if surplus >= 600 else 'STRESSED' if surplus < -1_000 else 'MODERATE')
#         liquidity_flag  = ('ADEQUATE' if surplus > 14_000 else 'LOW' if surplus < -32_000 else 'MODERATE')
#         bureau_risk     = ('HIGH' if (written_off >= 1 or doubtful >= 1 or dpd_60 >= 3 or score < 580)
#                            else 'MEDIUM' if (score < 650 or (dpd_30 >= 2 and cc_util > 0.60)) else 'LOW')
#         salary_stability = ('UNSTABLE' if tenure < 6 else 'STABLE' if (tenure >= 24 and score >= 700 and dpd_30 == 0) else 'MODERATE')
#     else:
#         dpd_90      = int(extraction_result.get('dpd_90_count_6m', 0) or 0)
#         bounces     = int(extraction_result.get('inward_bounce_count_3m', 0) or 0)
#         missing     = int(extraction_result.get('salary_missing_months', 0) or 0)
#         hard_reject = int(extraction_result.get('hard_reject_flag', 0) or 0)
#         surplus     = float(extraction_result.get('net_cash_surplus_6m') or extraction_result.get('net_surplus') or -50_000)
#         payment_discipline = ('POOR' if (dpd_90 >= 1 or bounces >= 2)
#                                else 'MODERATE' if (bounces == 1 or dpd_30 >= 3) else 'GOOD')
#         cashflow_health = ('HEALTHY' if surplus >= 14_000 else 'STABLE' if 600 <= surplus < 14_000
#                             else 'STRESSED' if surplus < -1_000 else 'MODERATE')
#         liquidity_flag  = 'ADEQUATE' if surplus > 14_000 else 'LOW' if surplus < -32_000 else 'MODERATE'
#         bureau_risk     = ('HIGH' if (hard_reject or dpd_90 >= 3 or written_off >= 1 or (dpd_90 >= 1 and dpd_30 >= 2))
#                            else 'MEDIUM' if (score < 580 or (dpd_30 >= 2 and cc_util > 0.60)) else 'LOW')
#         salary_stability = ('UNSTABLE' if missing >= 1
#                              else 'STABLE' if (missing == 0 and score >= 700 and dpd_30 == 0 and bounces == 0)
#                              else 'MODERATE')
#         surplus_for_return = surplus

#     return {
#         'payment_discipline_flag': payment_discipline,
#         'cashflow_health':         cashflow_health,
#         'liquidity_flag':          liquidity_flag,
#         'bureau_risk_flag':        bureau_risk,
#         'salary_stability_flag':   salary_stability,
#         '_inference_path':         'bureau_only' if is_bureau_only else 'bank_statement',
#     }

# # =============================================================================
# # CIBIL PDF EXTRACTION ENGINE (OCR + PATTERN MATCHING)
# # =============================================================================
# def extract_cibil_from_pdf(uploaded_file):
#     if not OCR_AVAILABLE:
#         return {'success': False, 'error': OCR_ERROR_MSG or 'OCR libraries not installed.'}
#     try:
#         pdf_bytes = uploaded_file.read()
#         images = convert_from_bytes(pdf_bytes, dpi=300)
#         full_text = ""
#         for image in images:
#             gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
#             _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             full_text += pytesseract.image_to_string(binary) + "\n"

#         credit_score = 720
#         for pat in [
#             r'\b(\d{3})\s*(?:EXCELLENT|VERY\s*GOOD|GOOD|FAIR|SUBPRIME|POOR|NH|NA)\b',
#             r'(?:cibil|credit)\s*score\s*[:\-\(]?\s*(\d{3})',
#             r'score.*?\((\d{3})\)',
#         ]:
#             m = re.search(pat, full_text, re.IGNORECASE)
#             if m:
#                 v = int(m.group(1))
#                 if 300 <= v <= 900:
#                     credit_score = v; break

#         monthly_income = 50000
#         m = re.search(r'(?:net\s+monthly\s+income|monthly\s+income|salary)[^\n\r]{0,30}?(?:rs\.?\s*|₹\s*)([\d,]+)', full_text, re.IGNORECASE)
#         if m:
#             v = int(m.group(1).replace(',', ''))
#             if v > 1000: monthly_income = v

#         cc_util_pct = 35
#         m = re.search(r'utilization\s*[\(:\-]?\s*(\d{1,3})\s*%', full_text, re.IGNORECASE)
#         if m: cc_util_pct = int(m.group(1))
#         cc_util = cc_util_pct / 100.0
#         high_util = 1 if cc_util_pct > 75 else 0

#         age_extracted = 35
#         m = re.search(r'(?:date\s+of\s+birth|dob)[:\s]+(\d{2}[-/]\w{3,9}[-/]\d{2,4})', full_text, re.IGNORECASE)
#         if m:
#             for fmt in ('%d-%b-%Y', '%d/%b/%Y', '%d-%m-%Y', '%d/%m/%Y'):
#                 try:
#                     dob = datetime.strptime(m.group(1), fmt)
#                     age_extracted = int((datetime.now() - dob).days / 365.25); break
#                 except Exception: continue

#         lines = full_text.split('\n')
#         accounts, enquiry_dates = [], []
#         in_accounts = in_enquiry = False
#         for line in lines:
#             lu = line.upper()
#             if 'ACCOUNT DETAILS' in lu: in_accounts, in_enquiry = True, False; continue
#             if 'ENQUIRY DETAILS' in lu: in_accounts, in_enquiry = False, True; continue
#             if in_accounts:
#                 if re.search(r'SUMMARY|SCORE|PERSONAL\s+INFO', lu): break
#                 stripped = line.strip()
#                 if not stripped: continue
#                 dpd_m = re.search(r'\b(0\d0|0\d\d|\d{3})\b', stripped)
#                 stat_m = re.search(r'\b(Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss)\s*$', stripped, re.IGNORECASE)
#                 if re.search(r'\bINR\b', stripped, re.IGNORECASE) or re.match(r'^[A-Z][a-zA-Z\s]+(?:Bank|Finance|Capital)', stripped):
#                     accounts.append({'dpd': int(dpd_m.group(1)) if dpd_m else 0,
#                                      'status': (stat_m.group(1) if stat_m else 'Active').lower()})
#             if in_enquiry:
#                 em = re.match(r'^\s*(\d{2}-[A-Za-z]{3}-\d{4})', line)
#                 if em: enquiry_dates.append(em.group(1))

#         written_off_count = settled_count = dpd_90_count = dpd_60_count = dpd_30_count = active_count = sub_std = 0
#         if accounts:
#             for acc in accounts:
#                 d, s = acc.get('dpd', 0), acc.get('status', '')
#                 if d >= 90: dpd_90_count += 1
#                 elif d >= 60: dpd_60_count += 1
#                 elif d >= 30: dpd_30_count += 1
#                 if 'written' in s: written_off_count += 1
#                 elif 'settled' in s: settled_count += 1
#                 elif 'active' in s: active_count += 1
#                 if d >= 30: sub_std += 1
#         else:
#             written_off_count = len(re.findall(r'\bwritten[-\s]?off\b', full_text, re.IGNORECASE))
#             settled_count     = len(re.findall(r'\bsettled\b', full_text, re.IGNORECASE))
#             dpd_90_count      = len(re.findall(r'\b090\b|90\+?\s*dpd', full_text, re.IGNORECASE))
#             dpd_60_count      = len(re.findall(r'\b060\b|60\+?\s*dpd', full_text, re.IGNORECASE))
#             dpd_30_count      = len(re.findall(r'\b030\b|30\+?\s*dpd', full_text, re.IGNORECASE))

#         enq_12m = len(enquiry_dates)
#         em2 = re.search(r'Enquiries?\s*\(?12M\)?\s*[:\s]+(\d+)', full_text, re.IGNORECASE)
#         if em2: enq_12m = max(enq_12m, int(em2.group(1)))

#         total_accounts = max(len(accounts), active_count + settled_count + written_off_count)
#         pct_active = active_count / total_accounts if total_accounts > 0 else 0.6

#         employment_tenure_months = 36
#         tm = re.search(r'(?:employed\s+for|employment\s+tenure|with\s+current\s+employer)[^\d]*(\d+)\s*(?:year|yr)', full_text, re.IGNORECASE)
#         if tm: employment_tenure_months = int(tm.group(1)) * 12

#         gender = 'F' if re.search(r'\bfemale\b', full_text, re.IGNORECASE) else 'M'
#         marital_status = 'Single' if re.search(r'\bsingle\b|\bunmarried\b', full_text, re.IGNORECASE) else 'Married'
#         education = 'GRADUATE'
#         for pat, val in [(r'post.?grad', 'POST-GRADUATE'), (r'professional', 'PROFESSIONAL'),
#                          (r'under.?grad', 'UNDER GRADUATE'), (r'\b12th\b|\bhsc\b', '12TH'), (r'\bssc\b|\b10th\b', 'SSC')]:
#             if re.search(pat, full_text, re.IGNORECASE): education = val; break

#         prod_map = {r'personal\s+loan': 'PL', r'credit\s+card': 'CC', r'home\s+loan': 'HL', r'auto\s+loan|car\s+loan': 'AL'}
#         last_prod = first_prod = 'others'
#         for pat, label in prod_map.items():
#             if re.search(pat, full_text, re.IGNORECASE): last_prod = first_prod = label; break

#         if credit_score >= 750 and (written_off_count > 0 or dpd_90_count > 0):
#             credit_score = 550

#         surplus_proxy = _infer_surplus_from_cibil(credit_score, dpd_60_count, dpd_30_count, float(monthly_income))

#         return {
#             'Credit_Score': credit_score,
#             'max_delinquency_level': max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30),
#             'max_recent_level_of_deliq': max(dpd_60_count*60, dpd_30_count*30),
#             'num_times_30p_dpd': dpd_30_count,
#             'num_times_60p_dpd': dpd_60_count,
#             'num_times_delinquent': dpd_30_count + dpd_60_count + dpd_90_count,
#             'num_deliq_6mts': dpd_30_count + dpd_60_count + dpd_90_count,
#             'num_deliq_12mts': dpd_30_count + dpd_60_count + dpd_90_count,
#             'num_deliq_6_12mts': 0,
#             'max_deliq_6mts': dpd_90_count if dpd_90_count > 0 else dpd_60_count,
#             'max_deliq_12mts': dpd_90_count if dpd_90_count > 0 else dpd_60_count,
#             'num_std': active_count, 'num_std_6mts': active_count, 'num_std_12mts': active_count,
#             'num_sub': sub_std, 'num_sub_6mts': sub_std, 'num_sub_12mts': sub_std,
#             'num_dbt': dpd_90_count, 'num_dbt_6mts': 0, 'num_dbt_12mts': 0,
#             'num_lss': written_off_count, 'num_lss_6mts': 0, 'num_lss_12mts': 0,
#             'enq_L3m': min(len(enquiry_dates), enq_12m), 'enq_L6m': enq_12m, 'enq_L12m': enq_12m,
#             'tot_enq': enq_12m,
#             'CC_enq': 0, 'CC_enq_L6m': 0, 'CC_enq_L12m': 0,
#             'PL_enq': 0, 'PL_enq_L6m': 0, 'PL_enq_L12m': 0,
#             'time_since_recent_enq': 30,
#             'pct_of_active_TLs_ever': round(pct_active, 2),
#             'pct_opened_TLs_L6m_of_L12m': 0.3,
#             'pct_currentBal_all_TL': 0.3,
#             'CC_utilization': round(cc_util, 2) if cc_util > 0 else -99999,
#             'PL_utilization': 0.25,
#             'max_unsec_exposure_inPct': cc_util_pct if cc_util_pct > 0 else 0,
#             'pct_PL_enq_L6m_of_L12m': 0.0, 'pct_CC_enq_L6m_of_L12m': 0.0,
#             'pct_PL_enq_L6m_of_ever': 0.0, 'pct_CC_enq_L6m_of_ever': 0.0,
#             'AGE': age_extracted,
#             'NETMONTHLYINCOME': monthly_income,
#             'Time_With_Curr_Empr': employment_tenure_months,
#             'GENDER': gender, 'MARITALSTATUS': marital_status, 'EDUCATION': education,
#             'CC_Flag': 1 if re.search(r'credit card', full_text, re.IGNORECASE) else 0,
#             'PL_Flag': 1 if re.search(r'personal loan', full_text, re.IGNORECASE) else 0,
#             'HL_Flag': 1 if re.search(r'home loan', full_text, re.IGNORECASE) else 0,
#             'GL_Flag': 1 if re.search(r'gold loan', full_text, re.IGNORECASE) else 0,
#             'last_prod_enq2': last_prod, 'first_prod_enq2': first_prod,
#             'time_since_recent_payment': 70,
#             'time_since_first_deliquency': -99999 if dpd_30_count == 0 else 180,
#             'time_since_recent_deliquency': -99999 if dpd_30_count == 0 else 90,
#             '_surplus_proxy': int(surplus_proxy),
#             'written_off_count': written_off_count,
#             'settled_count': settled_count,
#             'high_util_flag': high_util,
#             'dpd_90_count_6m': dpd_90_count,
#             'recent_deliq_flag': 1 if (dpd_90_count > 0 or dpd_60_count > 0) else 0,
#             'account_quality_score': max(0, 100 - written_off_count*20 - settled_count*10 - dpd_90_count*15 - dpd_30_count*5),
#             'raw_text': full_text,
#             'success': True,
#             'extraction_method': 'OCR+ExternalCIBIL',
#         }
#     except Exception as e:
#         return {'error': str(e), 'message': f'Error extracting CIBIL data: {str(e)}', 'success': False}

# # =============================================================================
# # FAIRNESS LOG HELPER
# # =============================================================================
# def log_decision_for_fairness(customer_data: dict, decision: str, risk_score: int, pd_pct: float,
#                                application_id: str = None, source: str = 'stage1'):
#     """
#     Append a minimal record to the in-session fairness log.
#     source = 'stage1' | 'stage2' | 'batch'
#     When Stage 2 completes, it REPLACES the Stage 1 record for the same application_id,
#     so the fairness dashboard always shows the FINAL binding decision.
#     """
#     record = {
#         'ts':              datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         'application_id':  application_id or customer_data.get('application_id', ''),
#         'source':          source,
#         'decision':        decision,
#         'risk_score':      risk_score,
#         'pd_pct':          pd_pct,
#         'gender':          customer_data.get('gender', 'Unknown'),
#         'city_tier':       customer_data.get('city_tier', 'Unknown'),
#         'employment_type': customer_data.get('employment_type', 'Unknown'),
#         'bureau_score':    customer_data.get('bureau_score', 0),
#         'age_band':        (
#             '24-30' if customer_data.get('age', 0) < 31 else
#             '31-40' if customer_data.get('age', 0) < 41 else
#             '41-50' if customer_data.get('age', 0) < 51 else '51+'
#         ),
#     }
#     st.session_state.fairness_log.append(record)

# # =============================================================================
# # HYBRID DECISION ENGINE
# # =============================================================================
# def make_hybrid_decision_enhanced(customer_dict):
#     fill_missing_ml_fields(customer_dict)
#     policy_checks = {}
#     age = customer_dict.get('age', 0)
#     employment_type = customer_dict.get('employment_type', 'Salaried')
#     kyc_verified = customer_dict.get('kyc_verified', True)
#     bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
#     fraud_flag = customer_dict.get('fraud_flag', False)

#     if age < 24 or age > 70:
#         policy_checks['age'] = f"❌ Age {age} (Required: 24-70)"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Age outside allowed range", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     policy_checks['age'] = f"✅ Age {age} (Valid)"

#     if not kyc_verified:
#         policy_checks['kyc'] = "❌ KYC Not Verified"
#         return {'decision': "REJECT", 'reason': "Policy Gate: KYC verification required", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     policy_checks['kyc'] = "✅ KYC Verified"

#     if not customer_dict.get('rbi_consent', False):
#         policy_checks['rbi_consent'] = "❌ RBI Consent not obtained"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Customer consent not obtained", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     policy_checks['rbi_consent'] = "✅ Consent Obtained"

#     if bankruptcy_flag:
#         policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Active bankruptcy", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     policy_checks['bankruptcy'] = "✅ No Bankruptcy"

#     if fraud_flag:
#         policy_checks['fraud'] = "❌ Fraud Flag"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Fraud detected", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     policy_checks['fraud'] = "✅ No Fraud History"

#     dependents = customer_dict.get('dependents', 0)
#     dependents_flag_review = dependents > 5
#     policy_checks['dependents'] = (f"⚠️ Dependents {dependents} (>5: Review Required)"
#                                    if dependents_flag_review else f"✅ Dependents {dependents} (Acceptable)")

#     monthly_income = customer_dict.get('avg_salary_6m', 0)
#     employment_tenure = customer_dict.get('employment_tenure_months', 0)
#     business_vintage = customer_dict.get('business_vintage_years', 0)

#     if monthly_income < 15000:
#         policy_checks['income'] = f"❌ Income ₹{monthly_income:,.0f} (Min: ₹15,000)"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Income below minimum", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     policy_checks['income'] = f"✅ Income ₹{monthly_income:,.0f}"

#     if employment_type == 'Salaried' and employment_tenure < 6:
#         policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient tenure", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
#         policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient business vintage", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     policy_checks['tenure'] = (f"✅ Tenure {employment_tenure} months" if employment_type == 'Salaried'
#                                 else f"✅ Business Vintage {business_vintage} years")

#     bureau_score = customer_dict.get('bureau_score', 0)
#     dpd_90 = customer_dict.get('dpd_90_count_6m', 0)
#     credit_utilization = customer_dict.get('credit_utilization_pct', 0)
#     recent_inquiries = customer_dict.get('recent_inquiries_3m', 0)

#         # --- DPD 90+ rule ---
#     dpd_90_flag_review = False
#     if dpd_90 > 5:
#         policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD (exceeds limit of 5)"
#         return {
#             'decision': "REJECT",
#             'reason': "Policy Gate: Severe delinquency > 5 instances of 90+ DPD",
#             'confidence': 0,
#             'class_probs': {'REJECT': 100},
#             'policy_checks': policy_checks,
#             'risk_score': 0,
#             'pd_percentage': 100.0,
#             'affordability_data': {}
#         }
#     elif dpd_90 > 1:
#         policy_checks['dpd'] = f"⚠️ {dpd_90} instances of 90+ DPD (2–5) → Review required"
#         dpd_90_flag_review = True
#     else:
#         policy_checks['dpd'] = f"✅ {dpd_90} instances of 90+ DPD (acceptable)"


#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"

#     if dpd_90 > 0:
#         policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Severe delinquency", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     policy_checks['dpd'] = "✅ No 90+ DPD"
#     policy_checks['utilization'] = (f"⚠️ High utilization {credit_utilization}%" if credit_utilization > 80
#                                     else f"✅ Utilization {credit_utilization}%")
#     policy_checks['inquiries'] = (f"⚠️ {recent_inquiries} recent inquiries" if recent_inquiries > 5
#                                   else f"✅ {recent_inquiries} inquiries")

#     active_loans = customer_dict.get('active_loans_count', 0)
#     active_loans_flag = active_loans >= 5
#     policy_checks['active_loans'] = (f"⚠️ High active loans ({int(active_loans)}) — Review"
#                                      if active_loans_flag else f"✅ Active loans: {int(active_loans)}")

#     salary_stability = customer_dict.get('salary_stability_flag', 'STABLE')
#     salary_flag = salary_stability == 'UNSTABLE'
#     policy_checks['salary'] = (
#         "⚠️ Unstable salary — Review required" if salary_stability == 'UNSTABLE' else
#         "⚠️ Moderate salary stability" if salary_stability == 'MODERATE' else "✅ Stable salary"
#     )

#     input_df = pd.DataFrame([customer_dict])
#     for col in TOP_FEATURES:
#         if col not in input_df.columns:
#             input_df[col] = "Unknown" if col in LE_MAP else 0
#     for col, le in LE_MAP.items():
#         if col in input_df.columns:
#             val = str(input_df[col].values[0])
#             try: input_df[col] = le.transform([val])[0]
#             except ValueError: input_df[col] = 0
#     final_input = input_df[TOP_FEATURES]
#     pred_idx = MODEL.predict(final_input)[0]
#     ml_decision = TARGET_LE.inverse_transform([pred_idx])[0]
#     try:
#         pred_proba = MODEL.predict_proba(final_input)[0]
#         confidence = max(pred_proba) * 100
#         class_probs = {cls: prob * 100 for cls, prob in zip(TARGET_LE.classes_, pred_proba)}
#     except Exception:
#         confidence = 75.0
#         class_probs = {ml_decision: 100.0}

#     loan_amount   = customer_dict.get('loan_amount', 0)
#     loan_tenure   = customer_dict.get('loan_tenure_months', 12)
#     interest_rate = customer_dict.get('interest_rate', 10.5)
#     existing_emi  = customer_dict.get('existing_emi', 0)
#     affordability_data = calculate_affordability(monthly_income, loan_amount, interest_rate, loan_tenure, existing_emi)
#     foir = affordability_data['foir_percentage']

#     if foir > 50:
#         ml_decision = "REJECT"
#         policy_checks['foir'] = f"❌ FOIR {foir:.1f}% exceeds maximum allowed (50%)"

#     if dependents_flag_review and ml_decision == "APPROVE": ml_decision = "REVIEW"
#     if active_loans_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"
#     if salary_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"

#     risk_score = calculate_final_risk_score(
#         bureau_score=bureau_score, ml_confidence=confidence, foir=foir,
#         dpd_90=dpd_90, dpd_30=customer_dict.get('dpd_30_count_6m', 0),
#         net_surplus=customer_dict.get('net_cash_surplus_6m', 0),
#         bounces=customer_dict.get('inward_bounce_count_3m', 0),
#         missing_months=customer_dict.get('salary_missing_months', 0),
#         active_loans=active_loans
#     )
#     pd_percentage = calculate_final_pd(
#         bureau_score=bureau_score, foir=foir, confidence=confidence,
#         dpd_90_count=dpd_90, dpd_30_count=customer_dict.get('dpd_30_count_6m', 0),
#         employment_type=employment_type, employment_tenure=employment_tenure,
#         business_vintage=business_vintage, recent_inquiries=recent_inquiries,
#         ml_decision=ml_decision
#     )
#     return {
#         'decision': ml_decision, 'ml_raw_decision': ml_decision,
#         'reason': "Decision based on comprehensive assessment",
#         'confidence': confidence, 'class_probs': class_probs,
#         'policy_checks': policy_checks, 'risk_score': risk_score,
#         'pd_percentage': round(pd_percentage, 2), 'affordability_data': affordability_data
#     }

# # =============================================================================
# # BATCH PREDICTION ENGINE
# # =============================================================================
# def process_batch_predictions(df):
#     results = []
#     required_fields = {
#         'age': 35, 'employment_type': 'Salaried', 'kyc_verified': True,
#         'bankruptcy_flag': False, 'fraud_flag': False, 'rbi_consent': True,
#         'employment_tenure_months': 24, 'business_vintage_years': 0,
#         'bureau_score': 700, 'dpd_90_count_6m': 0, 'dpd_30_count_6m': 0,
#         'credit_utilization_pct': 30, 'recent_inquiries_3m': 0,
#         'active_loans_count': 0, 'existing_emi': 0, 'avg_salary_6m': 50000,
#         'AMT_INCOME_TOTAL': 600000, 'net_cash_surplus_6m': 20000,
#         'salary_stability_flag': 'STABLE', 'loan_amount': 180000,
#         'loan_tenure_months': 24, 'interest_rate': 10.5, 'AMT_ANNUITY': 8500,
#         'dependents': 0, 'payment_discipline_flag': 'GOOD',
#         'liquidity_flag': 'LOW', 'cashflow_health': 'MODERATE',
#         'bureau_risk_flag': 'LOW', 'inward_bounce_count_3m': 0,
#         'salary_missing_months': 0, 'gender': 'Unknown', 'city_tier': 'Unknown',
#     }
#     for idx, row in df.iterrows():
#         customer_dict = row.to_dict()
#         for k, v in customer_dict.items():
#             if isinstance(v, str):
#                 if v.lower() in ['yes', 'true', '1']: customer_dict[k] = True
#                 elif v.lower() in ['no', 'false', '0']: customer_dict[k] = False
#         for field, default in required_fields.items():
#             if field not in customer_dict or pd.isna(customer_dict.get(field, None)):
#                 customer_dict[field] = default
#         try:
#             decision_data = make_hybrid_decision_enhanced(customer_dict)
#             reasons = generate_reason_codes(
#                 decision=decision_data.get('decision', 'ERROR'),
#                 customer_data=customer_dict,
#                 affordability_data=decision_data.get('affordability_data', {}),
#                 policy_checks=decision_data.get('policy_checks', {})
#             )
#             affordability = decision_data.get('affordability_data', {})
#             result = {
#                 'application_id': f"BATCH_{idx+1:04d}",
#                 'decision': decision_data.get('decision', 'ERROR'),
#                 'risk_score': decision_data.get('risk_score', 0),
#                 'pd_percentage': decision_data.get('pd_percentage', 0),
#                 'confidence': round(decision_data.get('confidence', 0), 2),
#                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 'reason_1': reasons[0] if len(reasons) > 0 else '',
#                 'reason_2': reasons[1] if len(reasons) > 1 else '',
#                 'reason_3': reasons[2] if len(reasons) > 2 else '',
#                 'age': customer_dict.get('age', ''),
#                 'gender': customer_dict.get('gender', ''),
#                 'city_tier': customer_dict.get('city_tier', ''),
#                 'employment_type': customer_dict.get('employment_type', ''),
#                 'bureau_score': customer_dict.get('bureau_score', ''),
#                 'monthly_income': customer_dict.get('avg_salary_6m', ''),
#                 'loan_amount': customer_dict.get('loan_amount', ''),
#                 'loan_tenure_months': customer_dict.get('loan_tenure_months', ''),
#                 'interest_rate': customer_dict.get('interest_rate', ''),
#                 'new_emi': affordability.get('new_emi', 0),
#                 'existing_emi': affordability.get('existing_emi', 0),
#                 'total_emi': affordability.get('total_emi', 0),
#                 'foir_percentage': round(affordability.get('foir_percentage', 0), 2),
#                 'net_disposable': affordability.get('net_disposable', 0),
#                 'affordability_status': affordability.get('status', 'N/A'),
#                 'dpd_90_count': customer_dict.get('dpd_90_count_6m', 0),
#                 'dpd_30_count': customer_dict.get('dpd_30_count_6m', 0),
#                 'credit_utilization': customer_dict.get('credit_utilization_pct', 0),
#                 'prob_approve': round(decision_data.get('class_probs', {}).get('APPROVE', 0), 2),
#                 'prob_review': round(decision_data.get('class_probs', {}).get('REVIEW', 0), 2),
#                 'prob_reject': round(decision_data.get('class_probs', {}).get('REJECT', 0), 2),
#             }
#         except Exception as e:
#             result = {
#                 'application_id': f"BATCH_{idx+1:04d}", 'decision': 'ERROR',
#                 'risk_score': 0, 'pd_percentage': 0, 'confidence': 0,
#                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 'reason_1': '', 'reason_2': '', 'reason_3': '',
#                 'age': customer_dict.get('age', ''), 'gender': customer_dict.get('gender', ''),
#                 'city_tier': customer_dict.get('city_tier', ''),
#                 'employment_type': customer_dict.get('employment_type', ''),
#                 'bureau_score': customer_dict.get('bureau_score', ''),
#                 'monthly_income': customer_dict.get('avg_salary_6m', ''),
#                 'loan_amount': customer_dict.get('loan_amount', ''),
#                 'error_message': str(e)
#             }
#         else:
#             # Log to fairness monitor (success path only)
#             log_decision_for_fairness(
#                 customer_dict,
#                 result['decision'],
#                 result['risk_score'],
#                 result['pd_percentage']
#             )
#         results.append(result)
#     return pd.DataFrame(results)

# # =============================================================================
# # MODERN UI COMPONENTS
# # =============================================================================
# def render_decision_header(decision_data, customer_data):
#     decision = decision_data.get('decision', 'ERROR')
#     risk_score = decision_data.get('risk_score', 0)
#     pd_score = decision_data.get('pd_percentage', 0)
#     approved_amount = customer_data.get('loan_amount', 0)
#     tenure = customer_data.get('loan_tenure_months', 24)
#     app_id = customer_data.get('application_id', 'N/A')
#     timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#     if decision == "APPROVE":
#         card_class = "decision-card decision-card-approved"; icon = "✓"; subtitle = "Application Approved Successfully"
#     elif decision == "REJECT":
#         card_class = "decision-card decision-card-rejected"; icon = "✗"; subtitle = "Application Not Approved"
#     else:
#         card_class = "decision-card decision-card-review"; icon = "⚠"; subtitle = "Requires Manual Review"
#     st.markdown(f'<div class="{card_class}"><div class="decision-title">{icon} {decision}</div><div class="decision-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)
#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1: st.markdown(f'<div class="stat-card"><div class="stat-number">{risk_score}</div><div class="stat-label">Risk Score</div></div>', unsafe_allow_html=True)
#     with col2: st.markdown(f'<div class="stat-card"><div class="stat-number">{pd_score}%</div><div class="stat-label">PD Score</div></div>', unsafe_allow_html=True)
#     with col3: st.markdown(f'<div class="stat-card"><div class="stat-number">₹{approved_amount:,.0f}</div><div class="stat-label">Loan Amount</div></div>', unsafe_allow_html=True)
#     with col4: st.markdown(f'<div class="stat-card"><div class="stat-number">{tenure}</div><div class="stat-label">Tenure (Months)</div></div>', unsafe_allow_html=True)
#     with col5: st.markdown(f'<div class="stat-card"><div class="stat-number">{decision_data.get("confidence", 0):.0f}%</div><div class="stat-label">Confidence</div></div>', unsafe_allow_html=True)
#     st.markdown("<br>", unsafe_allow_html=True)
#     col1, col2 = st.columns(2)
#     with col1: st.markdown(f'<div class="info-box"><strong>📋 Application ID:</strong> {app_id}</div>', unsafe_allow_html=True)
#     with col2: st.markdown(f'<div class="info-box"><strong>🕐 Decision Timestamp:</strong> {timestamp}</div>', unsafe_allow_html=True)

# def render_info_card(title, icon, data_dict, status_dict=None):
#     st.markdown(f'<div class="info-card"><div class="info-card-title">{icon} {title}</div><div class="info-card-content">', unsafe_allow_html=True)
#     for label, value in data_dict.items():
#         status = ""
#         if status_dict and label in status_dict:
#             if status_dict[label] == "pass": status = '<span class="status-badge badge-pass">✓</span>'
#             elif status_dict[label] == "fail": status = '<span class="status-badge badge-fail">✗</span>'
#             elif status_dict[label] == "warning": status = '<span class="status-badge badge-warning">⚠</span>'
#         st.markdown(f'<div class="data-row"><span class="data-label">{label}</span><span class="data-value">{value} {status}</span></div>', unsafe_allow_html=True)
#     st.markdown('</div></div>', unsafe_allow_html=True)

# def render_reason_codes(reasons):
#     st.markdown('<div class="info-card"><div class="info-card-title">📝 Decision Reasons</div><div class="info-card-content">', unsafe_allow_html=True)
#     for i, reason in enumerate(reasons, 1):
#         st.markdown(f'<div class="reason-item"><span class="reason-icon">{i}.</span>{reason}</div>', unsafe_allow_html=True)
#     st.markdown('</div></div>', unsafe_allow_html=True)

# def create_modern_gauge(value, title, max_value=100):
#     color = "#f56565" if value <= 50 else "#ed8936" if value <= 75 else "#48bb78"
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number", value=value,
#         title={'text': title, 'font': {'size': 18, 'color': '#2d3748'}},
#         number={'suffix': "%", 'font': {'size': 36, 'color': '#2d3748'}},
#         gauge={
#             'axis': {'range': [0, max_value]},
#             'bar': {'color': color, 'thickness': 0.75},
#             'bgcolor': 'white', 'borderwidth': 0,
#             'steps': [{'range': [0, 50], 'color': '#fed7d7'},
#                       {'range': [50, 75], 'color': '#feebc8'},
#                       {'range': [75, 100], 'color': '#c6f6d5'}]
#         }
#     ))
#     fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white')
#     return fig

# def create_modern_bar_chart(class_probs):
#     df = pd.DataFrame({'Decision': list(class_probs.keys()), 'Probability': list(class_probs.values())})
#     colors = {'REVIEW': '#ed8936', 'APPROVE': '#48bb78', 'REJECT': '#f56565'}
#     fig = px.bar(df, x='Decision', y='Probability', title='Decision Probabilities',
#                  color='Decision', color_discrete_map=colors, text='Probability')
#     fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
#     fig.update_layout(showlegend=False, yaxis_title='Probability (%)', xaxis_title='', height=300,
#                       margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white', plot_bgcolor='white',
#                       yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]})
#     return fig

# # =============================================================================
# # STAGE 2 BINARY RESOLVER
# # =============================================================================
# def resolve_stage2_to_binary(stage2_result: dict) -> dict:
#     result = stage2_result.copy()
#     tier  = result.get('stage2_tier', '')
#     raw   = result.get('final_decision', '')
#     score = result.get('combined_risk_score', 0) or 0
#     TIER_MAP = {'P1': 'APPROVE', 'P2': 'APPROVE', 'P3': 'REJECT', 'P4': 'REJECT'}
#     if raw == 'REJECT':
#         result['final_decision'] = 'REJECT'
#     elif raw == 'APPROVE':
#         result['final_decision'] = TIER_MAP.get(tier, 'APPROVE')
#     else:
#         if tier in TIER_MAP:
#             result['final_decision'] = TIER_MAP[tier]
#             result['reason'] = result.get('reason', '') + f" [REVIEW resolved to {TIER_MAP[tier]} via tier {tier}]"
#         else:
#             resolved = 'APPROVE' if score >= 600 else 'REJECT'
#             result['final_decision'] = resolved
#             result['reason'] = result.get('reason', '') + f" [REVIEW resolved to {resolved} via score {score}]"
#     if result['final_decision'] == 'APPROVE':
#         result.setdefault('interest_rate_range', {'P1': '9.5%–11%', 'P2': '11%–13%'}.get(tier, '11%–14%'))
#     else:
#         result['interest_rate_range'] = 'N/A — Rejected'
#     return result

# # =============================================================================
# # STAGE 2 RESULTS DISPLAY
# # =============================================================================
# def display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data):
#     st.markdown("---")
#     st.markdown('<p class="main-header">🎯 Stage 2 Final Results</p>', unsafe_allow_html=True)
#     final_decision    = stage2_result.get('final_decision', 'ERROR')
#     interest_range    = stage2_result.get('interest_rate_range', 'N/A')
#     stage2_tier       = stage2_result.get('stage2_tier', 'N/A')
#     stage2_confidence = stage2_result.get('stage2_confidence', 0)
#     combined_risk     = stage2_result.get('combined_risk_score', 0)

#     # ── Fairness log: use Stage 2 FINAL decision, remove the earlier Stage 1 entry ──
#     # Stage 1 logged a preliminary decision for this customer. Since Stage 2
#     # is the BINDING final decision, we replace that entry so the fairness
#     # dashboard always reflects the true outcome.
#     app_id = stage1_customer.get('application_id', None)
#     if app_id and 'fairness_log' in st.session_state:
#         st.session_state.fairness_log = [
#             r for r in st.session_state.fairness_log
#             if r.get('application_id') != app_id
#         ]
#     log_decision_for_fairness(
#         enhanced_customer_data,
#         final_decision,
#         combined_risk,
#         stage2_result.get('pd_percentage', stage1_data.get('pd_percentage', 0)),
#         application_id=app_id,
#         source='stage2'
#     )

#     if final_decision == "APPROVE":
#         st.markdown('<div class="decision-card decision-card-approved"><div class="decision-title">✓ APPROVE</div><div class="decision-subtitle">✅ Final Decision: Approved — Proceed to Disbursement</div></div>', unsafe_allow_html=True)
#     else:
#         st.markdown('<div class="decision-card decision-card-rejected"><div class="decision-title">✗ REJECT</div><div class="decision-subtitle">❌ Final Decision: Rejected — Application Declined</div></div>', unsafe_allow_html=True)

#     col1, col2, col3, col4 = st.columns(4)
#     with col1: st.metric("Risk Tier", stage2_tier)
#     with col2: st.metric("Interest Rate", interest_range)
#     with col3: st.metric("Combined Risk Score", combined_risk)
#     with col4: st.metric("Stage 2 Confidence", f"{stage2_confidence:.1f}%" if stage2_confidence else "N/A")

#     st.markdown("<br>", unsafe_allow_html=True)
#     tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔬 Analysis", "📋 Data", "📥 Download"])

#     with tab1:
#         s1_dec = st.session_state.get('stage1_decision', 'N/A')
#         s2_label = "✅ APPROVE" if final_decision == "APPROVE" else "❌ REJECT"
#         comparison_df = pd.DataFrame([
#             {'Stage': 'Stage 1 (Screening)', 'Decision': s1_dec, 'Risk Score': stage1_data.get('risk_score', 'N/A'), 'Tier': 'N/A', 'Note': 'APPROVE/REVIEW → proceed to Stage 2'},
#             {'Stage': 'Stage 2 — FINAL', 'Decision': s2_label, 'Risk Score': combined_risk, 'Tier': f"{stage2_tier} | {interest_range}", 'Note': 'Binding final decision'}
#         ])
#         st.dataframe(comparison_df, use_container_width=True, hide_index=True)
#         tier_info = {
#             'P1': {'name': 'Premium → APPROVED', 'color': '#10B981', 'desc': 'Excellent credit profile — lowest interest rate band'},
#             'P2': {'name': 'Standard → APPROVED', 'color': '#3B82F6', 'desc': 'Good credit profile — standard interest rate band'},
#             'P3': {'name': 'Subprime → REJECTED', 'color': '#F59E0B', 'desc': 'Fair credit with elevated risk — application declined'},
#             'P4': {'name': 'High Risk → REJECTED', 'color': '#EF4444', 'desc': 'High risk profile — application declined'},
#         }
#         if stage2_tier in tier_info:
#             td = tier_info[stage2_tier]
#             st.markdown(f'<div style="background:{td["color"]};color:white;padding:1rem;border-radius:0.5rem;"><h3 style="margin:0;color:white;">{stage2_tier}: {td["name"]}</h3><p style="margin:0.5rem 0 0 0;">{td["desc"]}</p></div>', unsafe_allow_html=True)
#         st.info(stage2_result.get('reason', 'N/A'))

#     with tab2:
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("**Tier Probabilities**")
#             if 'tier_probabilities' in stage2_result:
#                 for tier, prob in stage2_result['tier_probabilities'].items():
#                     st.metric(tier, f"{prob:.1f}%")
#         with col2:
#             st.markdown("**Stage Scores**")
#             st.metric("Stage 1 Risk Score", stage1_data.get('risk_score', 'N/A'))
#             st.metric("Stage 2 Risk Score", stage2_result.get('stage2_risk_score', 'N/A'))
#             st.metric("Combined Score", combined_risk)
#         with st.expander("Complete Stage 2 Result (JSON)"):
#             st.json(stage2_result)

#     with tab3:
#         col1, col2 = st.columns(2)
#         with col1:
#             with st.expander("Stage 1 Customer Data"): st.json(stage1_customer)
#         with col2:
#             with st.expander("Enhanced CIBIL Data"): st.json(enhanced_customer_data)

#     with tab4:
#         if PDF_AVAILABLE and generate_audit_pdf is not None:
#             try:
#                 _safe = lambda v, d='N/A': v if v is not None else d
#                 # Build full pd_calculation_factors from enhanced customer data
#                 _bs  = enhanced_customer_data.get('bureau_score', stage1_customer.get('bureau_score', 0))
#                 _foir = stage1_data.get('affordability_data', {}).get('foir_percentage', 0)
#                 _conf = stage1_data.get('confidence', 0)
#                 _dpd90 = enhanced_customer_data.get('dpd_90_count_6m', stage1_customer.get('dpd_90_count_6m', 0))
#                 _dpd30 = enhanced_customer_data.get('dpd_30_count_6m', stage1_customer.get('dpd_30_count_6m', 0))
#                 _emp_type = enhanced_customer_data.get('employment_type', stage1_customer.get('employment_type', 'Salaried'))
#                 _emp_ten  = enhanced_customer_data.get('employment_tenure_months', stage1_customer.get('employment_tenure_months', 24))
#                 _biz_vin  = enhanced_customer_data.get('business_vintage_years', stage1_customer.get('business_vintage_years', 0))
#                 _inq      = enhanced_customer_data.get('recent_inquiries_3m', stage1_customer.get('recent_inquiries_3m', 0))
#                 _base_pd   = bureau_score_to_pd(_bs)
#                 _foir_adj  = foir_to_pd_adjustment(_foir)
#                 _deliq_mul = delinquency_to_pd_multiplier(_dpd90, _dpd30)
#                 _emp_adj   = employment_stability_to_pd_adjustment(_emp_type, _emp_ten, _biz_vin)
#                 _inq_adj   = inquiry_pattern_to_pd_adjustment(_inq)
#                 _ml_adj    = ml_confidence_to_pd_adjustment(_conf, stage1_data.get('decision','REVIEW'))
#                 _final_pd  = stage1_data.get('pd_percentage', round(max(0.5, min(
#                     _base_pd * _deliq_mul + _foir_adj + _emp_adj + _inq_adj + _ml_adj, 25.0)), 2))

#                 report_data = {
#                     'application_id':  _safe(stage1_customer.get('application_id')),
#                     'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                     'model_version':   '8.7',
#                     'decision':        _safe(stage1_data.get('decision')),
#                     'stage2_final_decision':      _safe(final_decision),
#                     'stage2_tier':                _safe(stage2_tier),
#                     'stage2_interest_range':      _safe(interest_range),
#                     'stage2_combined_risk_score': _safe(combined_risk, 0),
#                     'stage2_confidence':          _safe(stage2_confidence, 0),
#                     'stage2_reason':              _safe(stage2_result.get('reason')),
#                     'stage2_tier_probabilities':  stage2_result.get('tier_probabilities') or {},
#                     'stage2_complete_analysis':   stage2_result,
#                     # Policy gate results
#                     'policy_checks': stage1_data.get('policy_checks', {}),
#                     # Full PD calculation breakdown
#                     'pd_calculation_factors': {
#                         'bureau_score':           _bs,
#                         'base_pd':                round(_base_pd, 2),
#                         'dpd_90':                 _dpd90,
#                         'dpd_30':                 _dpd30,
#                         'delinquency_multiplier': round(_deliq_mul, 2),
#                         'foir':                   round(_foir, 2),
#                         'foir_adjustment':        round(_foir_adj, 2),
#                         'employment_adjustment':  round(_emp_adj, 2),
#                         'inquiry_adjustment':     round(_inq_adj, 2),
#                         'ml_adjustment':          round(_ml_adj, 2),
#                         'final_pd':               _final_pd,
#                     },
#                     # Reason codes from Stage 1
#                     'reason_codes': stage1_customer.get('reason_codes', []),
#                     # Raw data refs
#                     'customer_data':          stage1_customer,
#                     'enhanced_customer_data': enhanced_customer_data,
#                 }
#                 pdf_buffer = generate_audit_pdf(report_data)
#                 st.download_button("📥 Download PDF Report", data=pdf_buffer,
#                                    file_name=f"stage2_report_{stage1_customer.get('application_id','X')}.pdf",
#                                    mime="application/pdf", use_container_width=True)
#             except Exception as e:
#                 st.error(f"PDF generation failed: {str(e)}")
#         else:
#             st.warning("PDF generation is not available.")

#     st.markdown("---")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         if st.button("🔄 New Assessment", key="new_assessment_stage2", use_container_width=True):
#             for k in ['stage1_complete','stage1_decision','stage1_data','current_customer_data']:
#                 st.session_state[k] = (False if k == 'stage1_complete' else None)
#             st.session_state.page_navigation = "👤 Assessment"
#             st.rerun()
#     with col2:
#         if st.button("← Back to Stage 1", key="back_to_stage1", use_container_width=True):
#             st.session_state.page_navigation = "👤 Assessment"
#             st.rerun()
#     with col3:
#         if st.button("🏠 Home", key="home_stage2", use_container_width=True):
#             st.session_state.page_navigation = "🏠 Home"
#             st.rerun()

# # =============================================================================
# # FAIRNESS MONITORING DASHBOARD
# # =============================================================================
# def render_fairness_dashboard():
#     st.markdown('<p class="main-header">⚖️ Fairness Monitoring</p>', unsafe_allow_html=True)
#     st.markdown("""
#         <div class="info-box">
#             <strong>RBI Fair Lending Compliance Dashboard</strong><br>
#             Tracks approval rates across demographic groups to detect potential disparate impact.
#             <strong>Fairness is measured on the FINAL binding decision</strong> — Stage 2 outcome
#             is used when available; Stage 1 (screening) entries are automatically replaced once
#             Stage 2 completes for the same application.
#             Data is session-based — decisions accumulate as applications are processed.
#         </div>
#     """, unsafe_allow_html=True)

#     log = st.session_state.get('fairness_log', [])

#     col1, col2 = st.columns([3, 1])
#     with col2:
#         if st.button("🗑️ Clear Log", use_container_width=True):
#             st.session_state.fairness_log = []
#             st.rerun()

#     if not log:
#         st.info("ℹ️ No decisions logged yet. Process some applications from the Assessment page to see fairness metrics here.")
#         st.markdown("### 📊 What will appear here:")
#         st.markdown("""
#         - **Approval rate by Gender** — tracks if male/female/other applicants are treated equitably
#         - **Approval rate by City Tier** — checks for geographic bias (Tier 1 vs Tier 3 vs Rural)
#         - **Approval rate by Age Band** — identifies potential age discrimination
#         - **Approval rate by Employment Type** — salaried vs self-employed equity check
#         - **Average Risk Score & PD by group** — confirms scoring is not systematically biased
#         """)
#         return

#     df = pd.DataFrame(log)
#     df['approved'] = (df['decision'] == 'APPROVE').astype(int)
#     n = len(df)

#     # Source breakdown
#     if 'source' in df.columns:
#         n_s2    = int((df['source'] == 'stage2').sum())
#         n_s1    = int((df['source'] == 'stage1').sum())
#         n_batch = int((df['source'] == 'batch').sum())
#         src_note = f"📌 {n_s2} Stage 2 (final) · {n_s1} Stage 1 (screening) · {n_batch} Batch"
#         st.caption(src_note)

#     st.markdown("---")
#     c1, c2, c3, c4 = st.columns(4)
#     with c1: st.metric("Total Decisions", n)
#     with c2: st.metric("Approvals", int(df['approved'].sum()), f"{df['approved'].mean()*100:.1f}%")
#     with c3: st.metric("Reviews", int((df['decision']=='REVIEW').sum()))
#     with c4: st.metric("Rejections", int((df['decision']=='REJECT').sum()))

#     st.markdown("---")
#     tab1, tab2, tab3, tab4 = st.tabs(["👥 Gender", "🏙️ City Tier", "📅 Age Band", "💼 Employment"])

#     COLOR_MAP = {'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}

#     def _approval_bar(group_col, title):
#         grp = df.groupby(group_col).agg(
#             Total=('decision', 'count'),
#             Approved=('approved', 'sum'),
#             Avg_Risk=('risk_score', 'mean'),
#             Avg_PD=('pd_pct', 'mean'),
#         ).reset_index()
#         grp['Approval Rate %'] = (grp['Approved'] / grp['Total'] * 100).round(1)
#         grp['Avg Risk Score'] = grp['Avg_Risk'].round(1)
#         grp['Avg PD %'] = grp['Avg_PD'].round(2)

#         col1, col2 = st.columns([2, 1])
#         with col1:
#             fig = px.bar(grp, x=group_col, y='Approval Rate %',
#                          title=title, text='Approval Rate %',
#                          color='Approval Rate %',
#                          color_continuous_scale=['#f56565', '#ed8936', '#48bb78'],
#                          range_color=[0, 100])
#             fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
#             fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10),
#                               coloraxis_showscale=False, paper_bgcolor='white', plot_bgcolor='white',
#                               yaxis={'range': [0, 110], 'gridcolor': '#e2e8f0'})
#             st.plotly_chart(fig, use_container_width=True)
#         with col2:
#             st.markdown("**Summary Table**")
#             display_df = grp[[group_col, 'Total', 'Approval Rate %', 'Avg Risk Score', 'Avg PD %']].copy()
#             # Flag groups with approval rate deviation > 15pp from overall
#             overall_rate = df['approved'].mean() * 100
#             def _flag(rate):
#                 diff = rate - overall_rate
#                 if abs(diff) > 15: return f"{'🔴' if diff < 0 else '🟢'} {rate:.1f}%"
#                 return f"✅ {rate:.1f}%"
#             display_df['Approval Rate %'] = display_df['Approval Rate %'].apply(_flag)
#             st.dataframe(display_df, use_container_width=True, hide_index=True)
#             overall_str = f"{overall_rate:.1f}%"
#             st.caption(f"Overall approval rate: **{overall_str}**. 🔴 = >15pp below average (potential bias). 🟢 = >15pp above average.")

#     with tab1:
#         if df['gender'].nunique() > 1:
#             _approval_bar('gender', 'Approval Rate by Gender')
#             # Decision mix donut per gender
#             fig2 = px.pie(df, names='decision', color='decision', color_discrete_map=COLOR_MAP,
#                           title='Decision Mix (all)', hole=0.5)
#             fig2.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
#             st.plotly_chart(fig2, use_container_width=True)
#         else:
#             st.info("Need 2+ gender values in decisions to show chart. Ensure Gender field is filled on the form.")

#     with tab2:
#         if df['city_tier'].nunique() > 1:
#             _approval_bar('city_tier', 'Approval Rate by City Tier')
#         else:
#             st.info("Need 2+ city tier values. Ensure City Tier field is filled on the form.")

#     with tab3:
#         if df['age_band'].nunique() > 1:
#             _approval_bar('age_band', 'Approval Rate by Age Band')
#         else:
#             st.info("Need decisions across multiple age bands (24-30, 31-40, 41-50, 51+).")

#     with tab4:
#         if df['employment_type'].nunique() > 1:
#             _approval_bar('employment_type', 'Approval Rate by Employment Type')
#         else:
#             st.info("Need 2+ employment types in decisions.")

#     st.markdown("---")
#     st.markdown("### 📥 Export Fairness Report")
#     col1, col2 = st.columns(2)
#     with col1:
#         csv_data = df.to_csv(index=False)
#         st.download_button("📥 Download Decision Log (CSV)", data=csv_data,
#                            file_name=f"fairness_log_{datetime.now().strftime('%Y%m%d')}.csv",
#                            mime="text/csv", use_container_width=True)
#     with col2:
#         st.caption("⚠️ **Note:** This log is session-based and resets when the app restarts. "
#                    "For persistent fairness monitoring, connect to a database or export regularly.")


# # =============================================================================
# # SIDEBAR
# # =============================================================================
# with st.sidebar:
#     st.markdown("# 🏦 Credit Risk Engine")
#     st.markdown("---")

#     navigation_options = ["🏠 Home", "👤 Assessment", "📊 Batch Process", "⚖️ Fairness", "📈 Model Info", "ℹ️ About"]

#     if (st.session_state.stage1_complete and st.session_state.stage1_decision in ['APPROVE', 'REVIEW']):
#         navigation_options.insert(2, "🔬 Stage 2 Analysis")
#         st.success(f"✅ Stage 1: {st.session_state.stage1_decision}")
#         st.info("🔬 Stage 2 Analysis unlocked!")
#     elif st.session_state.stage1_complete:
#         st.warning(f"⚠️ Stage 1: {st.session_state.stage1_decision}")
#         st.caption("Stage 2 only for APPROVE/REVIEW")

#     page = st.radio("**Navigation**", navigation_options,
#                     label_visibility="collapsed", key="page_navigation")

#     st.markdown("---")
#     stage2_indicator = '✅ Active' if STAGE2_AVAILABLE and is_stage2_available() else '❌ Inactive'
#     ocr_indicator = '✅ Ready' if OCR_AVAILABLE else '❌ Not Installed'
#     pdf_indicator = '✅ Ready' if PDF_AVAILABLE else '❌ Not Installed'
#     fairness_count = len(st.session_state.fairness_log)

#     st.markdown(f"""
#     <div class="info-card">
#         <div class="info-card-title">System Status</div>
#         <div class="info-card-content">
#             <div class="data-row"><span class="data-label">Model</span><span class="data-value">✅ Loaded</span></div>
#             <div class="data-row"><span class="data-label">Version</span><span class="data-value">8.6</span></div>
#             <div class="data-row"><span class="data-label">Stage 2</span><span class="data-value">{stage2_indicator}</span></div>
#             <div class="data-row"><span class="data-label">OCR</span><span class="data-value">{ocr_indicator}</span></div>
#             <div class="data-row"><span class="data-label">PDF Gen</span><span class="data-value">{pdf_indicator}</span></div>
#             <div class="data-row"><span class="data-label">Fairness Log</span><span class="data-value">{fairness_count} decisions</span></div>
#             <div class="data-row"><span class="data-label">Features</span><span class="data-value">{len(TOP_FEATURES)}</span></div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     with st.expander("🎯 **Top Features**"):
#         for i, feat in enumerate(TOP_FEATURES[:5], 1):
#             st.markdown(f"`{i}.` {feat}")

#     if st.session_state.stage1_complete:
#         st.markdown("---")
#         st.markdown("### 🚀 Quick Actions")
#         if st.button("🔄 New Assessment", key="new_assessment_sidebar", use_container_width=True):
#             for k in ['stage1_complete','stage1_decision','stage1_data','current_customer_data','extracted_cibil_data']:
#                 st.session_state[k] = False if k == 'stage1_complete' else None
#             st.rerun()

# # =============================================================================
# # PAGE ROUTING
# # =============================================================================
# if page == "🏠 Home":
#     st.markdown('<p class="main-header">Credit Risk Engine</p>', unsafe_allow_html=True)
#     st.markdown('<div class="info-box"><h3 style="margin-top:0;">🎯 AI-Powered Lending Decisions</h3><p style="margin-bottom:0;">Comprehensive credit risk evaluation combining hard policy rules, machine learning models, and affordability analysis.</p></div>', unsafe_allow_html=True)
#     st.markdown("<br>", unsafe_allow_html=True)
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.markdown('<div class="info-card"><div class="info-card-title">🛡️ Policy Gates</div><div class="info-card-content"><ul><li>Age & KYC verification</li><li>RBI consent check</li><li>Employment stability</li><li>Minimum income checks</li><li>Credit bureau thresholds</li><li>Bankruptcy & fraud detection</li></ul></div></div>', unsafe_allow_html=True)
#     with col2:
#         st.markdown('<div class="info-card"><div class="info-card-title">🤖 ML Assessment</div><div class="info-card-content"><ul><li>Random Forest classifier</li><li>60K+ training samples</li><li>Confidence scoring</li><li>Multi-class prediction</li><li>Feature importance</li></ul></div></div>', unsafe_allow_html=True)
#     with col3:
#         st.markdown('<div class="info-card"><div class="info-card-title">⚖️ Fairness Monitoring</div><div class="info-card-content"><ul><li>Approval rate by gender</li><li>Approval rate by city tier</li><li>Age band equity check</li><li>Employment type parity</li><li>RBI compliance ready</li></ul></div></div>', unsafe_allow_html=True)
#     st.markdown("<br>", unsafe_allow_html=True)
#     col1, col2, col3, col4 = st.columns(4)
#     with col1: st.metric("🎯 Accuracy", "85%", "+2%")
#     with col2: st.metric("⚡ Avg Response", "1.2s", "-0.3s")
#     with col3: st.metric("📊 Features", len(TOP_FEATURES))
#     with col4: st.metric("🔄 Version", "8.6", "Latest")
#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown("""
#         <div class="warning-box" style="background:#f0fff4;border:1px solid #9ae6b4;padding:1rem;border-radius:0.5rem;">
#             <strong>🆕 New in Version 8.6:</strong><br>
#             • <strong>Cleaned codebase</strong> — removed ~210 lines of duplicate function definitions<br>
#             • <strong>City Tier field</strong> — Tier 1/2/3/Rural captured on every application<br>
#             • <strong>Gender field</strong> — explicit gender capture for fairness logging<br>
#             • <strong>RBI Consent checkbox</strong> — required policy gate before assessment<br>
#             • <strong>Fairness Monitoring dashboard</strong> — approval rates by gender, city tier, age band, employment type<br>
#             • <strong>v8.5 features retained</strong> — dual-dataset OCR inference, categorical flag auto-fill
#         </div>
#     """, unsafe_allow_html=True)

# elif page == "👤 Assessment":
#     st.markdown('<p class="main-header">Credit Assessment</p>', unsafe_allow_html=True)

#     pdf_just_extracted = st.session_state.get('pdf_just_extracted', False)

#     with st.expander("📄 Upload CIBIL PDF to auto‑fill bureau fields",
#                      expanded=pdf_just_extracted or not st.session_state.get('pdf_bureau_score')):
#         if pdf_just_extracted:
#             ex = st.session_state.get('_last_extraction', {})
#             st.success("✅ CIBIL data extracted — form fields below have been updated automatically.")
#             c1, c2, c3, c4 = st.columns(4)
#             c1.metric("Credit Score", ex.get('Credit_Score', '—'))
#             c2.metric("Monthly Income", f"₹{ex.get('NETMONTHLYINCOME') or ex.get('avg_salary_6m', 0):,}")
#             c3.metric("DPD 60+ Count", ex.get('num_times_60p_dpd', 0))
#             c4.metric("CC Utilization", f"{max(0, float(ex.get('CC_utilization', 0) or 0))*100:.0f}%")
#             _inf = st.session_state.get('_last_inferred_flags', {})
#             if _inf:
#                 st.markdown("**📊 Inferred Categorical Flags:**")
#                 fc1, fc2, fc3, fc4, fc5 = st.columns(5)
#                 fc1.metric("Payment Discipline", _inf.get('payment_discipline_flag', '—'))
#                 fc2.metric("Cashflow Health", _inf.get('cashflow_health', '—'))
#                 fc3.metric("Liquidity", _inf.get('liquidity_flag', '—'))
#                 fc4.metric("Bureau Risk", _inf.get('bureau_risk_flag', '—'))
#                 fc5.metric("Salary Stability", _inf.get('salary_stability_flag', '—'))
#             if st.button("🔄 Upload a different PDF", key="reset_pdf"):
#                 st.session_state.pdf_just_extracted = False
#                 st.session_state.pop('_last_extraction', None)
#                 st.session_state.pop('_last_inferred_flags', None)
#                 st.rerun()
#         else:
#             st.markdown('<div class="info-box">💡 Complete the form below or upload a CIBIL PDF to auto‑fill bureau data.</div>', unsafe_allow_html=True)
#             uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="assessment_pdf")
#             if uploaded_pdf is not None:
#                 st.info(f"📄 File ready: **{uploaded_pdf.name}** ({uploaded_pdf.size/1024:.1f} KB)")
#                 if st.button("🔍 Extract & Auto-fill Form", key="extract_assessment", type="primary", use_container_width=True):
#                     with st.spinner("🔄 Running OCR on CIBIL PDF — this takes 10-30 seconds..."):
#                         extraction_result = extract_cibil_from_pdf(uploaded_pdf)
#                     if extraction_result.get('success', False):
#                         st.session_state.pdf_age               = int(extraction_result.get('AGE', 35))
#                         st.session_state.pdf_bureau_score      = int(extraction_result.get('Credit_Score', 720))
#                         st.session_state.pdf_dpd_90            = int(extraction_result.get('dpd_90_count_6m', 0))
#                         st.session_state.pdf_dpd_30            = int(extraction_result.get('num_times_30p_dpd', 0))
#                         st.session_state.pdf_credit_util       = int(max(0, float(extraction_result.get('CC_utilization', 0) or 0)) * 100)
#                         st.session_state.pdf_inquiries         = int(extraction_result.get('enq_L3m', 2))
#                         st.session_state.pdf_active_loans      = int(extraction_result.get('num_std', 1))
#                         st.session_state.pdf_existing_emi      = int(extraction_result.get('existing_emi', 15000))
#                         _income = int(extraction_result.get('NETMONTHLYINCOME') or extraction_result.get('avg_salary_6m') or 50000)
#                         st.session_state.pdf_monthly_income    = _income
#                         st.session_state.pdf_annual_income     = _income * 12
#                         _surplus = int(extraction_result.get('net_cash_surplus_6m') or extraction_result.get('_surplus_proxy') or 20000)
#                         st.session_state.pdf_net_surplus       = _surplus
#                         st.session_state.pdf_employment_tenure = int(extraction_result.get('Time_With_Curr_Empr', 24))
#                         _inferred = infer_categorical_flags(extraction_result)
#                         st.session_state.pdf_salary_stability   = _inferred['salary_stability_flag']
#                         st.session_state.pdf_payment_discipline = _inferred['payment_discipline_flag']
#                         st.session_state.pdf_cashflow_health    = _inferred['cashflow_health']
#                         st.session_state.pdf_liquidity_flag     = _inferred['liquidity_flag']
#                         st.session_state.pdf_bureau_risk_flag   = _inferred['bureau_risk_flag']
#                         st.session_state._last_inferred_flags   = _inferred
#                         st.session_state.pdf_just_extracted     = True
#                         st.session_state._last_extraction       = extraction_result
#                         st.rerun()
#                     else:
#                         st.error(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")

#     with st.form("assessment_form"):
#         # ── Identity & Eligibility ─────────────────────────────────────────
#         st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
#         col_name1, col_name2 = st.columns([2, 2])
#         with col_name1:
#             customer_name = st.text_input("Customer Name (Optional)", value="", placeholder="e.g. Ramesh Kumar")
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             age = st.number_input("Age", 24, 70, value=int(st.session_state.get('pdf_age', 35)))
#             employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business'],
#                 index=['Salaried','Self-Employed','Business'].index(st.session_state.get('pdf_employment_type','Salaried')))
#         with col2:
#             gender = st.selectbox("Gender", ['Male', 'Female', 'Non-binary / Other', 'Prefer not to say'], index=0)
#             dependents = st.number_input("Number of Dependents", 0, 20, value=int(st.session_state.get('pdf_dependents', 2)))
#         with col3:
#             # City Tier — NEW field for fairness monitoring
#             city_tier_label = st.selectbox("City Tier", list(CITY_TIERS.keys()), index=0)
#             city_tier = CITY_TIERS[city_tier_label]
#             kyc_verified = st.selectbox("KYC Verified", ['Yes', 'No'],
#                 index=0 if st.session_state.get('pdf_kyc', True) else 1) == 'Yes'
#         with col4:
#             bankruptcy_flag = st.selectbox("Bankruptcy Flag", ['No', 'Yes'],
#                 index=0 if not st.session_state.get('pdf_bankruptcy', False) else 1) == 'Yes'
#             fraud_flag = st.selectbox("Fraud Flag", ['No', 'Yes'],
#                 index=0 if not st.session_state.get('pdf_fraud', False) else 1) == 'Yes'

#         # RBI Consent — REQUIRED
#         st.markdown('<p class="section-header">📜 RBI Compliance</p>', unsafe_allow_html=True)
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             rbi_consent = st.checkbox(
#                 "✅ I confirm the customer has been informed of and consented to: (a) credit bureau enquiry, "
#                 "(b) data usage for credit assessment, (c) Key Fact Statement (KFS) terms, and "
#                 "(d) grievance redressal process. **(Required — RBI Digital Lending Guidelines)**",
#                 value=False
#             )
#         with col2:
#             st.markdown("""
#                 <div style="background:#fff3cd;border:1px solid #ffc107;padding:0.5rem;border-radius:0.4rem;font-size:0.82rem;">
#                     ⚠️ Without consent, the application cannot proceed per RBI DLG 2022.
#                 </div>
#             """, unsafe_allow_html=True)

#         # Employment tenure
#         st.markdown('<p class="section-header">💼 Employment</p>', unsafe_allow_html=True)
#         col1, col2 = st.columns(2)
#         with col1:
#             if employment_type == 'Salaried':
#                 employment_tenure = st.number_input("Employment Tenure (months)", 0, 600,
#                     value=int(st.session_state.get('pdf_employment_tenure', 24)))
#                 business_vintage = 0
#             else:
#                 business_vintage = st.number_input("Business Vintage (years)", 0, 50,
#                     value=int(st.session_state.get('pdf_business_vintage', 3)))
#                 employment_tenure = 0
#         with col2:
#             st.markdown("""
#                 <div class="info-box" style="margin-top:1rem;">
#                     <strong>Policy thresholds:</strong><br>
#                     Salaried: min 6 months<br>
#                     Self-Employed/Business: min 2 years
#                 </div>
#             """, unsafe_allow_html=True)

#         # Credit Bureau
#         st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             bureau_score = st.number_input("Bureau Score", 300, 900,
#                 value=int(st.session_state.get('pdf_bureau_score', 720)), step=10)
#             dpd_90_6m = st.number_input("DPD 90+ (Last 6M)", 0, 20,
#                 value=int(st.session_state.get('pdf_dpd_90', 0)))
#             dpd_30_6m = st.number_input("DPD 30+ (Last 6M)", 0, 20,
#                 value=int(st.session_state.get('pdf_dpd_30', 0)))
#         with col2:
#             credit_utilization = st.number_input("Credit Utilization (%)", 0, 100,
#                 value=int(st.session_state.get('pdf_credit_util', 30)))
#             recent_inquiries = st.number_input("Recent Inquiries (3M)", 0, 20,
#                 value=int(st.session_state.get('pdf_inquiries', 2)))
#         with col3:
#             active_loans = st.number_input("Active Loans", 0, 10,
#                 value=int(st.session_state.get('pdf_active_loans', 1)))
#             existing_emi = st.number_input("Existing Total EMI (₹)", 0, 200000,
#                 value=int(st.session_state.get('pdf_existing_emi', 15000)), step=1000)

#         # Income & Financial
#         st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             avg_salary = st.number_input("Monthly Income (₹)", 0, 1000000,
#                 value=int(st.session_state.get('pdf_monthly_income', 50000)), step=5000)
#             amt_income = st.number_input("Annual Income (₹)", 0, 10000000,
#                 value=int(st.session_state.get('pdf_annual_income', 600000)), step=10000)
#         with col2:
#             net_surplus = st.number_input("Net Cash Surplus (₹)", -100000, 500000,
#                 value=int(st.session_state.get('pdf_net_surplus', 20000)), step=5000)
#             _ss_opts = ['STABLE', 'MODERATE', 'UNSTABLE']
#             salary_stability = st.selectbox("Salary Stability", _ss_opts,
#                 index=_ss_opts.index(st.session_state.get('pdf_salary_stability', 'STABLE')))
#         with col3:
#             loan_amount = st.number_input("Loan Amount (₹)", 0, 5000000,
#                 value=int(st.session_state.get('pdf_loan_amount', 180000)), step=10000)
#             loan_tenure = st.number_input("Tenure (months)", 3, 360,
#                 value=int(st.session_state.get('pdf_loan_tenure', 24)))
#         with col4:
#             interest_rate = st.number_input("Interest Rate (%)", 8.0, 20.0,
#                 value=float(st.session_state.get('pdf_interest_rate', 10.5)), step=0.5)
#             amt_annuity = st.number_input("Requested EMI (₹)", 0, 200000,
#                 value=int(st.session_state.get('pdf_amt_annuity', 8500)), step=500)

#         # Additional Credit Behaviour
#         st.markdown('<p class="section-header">📋 Additional Credit Behaviour</p>', unsafe_allow_html=True)
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             _pd_opts = ['GOOD', 'MODERATE', 'POOR']
#             payment_discipline = st.selectbox("Payment Discipline", _pd_opts,
#                 index=_pd_opts.index(st.session_state.get('pdf_payment_discipline', 'GOOD')))
#             _lq_opts = ['LOW', 'ADEQUATE', 'MODERATE']
#             liquidity_flag = st.selectbox("Liquidity", _lq_opts,
#                 index=_lq_opts.index(st.session_state.get('pdf_liquidity_flag', 'LOW')))
#         with col2:
#             _cf_opts = ['MODERATE', 'HEALTHY', 'STRESSED', 'STABLE']
#             cashflow_health = st.selectbox("Cashflow Health", _cf_opts,
#                 index=_cf_opts.index(st.session_state.get('pdf_cashflow_health', 'MODERATE')))
#             _br_opts = ['LOW', 'MEDIUM', 'HIGH']
#             bureau_risk_flag = st.selectbox("Bureau Risk", _br_opts,
#                 index=_br_opts.index(st.session_state.get('pdf_bureau_risk_flag', 'LOW')))
#         with col3:
#             inward_bounce_count = st.number_input("Inward Bounce Count (3M)", 0, 10, 0)
#             salary_missing_months = st.number_input("Missing Salary Months (6M)", 0, 6, 0)

#         st.markdown("<br>", unsafe_allow_html=True)
#         submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)

#     if submitted:
#         timestamp = datetime.now()
#         app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
#         customer_data = {
#             'name': customer_name.strip() if customer_name.strip() else 'N/A',
#             'age': age, 'employment_type': employment_type,
#             'gender': gender, 'city_tier': city_tier,
#             'dependents': dependents, 'kyc_verified': kyc_verified,
#             'rbi_consent': rbi_consent,
#             'bankruptcy_flag': bankruptcy_flag, 'fraud_flag': fraud_flag,
#             'employment_tenure_months': employment_tenure,
#             'business_vintage_years': business_vintage,
#             'bureau_score': bureau_score,
#             'dpd_90_count_6m': dpd_90_6m, 'dpd_30_count_6m': dpd_30_6m,
#             'credit_utilization_pct': credit_utilization, 'max_utilization': credit_utilization,
#             'recent_inquiries_3m': recent_inquiries, 'active_loans_count': active_loans,
#             'avg_salary_6m': avg_salary, 'AMT_INCOME_TOTAL': amt_income,
#             'net_cash_surplus_6m': net_surplus, 'salary_stability_flag': salary_stability,
#             'loan_amount': loan_amount, 'loan_tenure_months': loan_tenure,
#             'interest_rate': interest_rate, 'existing_emi': existing_emi,
#             'AMT_ANNUITY': amt_annuity, 'application_id': app_id,
#             'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
#             'payment_discipline_flag': payment_discipline,
#             'liquidity_flag': liquidity_flag, 'cashflow_health': cashflow_health,
#             'bureau_risk_flag': bureau_risk_flag,
#             'inward_bounce_count_3m': inward_bounce_count,
#             'salary_missing_months': salary_missing_months,
#         }

#         with st.spinner("🔄 Processing Stage 1 assessment..."):
#             decision_data = make_hybrid_decision_enhanced(customer_data)

#         reasons = generate_reason_codes(
#             decision=decision_data.get('decision', 'ERROR'),
#             customer_data=customer_data,
#             affordability_data=decision_data.get('affordability_data', {}),
#             policy_checks=decision_data.get('policy_checks', {})
#         )
#         customer_data['reason_codes'] = reasons

#         # Log to fairness monitor (Stage 1 — may be replaced by Stage 2 final decision)
#         log_decision_for_fairness(customer_data, decision_data.get('decision','ERROR'),
#                                   decision_data.get('risk_score', 0), decision_data.get('pd_percentage', 0),
#                                   application_id=customer_data.get('application_id'),
#                                   source='stage1')

#         st.session_state.stage1_complete = True
#         st.session_state.stage1_decision = decision_data.get('decision', 'ERROR')
#         st.session_state.stage1_data = decision_data
#         st.session_state.current_customer_data = customer_data

#         for key in list(st.session_state.keys()):
#             if key.startswith('pdf_') or key in ('_last_extraction', '_last_inferred_flags'):
#                 del st.session_state[key]

#         tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])

#         with tab1:
#             st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
#             col1, col2 = st.columns(2)
#             with col1:
#                 render_info_card("👤 Identity", "👤",
#                                  {"Age": age, "Gender": gender, "City Tier": city_tier,
#                                   "Employment": employment_type, "Dependents": dependents,
#                                   "KYC Status": "Verified" if kyc_verified else "Not Verified",
#                                   "RBI Consent": "✅ Obtained" if rbi_consent else "❌ Not obtained"})
#                 render_info_card("💰 Financial", "💰",
#                                  {"Monthly Income": f"₹{avg_salary:,}", "Annual Income": f"₹{amt_income:,}",
#                                   "Net Surplus": f"₹{net_surplus:,}", "Stability": salary_stability})
#             with col2:
#                 render_info_card("🏦 Credit Bureau", "🏦",
#                                  {"Bureau Score": bureau_score, "DPD 90+": dpd_90_6m, "DPD 30+": dpd_30_6m,
#                                   "Utilization": f"{credit_utilization}%", "Recent Inquiries": recent_inquiries,
#                                   "Existing EMI": f"₹{existing_emi:,}"})
#                 render_info_card("📋 Loan Request", "📋",
#                                  {"Amount": f"₹{loan_amount:,}", "Tenure": f"{loan_tenure} months",
#                                   "Interest Rate": f"{interest_rate}%", "Requested EMI": f"₹{amt_annuity:,}"})

#         with tab2:
#             st.markdown('<p class="section-header">Decision Summary</p>', unsafe_allow_html=True)
#             render_decision_header(decision_data, customer_data)
#             st.markdown("<br>", unsafe_allow_html=True)

#             final_decision = decision_data.get('decision', 'ERROR')

#             if final_decision in ['APPROVE', 'REVIEW']:
#                 st.markdown("---")
#                 st.markdown('<div class="info-box" style="background:linear-gradient(135deg,#10B981,#059669);color:white;text-align:center;"><h3 style="margin:0;color:white;">✅ Eligible for Stage 2 Deep Dive</h3></div>', unsafe_allow_html=True)
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     if st.button("📝 Manual Entry", key="stage2_manual_btn", use_container_width=True, type="primary"):
#                         st.session_state.stage2_selected_tab = "Manual Entry"
#                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
#                         st.rerun()
#                 with col2:
#                     if st.button("📄 PDF Upload", key="stage2_pdf_btn", use_container_width=True, type="primary"):
#                         st.session_state.stage2_selected_tab = "PDF Upload"
#                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
#                         st.rerun()
#                 with col3:
#                     if st.button("📊 Batch Analysis", key="stage2_batch_btn", use_container_width=True, type="primary"):
#                         st.session_state.stage2_selected_tab = "Batch Analysis"
#                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
#                         st.rerun()
#             elif final_decision == 'REJECT':
#                 st.markdown("---")
#                 st.markdown('<div style="background:linear-gradient(135deg,#EF4444,#DC2626);color:white;padding:1rem;border-radius:0.5rem;text-align:center;"><h3 style="margin:0;color:white;">❌ Stage 2 Not Available</h3><p style="margin:0.5rem 0 0 0;">Application rejected. Stage 2 only for APPROVE/REVIEW.</p></div>', unsafe_allow_html=True)

#             st.markdown("<br>", unsafe_allow_html=True)
#             affordability = decision_data.get('affordability_data', {})
#             foir = affordability.get('foir_percentage', 0)
#             total_emi = affordability.get('total_emi', 0)
#             net_disp = affordability.get('net_disposable', 0)

#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 render_info_card("Identity & Eligibility", "👤",
#                     {f"Age: {age}": "", f"Employment: {employment_type}": "",
#                      f"City Tier: {city_tier}": "", f"Dependents: {dependents}": "",
#                      f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""},
#                     {f"Age: {age}": "pass" if 24 <= age <= 70 else "fail",
#                      f"Employment: {employment_type}": "pass",
#                      f"City Tier: {city_tier}": "pass",
#                      f"Dependents: {dependents}": "pass" if dependents <= 5 else "warning",
#                      f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_verified else "fail"})
#             with col2:
#                 render_info_card("Credit Bureau", "🏦",
#                     {f"Bureau Score: {bureau_score}": "", f"DPD 90+: {dpd_90_6m}": "",
#                      f"Utilization: {credit_utilization}%": ""},
#                     {f"Bureau Score: {bureau_score}": "pass" if bureau_score >= 550 else "fail",
#                      f"DPD 90+: {dpd_90_6m}": "pass" if dpd_90_6m == 0 else "fail",
#                      f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"})
#             with col3:
#                 render_info_card("Affordability", "💰",
#                     {f"Monthly Income: ₹{avg_salary:,}": "", f"FOIR: {foir:.1f}%": "",
#                      f"Total EMI: ₹{total_emi:,}": "", f"Net Disposable: ₹{net_disp:,}": ""},
#                     {f"Monthly Income: ₹{avg_salary:,}": "pass",
#                      f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
#                      f"Total EMI: ₹{total_emi:,}": "pass",
#                      f"Net Disposable: ₹{net_disp:,}": "pass" if net_disp >= 10000 else "warning"})

#             st.markdown("<br>", unsafe_allow_html=True)
#             render_reason_codes(reasons)
#             st.markdown("<br>", unsafe_allow_html=True)
#             col1, col2 = st.columns([1, 1])
#             with col1:
#                 if PDF_AVAILABLE and generate_decision_pdf is not None:
#                     try:
#                         pdf_buffer = generate_decision_pdf(
#                             decision_data=decision_data, customer_data=customer_data,
#                             affordability_data=decision_data.get('affordability_data', {}), reasons=reasons)
#                         st.download_button("📥 Decision Report (PDF)", data=pdf_buffer,
#                                            file_name=f"credit_decision_{app_id}.pdf", mime="application/pdf",
#                                            use_container_width=True)
#                     except Exception as e:
#                         st.error(f"Error generating PDF: {str(e)}")
#                 else:
#                     st.warning("PDF generation not available.")
#             with col2:
#                 if st.button("🔄 Re-Evaluate", key="reevaluate_btn", use_container_width=True):
#                     st.rerun()

#         with tab3:
#             st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.plotly_chart(create_modern_gauge(decision_data.get('confidence', 0), "Model Confidence"), use_container_width=True)
#             with col2:
#                 st.plotly_chart(create_modern_bar_chart(decision_data.get('class_probs', {"APPROVE": 0, "REVIEW": 0, "REJECT": 0})), use_container_width=True)
#             st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
#             policy_df = pd.DataFrame([{'Check': k, 'Result': v} for k, v in decision_data.get('policy_checks', {}).items()])
#             st.dataframe(policy_df, use_container_width=True, hide_index=True)
#             st.markdown('<p class="section-header">PD Calculation Breakdown</p>', unsafe_allow_html=True)
#             for factor, value in {
#                 'Bureau Score': f"{bureau_score} → Base PD: {bureau_score_to_pd(bureau_score):.1f}%",
#                 'Delinquency': f"DPD 90+: {dpd_90_6m}, DPD 30+: {dpd_30_6m} → Multiplier: {delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x",
#                 'FOIR Impact': f"{foir:.1f}% → Adjustment: {foir_to_pd_adjustment(foir):.1f}%",
#                 'Final PD': f"{decision_data.get('pd_percentage', 0)}%"
#             }.items():
#                 st.markdown(f"**{factor}:** {value}")

#         with tab4:
#             st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
#             audit_log = sanitize_for_json({
#                 'application_id': app_id,
#                 'timestamp': timestamp.isoformat(),
#                 'decision': decision_data.get('decision', 'ERROR'),
#                 'risk_score': decision_data.get('risk_score', 0),
#                 'pd_percentage': decision_data.get('pd_percentage', 0),
#                 'confidence': round(decision_data.get('confidence', 0), 2),
#                 'model_version': '8.6',
#                 'gender': gender, 'city_tier': city_tier,
#                 'rbi_consent': rbi_consent,
#                 'reason_codes': reasons,
#                 'policy_checks': decision_data.get('policy_checks', {}),
#                 'affordability': decision_data.get('affordability_data', {}),
#                 'customer_data': {k: v for k, v in customer_data.items() if k not in ['application_id','timestamp','reason_codes']},
#             })
#             with st.expander("📋 View Audit Log (JSON)"):
#                 st.json(audit_log)
#             col1, col2 = st.columns(2)
#             with col1:
#                 if PDF_AVAILABLE and generate_audit_pdf is not None:
#                     try:
#                         audit_pdf_buffer = generate_audit_pdf(audit_log)
#                         st.download_button("📥 Download Audit Trail (PDF)", data=audit_pdf_buffer,
#                                            file_name=f"audit_trail_{app_id}.pdf", mime="application/pdf",
#                                            use_container_width=True)
#                     except Exception as e:
#                         st.error(f"Error generating audit PDF: {str(e)}")
#                 else:
#                     st.warning("Audit PDF generation is not available.")
#             with col2:
#                 st.download_button("📥 Download Audit Log (JSON)",
#                                    data=json.dumps(audit_log, indent=2),
#                                    file_name=f"audit_{app_id}.json", mime="application/json",
#                                    use_container_width=True)

# elif page == "🔬 Stage 2 Analysis":
#     st.markdown('<p class="main-header">Stage 2: CIBIL Deep Dive</p>', unsafe_allow_html=True)

#     if not st.session_state.get('stage1_complete', False):
#         st.error("❌ You must complete Stage 1 Assessment first!")
#         if st.button("← Go to Assessment", use_container_width=True):
#             st.session_state.page_navigation = "👤 Assessment"
#             st.rerun()
#         st.stop()

#     if st.session_state.get('stage1_decision') not in ['APPROVE', 'REVIEW']:
#         st.error("❌ Stage 2 is only available for APPROVED or REVIEW applications!")
#         if st.button("← Go Back", use_container_width=True):
#             st.session_state.page_navigation = "👤 Assessment"
#             st.rerun()
#         st.stop()

#     if not (STAGE2_AVAILABLE and is_stage2_available()):
#         st.error("❌ Stage 2 model not available! Please ensure `stage2_cibil_model.pkl` is in the project directory.")
#         if st.button("← Go Back", use_container_width=True):
#             st.session_state.page_navigation = "👤 Assessment"
#             st.rerun()
#         st.stop()

#     stage1_data = st.session_state.get('stage1_data', {})
#     stage1_customer = st.session_state.get('current_customer_data', {})

#     st.markdown(f'<div class="info-box" style="background:linear-gradient(135deg,#3B82F6,#2563EB);color:white;"><h3 style="margin:0;color:white;">📊 Stage 1 Results</h3><p style="margin:0.5rem 0 0 0;"><strong>Decision:</strong> {st.session_state.get("stage1_decision","N/A")} | <strong>Risk Score:</strong> {stage1_data.get("risk_score","N/A")} | <strong>App ID:</strong> {stage1_customer.get("application_id","N/A")}</p></div>', unsafe_allow_html=True)
#     st.markdown("<br>", unsafe_allow_html=True)

#     tab_options = ["Manual Entry", "PDF Upload", "Batch Analysis"]
#     default_tab = st.session_state.get('stage2_selected_tab', 'Manual Entry')
#     selected_tab = st.radio("Select input method", tab_options,
#                             index=tab_options.index(default_tab) if default_tab in tab_options else 0,
#                             horizontal=True, label_visibility="collapsed")

#     if selected_tab == "Manual Entry":
#         st.markdown('<p class="section-header">Manual CIBIL Data Entry</p>', unsafe_allow_html=True)
#         with st.form("stage2_manual_form"):
#             st.markdown("### 👤 Demographics & Product Enquiries")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 gender_s2 = st.selectbox("Gender", ["Male", "Female", "Others"])
#                 marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Others"])
#                 education = st.selectbox("Education", ["Graduate", "Post Graduate", "Under Graduate", "Professional", "Others"])
#             with col2:
#                 st.markdown("**Credit Score & History**")
#                 cibil_score = st.number_input("Credit Score", 300, 900, 720, 10)
#                 max_delinquency = st.number_input("Max Delinquency Level", 0, 100, 0)
#                 num_times_30dpd = st.number_input("Times 30+ DPD", 0, 50, 0)
#                 num_times_60dpd = st.number_input("Times 60+ DPD", 0, 50, 0)
#                 num_times_delinquent = st.number_input("Total Delinquent", 0, 50, 0)
#             with col3:
#                 st.markdown("**Recent Behavior**")
#                 num_deliq_6m = st.number_input("Delinquencies (6M)", 0, 20, 0)
#                 num_deliq_12m = st.number_input("Delinquencies (12M)", 0, 20, 0)
#                 max_deliq_6m = st.number_input("Max Delinq (6M)", 0, 100, 0)
#                 max_deliq_12m = st.number_input("Max Delinq (12M)", 0, 100, 0)
#                 enq_L3m = st.number_input("Inquiries (3M)", 0, 20, 2)
#                 enq_L6m = st.number_input("Inquiries (6M)", 0, 30, 4)
#                 enq_L12m = st.number_input("Inquiries (12M)", 0, 50, 6)

#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.markdown("**Account Quality**")
#                 num_std = st.number_input("Standard Accounts", 0, 50, 3)
#                 num_std_6m = st.number_input("Standard (6M)", 0, 50, 3)
#                 num_std_12m = st.number_input("Standard (12M)", 0, 50, 3)
#                 num_sub = st.number_input("Sub-standard", 0, 20, 0)
#                 num_sub_6m = st.number_input("Sub-standard (6M)", 0, 20, 0)
#                 num_dbt = st.number_input("Doubtful", 0, 10, 0)
#                 num_lss = st.number_input("Loss", 0, 10, 0)
#             with col2:
#                 st.markdown("**Utilization**")
#                 pct_active_tls = st.number_input("% Active TLs", 0.0, 1.0, 0.60, 0.01)
#                 pct_current_bal = st.number_input("Current Balance %", 0.0, 1.0, 0.30, 0.01)
#                 cc_utilization = st.number_input("CC Utilization", 0.0, 1.0, 0.35, 0.01)
#                 pl_utilization = st.number_input("PL Utilization", 0.0, 1.0, 0.25, 0.01)
#                 max_unsec_exposure = st.number_input("Max Unsec Exposure %", 0, 100, 30)
#             with col3:
#                 st.markdown("**Demographics & Products**")
#                 age_cibil = st.number_input("Age", 24, 70, int(stage1_customer.get('age', 35)))
#                 net_monthly_income = st.number_input("Net Monthly Income", 0, 1000000, int(stage1_customer.get('avg_salary_6m', 50000)), 5000)
#                 time_curr_employer = st.number_input("Employment Tenure (months)", 0, 600, int(stage1_customer.get('employment_tenure_months', 24)))
#                 cc_flag = st.selectbox("Credit Card", ["Yes", "No"]) == "Yes"
#                 pl_flag = st.selectbox("Personal Loan", ["Yes", "No"]) == "No"
#                 hl_flag = st.selectbox("Home Loan", ["Yes", "No"]) == "No"
#                 gl_flag = st.selectbox("Gold Loan", ["Yes", "No"]) == "No"

#             st.markdown("<br>", unsafe_allow_html=True)
#             submitted_s2 = st.form_submit_button("🔬 Run Stage 2 Analysis", use_container_width=True, type="primary")

#         if submitted_s2:
#             with st.spinner("🔬 Running Stage 2 CIBIL Deep Analysis..."):
#                 enhanced_customer_data = stage1_customer.copy()
#                 _s1_inc = stage1_customer.get('avg_salary_6m', 50000)
#                 _s2_inc = net_monthly_income or 0
#                 _final_income = _s1_inc if (_s2_inc > 0 and _s2_inc < _s1_inc * 0.4) else (_s2_inc or _s1_inc)
#                 if _s2_inc > 0 and _s2_inc < _s1_inc * 0.4:
#                     st.warning(f'⚠️ CIBIL income ₹{_s2_inc:,} much lower than application income ₹{_s1_inc:,}. Using application income.')
#                 enhanced_customer_data.update({
#                     'bureau_score': cibil_score, 'age': age_cibil,
#                     'avg_salary_6m': _final_income, 'employment_tenure_months': time_curr_employer,
#                     'dpd_30_count_6m': num_times_30dpd, 'dpd_90_count_6m': num_times_60dpd,
#                     'max_delinquency_level': max_delinquency, 'num_times_delinquent': num_times_delinquent,
#                     'num_deliq_6mts': num_deliq_6m, 'num_deliq_12mts': num_deliq_12m,
#                     'max_deliq_6mts': max_deliq_6m, 'max_deliq_12mts': max_deliq_12m,
#                     'recent_inquiries_3m': enq_L3m, 'enq_L6m': enq_L6m, 'enq_L12m': enq_L12m,
#                     'active_loans_count': num_std, 'num_std_6mts': num_std_6m, 'num_std_12mts': num_std_12m,
#                     'num_sub': num_sub, 'num_sub_6mts': num_sub_6m,
#                     'num_dbt': num_dbt, 'num_lss': num_lss,
#                     'credit_utilization_pct': cc_utilization * 100,
#                     'pct_of_active_TLs_ever': pct_active_tls, 'pct_currentBal_all_TL': pct_current_bal,
#                     'CC_utilization': cc_utilization, 'PL_utilization': pl_utilization,
#                     'max_unsec_exposure_inPct': max_unsec_exposure,
#                     'CC_Flag': 1 if cc_flag else 0, 'PL_Flag': 1 if pl_flag else 0,
#                     'HL_Flag': 1 if hl_flag else 0, 'GL_Flag': 1 if gl_flag else 0,
#                     'GENDER': gender_s2, 'MARITALSTATUS': marital_status, 'EDUCATION': education,
#                 })
#                 enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)
#                 try:
#                     stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
#                     stage2_result = resolve_stage2_to_binary(stage2_result)
#                     display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
#                 except Exception as e:
#                     st.error(f"❌ Stage 2 analysis failed: {str(e)}")
#                     st.exception(e)

#     elif selected_tab == "PDF Upload":
#         st.markdown('<p class="section-header">📄 CIBIL PDF Upload</p>', unsafe_allow_html=True)
#         if not OCR_AVAILABLE:
#             st.error("❌ OCR not available. " + (OCR_ERROR_MSG or "Check packages.txt and requirements.txt."))
#             st.warning("Please use the **Manual Entry** tab.")
#         else:
#             uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="stage2_pdf")
#             if uploaded_pdf is not None:
#                 st.success(f"✅ File uploaded: {uploaded_pdf.name} ({uploaded_pdf.size/1024:.1f} KB)")
#                 if st.button("🔬 Extract & Analyze", key="extract_analyze_stage2", type="primary", use_container_width=True):
#                     with st.spinner("🔄 Extracting data from PDF..."):
#                         extraction_result = extract_cibil_from_pdf(uploaded_pdf)
#                     if extraction_result.get('success', False):
#                         st.success("✅ PDF extraction successful!")
#                         col1, col2, col3, col4 = st.columns(4)
#                         col1.metric("Credit Score", extraction_result.get('Credit_Score', 'N/A'))
#                         col2.metric("Times 30+ DPD", extraction_result.get('num_times_30p_dpd', 0))
#                         col3.metric("Times 60+ DPD", extraction_result.get('num_times_60p_dpd', 0))
#                         col4.metric("Active Accounts", extraction_result.get('num_std', 0))

#                         enhanced_customer_data = stage1_customer.copy()
#                         _s1_income = stage1_customer.get('avg_salary_6m', 50000)
#                         _s2_income = extraction_result.get('NETMONTHLYINCOME', 0)
#                         _use_income = _s1_income if (_s2_income > 0 and _s2_income < _s1_income * 0.4) else (_s2_income or _s1_income)
#                         enhanced_customer_data.update({
#                             'bureau_score': extraction_result.get('Credit_Score', 720),
#                             'age': extraction_result.get('AGE', stage1_customer.get('age', 35)),
#                             'avg_salary_6m': _use_income,
#                             'employment_tenure_months': extraction_result.get('Time_With_Curr_Empr', stage1_customer.get('employment_tenure_months', 24)),
#                             'dpd_30_count_6m': extraction_result.get('num_times_30p_dpd', 0),
#                             'dpd_90_count_6m': extraction_result.get('dpd_90_count_6m', 0),
#                             'max_delinquency_level': extraction_result.get('max_delinquency_level', 0),
#                             'num_times_delinquent': extraction_result.get('num_times_delinquent', 0),
#                             'num_deliq_6mts': extraction_result.get('num_deliq_6mts', 0),
#                             'num_deliq_12mts': extraction_result.get('num_deliq_12mts', 0),
#                             'max_deliq_6mts': extraction_result.get('max_deliq_6mts', 0),
#                             'max_deliq_12mts': extraction_result.get('max_deliq_12mts', 0),
#                             'recent_inquiries_3m': extraction_result.get('enq_L3m', 2),
#                             'enq_L6m': extraction_result.get('enq_L6m', 4),
#                             'enq_L12m': extraction_result.get('enq_L12m', 6),
#                             'active_loans_count': extraction_result.get('num_std', 1),
#                             'num_std_6mts': extraction_result.get('num_std_6mts', 0),
#                             'num_std_12mts': extraction_result.get('num_std_12mts', 0),
#                             'num_sub': extraction_result.get('num_sub', 0),
#                             'num_sub_6mts': extraction_result.get('num_sub_6mts', 0),
#                             'num_dbt': extraction_result.get('num_dbt', 0),
#                             'num_lss': extraction_result.get('num_lss', 0),
#                             'credit_utilization_pct': (0 if extraction_result.get('CC_utilization', 0) < 0 else extraction_result.get('CC_utilization', 0.35)) * 100,
#                             'CC_utilization': max(0, extraction_result.get('CC_utilization', 0.35) or 0),
#                             'PL_utilization': max(0, extraction_result.get('PL_utilization', 0.25) or 0),
#                             'pct_of_active_TLs_ever': extraction_result.get('pct_of_active_TLs_ever', 0.6),
#                             'pct_currentBal_all_TL': extraction_result.get('pct_currentBal_all_TL', 0.3),
#                             'max_unsec_exposure_inPct': extraction_result.get('max_unsec_exposure_inPct', 30),
#                             'CC_Flag': extraction_result.get('CC_Flag', 0),
#                             'PL_Flag': extraction_result.get('PL_Flag', 0),
#                             'HL_Flag': extraction_result.get('HL_Flag', 0),
#                             'GL_Flag': extraction_result.get('GL_Flag', 0),
#                             'written_off_count': extraction_result.get('written_off_count', 0),
#                         })
#                         enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)
#                         with st.spinner("🔬 Running Stage 2 analysis..."):
#                             try:
#                                 stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
#                                 stage2_result = resolve_stage2_to_binary(stage2_result)
#                                 display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
#                             except Exception as e:
#                                 st.error(f"❌ Analysis failed: {str(e)}")
#                     else:
#                         st.error("❌ PDF extraction failed: " + extraction_result.get('error', 'Unknown'))

#     elif selected_tab == "Batch Analysis":
#         st.info("📊 Stage 2 Batch analysis coming soon.")

# elif page == "⚖️ Fairness":
#     render_fairness_dashboard()

# elif page == "📊 Batch Process":
#     st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
#     st.markdown('<div class="info-box">📤 Upload a CSV file with customer data for bulk credit assessment.</div>', unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#             st.success(f"✅ Successfully loaded {len(df)} records")
#             with st.expander("📄 Preview Uploaded Data"):
#                 st.dataframe(df.head(), use_container_width=True)
#             required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
#             missing_cols = [col for col in required_cols if col not in df.columns]
#             if missing_cols:
#                 st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
#             else:
#                 if st.button("🚀 Process Batch Predictions", key="process_batch_btn", type="primary", use_container_width=True):
#                     with st.spinner(f"🔍 Processing {len(df)} records..."):
#                         results_df = process_batch_predictions(df)
#                     st.success(f"✅ Completed {len(results_df)} records!")
#                     tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
#                     with tab1:
#                         st.dataframe(results_df, use_container_width=True)
#                         c1, c2, c3, c4 = st.columns(4)
#                         with c1: st.metric("✅ Approved", len(results_df[results_df['decision'] == 'APPROVE']))
#                         with c2: st.metric("❌ Rejected", len(results_df[results_df['decision'] == 'REJECT']))
#                         with c3: st.metric("⚠️ Review", len(results_df[results_df['decision'] == 'REVIEW']))
#                         with c4: st.metric("📊 Avg Risk Score", f"{results_df['risk_score'].mean():.0f}")
#                     with tab2:
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             dc = results_df['decision'].value_counts()
#                             fig1 = px.pie(values=dc.values, names=dc.index, title="Decision Distribution",
#                                           color=dc.index, color_discrete_map={'APPROVE':'#48bb78','REVIEW':'#ed8936','REJECT':'#f56565'})
#                             st.plotly_chart(fig1, use_container_width=True)
#                         with col2:
#                             fig2 = px.histogram(results_df, x='risk_score', title="Risk Score Distribution",
#                                                 nbins=20, color_discrete_sequence=['#587042'])
#                             st.plotly_chart(fig2, use_container_width=True)
#                         # Fairness charts from batch
#                         if 'gender' in results_df.columns and results_df['gender'].nunique() > 1:
#                             results_df['approved_num'] = (results_df['decision'] == 'APPROVE').astype(int)
#                             grp = results_df.groupby('gender')['approved_num'].mean().reset_index()
#                             grp['Approval Rate %'] = (grp['approved_num'] * 100).round(1)
#                             fig3 = px.bar(grp, x='gender', y='Approval Rate %', title='Approval Rate by Gender (Batch)',
#                                           color='Approval Rate %', color_continuous_scale=['#f56565','#48bb78'], range_color=[0,100])
#                             st.plotly_chart(fig3, use_container_width=True)
#                         if 'city_tier' in results_df.columns and results_df['city_tier'].nunique() > 1:
#                             results_df['approved_num'] = (results_df['decision'] == 'APPROVE').astype(int)
#                             grp2 = results_df.groupby('city_tier')['approved_num'].mean().reset_index()
#                             grp2['Approval Rate %'] = (grp2['approved_num'] * 100).round(1)
#                             fig4 = px.bar(grp2, x='city_tier', y='Approval Rate %', title='Approval Rate by City Tier (Batch)',
#                                           color='Approval Rate %', color_continuous_scale=['#f56565','#48bb78'], range_color=[0,100])
#                             st.plotly_chart(fig4, use_container_width=True)
#                     with tab3:
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             st.download_button("📥 Download as CSV", data=results_df.to_csv(index=False),
#                                                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                                mime="text/csv", use_container_width=True)
#                         with col2:
#                             st.download_button("📥 Download as JSON", data=results_df.to_json(orient='records', indent=2),
#                                                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                                                mime="application/json", use_container_width=True)
#         except Exception as e:
#             st.error(f"❌ Error processing file: {str(e)}")
#     else:
#         st.markdown("---")
#         st.markdown("### 📋 CSV Template")
#         template_data = {
#             'age': [35, 42, 28], 'gender': ['Male', 'Female', 'Male'],
#             'city_tier': ['Tier 1', 'Tier 2', 'Tier 3'],
#             'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
#             'dependents': [2, 3, 6], 'kyc_verified': ['Yes', 'Yes', 'No'],
#             'bankruptcy_flag': ['No', 'No', 'No'], 'fraud_flag': ['No', 'No', 'No'],
#             'rbi_consent': ['Yes', 'Yes', 'Yes'],
#             'employment_tenure_months': [24, 0, 18], 'business_vintage_years': [0, 5, 0],
#             'bureau_score': [720, 680, 580], 'dpd_90_count_6m': [0, 1, 2],
#             'dpd_30_count_6m': [0, 2, 1], 'credit_utilization_pct': [30, 45, 75],
#             'recent_inquiries_3m': [2, 1, 5], 'active_loans_count': [1, 2, 3],
#             'avg_salary_6m': [50000, 75000, 35000], 'AMT_INCOME_TOTAL': [600000, 900000, 420000],
#             'net_cash_surplus_6m': [20000, 35000, 10000],
#             'salary_stability_flag': ['STABLE', 'MODERATE', 'UNSTABLE'],
#             'loan_amount': [180000, 250000, 100000], 'loan_tenure_months': [24, 36, 12],
#             'interest_rate': [10.5, 11.0, 12.0], 'existing_emi': [15000, 20000, 8000],
#             'AMT_ANNUITY': [8500, 9500, 4500],
#             'payment_discipline_flag': ['GOOD', 'MODERATE', 'POOR'],
#             'liquidity_flag': ['LOW', 'ADEQUATE', 'LOW'],
#             'cashflow_health': ['HEALTHY', 'MODERATE', 'STRESSED'],
#             'bureau_risk_flag': ['LOW', 'MEDIUM', 'HIGH'],
#             'inward_bounce_count_3m': [0, 1, 3], 'salary_missing_months': [0, 0, 2],
#         }
#         template_df = pd.DataFrame(template_data)
#         st.dataframe(template_df, use_container_width=True)
#         st.caption("📝 New columns: `gender`, `city_tier`, `rbi_consent` — required for fairness monitoring and compliance.")
#         st.download_button("📥 Download CSV Template", data=template_df.to_csv(index=False),
#                            file_name="credit_assessment_template_v8.6.csv",
#                            mime="text/csv", use_container_width=True)

# elif page == "📈 Model Info":
#     st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
#     col1, col2, col3 = st.columns(3)
#     with col1: st.markdown(f'<div class="stat-card"><div class="stat-number">RF</div><div class="stat-label">Model Type</div></div>', unsafe_allow_html=True)
#     with col2: st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TOP_FEATURES)}</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)
#     with col3: st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TARGET_LE.classes_)}</div><div class="stat-label">Classes</div></div>', unsafe_allow_html=True)
#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
#     feature_df = pd.DataFrame({'Rank': range(1, min(21, len(TOP_FEATURES)+1)), 'Feature': TOP_FEATURES[:20]})
#     st.dataframe(feature_df, use_container_width=True, hide_index=True)

# elif page == "ℹ️ About":
#     st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
#     st.markdown("""
#         <div class="info-card">
#             <div class="info-card-title">🏦 Credit Risk Assessment Platform</div>
#             <div class="info-card-content">
#                 <p><strong>Version:</strong> 8.6 — Cleaned codebase + Fairness Monitoring + City Tier + RBI Consent</p>
#                 <p><strong>Developer:</strong> Zen Meraki</p>
#                 <p><strong>Date:</strong> January 2026</p>
#                 <br>
#                 <p>A comprehensive credit risk evaluation system combining hard policy rules,
#                 machine learning, and affordability analysis for accurate and RBI-compliant lending decisions.</p>
#             </div>
#         </div>
#     """, unsafe_allow_html=True)
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("""
#             <div class="info-card">
#                 <div class="info-card-title">🎯 Key Features</div>
#                 <div class="info-card-content"><ul style="margin:0;padding-left:1.25rem;">
#                     <li>Three-layer decision engine</li>
#                     <li>Real-time risk assessment</li>
#                     <li>Industry-standard PD calculation</li>
#                     <li>FOIR calculation & validation</li>
#                     <li>Automated reason generation</li>
#                     <li>Complete audit trail (PDF)</li>
#                     <li>OCR auto-fill with categorical inference</li>
#                     <li>⚖️ Fairness monitoring dashboard</li>
#                     <li>🏙️ City Tier field for geographic equity</li>
#                     <li>📜 RBI consent gate (DLG 2022)</li>
#                 </ul></div>
#             </div>
#         """, unsafe_allow_html=True)
#     with col2:
#         st.markdown("""
#             <div class="info-card">
#                 <div class="info-card-title">🛠️ Technology Stack</div>
#                 <div class="info-card-content"><ul style="margin:0;padding-left:1.25rem;">
#                     <li>Streamlit (UI Framework)</li>
#                     <li>Scikit-learn (ML)</li>
#                     <li>Plotly (Visualizations)</li>
#                     <li>Pandas (Data Processing)</li>
#                     <li>ReportLab (PDF Generation)</li>
#                     <li>Tesseract OCR + pdf2image</li>
#                     <li>Python 3.8+</li>
#                 </ul></div>
#             </div>
#         """, unsafe_allow_html=True)







"""
Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
Enhanced with Modern UI/UX Design
Run with: streamlit run test.py (from inside the notebooks folder)
Author: Zen Meraki
Date: January 2026
VERSION: 8.4 - OCR AUTO-FILL FIX (all categorical dropdowns now update from PDF)
"""

import streamlit as st

# =============================================================================
# PAGE CONFIGURATION – MUST BE THE VERY FIRST STREAMLIT COMMAND
# =============================================================================
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STANDARD LIBRARY / THIRD-PARTY IMPORTS
# =============================================================================
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
from typing import Dict, List, Any, Union
import json
import sys
import os
from pathlib import Path
import re

# =============================================================================
# SUPPRESS SCIKIT-LEARN VERSION WARNINGS
# =============================================================================
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# =============================================================================
# DYNAMIC PATH RESOLUTION
# =============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
POSSIBLE_LOCATIONS = [
    CURRENT_DIR,
    PROJECT_ROOT,
    PROJECT_ROOT / "loan",
    PROJECT_ROOT / "utils",
    PROJECT_ROOT / "notebooks",
]

for loc in POSSIBLE_LOCATIONS:
    if loc.exists() and str(loc) not in sys.path:
        sys.path.insert(0, str(loc))

# =============================================================================
# OPTIONAL OCR DEPENDENCIES – GRACEFUL FALLBACK
# =============================================================================
OCR_AVAILABLE = False
OCR_ERROR_MSG = ""
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    import cv2
    from PIL import Image

    import shutil as _shutil
    _tess_cmd = (
        _shutil.which("tesseract")
        or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    if _tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd

    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True

except ImportError as _e:
    OCR_ERROR_MSG = (
        f"Missing Python package: {_e}. "
        "Add to requirements.txt: pytesseract  pdf2image  opencv-python-headless  pillow"
    )
except Exception as _e:
    _name = type(_e).__name__
    if "TesseractNotFound" in _name or "tesseract" in str(_e).lower():
        OCR_ERROR_MSG = (
            "Tesseract binary not found. "
            "Streamlit Cloud → add 'tesseract-ocr' and 'poppler-utils' to packages.txt. "
            "Linux → sudo apt install tesseract-ocr poppler-utils. "
            "Mac → brew install tesseract poppler."
        )
    else:
        OCR_ERROR_MSG = f"OCR init error ({_name}): {_e}"

# =============================================================================
# IMPORT CSS – WITH FALLBACK
# =============================================================================
try:
    from css_styles import CSS
except ImportError:
    CSS = """
    <style>
        .main-header { font-size: 2rem; font-weight: bold; color: #2d3748; }
        .section-header { font-size: 1.5rem; font-weight: 600; color: #2d3748; }
        .info-box { background: #f7fafc; padding: 1rem; border-radius: 0.5rem; }
        .decision-card { padding: 2rem; border-radius: 1rem; text-align: center; margin-bottom: 1rem; }
        .decision-card-approved { background: #c6f6d5; border-left: 5px solid #48bb78; }
        .decision-card-rejected { background: #fed7d7; border-left: 5px solid #f56565; }
        .decision-card-review { background: #feebc8; border-left: 5px solid #ed8936; }
        .decision-title { font-size: 2.5rem; font-weight: bold; }
        .decision-subtitle { font-size: 1rem; opacity: 0.8; }
        .stat-card { background: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
        .stat-number { font-size: 1.8rem; font-weight: bold; color: #2d3748; }
        .stat-label { font-size: 0.875rem; color: #718096; }
        .info-card { background: white; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .info-card-title { font-weight: 600; margin-bottom: 0.5rem; color: #2d3748; }
        .info-card-content { font-size: 0.875rem; }
        .data-row { display: flex; justify-content: space-between; padding: 0.25rem 0; border-bottom: 1px solid #e2e8f0; }
        .data-label { color: #4a5568; }
        .data-value { font-weight: 500; }
        .status-badge { padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; margin-left: 0.5rem; }
        .badge-pass { background: #c6f6d5; color: #22543d; }
        .badge-fail { background: #fed7d7; color: #742a2a; }
        .badge-warning { background: #feebc8; color: #744210; }
        .reason-item { padding: 0.25rem 0; }
        .reason-icon { color: #587042; font-weight: bold; margin-right: 0.5rem; }
    </style>
    """

st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    if 'stage1_complete' not in st.session_state:
        st.session_state.stage1_complete = False
    if 'stage1_decision' not in st.session_state:
        st.session_state.stage1_decision = None
    if 'stage1_data' not in st.session_state:
        st.session_state.stage1_data = None
    if 'current_customer_data' not in st.session_state:
        st.session_state.current_customer_data = None
    if 'page_navigation' not in st.session_state:
        st.session_state.page_navigation = "🏠 Home"
    if 'use_two_stage' not in st.session_state:
        st.session_state.use_two_stage = False
    if 'stage2_selected_tab' not in st.session_state:
        st.session_state.stage2_selected_tab = "Manual Entry"

init_session_state()

# =============================================================================
# IMPORT BUSINESS LOGIC MODULES
# =============================================================================
try:
    from affordability_engine import calculate_emi, calculate_affordability
    from reason_codes import generate_reason_codes
    from risk_engine import (
        calculate_final_risk_score, fill_missing_ml_fields,
        clean_sentinel_values, validate_cibil_identity
    )
    from affordability_engine import check_loan_to_income, check_net_disposable
except ImportError as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.info("""
    Please ensure the following files are placed in one of these directories:
    - `notebooks/` (same folder as test.py)
    - `loan/` (sibling of notebooks)
    - `utils/` (containing pdf_generator.py and __init__.py)
    - The project root (`credit_risk_engine/`)

    Required files:
    - affordability_engine.py
    - reason_codes.py
    - risk_engine.py
    - utils/__init__.py
    - utils/pdf_generator.py
    """)
    st.stop()

# =============================================================================
# STAGE 2 ENGINE – ROBUST FALLBACK
# =============================================================================
try:
    import stage2_engine
    from stage2_engine import make_two_stage_decision, is_stage2_available, get_stage2_status
    STAGE2_AVAILABLE = is_stage2_available()
except ImportError:
    stage2_engine = None
    STAGE2_AVAILABLE = False
    def make_two_stage_decision(*args, **kwargs):
        raise NotImplementedError("Stage 2 engine not available")
    def is_stage2_available():
        return False
    def get_stage2_status():
        return {"error": "Stage 2 engine module not found", "available": False}

# =============================================================================
# PDF GENERATION – SAFE FALLBACK
# =============================================================================
PDF_AVAILABLE = False
generate_decision_pdf = None
generate_audit_pdf = None
try:
    from utils.pdf_generator import generate_decision_pdf, generate_audit_pdf
    PDF_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# JSON SANITIZER
# =============================================================================
def sanitize_for_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)

# =============================================================================
# LOAD TRAINED MODEL ASSETS (Stage 1 Random Forest)
# =============================================================================
@st.cache_resource
def load_model_assets():
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
        return {'loaded': False, 'error': 'credit_risk_assets.pkl not found. Please run the training script first.'}
    except Exception as e:
        return {'loaded': False, 'error': f'Error loading model: {str(e)}'}

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
    if principal <= 0 or tenure_months <= 0:
        return 0
    monthly_rate = annual_rate / (12 * 100)
    if monthly_rate == 0:
        return principal / tenure_months
    emi = (principal * monthly_rate * (1 + monthly_rate)**tenure_months) / \
          ((1 + monthly_rate)**tenure_months - 1)
    return round(emi, 2)

def calculate_affordability(monthly_income, loan_amount, interest_rate, tenure_months, existing_emi):
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
    'strong_income': 'Strong monthly income (₹{income:,})',
    'low_utilization': 'Low credit utilization ({util}%)',
}
REJECTION_REASONS = {
    'low_bureau':       'Credit score below minimum ({score} < 550)',
    'high_foir':        'EMI burden too high (FOIR: {foir}% > 50%)',
    'severe_dpd':       'Severe payment delays ({dpd} instances of 90+ DPD)',
    'moderate_dpd':     'Frequent payment delays ({dpd} instances of 30+ DPD)',
    'low_income':       'Income below minimum threshold (₹{income:,} < ₹15,000)',
    'short_employment': 'Insufficient employment tenure ({tenure} months < 6)',
    'short_vintage':    'Insufficient business vintage ({vintage} years < 2 years)',
    'bankruptcy':       'Active bankruptcy detected',
    'kyc_failed':       'KYC verification not completed',
    'fraud_flag':       'Fraud flag present on application',
    'high_utilization': 'High credit utilization ({util}% > 80%)',
    'age_invalid':      'Age outside acceptable range ({age} years, must be 24–70)',
    'high_dependents':  'High number of dependents ({deps}) reducing net disposable income',
}
REVIEW_REASONS = {
    'borderline_bureau':  'Credit score in borderline range ({score})',
    'moderate_foir':      'EMI burden moderate (FOIR: {foir}%)',
    'mixed_signals':      'Mixed credit indicators requiring human review',
    'recent_employment':  'Recent employment change requiring verification',
    'high_loan_amount':   'Large loan amount requiring additional underwriting review',
    'moderate_dpd':       'Recent 30-day payment delays requiring review ({dpd} instances)',
    'moderate_dependents':'Moderate number of dependents ({deps}) may affect repayment',
}

def generate_reason_codes(decision, customer_data, affordability_data, policy_checks):
    reasons = []
    bureau_score      = customer_data.get('bureau_score', 0)
    foir              = affordability_data.get('foir_percentage', 0)
    dpd_90            = customer_data.get('dpd_90_count_6m', 0)
    dpd_30            = customer_data.get('dpd_30_count_6m', 0)
    income            = customer_data.get('avg_salary_6m', 0)
    employment_tenure = customer_data.get('employment_tenure_months', 0)
    business_vintage  = customer_data.get('business_vintage_years', 0)
    employment_type   = customer_data.get('employment_type', 'Salaried')
    credit_util       = customer_data.get('credit_utilization_pct', 0)
    age               = customer_data.get('age', 0)
    dependents        = customer_data.get('dependents', 0)

    if decision == "APPROVE":
        if bureau_score >= 750:
            reasons.append(APPROVAL_REASONS['high_bureau'].format(score=bureau_score))
        if employment_tenure >= 24:
            reasons.append(APPROVAL_REASONS['stable_employment'].format(tenure=employment_tenure))
        if foir <= 40:
            reasons.append(APPROVAL_REASONS['low_foir'].format(foir=round(foir, 1)))
        if dpd_90 == 0 and dpd_30 == 0:
            reasons.append(APPROVAL_REASONS['clean_payment'])
        if income >= 75000:
            reasons.append(APPROVAL_REASONS['strong_income'].format(income=income))
        if credit_util <= 30:
            reasons.append(APPROVAL_REASONS['low_utilization'].format(util=credit_util))

    elif decision == "REJECT":
        for check_name, check_result in policy_checks.items():
            if '❌' in str(check_result):
                cn = check_name.lower()
                if 'bureau' in cn:
                    reasons.append(REJECTION_REASONS['low_bureau'].format(score=bureau_score))
                elif 'dpd' in cn:
                    reasons.append(REJECTION_REASONS['severe_dpd'].format(dpd=dpd_90))
                elif 'income' in cn:
                    reasons.append(REJECTION_REASONS['low_income'].format(income=income))
                elif 'tenure' in cn:
                    if employment_type == 'Salaried':
                        reasons.append(REJECTION_REASONS['short_employment'].format(tenure=employment_tenure))
                    else:
                        reasons.append(REJECTION_REASONS['short_vintage'].format(vintage=business_vintage))
                elif 'kyc' in cn:
                    reasons.append(REJECTION_REASONS['kyc_failed'])
                elif 'bankruptcy' in cn:
                    reasons.append(REJECTION_REASONS['bankruptcy'])
                elif 'fraud' in cn:
                    reasons.append(REJECTION_REASONS['fraud_flag'])
                elif 'age' in cn:
                    reasons.append(REJECTION_REASONS['age_invalid'].format(age=age))
        if foir > 50:
            reasons.append(REJECTION_REASONS['high_foir'].format(foir=round(foir, 1)))
        if credit_util > 80:
            reasons.append(REJECTION_REASONS['high_utilization'].format(util=credit_util))
        if dpd_30 >= 3 and dpd_90 == 0:
            reasons.append(REJECTION_REASONS['moderate_dpd'].format(dpd=dpd_30))
        if dependents >= 4:
            reasons.append(REJECTION_REASONS['high_dependents'].format(deps=dependents))

    elif decision == "REVIEW":
        if 650 <= bureau_score < 700:
            reasons.append(REVIEW_REASONS['borderline_bureau'].format(score=bureau_score))
        if 40 < foir <= 50:
            reasons.append(REVIEW_REASONS['moderate_foir'].format(foir=round(foir, 1)))
        if employment_tenure < 12:
            reasons.append(REVIEW_REASONS['recent_employment'])
        if dpd_30 >= 1 and dpd_90 == 0:
            reasons.append(REVIEW_REASONS['moderate_dpd'].format(dpd=dpd_30))
        if 2 <= dependents < 4:
            reasons.append(REVIEW_REASONS['moderate_dependents'].format(deps=dependents))
        if not reasons:
            reasons.append(REVIEW_REASONS['mixed_signals'])

    return reasons[:3] if reasons else ['Decision based on comprehensive model assessment']

# =============================================================================
# PD CALCULATION
# =============================================================================
def bureau_score_to_pd(bureau_score):
    if bureau_score >= 800:
        return 0.5 + (900 - bureau_score) / 200 * 0.5
    elif bureau_score >= 750:
        return 1.0 + (800 - bureau_score) / 50 * 1.0
    elif bureau_score >= 700:
        return 2.0 + (750 - bureau_score) / 50 * 1.5
    elif bureau_score >= 650:
        return 3.5 + (700 - bureau_score) / 50 * 2.5
    elif bureau_score >= 600:
        return 6.0 + (650 - bureau_score) / 50 * 4.0
    elif bureau_score >= 550:
        return 10.0 + (600 - bureau_score) / 50 * 5.0
    else:
        return min(25.0, 15.0 + (550 - bureau_score) / 50 * 10.0)

def foir_to_pd_adjustment(foir_percentage):
    if foir_percentage <= 30:
        return -0.75
    elif foir_percentage <= 40:
        return 0.00
    elif foir_percentage <= 45:
        return 0.75
    elif foir_percentage <= 50:
        return 1.50
    elif foir_percentage <= 55:
        return 2.25
    elif foir_percentage <= 60:
        return 3.50
    else:
        return 6.00

def delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count=0):
    if dpd_90_count >= 3:
        return 5.0
    elif dpd_90_count == 2:
        return 3.0
    elif dpd_90_count == 1:
        return 2.0
    elif dpd_30_count >= 3:
        return 1.6
    elif dpd_30_count >= 1:
        return 1.3
    else:
        return 1.0

def employment_stability_to_pd_adjustment(employment_type, tenure_months, business_vintage_years=0):
    if employment_type == 'Salaried':
        if tenure_months >= 36:
            return -0.5
        elif tenure_months >= 12:
            return 0.0
        elif tenure_months >= 6:
            return 0.5
        else:
            return 2.0
    elif employment_type in ['Self-Employed', 'Business']:
        if business_vintage_years >= 5:
            return -0.5
        elif business_vintage_years >= 2:
            return 0.0
        else:
            return 1.5
    else:
        return 1.0

def inquiry_pattern_to_pd_adjustment(recent_inquiries_3m):
    if recent_inquiries_3m <= 1:
        return -0.3
    elif recent_inquiries_3m <= 3:
        return 0.0
    elif recent_inquiries_3m <= 5:
        return 0.8
    elif recent_inquiries_3m <= 8:
        return 1.5
    else:
        return 3.0

def ml_confidence_to_pd_adjustment(ml_confidence, ml_decision):
    if ml_decision == "APPROVE":
        if ml_confidence >= 90:
            return -0.5
        elif ml_confidence >= 70:
            return 0.0
        else:
            return 0.5
    elif ml_decision == "REVIEW":
        return 1.0
    else:
        return 5.0

def calculate_final_pd(bureau_score, foir, confidence, dpd_90_count=0, dpd_30_count=0,
                       employment_type='Salaried', employment_tenure=24, business_vintage=0,
                       recent_inquiries=2, ml_decision='APPROVE'):
    base_pd = bureau_score_to_pd(bureau_score)
    foir_adj = foir_to_pd_adjustment(foir)
    deliq_multiplier = delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count)
    employment_adj = employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage)
    inquiry_adj = inquiry_pattern_to_pd_adjustment(recent_inquiries)
    ml_adj = ml_confidence_to_pd_adjustment(confidence, ml_decision)
    adjusted_base_pd = base_pd * deliq_multiplier
    final_pd = adjusted_base_pd + foir_adj + employment_adj + inquiry_adj + ml_adj
    final_pd = max(0.5, min(final_pd, 25.0))
    return round(final_pd, 2)

# =============================================================================
# RISK SCORE CALCULATION
# =============================================================================
def calculate_final_risk_score(bureau_score, ml_confidence, foir,
                                dpd_90, dpd_30, net_surplus,
                                bounces=0, missing_months=0, active_loans=0):
    bureau_points = (bureau_score / 900) * 400
    ml_points = (ml_confidence / 100) * 300
    foir_points = max(0, (1 - foir / 50) * 150)
    dpd_penalty = min((dpd_90 * 50) + (dpd_30 * 20), 150)
    behavioral_penalty = min((bounces * 10) + (missing_months * 10), 100)
    if net_surplus > 50000:
        surplus_points = 50
    elif net_surplus > 0:
        surplus_points = 20
    elif net_surplus < -50000:
        surplus_points = -50
    else:
        surplus_points = -20
    total = (bureau_points + ml_points + foir_points
             + surplus_points - dpd_penalty - behavioral_penalty)
    return max(0, min(int(total), 1000))

# =============================================================================
# CATEGORICAL FLAG INFERENCE FROM CIBIL DATA
# v8.5: Dual-dataset calibration.
#
# Dataset A — train_60k_rule_accepted.csv (bank-statement enriched):
#   Has: net_cash_surplus_6m, inward_bounce_count_3m, salary_missing_months
#   payment_discipline: GOOD 99.9%,  MODERATE 0.02%, POOR 0.04%
#   cashflow_health   : MODERATE 90%, HEALTHY 8.8%, STRESSED 0.8%, STABLE 0.4%
#   liquidity_flag    : LOW 87.7%,   ADEQUATE 11.9%, MODERATE 0.4%
#   bureau_risk_flag  : LOW 97.9%,   HIGH 1.3%,      MEDIUM 0.75%
#   salary_stability  : MODERATE 85.8%, STABLE 12.1%, UNSTABLE 2.1%
#
# Dataset B — External_Cibil_Dataset.xlsx (bureau-only, 51,336 rows):
#   Has: num_times_30p_dpd, num_times_60p_dpd, num_lss, num_dbt,
#        NETMONTHLYINCOME, Time_With_Curr_Empr, Credit_Score
#   NO:  net_cash_surplus_6m, inward_bounce_count_3m, salary_missing_months
#   Income median: ₹23,000 (vs ₹50,000 in Dataset A — very different scale)
#   payment_discipline: POOR 10.5%, MODERATE 5.1%, GOOD 84.4%
#   bureau_risk_flag  : HIGH 5.0%,  MEDIUM 10.3%, LOW 84.7%
#   salary_stability  : UNSTABLE 0.04%, STABLE 11.2%, MODERATE 88.8%
#   Tier mapping      : P1(score 701+), P2(669-700), P3(subprime), P4(high risk)
#
# Auto-detection: if 'NETMONTHLYINCOME' key present → Dataset B path (bureau-only).
#                 Otherwise → Dataset A path (bank-statement enriched).
# =============================================================================
def _infer_surplus_from_cibil(score: int, dpd_60: int, dpd_30: int, income: float) -> float:
    """
    Estimate net cash surplus when no bank statement data is available.
    Used for External_Cibil_Dataset (bureau-only) OCR path.

    Calibrated against External_Cibil_Dataset tier distributions:
      - Score >= 700, clean DPD  -> income likely covers expenses  -> +30% income
      - Score 650-699, clean DPD -> borderline                      -> +10% income
      - Score < 650 OR 60+ DPD   -> stressed                        -> -20% income
      - 60+ DPD >= 3             -> severe stress                   -> -50% income
    """
    if dpd_60 >= 3:
        return income * -0.5
    elif score < 650 or dpd_60 >= 1:
        return income * -0.2
    elif score < 700:
        return income * 0.1
    else:
        return income * 0.3


def infer_categorical_flags(extraction_result: dict) -> dict:
    """
    Convert numeric CIBIL fields into the 5 categorical flags used by the
    Stage 1 assessment form.

    Automatically detects whether this is a bank-statement-enriched result
    (Dataset A / train_60k) or a bureau-only result (Dataset B / External CIBIL)
    and applies the appropriate calibrated thresholds for each.

    Args:
        extraction_result: dict returned by extract_cibil_from_pdf()

    Returns:
        dict with keys: payment_discipline_flag, cashflow_health,
                        liquidity_flag, bureau_risk_flag, salary_stability_flag
    """
    # ── Common fields (present in both datasets) ─────────────────────
    score       = int(extraction_result.get('Credit_Score', 700) or 700)
    dpd_30      = int(extraction_result.get('num_times_30p_dpd', 0) or 0)
    dpd_60      = int(extraction_result.get('num_times_60p_dpd', 0) or 0)
    written_off = int(extraction_result.get('num_lss', 0) or
                      extraction_result.get('written_off_count', 0) or 0)
    doubtful    = int(extraction_result.get('num_dbt', 0) or 0)
    cc_util_raw = extraction_result.get('CC_utilization', 0) or 0
    # Sentinel -99999 → 0 (no credit card on file)
    cc_util     = float(cc_util_raw) if cc_util_raw > 0 else 0.0
    income      = float(extraction_result.get('NETMONTHLYINCOME', 0) or
                        extraction_result.get('avg_salary_6m', 50_000) or 50_000)
    tenure      = int(extraction_result.get('Time_With_Curr_Empr', 24) or 24)

    # ── Detect dataset type ──────────────────────────────────────────
    # Dataset B (External CIBIL) uses NETMONTHLYINCOME key and lacks bank-stmt fields.
    # Dataset A (train_60k) uses avg_salary_6m and HAS surplus/bounce/missing.
    is_bureau_only = (
        'NETMONTHLYINCOME' in extraction_result
        and 'net_cash_surplus_6m' not in extraction_result
        and 'net_surplus' not in extraction_result
    )

    if is_bureau_only:
        # ── DATASET B PATH (External_Cibil_Dataset) ─────────────────
        # Income median ₹23k, score range 469-811, bureau fields only.
        # num_times_60p_dpd used as dpd_90 proxy (60+ includes 90+ DPD).

        dpd_90_proxy = dpd_60   # 60+ is the closest to 90+ in this dataset

        # Estimate surplus since no bank statement
        surplus = _infer_surplus_from_cibil(score, dpd_60, dpd_30, income)

        # 1. payment_discipline_flag
        # External CIBIL: POOR=10.5% (60+dpd>=1 OR 30+dpd>=3), MODERATE=5.1%, GOOD=84.4%
        if dpd_60 >= 1 or dpd_30 >= 3:
            payment_discipline = 'POOR'
        elif dpd_30 >= 1:
            payment_discipline = 'MODERATE'
        else:
            payment_discipline = 'GOOD'

        # 2. cashflow_health (derived from surplus proxy + DPD)
        # External CIBIL distribution via proxy: STABLE 84%, STRESSED 14%, HEALTHY 1.2%
        if surplus >= 14_000:
            cashflow_health = 'HEALTHY'
        elif surplus >= 600:
            cashflow_health = 'STABLE'
        elif surplus < -1_000:
            cashflow_health = 'STRESSED'
        else:
            cashflow_health = 'MODERATE'

        # 3. liquidity_flag (derived from surplus proxy)
        # External CIBIL proxy: ADEQUATE 1.2%, MODERATE 98.6%, LOW 0.1%
        # Note: income-based surplus rarely reaches extremes → mostly MODERATE
        if surplus > 14_000:
            liquidity_flag = 'ADEQUATE'
        elif surplus > -32_000:
            liquidity_flag = 'MODERATE'
        else:
            liquidity_flag = 'LOW'

        # 4. bureau_risk_flag
        # External CIBIL: HIGH=5.0%, MEDIUM=10.3%, LOW=84.7%
        # num_lss (written-off) and num_dbt (doubtful) are strong HIGH signals.
        if written_off >= 1 or doubtful >= 1 or dpd_60 >= 3 or score < 580:
            bureau_risk = 'HIGH'
        elif score < 650 or (dpd_30 >= 2 and cc_util > 0.60):
            bureau_risk = 'MEDIUM'
        else:
            bureau_risk = 'LOW'

        # 5. salary_stability_flag
        # External CIBIL: UNSTABLE=0.04%(tenure<6m), STABLE=11.2%(tenure>=24,score>=700)
        # No salary_missing_months → use employment tenure + score + DPD
        if tenure < 6:
            salary_stability = 'UNSTABLE'
        elif tenure >= 24 and score >= 700 and dpd_30 == 0:
            salary_stability = 'STABLE'
        else:
            salary_stability = 'MODERATE'

    else:
        # ── DATASET A PATH (train_60k / bank-statement enriched) ────
        # Has actual surplus, bounce count, and missing salary months.
        dpd_90      = int(extraction_result.get('dpd_90_count_6m', 0) or 0)
        bounces     = int(extraction_result.get('inward_bounce_count_3m', 0) or 0)
        missing     = int(extraction_result.get('salary_missing_months', 0) or 0)
        hard_reject = int(extraction_result.get('hard_reject_flag', 0) or 0)
        surplus     = float(
            extraction_result.get('net_cash_surplus_6m')
            or extraction_result.get('net_surplus')
            or -50_000
        )

        # 1. payment_discipline_flag
        # train_60k: POOR/MODERATE rows bounce mean ~1.0, GOOD mean=0.008.
        if dpd_90 >= 1 or bounces >= 2:
            payment_discipline = 'POOR'
        elif bounces == 1 or dpd_30 >= 3:
            payment_discipline = 'MODERATE'
        else:
            payment_discipline = 'GOOD'

        # 2. cashflow_health
        # train_60k: HEALTHY min surplus=14k, STABLE min=600, STRESSED max=-1k.
        if surplus >= 14_000:
            cashflow_health = 'HEALTHY'
        elif 600 <= surplus < 14_000:
            cashflow_health = 'STABLE'
        elif surplus < -1_000:
            cashflow_health = 'STRESSED'
        else:
            cashflow_health = 'MODERATE'

        # 3. liquidity_flag
        # train_60k: ADEQUATE median=+83k, MODERATE median=-32k, LOW median=-109k.
        if surplus > 14_000:
            liquidity_flag = 'ADEQUATE'
        elif surplus > -32_000:
            liquidity_flag = 'MODERATE'
        else:
            liquidity_flag = 'LOW'

        # 4. bureau_risk_flag
        # train_60k: HIGH ~99% hard_rejected, dpd_90 mean=6.1; MEDIUM score median=539.
        if hard_reject or dpd_90 >= 3 or written_off >= 1 or (dpd_90 >= 1 and dpd_30 >= 2):
            bureau_risk = 'HIGH'
        elif score < 580 or (dpd_30 >= 2 and cc_util > 0.60):
            bureau_risk = 'MEDIUM'
        else:
            bureau_risk = 'LOW'

        # 5. salary_stability_flag
        # train_60k: UNSTABLE missing>=1, STABLE cv~0.05+zero missing, MODERATE rest.
        if missing >= 1:
            salary_stability = 'UNSTABLE'
        elif missing == 0 and score >= 700 and dpd_30 == 0 and bounces == 0:
            salary_stability = 'STABLE'
        else:
            salary_stability = 'MODERATE'

    return {
        'payment_discipline_flag': payment_discipline,
        'cashflow_health':         cashflow_health,
        'liquidity_flag':          liquidity_flag,
        'bureau_risk_flag':        bureau_risk,
        'salary_stability_flag':   salary_stability,
        '_inference_path':         'bureau_only' if is_bureau_only else 'bank_statement',
        '_surplus_used':           surplus if is_bureau_only else locals().get('surplus', 0),
    }

# =============================================================================
# CIBIL PDF EXTRACTION ENGINE (OCR + PATTERN MATCHING) – OPTIONAL
# =============================================================================
def extract_cibil_from_pdf(uploaded_file):
    if not OCR_AVAILABLE:
        return {'success': False, 'error': OCR_ERROR_MSG or 'OCR libraries not installed. Check packages.txt and requirements.txt.'}

    try:
        pdf_bytes = uploaded_file.read()
        images = convert_from_bytes(pdf_bytes, dpi=300)
        full_text = ""
        for image in images:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            full_text += pytesseract.image_to_string(binary) + "\n"

        credit_score = 720
        score_match = re.search(
            r'\b(\d{3})\s*(?:EXCELLENT|VERY\s*GOOD|GOOD|FAIR|SUBPRIME|POOR|NH|NA)\b',
            full_text, re.IGNORECASE
        )
        if score_match:
            val = int(score_match.group(1))
            if 300 <= val <= 900:
                credit_score = val
        if credit_score == 720:
            score_match2 = re.search(
                r'(?:cibil|credit)\s*score\s*[:\-\(]?\s*(\d{3})',
                full_text, re.IGNORECASE
            )
            if score_match2:
                val = int(score_match2.group(1))
                if 300 <= val <= 900:
                    credit_score = val
        if credit_score == 720:
            score_match3 = re.search(r'score.*?\((\d{3})\)', full_text, re.IGNORECASE)
            if score_match3:
                val = int(score_match3.group(1))
                if 300 <= val <= 900:
                    credit_score = val

        monthly_income = 50000
        income_match = re.search(
            r'(?:net\s+monthly\s+income|monthly\s+income|net\s+income|salary)[^\n\r]{0,30}?'
            r'(?:rs\.?\s*|inr\s*|₹\s*)([\d,]+)',
            full_text, re.IGNORECASE
        )
        if income_match:
            val = int(income_match.group(1).replace(',', ''))
            if val > 1000:
                monthly_income = val
        if monthly_income == 50000:
            income_match2 = re.search(r'(?:rs\.?\s*|₹\s*)([\d,]{4,})', full_text, re.IGNORECASE)
            if income_match2:
                val = int(income_match2.group(1).replace(',', ''))
                if 5000 <= val <= 1000000:
                    monthly_income = val

        cc_util_pct = 35
        util_match = re.search(r'utilization\s*[\(:\-]?\s*(\d{1,3})\s*%', full_text, re.IGNORECASE)
        if util_match:
            cc_util_pct = int(util_match.group(1))
        cc_util = cc_util_pct / 100.0
        high_util = 1 if cc_util_pct > 75 else 0

        age_extracted = 35
        dob_match = re.search(
            r'(?:date\s+of\s+birth|dob)[:\s]+(\d{2}[-/]\w{3,9}[-/]\d{2,4})',
            full_text, re.IGNORECASE
        )
        if dob_match:
            try:
                from datetime import datetime as _dt
                dob_str = dob_match.group(1)
                for fmt in ('%d-%b-%Y', '%d/%b/%Y', '%d-%m-%Y', '%d/%m/%Y'):
                    try:
                        dob = _dt.strptime(dob_str, fmt)
                        age_extracted = int((datetime.now() - dob).days / 365.25)
                        break
                    except Exception:
                        continue
            except Exception:
                pass

        biz_vintage = 3
        biz_match = re.search(r'business\s+vintage.*?(\d+)', full_text, re.IGNORECASE)
        if biz_match:
            biz_vintage = int(biz_match.group(1))

        lines = full_text.split('\n')
        in_accounts = False
        in_enquiry = False
        accounts = []
        enquiry_dates = []

        for line in lines:
            line_up = line.upper()
            if 'ACCOUNT DETAILS' in line_up:
                in_accounts = True
                in_enquiry = False
                continue
            if 'ENQUIRY DETAILS' in line_up:
                in_accounts = False
                in_enquiry = True
                continue

            if in_accounts:
                if re.search(r'SUMMARY|SCORE|PERSONAL\s+INFO', line_up):
                    break
                if re.search(r'\bLender\b|\bAccount\s*No\b|\bOpen\s*Date\b|\bDPD\b|\bStatus\b', line, re.IGNORECASE):
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                dpd_match = re.search(r'\b(0\d0|0\d\d|\d{3})\b', stripped)
                status_match = re.search(
                    r'\b(Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss)\s*$',
                    stripped, re.IGNORECASE
                )
                if (re.search(r'\bINR\b', stripped, re.IGNORECASE) or
                        re.match(r'^[A-Z][a-zA-Z\s]+(?:Bank|Finance|Capital|Fincorp|SBI|ICICI|HDFC|Axis|Bajaj|Tata|Kotak)', stripped)):
                    dpd_val = int(dpd_match.group(1)) if dpd_match else 0
                    status_str = status_match.group(1) if status_match else 'Active'
                    accounts.append({'dpd': dpd_val, 'status': status_str.lower()})

            if in_enquiry:
                enq_date = re.match(r'^\s*(\d{2}-[A-Za-z]{3}-\d{4})', line)
                if enq_date:
                    enquiry_dates.append(enq_date.group(1))

        written_off_count = 0
        settled_count = 0
        dpd_90_count = 0
        dpd_60_count = 0
        dpd_30_count = 0
        active_count = 0
        sub_standard_count = 0

        if accounts:
            for acc in accounts:
                dpd = acc.get('dpd', 0)
                status = acc.get('status', '')
                if dpd >= 90:
                    dpd_90_count += 1
                elif dpd >= 60:
                    dpd_60_count += 1
                elif dpd >= 30:
                    dpd_30_count += 1
                if 'written' in status:
                    written_off_count += 1
                elif 'settled' in status:
                    settled_count += 1
                elif 'active' in status:
                    active_count += 1
                if dpd >= 30:
                    sub_standard_count += 1
        else:
            written_off_count = len(re.findall(r'\bwritten[-\s]?off\b', full_text, re.IGNORECASE))
            settled_count     = len(re.findall(r'\bsettled\b', full_text, re.IGNORECASE))
            dpd_90_count      = len(re.findall(r'\b090\b|90\+?\s*dpd', full_text, re.IGNORECASE))
            dpd_60_count      = len(re.findall(r'\b060\b|60\+?\s*dpd', full_text, re.IGNORECASE))
            dpd_30_count      = len(re.findall(r'\b030\b|30\+?\s*dpd', full_text, re.IGNORECASE))
            active_sum = re.search(r'Total\s+Accounts\s+Active.*?(\d+)\s+(\d+)', full_text, re.IGNORECASE)
            if active_sum:
                active_count = int(active_sum.group(2))

        if active_count == 0:
            summary_match = re.search(
                r'Total\s+Accounts\s+Active[^\n]*\n\s*(\d+)\s+(\d+)',
                full_text, re.IGNORECASE
            )
            if summary_match:
                active_count = int(summary_match.group(2))
            else:
                inline = re.search(
                    r'(?:Total\s+Accounts.*?Active.*?Closed.*?\n|(\d+)\s+(\d+)\s+(\d+)\s+[\d,]+\s+\d+)',
                    full_text, re.IGNORECASE
                )
                if inline and inline.group(2):
                    active_count = int(inline.group(2))

        enq_12m_total = len(enquiry_dates)
        enq_sum_match = re.search(r'Enquiries?\s*\(?12M\)?\s*[:\s]+(\d+)', full_text, re.IGNORECASE)
        if enq_sum_match:
            enq_12m_total = max(enq_12m_total, int(enq_sum_match.group(1)))

        enq_L3m = min(len(enquiry_dates), enq_12m_total)
        enq_L6m = enq_12m_total
        enq_L12m = enq_12m_total

        if credit_score >= 750 and (written_off_count > 0 or dpd_90_count > 0):
            credit_score = 550

        total_accounts = max(len(accounts), active_count + settled_count + written_off_count)
        pct_active = active_count / total_accounts if total_accounts > 0 else 0.6

        # ── Employment tenure extraction ──────────────────────────────
        # Try: "X years Y months at <employer>", "employed since <date>",
        # "total employment X months", or fallback to biz_vintage * 12
        employment_tenure_months = biz_vintage * 12
        tenure_match = re.search(
            r'(?:employed\s+for|employment\s+tenure|with\s+current\s+employer)[^\d]*(\d+)\s*(?:year|yr)',
            full_text, re.IGNORECASE
        )
        if tenure_match:
            employment_tenure_months = int(tenure_match.group(1)) * 12
        else:
            tenure_m = re.search(
                r'(?:employed\s+for|employment\s+tenure)[^\d]*(\d+)\s*month',
                full_text, re.IGNORECASE
            )
            if tenure_m:
                employment_tenure_months = int(tenure_m.group(1))

        # ── Gender / Marital / Education extraction ───────────────────
        gender = 'M'
        if re.search(r'\bfemale\b|\bF\b', full_text, re.IGNORECASE):
            gender = 'F'

        marital_status = 'Married'
        if re.search(r'\bsingle\b|\bunmarried\b', full_text, re.IGNORECASE):
            marital_status = 'Single'

        education = 'GRADUATE'
        for edu_pat, edu_val in [
            (r'post.?grad', 'POST-GRADUATE'),
            (r'professional', 'PROFESSIONAL'),
            (r'under.?grad', 'UNDER GRADUATE'),
            (r'\b12th\b|\bhsc\b', '12TH'),
            (r'\bssc\b|\b10th\b', 'SSC'),
        ]:
            if re.search(edu_pat, full_text, re.IGNORECASE):
                education = edu_val
                break

        # ── Last / first product enquiry ─────────────────────────────
        prod_enq_map = {
            r'personal\s+loan': 'PL',
            r'credit\s+card':   'CC',
            r'home\s+loan':     'HL',
            r'auto\s+loan|car\s+loan': 'AL',
            r'consumer\s+loan': 'ConsumerLoan',
        }
        last_prod_enq = 'others'
        first_prod_enq = 'others'
        for pat, label in prod_enq_map.items():
            if re.search(pat, full_text, re.IGNORECASE):
                last_prod_enq = label
                first_prod_enq = label
                break

        # ── Compute net surplus proxy (since no bank statement in PDF) ─
        # Uses calibrated income-based formula from External_Cibil_Dataset analysis.
        # Available if income was extracted; used by infer_categorical_flags().
        # dpd_60_count is the 60+ DPD proxy for this dataset.
        surplus_proxy = _infer_surplus_from_cibil(
            score=credit_score,
            dpd_60=dpd_60_count,
            dpd_30=dpd_30_count,
            income=float(monthly_income)
        )

        extracted_data = {
            # ── Core Credit Score ──────────────────────────────────────
            'Credit_Score': credit_score,

            # ── Delinquency (External CIBIL naming convention) ─────────
            'max_delinquency_level':    max(dpd_90_count * 90, dpd_60_count * 60, dpd_30_count * 30),
            'max_recent_level_of_deliq': max(dpd_60_count * 60, dpd_30_count * 30),
            'recent_level_of_deliq':    max(dpd_60_count * 60, dpd_30_count * 30),
            'num_times_30p_dpd':        dpd_30_count,
            'num_times_60p_dpd':        dpd_60_count,   # 60+ used as dpd_90 proxy
            'num_times_delinquent':     dpd_30_count + dpd_60_count + dpd_90_count,
            'num_deliq_6mts':           dpd_30_count + dpd_60_count + dpd_90_count,
            'num_deliq_12mts':          dpd_30_count + dpd_60_count + dpd_90_count,
            'num_deliq_6_12mts':        0,
            'max_deliq_6mts':           dpd_90_count if dpd_90_count > 0 else dpd_60_count,
            'max_deliq_12mts':          dpd_90_count if dpd_90_count > 0 else dpd_60_count,

            # ── Account Quality (External CIBIL naming) ────────────────
            'num_std':      active_count,
            'num_std_6mts': active_count,
            'num_std_12mts': active_count,
            'num_sub':      sub_standard_count,
            'num_sub_6mts': sub_standard_count,
            'num_sub_12mts': sub_standard_count,
            'num_dbt':      dpd_90_count,       # doubtful ≈ 90+ DPD proxy
            'num_dbt_6mts': 0,
            'num_dbt_12mts': 0,
            'num_lss':      written_off_count,  # loss/written-off
            'num_lss_6mts': 0,
            'num_lss_12mts': 0,

            # ── Enquiry fields ─────────────────────────────────────────
            'enq_L3m':  enq_L3m,
            'enq_L6m':  enq_L6m,
            'enq_L12m': enq_L12m,
            'tot_enq':  enq_L12m,
            'CC_enq':   0,  'CC_enq_L6m': 0,  'CC_enq_L12m': 0,
            'PL_enq':   0,  'PL_enq_L6m': 0,  'PL_enq_L12m': 0,
            'time_since_recent_enq': 30,

            # ── Utilization ────────────────────────────────────────────
            'pct_of_active_TLs_ever':      round(pct_active, 2),
            'pct_opened_TLs_L6m_of_L12m':  0.3,
            'pct_currentBal_all_TL':        0.3,
            'CC_utilization':               round(cc_util, 2) if cc_util > 0 else -99999,
            'PL_utilization':               0.25,
            'max_unsec_exposure_inPct':     cc_util_pct if cc_util_pct > 0 else 0,
            'pct_PL_enq_L6m_of_L12m':      0.0,
            'pct_CC_enq_L6m_of_L12m':      0.0,
            'pct_PL_enq_L6m_of_ever':      0.0,
            'pct_CC_enq_L6m_of_ever':      0.0,

            # ── Demographics (External CIBIL fields) ───────────────────
            'AGE':                  age_extracted,
            'NETMONTHLYINCOME':     monthly_income,    # ← External CIBIL key (not avg_salary_6m)
            'Time_With_Curr_Empr':  employment_tenure_months,
            'GENDER':               gender,
            'MARITALSTATUS':        marital_status,
            'EDUCATION':            education,

            # ── Product flags ──────────────────────────────────────────
            'CC_Flag': 1 if re.search(r'credit card', full_text, re.IGNORECASE) else 0,
            'PL_Flag': 1 if re.search(r'personal loan', full_text, re.IGNORECASE) else 0,
            'HL_Flag': 1 if re.search(r'home loan', full_text, re.IGNORECASE) else 0,
            'GL_Flag': 1 if re.search(r'gold loan', full_text, re.IGNORECASE) else 0,
            'last_prod_enq2':  last_prod_enq,
            'first_prod_enq2': first_prod_enq,

            # ── Time-since fields (sentinel if no event found) ─────────
            'time_since_recent_payment':     70,
            'time_since_first_deliquency':   -99999 if dpd_30_count == 0 else 180,
            'time_since_recent_deliquency':  -99999 if dpd_30_count == 0 else 90,

            # ── Surplus proxy (for infer_categorical_flags auto-path) ──
            # NOTE: NETMONTHLYINCOME key is set above, which triggers bureau_only path.
            # surplus_proxy stored here for session display only.
            '_surplus_proxy': int(surplus_proxy),

            # ── Legacy / internal fields ───────────────────────────────
            'written_off_count':    written_off_count,   # legacy alias
            'settled_count':        settled_count,
            'high_util_flag':       high_util,
            'dpd_90_count_6m':      dpd_90_count,        # Stage 1 form field name
            'recent_deliq_flag':    1 if (dpd_90_count > 0 or dpd_60_count > 0) else 0,
            'account_quality_score': max(0, 100
                - written_off_count * 20
                - settled_count * 10
                - dpd_90_count * 15
                - dpd_30_count * 5),

            # ── Metadata ──────────────────────────────────────────────
            'raw_text':          full_text,
            'success':           True,
            'extraction_method': 'OCR+ExternalCIBIL',
        }
        return extracted_data
    except Exception as e:
        return {'error': str(e), 'message': f'Error extracting CIBIL data: {str(e)}', 'success': False}

# =============================================================================
# HYBRID DECISION ENGINE
# =============================================================================
def make_hybrid_decision_enhanced(customer_dict):
    fill_missing_ml_fields(customer_dict)

    policy_checks = {}
    age = customer_dict.get('age', 0)
    employment_type = customer_dict.get('employment_type', 'Salaried')
    kyc_verified = customer_dict.get('kyc_verified', True)
    bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
    fraud_flag = customer_dict.get('fraud_flag', False)
    age_min, age_max = 24, 70
    if age < age_min or age > age_max:
        policy_checks['age'] = f"❌ Age {age} (Required: {age_min}-{age_max})"
        return {'decision': "REJECT", 'reason': "Policy Gate: Age outside allowed range", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 100.0, 'affordability_data': {}}
    policy_checks['age'] = f"✅ Age {age} (Valid)"
    if not kyc_verified:
        policy_checks['kyc'] = "❌ KYC Not Verified"
        return {'decision': "REJECT", 'reason': "Policy Gate: KYC verification required", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 100.0, 'affordability_data': {}}
    policy_checks['kyc'] = "✅ KYC Verified"
    if bankruptcy_flag:
        policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
        return {'decision': "REJECT", 'reason': "Policy Gate: Active bankruptcy", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 100.0, 'affordability_data': {}}
    policy_checks['bankruptcy'] = "✅ No Bankruptcy"
    if fraud_flag:
        policy_checks['fraud'] = "❌ Fraud Flag"
        return {'decision': "REJECT", 'reason': "Policy Gate: Fraud detected", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 100.0, 'affordability_data': {}}
    policy_checks['fraud'] = "✅ No Fraud History"

    dependents = customer_dict.get('dependents', 0)
    dependents_flag_review = False
    if dependents > 5:
        policy_checks['dependents'] = f"⚠️ Dependents {dependents} (>5: Review Required)"
        dependents_flag_review = True
    else:
        policy_checks['dependents'] = f"✅ Dependents {dependents} (Acceptable)"

    monthly_income = customer_dict.get('avg_salary_6m', 0)
    employment_tenure = customer_dict.get('employment_tenure_months', 0)
    business_vintage = customer_dict.get('business_vintage_years', 0)
    if monthly_income < 15000:
        policy_checks['income'] = f"❌ Income ₹{monthly_income:,.0f} (Min: ₹15,000)"
        return {'decision': "REJECT", 'reason': "Policy Gate: Income below minimum", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 100.0, 'affordability_data': {}}
    policy_checks['income'] = f"✅ Income ₹{monthly_income:,.0f}"
    if employment_type == 'Salaried' and employment_tenure < 6:
        policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
        return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient tenure", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 100.0, 'affordability_data': {}}
    elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
        policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
        return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient business vintage", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 100.0, 'affordability_data': {}}
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
        return {'decision': "REJECT", 'reason': "Policy Gate: Bureau score too low", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 100.0, 'affordability_data': {}}
    policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"
    dpd_90_flag_review = False
    if dpd_90 > 5:
        policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD (exceeds limit of 5)"
        return {'decision': "REJECT", 'reason': "Policy Gate: Severe delinquency > 5 instances of 90+ DPD",
                'confidence': 0, 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks,
                'risk_score': 0, 'pd_percentage': 100.0, 'affordability_data': {}}
    elif dpd_90 > 1:
        policy_checks['dpd'] = f"⚠️ {dpd_90} instances of 90+ DPD (2–5) → Review required"
        dpd_90_flag_review = True
    else:
        policy_checks['dpd'] = f"✅ {dpd_90} instances of 90+ DPD (acceptable)"
    if credit_utilization > 80:
        policy_checks['utilization'] = f"⚠️ High utilization {credit_utilization}%"
    else:
        policy_checks['utilization'] = f"✅ Utilization {credit_utilization}%"
    if recent_inquiries > 5:
        policy_checks['inquiries'] = f"⚠️ {recent_inquiries} recent inquiries"
    else:
        policy_checks['inquiries'] = f"✅ {recent_inquiries} inquiries"

    active_loans = customer_dict.get('active_loans_count', 0)
    if active_loans >= 5:
        policy_checks['active_loans'] = f"⚠️ High active loans ({int(active_loans)}) — Review"
        active_loans_flag = True
    else:
        policy_checks['active_loans'] = f"✅ Active loans: {int(active_loans)}"
        active_loans_flag = False

    salary_stability = customer_dict.get('salary_stability_flag', 'STABLE')
    if salary_stability == 'UNSTABLE':
        policy_checks['salary'] = "⚠️ Unstable salary — Review required"
        salary_flag = True
    elif salary_stability == 'MODERATE':
        policy_checks['salary'] = "⚠️ Moderate salary stability"
        salary_flag = False
    else:
        policy_checks['salary'] = "✅ Stable salary"
        salary_flag = False

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
    ml_raw_decision = ml_decision
    try:
        pred_proba = MODEL.predict_proba(final_input)[0]
        confidence = max(pred_proba) * 100
        class_probs = {cls: prob * 100 for cls, prob in zip(TARGET_LE.classes_, pred_proba)}
    except Exception:
        confidence = 75.0
        class_probs = {ml_decision: 100.0}

    loan_amount = customer_dict.get('loan_amount', 0)
    loan_tenure = customer_dict.get('loan_tenure_months', 12)
    interest_rate = customer_dict.get('interest_rate', 10.5)
    existing_emi = customer_dict.get('existing_emi', 0)
    affordability_data = calculate_affordability(monthly_income, loan_amount, interest_rate, loan_tenure, existing_emi)
    foir = affordability_data['foir_percentage']

    if foir > 50:
        ml_decision = "REJECT"
        policy_checks['foir'] = f"❌ FOIR {foir:.1f}% exceeds maximum allowed (50%)"

    if dependents_flag_review and ml_decision == "APPROVE":
        ml_decision = "REVIEW"
    if active_loans_flag and ml_decision == "APPROVE":
        ml_decision = "REVIEW"
    if salary_flag and ml_decision == "APPROVE":
        ml_decision = "REVIEW"
    if dpd_90_flag_review and ml_decision == "APPROVE":
        ml_decision = "REVIEW"

    risk_score = calculate_final_risk_score(
        bureau_score=bureau_score,
        ml_confidence=confidence,
        foir=foir,
        dpd_90=dpd_90,
        dpd_30=customer_dict.get('dpd_30_count_6m', 0),
        net_surplus=customer_dict.get('net_cash_surplus_6m', 0),
        bounces=customer_dict.get('inward_bounce_count_3m', 0),
        missing_months=customer_dict.get('salary_missing_months', 0),
        active_loans=active_loans
    )

    pd_percentage = calculate_final_pd(
        bureau_score=bureau_score,
        foir=foir,
        confidence=confidence,
        dpd_90_count=dpd_90,
        dpd_30_count=customer_dict.get('dpd_30_count_6m', 0),
        employment_type=employment_type,
        employment_tenure=employment_tenure,
        business_vintage=business_vintage,
        recent_inquiries=recent_inquiries,
        ml_decision=ml_decision
    )

    return {
        'decision': ml_decision,
        'ml_raw_decision': ml_raw_decision,
        'reason': "Decision based on comprehensive assessment",
        'confidence': confidence,
        'class_probs': class_probs,
        'policy_checks': policy_checks,
        'risk_score': risk_score,
        'pd_percentage': round(pd_percentage, 2),
        'affordability_data': affordability_data
    }

# =============================================================================
# BATCH PREDICTION ENGINE
# =============================================================================
def process_batch_predictions(df):
    results = []
    for idx, row in df.iterrows():
        customer_dict = row.to_dict()
        for key, value in customer_dict.items():
            if isinstance(value, str):
                if value.lower() in ['yes', 'true', '1']:
                    customer_dict[key] = True
                elif value.lower() in ['no', 'false', '0']:
                    customer_dict[key] = False
        required_fields = {
            'age': 35, 'employment_type': 'Salaried', 'kyc_verified': True,
            'bankruptcy_flag': False, 'fraud_flag': False, 'employment_tenure_months': 24,
            'business_vintage_years': 0, 'bureau_score': 700, 'dpd_90_count_6m': 0,
            'dpd_30_count_6m': 0, 'credit_utilization_pct': 30, 'recent_inquiries_3m': 0,
            'active_loans_count': 0, 'existing_emi': 0, 'avg_salary_6m': 50000,
            'AMT_INCOME_TOTAL': 600000, 'net_cash_surplus_6m': 20000, 'salary_stability_flag': 'STABLE',
            'loan_amount': 180000, 'loan_tenure_months': 24, 'interest_rate': 10.5, 'AMT_ANNUITY': 8500,
            'dependents': 0,
            'payment_discipline_flag': 'GOOD',
            'liquidity_flag': 'LOW',
            'cashflow_health': 'MODERATE',
            'bureau_risk_flag': 'LOW',
            'inward_bounce_count_3m': 0,
            'salary_missing_months': 0,
        }
        for field, default in required_fields.items():
            if field not in customer_dict or pd.isna(customer_dict[field]):
                customer_dict[field] = default
        try:
            decision_data = make_hybrid_decision_enhanced(customer_dict)
            reasons = generate_reason_codes(
                decision=decision_data.get('decision', 'ERROR'),
                customer_data=customer_dict,
                affordability_data=decision_data.get('affordability_data', {}),
                policy_checks=decision_data.get('policy_checks', {})
            )
            app_id = f"BATCH_{idx+1:04d}"
            affordability = decision_data.get('affordability_data', {})
            result = {
                'application_id': app_id,
                'decision': decision_data.get('decision', 'ERROR'),
                'risk_score': decision_data.get('risk_score', 0),
                'pd_percentage': decision_data.get('pd_percentage', 0),
                'confidence': round(decision_data.get('confidence', 0), 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'reason_1': reasons[0] if len(reasons) > 0 else '',
                'reason_2': reasons[1] if len(reasons) > 1 else '',
                'reason_3': reasons[2] if len(reasons) > 2 else '',
                'age': customer_dict.get('age', ''),
                'employment_type': customer_dict.get('employment_type', ''),
                'bureau_score': customer_dict.get('bureau_score', ''),
                'monthly_income': customer_dict.get('avg_salary_6m', ''),
                'loan_amount': customer_dict.get('loan_amount', ''),
                'loan_tenure_months': customer_dict.get('loan_tenure_months', ''),
                'interest_rate': customer_dict.get('interest_rate', ''),
                'new_emi': affordability.get('new_emi', 0),
                'existing_emi': affordability.get('existing_emi', 0),
                'total_emi': affordability.get('total_emi', 0),
                'foir_percentage': round(affordability.get('foir_percentage', 0), 2),
                'net_disposable': affordability.get('net_disposable', 0),
                'affordability_status': affordability.get('status', 'N/A'),
                'dpd_90_count': customer_dict.get('dpd_90_count_6m', 0),
                'dpd_30_count': customer_dict.get('dpd_30_count_6m', 0),
                'credit_utilization': customer_dict.get('credit_utilization_pct', 0),
                'recent_inquiries': customer_dict.get('recent_inquiries_3m', 0),
                'active_loans': customer_dict.get('active_loans_count', 0),
                'employment_tenure': customer_dict.get('employment_tenure_months', 0),
                'business_vintage': customer_dict.get('business_vintage_years', 0),
                'salary_stability': customer_dict.get('salary_stability_flag', ''),
                'kyc_status': 'Verified' if customer_dict.get('kyc_verified', True) else 'Not Verified',
                'bankruptcy': 'Yes' if customer_dict.get('bankruptcy_flag', False) else 'No',
                'fraud': 'Yes' if customer_dict.get('fraud_flag', False) else 'No',
                'prob_approve': round(decision_data.get('class_probs', {}).get('APPROVE', 0), 2),
                'prob_review': round(decision_data.get('class_probs', {}).get('REVIEW', 0), 2),
                'prob_reject': round(decision_data.get('class_probs', {}).get('REJECT', 0), 2),
            }
        except Exception as e:
            result = {
                'application_id': f"BATCH_{idx+1:04d}",
                'decision': 'ERROR',
                'risk_score': 0, 'pd_percentage': 0, 'confidence': 0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'reason_1': '', 'reason_2': '', 'reason_3': '',
                'age': customer_dict.get('age', ''),
                'employment_type': customer_dict.get('employment_type', ''),
                'bureau_score': customer_dict.get('bureau_score', ''),
                'monthly_income': customer_dict.get('avg_salary_6m', ''),
                'loan_amount': customer_dict.get('loan_amount', ''),
                'error_message': str(e)
            }
        results.append(result)
    return pd.DataFrame(results)

def create_download_link(df, filename="batch_results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;">📥 Download CSV</a>'

# =============================================================================
# MODERN UI COMPONENTS
# =============================================================================
def render_decision_header(decision_data, customer_data):
    decision = decision_data.get('decision', 'ERROR')
    risk_score = decision_data.get('risk_score', 0)
    pd_score = decision_data.get('pd_percentage', 0)
    approved_amount = customer_data.get('loan_amount', 0)
    tenure = customer_data.get('loan_tenure_months', 24)
    app_id = customer_data.get('application_id', 'N/A')
    timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
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
            <div class="decision-title"><span>{icon}</span><span>{decision}</span></div>
            <div class="decision-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{risk_score}</div><div class="stat-label">Risk Score</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{pd_score}%</div><div class="stat-label">PD Score</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">₹{approved_amount:,.0f}</div><div class="stat-label">Loan Amount</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{tenure}</div><div class="stat-label">Tenure (Months)</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{decision_data.get("confidence", 0):.0f}%</div><div class="stat-label">Confidence</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="info-box"><strong>📋 Application ID:</strong> {app_id}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="info-box"><strong>🕐 Decision Timestamp:</strong> {timestamp}</div>', unsafe_allow_html=True)

def render_info_card(title, icon, data_dict, status_dict=None):
    st.markdown(f'<div class="info-card"><div class="info-card-title"><span class="icon">{icon}</span><span>{title}</span></div><div class="info-card-content">', unsafe_allow_html=True)
    for label, value in data_dict.items():
        status = ""
        if status_dict and label in status_dict:
            if status_dict[label] == "pass":
                status = '<span class="status-badge badge-pass">✓ Passed</span>'
            elif status_dict[label] == "fail":
                status = '<span class="status-badge badge-fail">✗ Failed</span>'
            elif status_dict[label] == "warning":
                status = '<span class="status-badge badge-warning">⚠ Warning</span>'
        st.markdown(f'<div class="data-row"><span class="data-label">{label}</span><span class="data-value">{value} {status}</span></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

def render_reason_codes(reasons):
    st.markdown('<div class="info-card"><div class="info-card-title"><span class="icon">📝</span><span>Decision Reasons</span></div><div class="info-card-content">', unsafe_allow_html=True)
    for i, reason in enumerate(reasons, 1):
        st.markdown(f'<div class="reason-item"><span class="reason-icon">{i}.</span><span>{reason}</span></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

def create_modern_gauge(value, title, max_value=100):
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
            'bgcolor': 'white', 'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': '#fed7d7'},
                {'range': [50, 75], 'color': '#feebc8'},
                {'range': [75, 100], 'color': '#c6f6d5'}
            ]
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white',
                      font={'family': 'Inter', 'color': '#2d3748'})
    return fig

def create_modern_bar_chart(class_probs):
    df = pd.DataFrame({'Decision': list(class_probs.keys()), 'Probability': list(class_probs.values())})
    colors = {'REVIEW': '#ed8936', 'APPROVE': '#48bb78', 'REJECT': '#f56565'}
    fig = px.bar(df, x='Decision', y='Probability', title='Decision Probabilities', color='Decision',
                 color_discrete_map=colors, text='Probability')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
    fig.update_layout(
        showlegend=False, yaxis_title='Probability (%)', xaxis_title='', height=300,
        margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white', plot_bgcolor='white',
        font={'family': 'Inter', 'color': '#2d3748'},
        yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]},
        xaxis={'tickfont': {'size': 14, 'color': '#2d3748'}}
    )
    return fig

# =============================================================================
# STAGE 2 BINARY RESOLVER
# =============================================================================
def resolve_stage2_to_binary(stage2_result: dict) -> dict:
    result = stage2_result.copy()
    tier   = result.get('stage2_tier', '')
    raw    = result.get('final_decision', '')
    score  = result.get('combined_risk_score', 0) or 0

    TIER_TO_DECISION = {
        'P1': 'APPROVE',
        'P2': 'APPROVE',
        'P3': 'REJECT',
        'P4': 'REJECT',
    }

    if raw == 'REJECT':
        result['final_decision'] = 'REJECT'
    elif raw == 'APPROVE':
        if tier in TIER_TO_DECISION:
            result['final_decision'] = TIER_TO_DECISION[tier]
        else:
            result['final_decision'] = 'APPROVE'
    else:
        if tier in TIER_TO_DECISION:
            result['final_decision'] = TIER_TO_DECISION[tier]
            result['reason'] = (
                result.get('reason', '') +
                f" [REVIEW resolved to {TIER_TO_DECISION[tier]} via risk tier {tier}]"
            )
        else:
            resolved = 'APPROVE' if score >= 600 else 'REJECT'
            result['final_decision'] = resolved
            result['reason'] = (
                result.get('reason', '') +
                f" [REVIEW resolved to {resolved} via combined risk score {score}]"
            )

    if result['final_decision'] == 'APPROVE':
        result.setdefault('interest_rate_range',
            {'P1': '9.5% – 11%', 'P2': '11% – 13%'}.get(tier, '11% – 14%'))
    else:
        result['interest_rate_range'] = 'N/A — Rejected'

    return result

# =============================================================================
# STAGE 2 RESULTS DISPLAY
# =============================================================================
def display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data):
    st.markdown("---")
    st.markdown('<p class="main-header">🎯 Stage 2 Final Results</p>', unsafe_allow_html=True)

    final_decision = stage2_result.get('final_decision', 'ERROR')
    interest_range = stage2_result.get('interest_rate_range', 'N/A')
    stage2_tier = stage2_result.get('stage2_tier', 'N/A')
    stage2_confidence = stage2_result.get('stage2_confidence', 0)
    combined_risk_score = stage2_result.get('combined_risk_score', 0)

    if final_decision == "APPROVE":
        card_class = "decision-card decision-card-approved"
        icon = "✓"
        subtitle = "✅ Final Decision: Approved — Proceed to Disbursement"
    else:
        card_class = "decision-card decision-card-rejected"
        icon = "✗"
        subtitle = "❌ Final Decision: Rejected — Application Declined"

    st.markdown(f"""
        <div class="{card_class}">
            <div class="decision-title"><span>{icon}</span><span>{final_decision}</span></div>
            <div class="decision-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Risk Tier", stage2_tier)
    with col2:
        st.metric("Interest Rate", interest_range)
    with col3:
        st.metric("Combined Risk Score", combined_risk_score)
    with col4:
        confidence_display = f"{stage2_confidence:.1f}%" if stage2_confidence is not None else "N/A"
        st.metric("Stage 2 Confidence", confidence_display)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔬 Analysis", "📋 Data", "📥 Download"])

    with tab1:
        st.markdown("### 📊 Decision Comparison")
        s1_dec   = st.session_state.get('stage1_decision', 'N/A')
        s2_label = "✅ APPROVE" if final_decision == "APPROVE" else "❌ REJECT"
        comparison_df = pd.DataFrame([
            {'Stage': 'Stage 1 (Screening)', 'Decision': s1_dec,
             'Risk Score': stage1_data.get('risk_score', 'N/A'), 'Tier': 'N/A',
             'Note': 'APPROVE / REVIEW → proceed to Stage 2'},
            {'Stage': 'Stage 2 — FINAL (CIBIL Deep)', 'Decision': s2_label,
             'Risk Score': combined_risk_score, 'Tier': f"{stage2_tier} | {interest_range}",
             'Note': 'Binding final decision'}
        ])
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("### 🎯 Risk Tier Details")
        tier_info = {
            'P1': {'name': 'Premium  → APPROVED',  'color': '#10B981',
                   'desc': 'Excellent credit profile — lowest interest rate band'},
            'P2': {'name': 'Standard → APPROVED',  'color': '#3B82F6',
                   'desc': 'Good credit profile — standard interest rate band'},
            'P3': {'name': 'Subprime → REJECTED',  'color': '#F59E0B',
                   'desc': 'Fair credit with elevated risk — application declined'},
            'P4': {'name': 'High Risk → REJECTED', 'color': '#EF4444',
                   'desc': 'High risk profile — application declined'},
        }
        if stage2_tier in tier_info:
            tier_data = tier_info[stage2_tier]
            st.markdown(f"""
                <div style="background: {tier_data['color']}; color: white; padding: 1rem; border-radius: 0.5rem;">
                    <h3 style="margin: 0; color: white;">{stage2_tier}: {tier_data['name']}</h3>
                    <p style="margin: 0.5rem 0;">Interest Rate: {interest_range}</p>
                    <p style="margin: 0;">{tier_data['desc']}</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("### 📝 Decision Reasoning")
        st.info(stage2_result.get('reason', 'N/A'))

    with tab2:
        st.markdown("### 🔬 Detailed Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Tier Probabilities**")
            if 'tier_probabilities' in stage2_result:
                for tier, prob in stage2_result['tier_probabilities'].items():
                    st.metric(tier, f"{prob:.1f}%")
        with col2:
            st.markdown("**Stage Scores**")
            st.metric("Stage 1 Risk Score", stage1_data.get('risk_score', 'N/A'))
            st.metric("Stage 2 Risk Score", stage2_result.get('stage2_risk_score', 'N/A'))
            st.metric("Combined Score", combined_risk_score)
        with st.expander("📋 Complete Stage 2 Result"):
            st.json(stage2_result)

    with tab3:
        st.markdown("### 📋 Input Data")
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Stage 1 Customer Data"):
                st.json(stage1_customer)
        with col2:
            with st.expander("Enhanced CIBIL Data"):
                st.json(enhanced_customer_data)

    with tab4:
        st.markdown("### 📥 Download Reports")
        bureau_score = stage1_customer.get('bureau_score', 0)
        dpd_90 = stage1_customer.get('dpd_90_count_6m', 0)
        dpd_30 = stage1_customer.get('dpd_30_count_6m', 0)
        foir = stage1_data.get('affordability_data', {}).get('foir_percentage', 0)
        employment_type = stage1_customer.get('employment_type', 'Salaried')
        employment_tenure = stage1_customer.get('employment_tenure_months', 0)
        business_vintage = stage1_customer.get('business_vintage_years', 0)
        ml_decision = stage1_data.get('decision', 'ERROR')
        confidence = stage1_data.get('confidence', 0)

        def _safe(v, default='N/A'):
            return v if v is not None else default

        report_data = {
            'application_id': _safe(stage1_customer.get('application_id'), 'N/A'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'decision': _safe(stage1_data.get('decision'), 'N/A'),
            'risk_score': _safe(stage1_data.get('risk_score'), 0),
            'pd_percentage': _safe(stage1_data.get('pd_percentage'), 0),
            'confidence': _safe(stage1_data.get('confidence'), 0),
            'policy_checks': stage1_data.get('policy_checks', {}),
            'affordability_data': stage1_data.get('affordability_data', {}),
            'customer_data': stage1_customer,
            'reason_codes': stage1_customer.get('reason_codes', []),
            'pd_calculation_factors': {
                'bureau_score': bureau_score,
                'base_pd': bureau_score_to_pd(bureau_score),
                'dpd_90': dpd_90, 'dpd_30': dpd_30,
                'delinquency_multiplier': delinquency_to_pd_multiplier(dpd_90, dpd_30),
                'foir': foir,
                'foir_adjustment': foir_to_pd_adjustment(foir),
                'employment_adjustment': employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage),
                'ml_adjustment': ml_confidence_to_pd_adjustment(confidence, ml_decision),
                'final_pd': stage1_data.get('pd_percentage', 0)
            },
            'stage2_final_decision': _safe(final_decision, 'N/A'),
            'stage2_tier': _safe(stage2_tier, 'N/A'),
            'stage2_interest_range': _safe(interest_range, 'N/A'),
            'stage2_combined_risk_score': _safe(combined_risk_score, 0),
            'stage2_confidence': _safe(stage2_confidence, 0),
            'stage2_reason': _safe(stage2_result.get('reason'), 'N/A'),
            'stage2_tier_probabilities': stage2_result.get('tier_probabilities') or {},
            'stage2_complete_analysis': stage2_result,
            'stage1_data': stage1_data,
            'enhanced_customer_data': enhanced_customer_data
        }

        if PDF_AVAILABLE and generate_audit_pdf is not None:
            try:
                pdf_buffer = generate_audit_pdf(report_data)
                st.download_button(
                    "📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"stage2_report_{stage1_customer.get('application_id', 'unknown')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF generation failed: {str(e)}")
        else:
            st.warning("PDF generation is not available. Please install the required PDF generator module.")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 New Assessment", key="new_assessment_stage2", use_container_width=True):
            st.session_state.stage1_complete = False
            st.session_state.stage1_decision = None
            st.session_state.stage1_data = None
            st.session_state.current_customer_data = None
            st.session_state.page_navigation = "👤 Assessment"
            st.rerun()
    with col2:
        if st.button("← Back to Stage 1", key="back_to_stage1", use_container_width=True):
            st.session_state.page_navigation = "👤 Assessment"
            st.rerun()
    with col3:
        if st.button("🏠 Home", key="home_stage2", use_container_width=True):
            st.session_state.page_navigation = "🏠 Home"
            st.rerun()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("# 🏦 Credit Risk Engine")
    st.markdown("---")

    navigation_options = ["🏠 Home", "👤 Assessment", "📊 Batch Process", "📈 Model Info", "ℹ️ About"]

    if (st.session_state.stage1_complete and
            st.session_state.stage1_decision in ['APPROVE', 'REVIEW']):
        navigation_options.insert(2, "🔬 Stage 2 Analysis")
        st.success(f"✅ Stage 1: {st.session_state.stage1_decision}")
        st.info("🔬 Stage 2 Analysis unlocked!")
    elif st.session_state.stage1_complete:
        st.warning(f"⚠️ Stage 1: {st.session_state.stage1_decision}")
        st.caption("Stage 2 only for APPROVE/REVIEW")

    page = st.radio(
        "**Navigation**",
        navigation_options,
        label_visibility="collapsed",
        key="page_navigation"
    )

    st.markdown("---")

    stage2_indicator = '✅ Active' if STAGE2_AVAILABLE and is_stage2_available() else '❌ Inactive'
    ocr_indicator = '✅ Ready' if OCR_AVAILABLE else '❌ Not Installed'
    if not OCR_AVAILABLE and OCR_ERROR_MSG:
        ocr_indicator += ' ⚠️'
    pdf_indicator = '✅ Ready' if PDF_AVAILABLE else '❌ Not Installed'

    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">System Status</div>
        <div class="info-card-content">
            <div class="data-row"><span class="data-label">Model</span><span class="data-value">✅ Loaded</span></div>
            <div class="data-row"><span class="data-label">Version</span><span class="data-value">8.4</span></div>
            <div class="data-row"><span class="data-label">Stage 2</span><span class="data-value">{stage2_indicator}</span></div>
            <div class="data-row"><span class="data-label">OCR</span><span class="data-value">{ocr_indicator}</span></div>
            <div class="data-row"><span class="data-label">PDF Gen</span><span class="data-value">{pdf_indicator}</span></div>
            <div class="data-row"><span class="data-label">Features</span><span class="data-value">{len(TOP_FEATURES)}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🎯 **Top Features**"):
        for i, feat in enumerate(TOP_FEATURES[:5], 1):
            st.markdown(f"`{i}.` {feat}")

    if st.session_state.stage1_complete:
        st.markdown("---")
        st.markdown("### 🚀 Quick Actions")
        if st.button("🔄 New Assessment", key="new_assessment_sidebar", use_container_width=True):
            st.session_state.stage1_complete = False
            st.session_state.stage1_decision = None
            st.session_state.stage1_data = None
            st.session_state.current_customer_data = None
            st.session_state.extracted_cibil_data = None
            st.rerun()

# =============================================================================
# PAGE ROUTING
# =============================================================================

if page == "🏠 Home":
    st.markdown('<p class="main-header">Credit Risk Engine</p>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-box">
            <h3 style="margin-top: 0;">🎯 AI-Powered Lending Decisions</h3>
            <p style="margin-bottom: 0;">Comprehensive credit risk evaluation combining hard policy rules,
            machine learning models, and affordability analysis for accurate lending decisions.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="info-card"><div class="info-card-title"><span class="icon">🛡️</span><span>Policy Gates</span></div>
            <div class="info-card-content"><ul><li>Age & KYC verification</li><li>Employment stability</li>
            <li>Minimum income checks</li><li>Credit bureau thresholds</li><li>Bankruptcy & fraud detection</li></ul></div></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="info-card"><div class="info-card-title"><span class="icon">🤖</span><span>ML Assessment</span></div>
            <div class="info-card-content"><ul><li>Random Forest classifier</li><li>60K+ training samples</li>
            <li>Confidence scoring</li><li>Multi-class prediction</li><li>Feature importance</li></ul></div></div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="info-card"><div class="info-card-title"><span class="icon">💰</span><span>Affordability</span></div>
            <div class="info-card-content"><ul><li>EMI calculation</li><li>FOIR analysis (max 50%)</li>
            <li>Net disposable income</li><li>Debt burden assessment</li><li>Affordability scoring</li></ul></div></div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🎯 Accuracy", "85%", "+2%")
    with col2: st.metric("⚡ Avg Response", "1.2s", "-0.3s")
    with col3: st.metric("📊 Features", len(TOP_FEATURES))
    with col4: st.metric("🔄 Version", "8.4", "Latest")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="warning-box">
            <strong>🆕 New in Version 8.4:</strong><br>
            • OCR Auto-fill Fix: All 5 categorical dropdowns now update correctly from PDF<br>
            • Payment Discipline inferred from DPD + bounce data (60K dataset calibrated)<br>
            • Cashflow Health inferred from net cash surplus thresholds<br>
            • Liquidity Flag inferred from net cash surplus<br>
            • Bureau Risk Flag inferred from score + DPD + hard-reject signals<br>
            • Salary Stability now uses data-driven inference (not hardcoded STABLE)
        </div>
    """, unsafe_allow_html=True)

elif page == "👤 Assessment":
    st.markdown('<p class="main-header">Credit Assessment</p>', unsafe_allow_html=True)

    pdf_just_extracted = st.session_state.get('pdf_just_extracted', False)

    with st.expander("📄 Upload CIBIL PDF to auto‑fill bureau fields",
                     expanded=pdf_just_extracted or not st.session_state.get('pdf_bureau_score')):

        if pdf_just_extracted:
            ex = st.session_state.get('_last_extraction', {})
            st.success("✅ CIBIL data extracted — form fields below have been updated automatically.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Credit Score",    ex.get('Credit_Score', '—'))
            c2.metric("Monthly Income",  f"₹{ex.get('NETMONTHLYINCOME') or ex.get('avg_salary_6m', 0):,}")
            c3.metric("DPD 60+ Count",   ex.get('num_times_60p_dpd', 0))
            c4.metric("CC Utilization",  f"{max(0, float(ex.get('CC_utilization', 0) or 0))*100:.0f}%")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("DPD 30+ Count",  ex.get('num_times_30p_dpd', 0))
            c2.metric("Inquiries (3M)", ex.get('enq_L3m', 0))
            c3.metric("Active Accounts", ex.get('num_std', 0))
            c4.metric("Written-Off",    ex.get('num_lss', ex.get('written_off_count', 0)))
            # Show surplus proxy and inference path
            _inf_path = ex.get('_surplus_proxy', None)
            if _inf_path is not None:
                surplus_val = ex.get('_surplus_proxy', 0)
                st.info(f"💡 **Bureau-only PDF** — net surplus estimated from income: ₹{surplus_val:,} "
                        f"(no bank statement in CIBIL report). Used for cashflow/liquidity inference.")
            if ex.get('written_off_count', 0) > 0 or ex.get('settled_count', 0) > 0:
                st.warning(f"⚠️ Severe negatives detected: "
                           f"{ex.get('written_off_count', 0)} written-off, "
                           f"{ex.get('settled_count', 0)} settled accounts. "
                           f"Score overridden to {ex.get('Credit_Score', '?')}.")

            # ── FIX v8.4: Show inferred categorical flags in summary ──
            _inf = st.session_state.get('_last_inferred_flags', {})
            if _inf:
                st.markdown("**📊 Inferred Categorical Flags (from CIBIL data):**")
                fc1, fc2, fc3, fc4, fc5 = st.columns(5)
                fc1.metric("Payment Discipline", _inf.get('payment_discipline_flag', '—'))
                fc2.metric("Cashflow Health",    _inf.get('cashflow_health', '—'))
                fc3.metric("Liquidity",          _inf.get('liquidity_flag', '—'))
                fc4.metric("Bureau Risk",         _inf.get('bureau_risk_flag', '—'))
                fc5.metric("Salary Stability",   _inf.get('salary_stability_flag', '—'))

            if st.session_state.get('stage1_complete') and st.session_state.get('current_customer_data'):
                app_id_s1 = st.session_state.current_customer_data.get('application_id', 'Pending submission')
                st.markdown(f"""
                    <div style="background:#1e3a5f;color:white;padding:0.5rem 1rem;border-radius:0.4rem;margin-bottom:0.5rem;font-size:0.9rem;">
                        <strong>📋 Application ID:</strong> {app_id_s1}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("No active assessment. Please submit the form below.")
            if st.toggle("📋 Show full extracted JSON"):
                st.json({k: v for k, v in ex.items() if k != 'raw_text'})
            st.markdown("---")
            if st.button("🔄 Upload a different PDF", key="reset_pdf"):
                st.session_state.pdf_just_extracted = False
                st.session_state.pop('_last_extraction', None)
                st.session_state.pop('_last_inferred_flags', None)
                st.rerun()
        else:
            st.markdown('<div class="info-box">💡 Complete the form below or upload a CIBIL PDF to auto‑fill bureau data.</div>', unsafe_allow_html=True)
            uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="assessment_pdf")
            if uploaded_pdf is not None:
                st.info(f"📄 File ready: **{uploaded_pdf.name}** ({uploaded_pdf.size/1024:.1f} KB)")
                if st.button("🔍 Extract & Auto-fill Form", key="extract_assessment", type="primary", use_container_width=True):
                    with st.spinner("🔄 Running OCR on CIBIL PDF — this takes 10-30 seconds..."):
                        extraction_result = extract_cibil_from_pdf(uploaded_pdf)
                    if extraction_result.get('success', False):
                        st.session_state.pdf_age               = int(extraction_result.get('AGE', 35))
                        st.session_state.pdf_employment_type   = 'Salaried'
                        st.session_state.pdf_kyc               = True
                        st.session_state.pdf_bankruptcy        = False
                        st.session_state.pdf_fraud             = False
                        st.session_state.pdf_bureau_score      = int(extraction_result.get('Credit_Score', 720))
                        st.session_state.pdf_dpd_90            = int(extraction_result.get('dpd_90_count_6m', 0))
                        st.session_state.pdf_dpd_30            = int(extraction_result.get('num_times_30p_dpd', 0))
                        st.session_state.pdf_credit_util       = int(max(0, float(extraction_result.get('CC_utilization', 0) or 0)) * 100)
                        st.session_state.pdf_inquiries         = int(extraction_result.get('enq_L3m', 2))
                        st.session_state.pdf_active_loans      = int(extraction_result.get('num_std', 1))
                        st.session_state.pdf_existing_emi      = int(extraction_result.get('existing_emi', 15000))
                        # ── Income: External CIBIL uses NETMONTHLYINCOME (median ₹23k) ──
                        # train_60k uses avg_salary_6m (median ₹50k). Use whichever is present.
                        _income = int(
                            extraction_result.get('NETMONTHLYINCOME')
                            or extraction_result.get('avg_salary_6m')
                            or 50000
                        )
                        st.session_state.pdf_monthly_income    = _income
                        st.session_state.pdf_annual_income     = _income * 12
                        # ── Net surplus: use proxy if bureau-only, else actual ─────────
                        _surplus = int(
                            extraction_result.get('net_cash_surplus_6m')
                            or extraction_result.get('net_surplus')
                            or extraction_result.get('_surplus_proxy')
                            or 20000
                        )
                        st.session_state.pdf_net_surplus       = _surplus
                        st.session_state.pdf_loan_amount       = int(extraction_result.get('loan_amount', 180000))
                        st.session_state.pdf_loan_tenure       = int(extraction_result.get('loan_tenure', 24))
                        st.session_state.pdf_interest_rate     = float(extraction_result.get('interest_rate', 10.5))
                        st.session_state.pdf_amt_annuity       = int(extraction_result.get('amt_annuity', 8500))
                        st.session_state.pdf_employment_tenure = int(extraction_result.get('Time_With_Curr_Empr', 24))
                        st.session_state.pdf_business_vintage  = int(extraction_result.get('business_vintage', 3))
                        st.session_state.pdf_dependents        = int(extraction_result.get('dependents', 2))

                        # ── FIX v8.4: Infer all 5 categorical flags from CIBIL data ──
                        _inferred = infer_categorical_flags(extraction_result)
                        st.session_state.pdf_salary_stability   = _inferred['salary_stability_flag']
                        st.session_state.pdf_payment_discipline = _inferred['payment_discipline_flag']
                        st.session_state.pdf_cashflow_health    = _inferred['cashflow_health']
                        st.session_state.pdf_liquidity_flag     = _inferred['liquidity_flag']
                        st.session_state.pdf_bureau_risk_flag   = _inferred['bureau_risk_flag']
                        # Store inferred flags for display in the summary banner
                        st.session_state._last_inferred_flags   = _inferred

                        st.session_state.pdf_just_extracted    = True
                        st.session_state._last_extraction      = extraction_result
                        st.rerun()
                    else:
                        st.error(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")
                        st.info("Tip: Make sure Tesseract and Poppler are installed and paths are set correctly.")

    with st.form("assessment_form"):
        st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input(
                "Age", 24, 70,
                value=int(st.session_state.get('pdf_age', 35)),
                help="Customer's age in years (Minimum: 24, Maximum: 70)"
            )
            employment_type = st.selectbox(
                "Employment Type",
                ['Salaried', 'Self-Employed', 'Business'],
                index=['Salaried', 'Self-Employed', 'Business'].index(
                    st.session_state.get('pdf_employment_type', 'Salaried')
                )
            )
        with col2:
            dependents = st.number_input(
                "Number of Dependents", 0, 20,
                value=int(st.session_state.get('pdf_dependents', 2)),
                help="1-5: Approve eligible | >5: Review required"
            )
            kyc_verified = st.selectbox(
                "KYC Verified", ['Yes', 'No'],
                index=0 if st.session_state.get('pdf_kyc', True) else 1
            ) == 'Yes'
        with col3:
            bankruptcy_flag = st.selectbox(
                "Bankruptcy Flag", ['No', 'Yes'],
                index=0 if not st.session_state.get('pdf_bankruptcy', False) else 1
            ) == 'Yes'
            fraud_flag = st.selectbox(
                "Fraud Flag", ['No', 'Yes'],
                index=0 if not st.session_state.get('pdf_fraud', False) else 1
            ) == 'Yes'
            if employment_type == 'Salaried':
                employment_tenure = st.number_input(
                    "Employment Tenure (months)", 0, 600,
                    value=int(st.session_state.get('pdf_employment_tenure', 24))
                )
                business_vintage = 0
            else:
                business_vintage = st.number_input(
                    "Business Vintage (years)", 0, 50,
                    value=int(st.session_state.get('pdf_business_vintage', 3))
                )
                employment_tenure = 0

        st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            bureau_score = st.number_input(
                "Bureau Score", 300, 900,
                value=int(st.session_state.get('pdf_bureau_score', 720)), step=10
            )
            dpd_90_6m = st.number_input(
                "DPD 90+ (Last 6M)", 0, 20,
                value=int(st.session_state.get('pdf_dpd_90', 0))
            )
            dpd_30_6m = st.number_input(
                "DPD 30+ (Last 6M)", 0, 20,
                value=int(st.session_state.get('pdf_dpd_30', 0))
            )
        with col2:
            credit_utilization = st.number_input(
                "Credit Utilization (%)", 0, 100,
                value=int(st.session_state.get('pdf_credit_util', 30))
            )
            recent_inquiries = st.number_input(
                "Recent Inquiries (3M)", 0, 20,
                value=int(st.session_state.get('pdf_inquiries', 2))
            )
        with col3:
            active_loans = st.number_input(
                "Active Loans", 0, 10,
                value=int(st.session_state.get('pdf_active_loans', 1))
            )
            existing_emi = st.number_input(
                "Existing Total EMI (₹)", 0, 200000,
                value=int(st.session_state.get('pdf_existing_emi', 15000)), step=1000
            )

        st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_salary = st.number_input(
                "Monthly Income (₹)", 0, 1000000,
                value=int(st.session_state.get('pdf_monthly_income', 50000)), step=5000
            )
            amt_income = st.number_input(
                "Annual Income (₹)", 0, 10000000,
                value=int(st.session_state.get('pdf_annual_income', 600000)), step=10000
            )
        with col2:
            net_surplus = st.number_input(
                "Net Cash Surplus (₹)", -100000, 500000,
                value=int(st.session_state.get('pdf_net_surplus', 20000)), step=5000
            )
            # ── FIX v8.4: Salary Stability reads inferred session_state key ──
            _ss_opts = ['STABLE', 'MODERATE', 'UNSTABLE']
            salary_stability = st.selectbox(
                "Salary Stability",
                _ss_opts,
                index=_ss_opts.index(st.session_state.get('pdf_salary_stability', 'STABLE'))
            )
        with col3:
            loan_amount = st.number_input(
                "Loan Amount (₹)", 0, 5000000,
                value=int(st.session_state.get('pdf_loan_amount', 180000)), step=10000
            )
            loan_tenure = st.number_input(
                "Tenure (months)", 3, 360,
                value=int(st.session_state.get('pdf_loan_tenure', 24))
            )
        with col4:
            interest_rate = st.number_input(
                "Interest Rate (%)", 8.0, 20.0,
                value=float(st.session_state.get('pdf_interest_rate', 10.5)), step=0.5
            )
            amt_annuity = st.number_input(
                "Requested EMI (₹)", 0, 200000,
                value=int(st.session_state.get('pdf_amt_annuity', 8500)), step=500
            )

        # ── Additional Credit Behaviour ──────────────────────────────────────
        st.markdown('<p class="section-header">📋 Additional Credit Behaviour</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            # ── FIX v8.4: payment_discipline reads inferred session_state key ──
            _pd_opts = ['GOOD', 'MODERATE', 'POOR']
            payment_discipline = st.selectbox(
                "Payment Discipline", _pd_opts,
                index=_pd_opts.index(st.session_state.get('pdf_payment_discipline', 'GOOD')),
                help="Overall payment behavior pattern"
            )
            # ── FIX v8.4: liquidity_flag reads inferred session_state key ──
            _lq_opts = ['LOW', 'ADEQUATE', 'MODERATE']
            liquidity_flag = st.selectbox(
                "Liquidity", _lq_opts,
                index=_lq_opts.index(st.session_state.get('pdf_liquidity_flag', 'LOW')),
                help="Cash liquidity position"
            )
        with col2:
            # ── FIX v8.4: cashflow_health reads inferred session_state key ──
            _cf_opts = ['MODERATE', 'HEALTHY', 'STRESSED', 'STABLE']
            cashflow_health = st.selectbox(
                "Cashflow Health", _cf_opts,
                index=_cf_opts.index(st.session_state.get('pdf_cashflow_health', 'MODERATE')),
                help="Overall cashflow health assessment"
            )
            # ── FIX v8.4: bureau_risk_flag reads inferred session_state key ──
            _br_opts = ['LOW', 'MEDIUM', 'HIGH']
            bureau_risk_flag = st.selectbox(
                "Bureau Risk", _br_opts,
                index=_br_opts.index(st.session_state.get('pdf_bureau_risk_flag', 'LOW')),
                help="External bureau risk rating"
            )
        with col3:
            inward_bounce_count = st.number_input(
                "Inward Bounce Count (3M)", 0, 10, 0,
                help="Number of bounced inward cheques last 3 months"
            )
            salary_missing_months = st.number_input(
                "Missing Salary Months (6M)", 0, 6, 0,
                help="Months without salary credit"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)

    if submitted:
        timestamp = datetime.now()
        app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
        customer_data = {
            'age': age,
            'employment_type': employment_type,
            'dependents': dependents,
            'kyc_verified': kyc_verified,
            'bankruptcy_flag': bankruptcy_flag,
            'fraud_flag': fraud_flag,
            'employment_tenure_months': employment_tenure,
            'business_vintage_years': business_vintage,
            'bureau_score': bureau_score,
            'dpd_90_count_6m': dpd_90_6m,
            'dpd_30_count_6m': dpd_30_6m,
            'credit_utilization_pct': credit_utilization,
            'max_utilization': credit_utilization,
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
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'payment_discipline_flag': payment_discipline,
            'liquidity_flag': liquidity_flag,
            'cashflow_health': cashflow_health,
            'bureau_risk_flag': bureau_risk_flag,
            'inward_bounce_count_3m': inward_bounce_count,
            'salary_missing_months': salary_missing_months,
        }

        with st.spinner("🔄 Processing Stage 1 assessment..."):
            decision_data = make_hybrid_decision_enhanced(customer_data)

        reasons = generate_reason_codes(
            decision=decision_data.get('decision', 'ERROR'),
            customer_data=customer_data,
            affordability_data=decision_data.get('affordability_data', {}),
            policy_checks=decision_data.get('policy_checks', {})
        )
        customer_data['reason_codes'] = reasons

        st.session_state.stage1_complete = True
        st.session_state.stage1_decision = decision_data.get('decision', 'ERROR')
        st.session_state.stage1_data = decision_data
        st.session_state.current_customer_data = customer_data

        for key in list(st.session_state.keys()):
            if key.startswith('pdf_') or key in ('_last_extraction', '_last_inferred_flags'):
                del st.session_state[key]

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])

        with tab1:
            st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                render_info_card("👤 Identity", "👤",
                                 {"Age": age, "Employment": employment_type, "Dependents": dependents,
                                  "KYC Status": "Verified" if kyc_verified else "Not Verified",
                                  "Tenure": f"{employment_tenure} months" if employment_type == 'Salaried' else f"{business_vintage} years"})
                render_info_card("💰 Financial", "💰",
                                 {"Monthly Income": f"₹{avg_salary:,}", "Annual Income": f"₹{amt_income:,}",
                                  "Net Surplus": f"₹{net_surplus:,}", "Stability": salary_stability})
            with col2:
                render_info_card("🏦 Credit Bureau", "🏦",
                                 {"Bureau Score": bureau_score, "DPD 90+": dpd_90_6m, "DPD 30+": dpd_30_6m,
                                  "Utilization": f"{credit_utilization}%", "Recent Inquiries": recent_inquiries,
                                  "Existing EMI": f"₹{existing_emi:,}"})
                render_info_card("📋 Loan Request", "📋",
                                 {"Amount": f"₹{loan_amount:,}", "Tenure": f"{loan_tenure} months",
                                  "Interest Rate": f"{interest_rate}%", "Requested EMI": f"₹{amt_annuity:,}"})

        with tab2:
            st.markdown('<p class="section-header">Decision Summary</p>', unsafe_allow_html=True)
            render_decision_header(decision_data, customer_data)
            st.markdown("<br>", unsafe_allow_html=True)

            final_decision = decision_data.get('decision', 'ERROR')

            if final_decision in ['APPROVE', 'REVIEW']:
                st.markdown("---")
                st.markdown("""
                    <div class="info-box" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white; text-align: center;">
                        <h3 style="margin: 0; color: white;">✅ Eligible for Stage 2 Deep Dive</h3>
                        <p style="margin: 0.5rem 0 0 0;">Choose an input method to proceed:</p>
                    </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📝 Manual Entry", key="stage2_manual_btn", use_container_width=True, type="primary"):
                        st.session_state.stage2_selected_tab = "Manual Entry"
                        st.session_state.page_navigation = "🔬 Stage 2 Analysis"
                        st.rerun()
                with col2:
                    if st.button("📄 PDF Upload", key="stage2_pdf_btn", use_container_width=True, type="primary"):
                        st.session_state.stage2_selected_tab = "PDF Upload"
                        st.session_state.page_navigation = "🔬 Stage 2 Analysis"
                        st.rerun()
                with col3:
                    if st.button("📊 Batch Analysis", key="stage2_batch_btn", use_container_width=True, type="primary"):
                        st.session_state.stage2_selected_tab = "Batch Analysis"
                        st.session_state.page_navigation = "🔬 Stage 2 Analysis"
                        st.rerun()
            elif final_decision == 'REJECT':
                st.markdown("---")
                st.markdown("""
                    <div class="warning-box" style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; text-align: center;">
                        <h3 style="margin: 0; color: white;">❌ Stage 2 Not Available</h3>
                        <p style="margin: 0.5rem 0 0 0;">Application rejected. Stage 2 only for APPROVE/REVIEW.</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            affordability = decision_data.get('affordability_data', {})
            foir = affordability.get('foir_percentage', 0)
            total_emi = affordability.get('total_emi', 0)
            net_disp = affordability.get('net_disposable', 0)

            col1, col2, col3 = st.columns(3)
            with col1:
                render_info_card("Identity & Eligibility", "👤",
                                 {f"Age: {age}": "", f"Employment: {employment_type}": "",
                                  f"Dependents: {dependents}": "",
                                  f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""},
                                 {f"Age: {age}": "pass" if 24 <= age <= 70 else "fail",
                                  f"Employment: {employment_type}": "pass",
                                  f"Dependents: {dependents}": "pass" if dependents <= 5 else "warning",
                                  f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_verified else "fail"})
            with col2:
                bureau_pass = bureau_score >= 550
                dpd_pass = dpd_90_6m == 0
                render_info_card("Credit Bureau", "🏦",
                                 {f"Bureau Score: {bureau_score}": "", f"DPD 90+: {dpd_90_6m}": "",
                                  f"Utilization: {credit_utilization}%": ""},
                                 {f"Bureau Score: {bureau_score}": "pass" if bureau_pass else "fail",
                                  f"DPD 90+: {dpd_90_6m}": "pass" if dpd_pass else "fail",
                                  f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"})
            with col3:
                render_info_card("Affordability", "💰",
                                 {f"Monthly Income: ₹{avg_salary:,}": "", f"FOIR: {foir:.1f}%": "",
                                  f"Total EMI: ₹{total_emi:,}": "", f"Net Disposable: ₹{net_disp:,}": ""},
                                 {f"Monthly Income: ₹{avg_salary:,}": "pass",
                                  f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
                                  f"Total EMI: ₹{total_emi:,}": "pass",
                                  f"Net Disposable: ₹{net_disp:,}": "pass" if net_disp >= 10000 else "warning"})

            st.markdown("<br>", unsafe_allow_html=True)
            render_reason_codes(reasons)
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                if PDF_AVAILABLE and generate_decision_pdf is not None:
                    try:
                        pdf_buffer = generate_decision_pdf(
                            decision_data=decision_data, customer_data=customer_data,
                            affordability_data=decision_data.get('affordability_data', {}), reasons=reasons)
                        st.download_button("📥 Decision Report (PDF)", data=pdf_buffer,
                                           file_name=f"credit_decision_{app_id}.pdf", mime="application/pdf",
                                           use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
                else:
                    st.warning("PDF generation not available.")
            with col2:
                if st.button("🔄 Re-Evaluate", key="reevaluate_btn", use_container_width=True):
                    st.rerun()

        with tab3:
            st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_modern_gauge(decision_data.get('confidence', 0), "Model Confidence")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                class_probs = decision_data.get('class_probs', {"APPROVE": 0, "REVIEW": 0, "REJECT": 0})
                fig2 = create_modern_bar_chart(class_probs)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
            policy_df = pd.DataFrame([{'Check': k, 'Result': v} for k, v in decision_data.get('policy_checks', {}).items()])
            st.dataframe(policy_df, use_container_width=True, hide_index=True)

            st.markdown('<p class="section-header">PD Calculation Breakdown</p>', unsafe_allow_html=True)
            pd_factors_display = {
                'Bureau Score': f"{bureau_score} → Base PD: {bureau_score_to_pd(bureau_score):.1f}%",
                'Delinquency': f"DPD 90+: {dpd_90_6m}, DPD 30+: {dpd_30_6m} → Multiplier: {delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x",
                'FOIR Impact': f"{foir:.1f}% → Adjustment: {foir_to_pd_adjustment(foir):.1f}%",
                'Employment Stability': f"{employment_type}, {employment_tenure if employment_type == 'Salaried' else business_vintage}{' months' if employment_type == 'Salaried' else ' years'} → Adjustment: {employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage):.1f}%",
                'ML Confidence': f"{decision_data.get('confidence', 0):.1f}% → Adjustment: {ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')):.1f}%",
                'Final PD': f"{decision_data.get('pd_percentage', 0)}%"
            }
            for factor, value in pd_factors_display.items():
                st.markdown(f"**{factor}:** {value}")

        with tab4:
            st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
            audit_log_raw = {
                'application_id': app_id,
                'timestamp': timestamp.isoformat(),
                'decision': decision_data.get('decision', 'ERROR'),
                'risk_score': decision_data.get('risk_score', 0),
                'pd_percentage': decision_data.get('pd_percentage', 0),
                'confidence': round(decision_data.get('confidence', 0), 2),
                'model_version': '8.4',
                'reason_codes': reasons,
                'policy_checks': decision_data.get('policy_checks', {}),
                'affordability': decision_data.get('affordability_data', {}),
                'customer_data': {k: v for k, v in customer_data.items() if k not in ['application_id', 'timestamp', 'reason_codes']},
                'pd_calculation_factors': {
                    'bureau_score': bureau_score,
                    'base_pd': bureau_score_to_pd(bureau_score),
                    'dpd_90': dpd_90_6m, 'dpd_30': dpd_30_6m,
                    'delinquency_multiplier': delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m),
                    'foir': foir,
                    'foir_adjustment': foir_to_pd_adjustment(foir),
                    'employment_adjustment': employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage),
                    'ml_adjustment': ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')),
                    'final_pd': decision_data.get('pd_percentage', 0)
                }
            }
            audit_log = sanitize_for_json(audit_log_raw)

            with st.expander("📋 View Audit Log (JSON)"):
                st.json(audit_log)

            col1, col2 = st.columns(2)
            with col1:
                if PDF_AVAILABLE and generate_audit_pdf is not None:
                    try:
                        audit_pdf_buffer = generate_audit_pdf(audit_log)
                        st.download_button("📥 Download Audit Trail (PDF)",
                                           data=audit_pdf_buffer,
                                           file_name=f"audit_trail_{app_id}.pdf",
                                           mime="application/pdf",
                                           use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating audit PDF: {str(e)}")
                else:
                    st.warning("Audit PDF generation is not available.")
            with col2:
                audit_json = json.dumps(audit_log, indent=2)
                st.download_button("📥 Download Audit Log (JSON)",
                                   data=audit_json,
                                   file_name=f"audit_{app_id}.json",
                                   mime="application/json",
                                   use_container_width=True)

            st.markdown('<p class="section-header">PD Calculation Summary</p>', unsafe_allow_html=True)
            pd_table = pd.DataFrame([
                {"Factor": "Bureau Score", "Value": f"{bureau_score}", "Impact": f"{bureau_score_to_pd(bureau_score):.1f}% base PD"},
                {"Factor": "Delinquency (DPD 90+)", "Value": f"{dpd_90_6m} times", "Impact": f"{delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x multiplier"},
                {"Factor": "FOIR", "Value": f"{foir:.1f}%", "Impact": f"{foir_to_pd_adjustment(foir):.1f}% adjustment"},
                {"Factor": "Employment Stability",
                 "Value": f"{employment_type} ({employment_tenure if employment_type == 'Salaried' else business_vintage}{' months' if employment_type == 'Salaried' else ' years'})",
                 "Impact": f"{employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage):.1f}% adjustment"},
                {"Factor": "ML Decision Confidence",
                 "Value": f"{decision_data.get('confidence', 0):.1f}% ({decision_data.get('decision', 'ERROR')})",
                 "Impact": f"{ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')):.1f}% adjustment"},
                {"Factor": "Final PD", "Value": f"{decision_data.get('pd_percentage', 0)}%", "Impact": "Industry-standard calculation"}
            ])
            st.dataframe(pd_table, use_container_width=True, hide_index=True)

elif page == "🔬 Stage 2 Analysis":
    st.markdown('<p class="main-header">Stage 2: CIBIL Deep Dive</p>', unsafe_allow_html=True)

    if not st.session_state.get('stage1_complete', False):
        st.error("❌ You must complete Stage 1 Assessment first!")
        st.info("Please go to the 👤 Assessment page and submit an application.")
        if st.button("← Go to Assessment", use_container_width=True):
            st.session_state.page_navigation = "👤 Assessment"
            st.rerun()
        st.stop()

    if st.session_state.get('stage1_decision') not in ['APPROVE', 'REVIEW']:
        st.error("❌ Stage 2 is only available for APPROVED or REVIEW applications!")
        st.warning(f"Your Stage 1 decision: {st.session_state.get('stage1_decision', 'Unknown')}")
        if st.button("← Go Back", use_container_width=True):
            st.session_state.page_navigation = "👤 Assessment"
            st.rerun()
        st.stop()

    if not (STAGE2_AVAILABLE and is_stage2_available()):
        st.error("❌ Stage 2 model not available!")
        st.info("Please ensure `stage2_cibil_model.pkl` is in the project directory.")
        if st.button("← Go Back", use_container_width=True):
            st.session_state.page_navigation = "👤 Assessment"
            st.rerun()
        st.stop()

    stage1_data = st.session_state.get('stage1_data', {})
    stage1_customer = st.session_state.get('current_customer_data', {})

    st.markdown(f"""
        <div class="info-box" style="background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); color: white;">
            <h3 style="margin: 0; color: white;">📊 Stage 1 Results</h3>
            <p style="margin: 0.5rem 0 0 0;">
                <strong>Decision:</strong> {st.session_state.get('stage1_decision', 'N/A')} |
                <strong>Risk Score:</strong> {stage1_data.get('risk_score', 'N/A')} |
                <strong>Application ID:</strong> {stage1_customer.get('application_id', 'N/A')}
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_options = ["Manual Entry", "PDF Upload", "Batch Analysis"]
    default_tab = st.session_state.get('stage2_selected_tab', 'Manual Entry')
    if default_tab not in tab_options:
        default_tab = "Manual Entry"
    selected_tab = st.radio(
        "Select input method",
        tab_options,
        index=tab_options.index(default_tab),
        horizontal=True,
        label_visibility="collapsed"
    )

    if selected_tab == "Manual Entry":
        st.markdown('<p class="section-header">Manual CIBIL Data Entry</p>', unsafe_allow_html=True)
        st.markdown("""
            <div class="info-box">
                📝 <strong>Manual Data Entry</strong><br>
                Enter CIBIL bureau data to enhance Stage 1 customer profile.<br>
                The Stage 2 model will use this data to predict risk tier (P1/P2/P3/P4).
            </div>
        """, unsafe_allow_html=True)

        with st.form("stage2_manual_form"):
            st.markdown("### 📋 Application Reference")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Application ID", value=stage1_customer.get('application_id', 'N/A'), disabled=True)
                st.text_input("Stage 1 Decision", value=st.session_state.get('stage1_decision', 'N/A'), disabled=True)
            with col2:
                st.text_input("Customer Name (Optional)", "")
                st.number_input("Stage 1 Risk Score", value=int(stage1_data.get('risk_score', 750)), disabled=True)

            st.markdown("---")
            st.markdown("### 🏦 CIBIL Bureau Data")

            st.markdown("---")
            st.markdown("### 👤 Demographics & Product Enquiries")

            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female", "Others"],
                    help="Select gender as per CIBIL report"
                )
            with col2:
                marital_status = st.selectbox(
                    "Marital Status",
                    ["Married", "Single", "Divorced", "Widowed", "Others"],
                    help="Marital status from bureau data"
                )
            with col3:
                education = st.selectbox(
                    "Education",
                    ["Graduate", "Post Graduate", "Under Graduate", "Professional", "Others"],
                    help="Highest education level"
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Credit Score & History**")
                cibil_score = st.number_input("Credit Score", 300, 900, 720, 10)
                max_delinquency = st.number_input("Max Delinquency Level", 0, 100, 0)
                num_times_30dpd = st.number_input("Times 30+ DPD", 0, 50, 0)
                num_times_60dpd = st.number_input("Times 60+ DPD", 0, 50, 0)
                num_times_delinquent = st.number_input("Total Delinquent", 0, 50, 0)
            with col2:
                st.markdown("**Recent Behavior (6-12M)**")
                num_deliq_6m = st.number_input("Delinquencies (6M)", 0, 20, 0)
                num_deliq_12m = st.number_input("Delinquencies (12M)", 0, 20, 0)
                max_deliq_6m = st.number_input("Max Delinq (6M)", 0, 100, 0)
                max_deliq_12m = st.number_input("Max Delinq (12M)", 0, 100, 0)
                enq_L3m = st.number_input("Inquiries (3M)", 0, 20, 2)
                enq_L6m = st.number_input("Inquiries (6M)", 0, 30, 4)
                enq_L12m = st.number_input("Inquiries (12M)", 0, 50, 6)
            with col3:
                st.markdown("**Account Quality**")
                num_std = st.number_input("Standard Accounts", 0, 50, 3)
                num_std_6m = st.number_input("Standard (6M)", 0, 50, 3)
                num_std_12m = st.number_input("Standard (12M)", 0, 50, 3)
                num_sub = st.number_input("Sub-standard", 0, 20, 0)
                num_sub_6m = st.number_input("Sub-standard (6M)", 0, 20, 0)
                num_dbt = st.number_input("Doubtful", 0, 10, 0)
                num_lss = st.number_input("Loss", 0, 10, 0)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Utilization**")
                pct_active_tls = st.number_input("% Active TLs", 0.0, 1.0, 0.60, 0.01)
                pct_current_bal = st.number_input("Current Balance %", 0.0, 1.0, 0.30, 0.01)
                cc_utilization = st.number_input("CC Utilization", 0.0, 1.0, 0.35, 0.01)
                pl_utilization = st.number_input("PL Utilization", 0.0, 1.0, 0.25, 0.01)
                max_unsec_exposure = st.number_input("Max Unsec Exposure %", 0, 100, 30)
            with col2:
                st.markdown("**Demographics**")
                age_cibil = st.number_input("Age", 24, 70, int(stage1_customer.get('age', 35)))
                net_monthly_income = st.number_input("Net Monthly Income", 0, 1000000,
                                                      int(stage1_customer.get('avg_salary_6m', 50000)), 5000)
                time_curr_employer = st.number_input("Employment Tenure (months)", 0, 600,
                                                      int(stage1_customer.get('employment_tenure_months', 24)))
            with col3:
                st.markdown("**Product Flags**")
                cc_flag = st.selectbox("Credit Card", ["Yes", "No"]) == "Yes"
                pl_flag = st.selectbox("Personal Loan", ["Yes", "No"]) == "No"
                hl_flag = st.selectbox("Home Loan", ["Yes", "No"]) == "No"
                gl_flag = st.selectbox("Gold Loan", ["Yes", "No"]) == "No"

            st.markdown("<br>", unsafe_allow_html=True)
            submitted_s2 = st.form_submit_button("🔬 Run Stage 2 Analysis", use_container_width=True, type="primary")

        if submitted_s2:
            with st.spinner("🔬 Running Stage 2 CIBIL Deep Analysis..."):
                enhanced_customer_data = stage1_customer.copy()
                _s1_inc = stage1_customer.get('avg_salary_6m', 50000)
                _s2_inc = net_monthly_income or 0
                _final_income = _s1_inc if (_s2_inc > 0 and _s2_inc < _s1_inc * 0.4) else (_s2_inc or _s1_inc)
                if _s2_inc > 0 and _s2_inc < _s1_inc * 0.4:
                    st.warning(f'⚠️ CIBIL income ₹{_s2_inc:,} is much lower than application income ₹{_s1_inc:,}. Using application income for FOIR.')
                enhanced_customer_data.update({
                    'bureau_score': cibil_score,
                    'age': age_cibil,
                    'avg_salary_6m': _final_income,
                    'employment_tenure_months': time_curr_employer,
                    'dpd_30_count_6m': num_times_30dpd,
                    'dpd_90_count_6m': num_times_60dpd,
                    'max_delinquency_level': max_delinquency,
                    'num_times_delinquent': num_times_delinquent,
                    'num_deliq_6mts': num_deliq_6m,
                    'num_deliq_12mts': num_deliq_12m,
                    'max_deliq_6mts': max_deliq_6m,
                    'max_deliq_12mts': max_deliq_12m,
                    'recent_inquiries_3m': enq_L3m,
                    'enq_L6m': enq_L6m,
                    'enq_L12m': enq_L12m,
                    'active_loans_count': num_std,
                    'num_std_6mts': num_std_6m,
                    'num_std_12mts': num_std_12m,
                    'num_sub': num_sub,
                    'num_sub_6mts': num_sub_6m,
                    'num_dbt': num_dbt,
                    'num_lss': num_lss,
                    'credit_utilization_pct': cc_utilization * 100,
                    'pct_of_active_TLs_ever': pct_active_tls,
                    'pct_currentBal_all_TL': pct_current_bal,
                    'CC_utilization': cc_utilization,
                    'PL_utilization': pl_utilization,
                    'max_unsec_exposure_inPct': max_unsec_exposure,
                    'CC_Flag': 1 if cc_flag else 0,
                    'PL_Flag': 1 if pl_flag else 0,
                    'HL_Flag': 1 if hl_flag else 0,
                    'GL_Flag': 1 if gl_flag else 0,
                    'GENDER': gender,
                    'MARITALSTATUS': marital_status,
                    'EDUCATION': education,
                })
                enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)
                try:
                    stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
                    stage2_result = resolve_stage2_to_binary(stage2_result)
                    display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
                except Exception as e:
                    st.error(f"❌ Stage 2 analysis failed: {str(e)}")
                    st.exception(e)

    elif selected_tab == "PDF Upload":
        st.markdown('<p class="section-header">📄 CIBIL PDF Upload</p>', unsafe_allow_html=True)
        if not OCR_AVAILABLE:
            st.error("❌ OCR not available. " + (OCR_ERROR_MSG or "Check packages.txt and requirements.txt."))
            st.warning("For now, please use the **Manual Entry** tab.")
        else:
            st.markdown("""
                <div class="info-box">
                    📄 <strong>CIBIL PDF Extraction</strong><br>
                    Upload a CIBIL bureau report PDF for automatic extraction and analysis.
                </div>
            """, unsafe_allow_html=True)
            uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="stage2_pdf")
            if uploaded_pdf is not None:
                st.success(f"✅ File uploaded: {uploaded_pdf.name} ({uploaded_pdf.size / 1024:.1f} KB)")
                if st.button("🔬 Extract & Analyze", key="extract_analyze_stage2", type="primary", use_container_width=True):
                    with st.spinner("🔄 Extracting data from PDF..."):
                        extraction_result = extract_cibil_from_pdf(uploaded_pdf)
                    if extraction_result.get('success', False):
                        st.success("✅ PDF extraction successful!")

                        app_id_display  = stage1_customer.get('application_id', 'N/A')
                        cust_name       = stage1_customer.get('customer_name', 'N/A')
                        s1_decision     = st.session_state.get('stage1_decision', 'N/A')
                        s1_risk         = stage1_data.get('risk_score', 'N/A')
                        st.markdown(f"""
                            <div style="background:#1e3a5f;color:white;padding:0.75rem 1rem;border-radius:0.5rem;margin-bottom:0.75rem;">
                                <strong>📋 Application ID:</strong> {app_id_display} &nbsp;|&nbsp;
                                <strong>Stage 1:</strong> {s1_decision} &nbsp;|&nbsp;
                                <strong>Risk Score:</strong> {s1_risk}
                            </div>
                        """, unsafe_allow_html=True)

                        st.markdown("### 📋 Extracted CIBIL Data (Summary)")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Credit Score", extraction_result.get('Credit_Score', 'N/A'))
                            st.metric("Max Delinquency Level", extraction_result.get('max_delinquency_level', 0))
                        with col2:
                            st.metric("Times 30+ DPD", extraction_result.get('num_times_30p_dpd', 0))
                            st.metric("Times 60+ DPD", extraction_result.get('num_times_60p_dpd', 0))
                        with col3:
                            st.metric("Total Delinquent", extraction_result.get('num_times_delinquent', 0))
                            st.metric("DPD 90+ (6M)", extraction_result.get('dpd_90_count_6m', 0))
                        with col4:
                            st.metric("Active Accounts", extraction_result.get('num_std', 0))
                            st.metric("Written Off", extraction_result.get('written_off_count', 0))

                        with st.expander("🔍 View All Extracted Features (with internal IDs)", expanded=False):
                            friendly_names = {
                                'Credit_Score': 'Credit Score',
                                'AGE': 'Age',
                                'max_delinquency_level': 'Max Delinquency Level',
                                'num_times_30p_dpd': 'Times 30+ DPD',
                                'num_times_60p_dpd': 'Times 60+ DPD',
                                'num_times_delinquent': 'Total Times Delinquent',
                                'dpd_90_count_6m': 'DPD 90+ Count (6M)',
                                'num_deliq_6mts': 'Delinquent Count (6M)',
                                'num_deliq_12mts': 'Delinquent Count (12M)',
                                'max_deliq_6mts': 'Max Delinquency (6M)',
                                'max_deliq_12mts': 'Max Delinquency (12M)',
                                'enq_L3m': 'Recent Inquiries (3M)',
                                'enq_L6m': 'Inquiries (6M)',
                                'enq_L12m': 'Inquiries (12M)',
                                'num_std': 'Standard / Active Accounts',
                                'num_std_6mts': 'Standard Accounts (6M)',
                                'num_std_12mts': 'Standard Accounts (12M)',
                                'num_sub': 'Sub-standard Accounts',
                                'num_sub_6mts': 'Sub-standard (6M)',
                                'num_dbt': 'Doubtful Accounts',
                                'num_lss': 'Loss / Written-Off Accounts',
                                'CC_utilization': 'Credit Card Utilization (0–1)',
                                'PL_utilization': 'Personal Loan Utilization (0–1)',
                                'CC_Flag': 'Has Credit Card (1=Yes)',
                                'PL_Flag': 'Has Personal Loan (1=Yes)',
                                'HL_Flag': 'Has Home Loan (1=Yes)',
                                'GL_Flag': 'Has Gold Loan (1=Yes)',
                                'written_off_count': 'Written Off Count',
                                'settled_count': 'Settled Account Count',
                                'high_util_flag': 'High Utilization Flag (1=Yes)',
                                'recent_deliq_flag': 'Recent Delinquency Flag (1=Yes)',
                                'account_quality_score': 'Account Quality Score (0–100)',
                                'Time_With_Curr_Empr': 'Employment Tenure (months)',
                                'NETMONTHLYINCOME': 'Net Monthly Income (₹)',
                                'pct_of_active_TLs_ever': '% Active Trade Lines Ever',
                                'pct_currentBal_all_TL': '% Current Balance / All TL',
                                'max_unsec_exposure_inPct': 'Max Unsecured Exposure (%)',
                                'extraction_method': 'Extraction Method',
                            }
                            exclude_keys = {'success', 'error', 'raw_text'}
                            data_items = []
                            for key, val in extraction_result.items():
                                if key in exclude_keys:
                                    continue
                                fname = friendly_names.get(key, key.replace('_', ' ').title())
                                data_items.append({"Feature Name": fname, "Internal ID": key, "Extracted Value": str(val)})
                            data_items.sort(key=lambda x: x["Feature Name"])
                            data_items = [
                                {"Feature Name": "── Application ID", "Internal ID": "application_id", "Extracted Value": app_id_display},
                                {"Feature Name": "── Customer Name", "Internal ID": "customer_name", "Extracted Value": cust_name},
                                {"Feature Name": "── Stage 1 Decision", "Internal ID": "stage1_decision", "Extracted Value": s1_decision},
                                {"Feature Name": "── Stage 1 Risk Score", "Internal ID": "stage1_risk_score", "Extracted Value": str(s1_risk)},
                            ] + data_items
                            import pandas as _pd
                            df_all = _pd.DataFrame(data_items)
                            st.dataframe(df_all, use_container_width=True, hide_index=True)

                        enhanced_customer_data = stage1_customer.copy()
                        _s1_income = stage1_customer.get('avg_salary_6m', 50000)
                        _s2_income = extraction_result.get('NETMONTHLYINCOME', 0)
                        _use_income = _s1_income if (_s2_income > 0 and _s2_income < _s1_income * 0.4) else (_s2_income or _s1_income)
                        if _s2_income > 0 and _s2_income < _s1_income * 0.4:
                            st.warning(f'⚠️ CIBIL income ₹{_s2_income:,} is much lower than application income ₹{_s1_income:,}. Using application income for FOIR.')

                        enhanced_customer_data.update({
                            'bureau_score': extraction_result.get('Credit_Score', 720),
                            'age': extraction_result.get('AGE', stage1_customer.get('age', 35)),
                            'avg_salary_6m': _use_income,
                            'employment_tenure_months': extraction_result.get('Time_With_Curr_Empr', stage1_customer.get('employment_tenure_months', 24)),
                            'dpd_30_count_6m': extraction_result.get('num_times_30p_dpd', 0),
                            'dpd_90_count_6m': extraction_result.get('dpd_90_count_6m', 0),
                            'max_delinquency_level': extraction_result.get('max_delinquency_level', 0),
                            'num_times_delinquent': extraction_result.get('num_times_delinquent', 0),
                            'num_deliq_6mts': extraction_result.get('num_deliq_6mts', 0),
                            'num_deliq_12mts': extraction_result.get('num_deliq_12mts', 0),
                            'max_deliq_6mts': extraction_result.get('max_deliq_6mts', 0),
                            'max_deliq_12mts': extraction_result.get('max_deliq_12mts', 0),
                            'recent_inquiries_3m': extraction_result.get('enq_L3m', 2),
                            'enq_L6m': extraction_result.get('enq_L6m', 4),
                            'enq_L12m': extraction_result.get('enq_L12m', 6),
                            'active_loans_count': extraction_result.get('num_std', 1),
                            'num_std_6mts': extraction_result.get('num_std_6mts', 0),
                            'num_std_12mts': extraction_result.get('num_std_12mts', 0),
                            'num_sub': extraction_result.get('num_sub', 0),
                            'num_sub_6mts': extraction_result.get('num_sub_6mts', 0),
                            'num_dbt': extraction_result.get('num_dbt', 0),
                            'num_lss': extraction_result.get('num_lss', 0),
                            'credit_utilization_pct': (0 if extraction_result.get('CC_utilization', 0) < 0 else extraction_result.get('CC_utilization', 0.35)) * 100,
                            'pct_of_active_TLs_ever': extraction_result.get('pct_of_active_TLs_ever', 0.6),
                            'pct_currentBal_all_TL': extraction_result.get('pct_currentBal_all_TL', 0.3),
                            'CC_utilization': 0 if extraction_result.get('CC_utilization', 0) < 0 else extraction_result.get('CC_utilization', 0.35),
                            'PL_utilization': 0 if extraction_result.get('PL_utilization', 0) < 0 else extraction_result.get('PL_utilization', 0.25),
                            'max_unsec_exposure_inPct': extraction_result.get('max_unsec_exposure_inPct', 30),
                            'CC_Flag': extraction_result.get('CC_Flag', 0),
                            'PL_Flag': extraction_result.get('PL_Flag', 0),
                            'HL_Flag': extraction_result.get('HL_Flag', 0),
                            'GL_Flag': extraction_result.get('GL_Flag', 0),
                            'written_off_count': extraction_result.get('written_off_count', 0),
                            'settled_count': extraction_result.get('settled_count', 0),
                            'high_util_flag': extraction_result.get('high_util_flag', 0),
                            'recent_deliq_flag': extraction_result.get('recent_deliq_flag', 0),
                            'account_quality_score': extraction_result.get('account_quality_score', 0)
                        })

                        enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)

                        with st.spinner("🔬 Running Stage 2 analysis..."):
                            try:
                                stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
                                stage2_result = resolve_stage2_to_binary(stage2_result)
                                display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
                            except Exception as e:
                                st.error(f"❌ Analysis failed: {str(e)}")
                    else:
                        st.error("❌ PDF extraction failed! Error: " + extraction_result.get('error', 'Unknown'))

    elif selected_tab == "Batch Analysis":
        st.markdown('<p class="section-header">📊 Batch CIBIL Analysis</p>', unsafe_allow_html=True)
        st.info("📊 Batch analysis feature coming soon! (Upload a CSV with all required CIBIL fields)")

elif page == "📊 Batch Process":
    st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-box">
            📤 Upload a CSV file with customer data for bulk credit assessment.
            The file should include all required fields for prediction.
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} records")
            with st.expander("📄 Preview Uploaded Data"):
                st.dataframe(df.head(), use_container_width=True)
                st.write(f"**Total Records:** {len(df)}")
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your CSV includes at least these columns: age, employment_type, avg_salary_6m, bureau_score, loan_amount")
            else:
                if st.button("🚀 Process Batch Predictions", key="process_batch_btn", type="primary", use_container_width=True):
                    with st.spinner(f"🔍 Processing {len(df)} records..."):
                        progress_bar = st.progress(0)
                        results_df = process_batch_predictions(df)
                        progress_bar.progress(100)
                        st.success(f"✅ Completed processing {len(results_df)} records!")
                        tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
                        with tab1:
                            st.dataframe(results_df, use_container_width=True)
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("✅ Approved", len(results_df[results_df['decision'] == 'APPROVE']))
                            with col2:
                                st.metric("❌ Rejected", len(results_df[results_df['decision'] == 'REJECT']))
                            with col3:
                                st.metric("⚠️ Review", len(results_df[results_df['decision'] == 'REVIEW']))
                            with col4:
                                st.metric("📊 Avg Risk Score", f"{results_df['risk_score'].mean():.0f}")
                        with tab2:
                            col1, col2 = st.columns(2)
                            with col1:
                                decision_counts = results_df['decision'].value_counts()
                                fig1 = px.pie(values=decision_counts.values, names=decision_counts.index,
                                              title="Decision Distribution", color=decision_counts.index,
                                              color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
                                st.plotly_chart(fig1, use_container_width=True)
                            with col2:
                                fig2 = px.histogram(results_df, x='risk_score', title="Risk Score Distribution",
                                                    nbins=20, color_discrete_sequence=['#587042'])
                                st.plotly_chart(fig2, use_container_width=True)
                            fig3 = px.scatter(results_df, x='monthly_income', y='loan_amount', color='decision',
                                              size='risk_score', title="Income vs Loan Amount (Colored by Decision)",
                                              hover_data=['application_id', 'foir_percentage'],
                                              color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
                            st.plotly_chart(fig3, use_container_width=True)
                            fig4 = px.box(results_df, x='decision', y='pd_percentage',
                                          title="PD Distribution by Decision", color='decision',
                                          color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
                            st.plotly_chart(fig4, use_container_width=True)
                        with tab3:
                            st.markdown("### Download Results")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    "📥 Download as CSV",
                                    data=results_df.to_csv(index=False),
                                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            with col2:
                                st.download_button(
                                    "📥 Download as JSON",
                                    data=results_df.to_json(orient='records', indent=2),
                                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                            st.markdown("---")
                            st.markdown("#### Filtered Downloads")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                approved_df = results_df[results_df['decision'] == 'APPROVE']
                                if len(approved_df) > 0:
                                    st.download_button(
                                        f"✅ Approved Only ({len(approved_df)})",
                                        data=approved_df.to_csv(index=False),
                                        file_name=f"approved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            with col2:
                                rejected_df = results_df[results_df['decision'] == 'REJECT']
                                if len(rejected_df) > 0:
                                    st.download_button(
                                        f"❌ Rejected Only ({len(rejected_df)})",
                                        data=rejected_df.to_csv(index=False),
                                        file_name=f"rejected_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            with col3:
                                review_df = results_df[results_df['decision'] == 'REVIEW']
                                if len(review_df) > 0:
                                    st.download_button(
                                        f"⚠️ Review Only ({len(review_df)})",
                                        data=review_df.to_csv(index=False),
                                        file_name=f"review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure the CSV file is properly formatted and contains the required columns.")
    else:
        st.markdown("---")
        st.markdown("### 📋 CSV Template")
        template_data = {
            'age': [35, 42, 28],
            'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
            'dependents': [2, 3, 6],
            'kyc_verified': ['Yes', 'Yes', 'No'],
            'bankruptcy_flag': ['No', 'No', 'No'],
            'fraud_flag': ['No', 'No', 'No'],
            'employment_tenure_months': [24, 0, 18],
            'business_vintage_years': [0, 5, 0],
            'bureau_score': [720, 680, 580],
            'dpd_90_count_6m': [0, 1, 2],
            'dpd_30_count_6m': [0, 2, 1],
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
            'AMT_ANNUITY': [8500, 9500, 4500],
            'payment_discipline_flag': ['GOOD', 'MODERATE', 'POOR'],
            'liquidity_flag': ['LOW', 'ADEQUATE', 'LOW'],
            'cashflow_health': ['HEALTHY', 'MODERATE', 'STRESSED'],
            'bureau_risk_flag': ['LOW', 'MEDIUM', 'HIGH'],
            'inward_bounce_count_3m': [0, 1, 3],
            'salary_missing_months': [0, 0, 2],
        }
        template_df = pd.DataFrame(template_data)
        st.dataframe(template_df, use_container_width=True)
        st.caption("📝 Note: `dependents > 5` will automatically trigger REVIEW regardless of other factors.")
        st.download_button(
            "📥 Download CSV Template",
            data=template_df.to_csv(index=False),
            file_name="credit_assessment_template.csv",
            mime="text/csv",
            use_container_width=True
        )

elif page == "📈 Model Info":
    st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">RF</div><div class="stat-label">Model Type</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TOP_FEATURES)}</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TARGET_LE.classes_)}</div><div class="stat-label">Classes</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
    feature_df = pd.DataFrame({'Rank': range(1, min(21, len(TOP_FEATURES) + 1)), 'Feature': TOP_FEATURES[:20]})
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

elif page == "ℹ️ About":
    st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-card">
            <div class="info-card-title"><span class="icon">🏦</span><span>Credit Risk Assessment Platform</span></div>
            <div class="info-card-content">
                <p><strong>Version:</strong> 8.4 - OCR AUTO-FILL FIX (categorical dropdowns now update from PDF)</p>
                <p><strong>Developer:</strong> Zen Meraki</p>
                <p><strong>Date:</strong> January 2026</p>
                <br>
                <p>A comprehensive credit risk evaluation system combining hard policy rules,
                machine learning models, and affordability analysis for accurate and compliant lending decisions.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="info-card">
                <div class="info-card-title"><span class="icon">🎯</span><span>Key Features</span></div>
                <div class="info-card-content">
                    <ul style="margin: 0; padding-left: 1.25rem;">
                        <li>Three-layer decision engine</li>
                        <li>Real-time risk assessment</li>
                        <li>Industry-standard PD calculation</li>
                        <li>FOIR calculation & validation</li>
                        <li>Automated reason generation</li>
                        <li>Complete audit trail (PDF)</li>
                        <li>Professional UI/UX</li>
                        <li>OCR auto-fill with full categorical inference</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="info-card">
                <div class="info-card-title"><span class="icon">🛠️</span><span>Technology Stack</span></div>
                <div class="info-card-content">
                    <ul style="margin: 0; padding-left: 1.25rem;">
                        <li>Streamlit (UI Framework)</li>
                        <li>Scikit-learn (ML)</li>
                        <li>Plotly (Visualizations)</li>
                        <li>Pandas (Data Processing)</li>
                        <li>ReportLab (PDF Generation)</li>
                        <li>Python 3.8+</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
