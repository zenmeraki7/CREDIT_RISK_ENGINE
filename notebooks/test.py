
# # CORRECTED test.py - VERSION 8.2 (FIXED: use_two_stage session state, tab4 indentation, page fallback)
# """
# Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
# Enhanced with Modern UI/UX Design
# Run with: streamlit run test.py (from inside the notebooks folder)
# Author: Zen Meraki
# Date: January 2026
# VERSION: 8.2 - COMPLETELY FIXED PD CALCULATION & AUDIT PDF
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
# from typing import Dict, List, Any, Union
# import json
# import sys
# import os
# from pathlib import Path
# import css_styles
# sys.path.insert(0, str(Path(__file__).parent.parent))

# # =============================================================================
# # 2. PAGE CONFIG – MUST BE FIRST STREAMLIT COMMAND
# # =============================================================================
# st.set_page_config(
#     page_title="Credit Risk Assessment",
#     page_icon="💳",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # =============================================================================
# # 3. EVERYTHING ELSE (CSS, functions, etc.)
# # =============================================================================
# from css_styles import CSS
# st.markdown(CSS, unsafe_allow_html=True)

# warnings.filterwarnings('ignore')

# # =============================================================================
# # STAGE 2 ENGINE – ROBUST FALLBACK
# # =============================================================================
# try:
#     import stage2_engine
#     from stage2_engine import make_two_stage_decision, is_stage2_available, get_stage2_status
#     STAGE2_AVAILABLE = True
# except ImportError:
#     stage2_engine = None
#     STAGE2_AVAILABLE = False
#     def make_two_stage_decision(*args, **kwargs):
#         raise NotImplementedError("Stage 2 engine not available")
#     def is_stage2_available():
#         return False
#     def get_stage2_status():
#         return {"error": "Stage 2 engine module not found", "available": False}

# # =============================================================================
# # OCR IMPORTS FOR CIBIL PDF EXTRACTION
# # =============================================================================
# try:
#     import pytesseract
#     from pdf2image import convert_from_bytes
#     import cv2
#     from PIL import Image
#     import re
#     OCR_AVAILABLE = True
# except ImportError:
#     OCR_AVAILABLE = False

# # =============================================================================
# # PDF GENERATION – SAFE FALLBACK
# # =============================================================================
# PDF_AVAILABLE = False
# generate_decision_pdf = None
# generate_audit_pdf = None
# try:
#     from utils.pdf_generator import generate_decision_pdf, generate_audit_pdf
#     PDF_AVAILABLE = True
# except ImportError as e:
#     PDF_AVAILABLE = False
#     pass

# # =============================================================================
# # JSON SANITIZER
# # =============================================================================
# def sanitize_for_json(obj: Any) -> Any:
#     if obj is None or isinstance(obj, (str, int, float, bool)):
#         return obj
#     if isinstance(obj, set):
#         return list(obj)
#     if isinstance(obj, datetime):
#         return obj.isoformat()
#     if isinstance(obj, np.integer):
#         return int(obj)
#     if isinstance(obj, np.floating):
#         return float(obj)
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     if isinstance(obj, dict):
#         return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [sanitize_for_json(item) for item in obj]
#     try:
#         json.dumps(obj)
#         return obj
#     except (TypeError, ValueError):
#         return str(obj)

# # =============================================================================
# # SESSION STATE INITIALIZATION
# # =============================================================================
# def init_session_state():
#     """Initialize all session state variables"""
#     if 'stage1_complete' not in st.session_state:
#         st.session_state.stage1_complete = False
#     if 'stage1_decision' not in st.session_state:
#         st.session_state.stage1_decision = None
#     if 'stage1_data' not in st.session_state:
#         st.session_state.stage1_data = None
#     if 'current_customer_data' not in st.session_state:
#         st.session_state.current_customer_data = None
#     if 'page_navigation' not in st.session_state:
#         st.session_state.page_navigation = "🏠 Home"
#     if 'use_two_stage' not in st.session_state:
#         st.session_state.use_two_stage = False
#     if 'stage2_selected_tab' not in st.session_state:
#         st.session_state.stage2_selected_tab = "Manual Entry"   # default tab

# # Initialize session state
# init_session_state()

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
#         return {'loaded': False, 'error': 'credit_risk_assets.pkl not found. Please run the training script first.'}
#     except Exception as e:
#         return {'loaded': False, 'error': f'Error loading model: {str(e)}'}

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
# # AFFORDABILITY CALCULATION ENGINE (FIXED EMI OVERFLOW)
# # =============================================================================
# def calculate_emi(principal, annual_rate, tenure_months):
#     if principal <= 0 or tenure_months <= 0:
#         return 0
#     monthly_rate = annual_rate / (12 * 100)  # annual_rate is in percent, e.g., 10.5
#     if monthly_rate == 0:
#         return principal / tenure_months
#     emi = (principal * monthly_rate * (1 + monthly_rate)**tenure_months) / \
#           ((1 + monthly_rate)**tenure_months - 1)
#     return round(emi, 2)

# def calculate_affordability(monthly_income, loan_amount, interest_rate, tenure_months, existing_emi):
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
# # PD CALCULATION
# # =============================================================================
# def bureau_score_to_pd(bureau_score):
#     if bureau_score >= 800:
#         return 0.5 + (900 - bureau_score) / 200 * 0.5
#     elif bureau_score >= 750:
#         return 1.0 + (800 - bureau_score) / 50 * 1.0
#     elif bureau_score >= 700:
#         return 2.0 + (750 - bureau_score) / 50 * 1.5
#     elif bureau_score >= 650:
#         return 3.5 + (700 - bureau_score) / 50 * 2.5
#     elif bureau_score >= 600:
#         return 6.0 + (650 - bureau_score) / 50 * 4.0
#     elif bureau_score >= 550:
#         return 10.0 + (600 - bureau_score) / 50 * 5.0
#     else:
#         return min(25.0, 15.0 + (550 - bureau_score) / 50 * 10.0)

# def foir_to_pd_adjustment(foir_percentage):
#     if foir_percentage <= 30:
#         return -0.5
#     elif foir_percentage <= 40:
#         return 0.0
#     elif foir_percentage <= 50:
#         return 1.0
#     elif foir_percentage <= 60:
#         return 2.5
#     else:
#         return 5.0

# def delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count=0):
#     if dpd_90_count >= 3:
#         return 5.0
#     elif dpd_90_count == 2:
#         return 3.0
#     elif dpd_90_count == 1:
#         return 2.0
#     elif dpd_30_count >= 3:
#         return 1.6
#     elif dpd_30_count >= 1:
#         return 1.3
#     else:
#         return 1.0

# def employment_stability_to_pd_adjustment(employment_type, tenure_months, business_vintage_years=0):
#     if employment_type == 'Salaried':
#         if tenure_months >= 36:
#             return -0.5
#         elif tenure_months >= 12:
#             return 0.0
#         elif tenure_months >= 6:
#             return 0.5
#         else:
#             return 2.0
#     elif employment_type in ['Self-Employed', 'Business']:
#         if business_vintage_years >= 5:
#             return -0.5
#         elif business_vintage_years >= 2:
#             return 0.0
#         else:
#             return 1.5
#     else:
#         return 1.0

# def inquiry_pattern_to_pd_adjustment(recent_inquiries_3m):
#     if recent_inquiries_3m <= 1:
#         return -0.3
#     elif recent_inquiries_3m <= 3:
#         return 0.0
#     elif recent_inquiries_3m <= 5:
#         return 0.8
#     elif recent_inquiries_3m <= 8:
#         return 1.5
#     else:
#         return 3.0

# def ml_confidence_to_pd_adjustment(ml_confidence, ml_decision):
#     if ml_decision == "APPROVE":
#         if ml_confidence >= 90:
#             return -0.5
#         elif ml_confidence >= 70:
#             return 0.0
#         else:
#             return 0.5
#     elif ml_decision == "REVIEW":
#         return 1.0
#     else:
#         return 5.0

# def calculate_final_pd(bureau_score, foir, confidence, dpd_90_count=0, dpd_30_count=0,
#                       employment_type='Salaried', employment_tenure=24, business_vintage=0,
#                       recent_inquiries=2, ml_decision='APPROVE'):
#     base_pd = bureau_score_to_pd(bureau_score)
#     foir_adj = foir_to_pd_adjustment(foir)
#     deliq_multiplier = delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count)
#     employment_adj = employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage)
#     inquiry_adj = inquiry_pattern_to_pd_adjustment(recent_inquiries)
#     ml_adj = ml_confidence_to_pd_adjustment(confidence, ml_decision)
#     adjusted_base_pd = base_pd * deliq_multiplier
#     final_pd = adjusted_base_pd + foir_adj + employment_adj + inquiry_adj + ml_adj
#     final_pd = max(0.5, min(final_pd, 25.0))
#     return round(final_pd, 2)

# # =============================================================================
# # RISK SCORE CALCULATION
# # =============================================================================
# def calculate_final_risk_score(bureau_score, ml_confidence, foir):
#     bureau_points = (bureau_score / 900) * 400
#     ml_points = (ml_confidence / 100) * 400
#     foir_points = max(0, (1 - foir/50) * 200)
#     total_score = int(bureau_points + ml_points + foir_points)
#     return min(max(total_score, 0), 1000)

# # =============================================================================
# # CIBIL PDF EXTRACTION ENGINE (OCR + PATTERN MATCHING)
# # =============================================================================
# def extract_cibil_from_pdf(uploaded_file):
#     """Extract CIBIL bureau data from PDF using OCR and pattern matching"""
#     if not OCR_AVAILABLE:
#         return {
#             'success': False,
#             'error': 'OCR libraries not installed',
#             'message': 'Please install: pip install pytesseract pdf2image opencv-python pillow'
#         }
#     try:
#         pdf_bytes = uploaded_file.read()
#         images = convert_from_bytes(pdf_bytes, dpi=300)
#         full_text = ""
#         for page_num, image in enumerate(images):
#             img_array = np.array(image)
#             gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#             _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             page_text = pytesseract.image_to_string(binary)
#             full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
#         # Extraction helper functions (same as before)
#         def extract_credit_score(text):
#             patterns = [r'credit\s*score[:\s]*(\d{3})', r'score[:\s]*(\d{3})', r'cibil\s*score[:\s]*(\d{3})']
#             for pattern in patterns:
#                 match = re.search(pattern, text, re.IGNORECASE)
#                 if match:
#                     score = int(match.group(1))
#                     if 300 <= score <= 900:
#                         return score
#             return 720

#         def extract_delinquency_level(text):
#             patterns = [r'max\s*delinquency[:\s]*(\d+)', r'delinquency\s*level[:\s]*(\d+)']
#             for pattern in patterns:
#                 match = re.search(pattern, text, re.IGNORECASE)
#                 if match:
#                     return int(match.group(1))
#             return 0

#         def extract_dpd_count(text, dpd_type):
#             patterns = [rf'{dpd_type}\+?\s*dpd[:\s]*(\d+)', rf'dpd\s*{dpd_type}[:\s]*(\d+)']
#             for pattern in patterns:
#                 match = re.search(pattern, text, re.IGNORECASE)
#                 if match:
#                     return int(match.group(1))
#             return 0

#         def extract_total_delinquencies(text):
#             match = re.search(r'total\s*delinquencies[:\s]*(\d+)', text, re.IGNORECASE)
#             return int(match.group(1)) if match else 0

#         def extract_recent_delinquencies(text, months):
#             patterns = [rf'delinquencies?\s*\(?{months}\s*months?\)?[:\s]*(\d+)', rf'{months}m\s*delinq[:\s]*(\d+)']
#             for pattern in patterns:
#                 match = re.search(pattern, text, re.IGNORECASE)
#                 if match:
#                     return int(match.group(1))
#             return 0

#         def extract_max_delinquency_period(text, months):
#             match = re.search(rf'max\s*delinq\s*{months}m[:\s]*(\d+)', text, re.IGNORECASE)
#             return int(match.group(1)) if match else 0

#         def extract_inquiries(text, months):
#             patterns = [rf'inquiries?\s*\(?{months}\s*months?\)?[:\s]*(\d+)', rf'{months}m\s*inquir[y|ies][:\s]*(\d+)', rf'enq\s*{months}m[:\s]*(\d+)']
#             for pattern in patterns:
#                 match = re.search(pattern, text, re.IGNORECASE)
#                 if match:
#                     return int(match.group(1))
#             return 0

#         def extract_account_count(text, account_type):
#             match = re.search(rf'{account_type}\s*accounts?[:\s]*(\d+)', text, re.IGNORECASE)
#             return int(match.group(1)) if match else 0

#         def extract_account_count_period(text, account_type, months):
#             match = re.search(rf'{account_type}\s*\({months}m\)[:\s]*(\d+)', text, re.IGNORECASE)
#             return int(match.group(1)) if match else 0

#         def extract_active_tl_percentage(text):
#             match = re.search(r'active\s*tl[s]?[:\s]*(\d+\.?\d*)%?', text, re.IGNORECASE)
#             if match:
#                 val = float(match.group(1))
#                 return val / 100 if val > 1 else val
#             return 0.60

#         def extract_current_balance_pct(text):
#             match = re.search(r'current\s*balance[:\s]*(\d+\.?\d*)%?', text, re.IGNORECASE)
#             if match:
#                 val = float(match.group(1))
#                 return val / 100 if val > 1 else val
#             return 0.30

#         def extract_cc_utilization(text):
#             patterns = [r'cc\s*utilization[:\s]*(\d+\.?\d*)%?', r'credit\s*card\s*util[:\s]*(\d+\.?\d*)%?']
#             for pattern in patterns:
#                 match = re.search(pattern, text, re.IGNORECASE)
#                 if match:
#                     val = float(match.group(1))
#                     return val / 100 if val > 1 else val
#             return 0.35

#         def extract_pl_utilization(text):
#             match = re.search(r'pl\s*utilization[:\s]*(\d+\.?\d*)%?', text, re.IGNORECASE)
#             if match:
#                 val = float(match.group(1))
#                 return val / 100 if val > 1 else val
#             return 0.25

#         def extract_unsec_exposure(text):
#             match = re.search(r'unsec(?:ured)?\s*exposure[:\s]*(\d+\.?\d*)%?', text, re.IGNORECASE)
#             return int(float(match.group(1))) if match else 30

#         def extract_age(text):
#             patterns = [r'age[:\s]*(\d{2})', r'dob.*?(\d{2})\s*years?']
#             for pattern in patterns:
#                 match = re.search(pattern, text, re.IGNORECASE)
#                 if match:
#                     age = int(match.group(1))
#                     if 18 <= age <= 100:
#                         return age
#             return 35

#         def extract_monthly_income(text):
#             patterns = [r'monthly\s*income[:\s]*(?:rs\.?|₹)?\s*(\d+(?:,\d+)*)', r'net\s*monthly[:\s]*(?:rs\.?|₹)?\s*(\d+(?:,\d+)*)']
#             for pattern in patterns:
#                 match = re.search(pattern, text, re.IGNORECASE)
#                 if match:
#                     income_str = match.group(1).replace(',', '')
#                     return int(income_str)
#             return 50000

#         def extract_employment_tenure(text):
#             patterns = [r'current\s*employer[:\s]*(\d+)\s*months?', r'time\s*with\s*employer[:\s]*(\d+)']
#             for pattern in patterns:
#                 match = re.search(pattern, text, re.IGNORECASE)
#                 if match:
#                     return int(match.group(1))
#             return 24

#         def detect_product_flag(text, product_type):
#             return product_type.lower() in text.lower()

#         extracted_data = {
#             'Credit_Score': extract_credit_score(full_text),
#             'max_delinquency_level': extract_delinquency_level(full_text),
#             'num_times_30p_dpd': extract_dpd_count(full_text, '30'),
#             'num_times_60p_dpd': extract_dpd_count(full_text, '60'),
#             'num_times_delinquent': extract_total_delinquencies(full_text),
#             'num_deliq_6mts': extract_recent_delinquencies(full_text, '6'),
#             'num_deliq_12mts': extract_recent_delinquencies(full_text, '12'),
#             'max_deliq_6mts': extract_max_delinquency_period(full_text, '6'),
#             'max_deliq_12mts': extract_max_delinquency_period(full_text, '12'),
#             'enq_L3m': extract_inquiries(full_text, '3'),
#             'enq_L6m': extract_inquiries(full_text, '6'),
#             'enq_L12m': extract_inquiries(full_text, '12'),
#             'num_std': extract_account_count(full_text, 'standard'),
#             'num_std_6mts': extract_account_count_period(full_text, 'standard', '6'),
#             'num_std_12mts': extract_account_count_period(full_text, 'standard', '12'),
#             'num_sub': extract_account_count(full_text, 'sub-standard'),
#             'num_sub_6mts': extract_account_count_period(full_text, 'sub-standard', '6'),
#             'num_dbt': extract_account_count(full_text, 'doubtful'),
#             'num_lss': extract_account_count(full_text, 'loss'),
#             'pct_of_active_TLs_ever': extract_active_tl_percentage(full_text),
#             'pct_currentBal_all_TL': extract_current_balance_pct(full_text),
#             'CC_utilization': extract_cc_utilization(full_text),
#             'PL_utilization': extract_pl_utilization(full_text),
#             'max_unsec_exposure_inPct': extract_unsec_exposure(full_text),
#             'AGE': extract_age(full_text),
#             'NETMONTHLYINCOME': extract_monthly_income(full_text),
#             'Time_With_Curr_Empr': extract_employment_tenure(full_text),
#             'CC_Flag': 1 if detect_product_flag(full_text, 'credit card') else 0,
#             'PL_Flag': 1 if detect_product_flag(full_text, 'personal loan') else 0,
#             'HL_Flag': 1 if detect_product_flag(full_text, 'home loan') else 0,
#             'GL_Flag': 1 if detect_product_flag(full_text, 'gold loan') else 0,
#             'raw_text': full_text,
#             'success': True,
#             'extraction_method': 'OCR'
#         }
#         extracted_data['delinq_severity_score'] = extracted_data['max_delinquency_level'] / 3 if extracted_data['max_delinquency_level'] > 0 else 0
#         extracted_data['high_dpd_risk'] = 1 if (extracted_data['num_times_30p_dpd'] > 2 or extracted_data['num_times_60p_dpd'] > 0) else 0
#         extracted_data['recent_deliq_flag'] = 1 if extracted_data['num_deliq_6mts'] > 0 else 0
#         extracted_data['credit_hungry'] = 1 if extracted_data['enq_L3m'] > 3 else 0
#         extracted_data['account_quality_score'] = (
#             extracted_data['num_std'] * 10 + 
#             extracted_data['num_sub'] * -5 + 
#             extracted_data['num_dbt'] * -15 + 
#             extracted_data['num_lss'] * -25
#         )
#         extracted_data['high_util_flag'] = 1 if extracted_data['pct_currentBal_all_TL'] > 0.75 else 0
#         extracted_data['employment_stable'] = 1 if extracted_data['Time_With_Curr_Empr'] >= 24 else 0
#         return extracted_data
#     except Exception as e:
#         return {'error': str(e), 'message': f'Error extracting CIBIL data: {str(e)}', 'success': False}

# # =============================================================================
# # HYBRID DECISION ENGINE
# # =============================================================================
# def make_hybrid_decision_enhanced(customer_dict):
#     policy_checks = {}
#     age = customer_dict.get('age', 0)
#     employment_type = customer_dict.get('employment_type', 'Salaried')
#     kyc_verified = customer_dict.get('kyc_verified', True)
#     bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
#     fraud_flag = customer_dict.get('fraud_flag', False)
#     age_min, age_max = 24, 70
#     if age < age_min or age > age_max:
#         policy_checks['age'] = f"❌ Age {age} (Required: {age_min}-{age_max})"
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
    
#     # ============================================================
#     # DEPENDENTS CHECK - ADDED HERE
#     # ============================================================
#     dependents = customer_dict.get('dependents', 0)
#     dependents_flag_review = False
#     if dependents > 5:
#         policy_checks['dependents'] = f"⚠️ Dependents {dependents} (>5: Review Required)"
#         dependents_flag_review = True
#     else:
#         policy_checks['dependents'] = f"✅ Dependents {dependents} (1-5: Acceptable)"
#     # ============================================================
    
#     monthly_income = customer_dict.get('avg_salary_6m', 0)
#     employment_tenure = customer_dict.get('employment_tenure_months', 0)
#     business_vintage = customer_dict.get('business_vintage_years', 0)
#     if monthly_income < 15000:
#         policy_checks['income'] = f"❌ Income Rs.{monthly_income:,.0f} (Min: Rs.15,000)"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Income below minimum", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     policy_checks['income'] = f"✅ Income Rs.{monthly_income:,.0f}"
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
#         return {'decision': "REJECT", 'reason': "Policy Gate: Bureau score too low", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
#     policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"
#     if dpd_90 > 0:
#         policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Severe delinquency", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 100.0, 'affordability_data': {}}
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
#         class_probs = {cls: prob * 100 for cls, prob in zip(TARGET_LE.classes_, pred_proba)}
#     except:
#         confidence = 75.0
#         class_probs = {ml_decision: 100.0}
#     # Affordability
#     loan_amount = customer_dict.get('loan_amount', 0)
#     loan_tenure = customer_dict.get('loan_tenure_months', 12)
#     interest_rate = customer_dict.get('interest_rate', 10.5)
#     existing_emi = customer_dict.get('existing_emi', 0)
#     affordability_data = calculate_affordability(monthly_income, loan_amount, interest_rate, loan_tenure, existing_emi)
#     foir = affordability_data['foir_percentage']
#     if ml_decision == "APPROVE" and foir > 45:
#         ml_decision = "REVIEW"
#     # ============================================================
#     # DEPENDENTS RULE APPLICATION - ADDED HERE
#     # ============================================================
#     # Apply dependents rule: >5 dependents forces REVIEW (unless already REJECT)
#     if dependents_flag_review and ml_decision == "APPROVE":
#         ml_decision = "REVIEW"
#     # ============================================================
#     risk_score = calculate_final_risk_score(bureau_score, confidence, foir)
#     pd_percentage = calculate_final_pd(
#         bureau_score=bureau_score, foir=foir, confidence=confidence,
#         dpd_90_count=dpd_90, dpd_30_count=customer_dict.get('dpd_30_count_6m', 0),
#         employment_type=employment_type, employment_tenure=employment_tenure,
#         business_vintage=business_vintage, recent_inquiries=recent_inquiries,
#         ml_decision=ml_decision
#     )
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
# # BATCH PREDICTION ENGINE
# # =============================================================================
# def process_batch_predictions(df):
#     results = []
#     for idx, row in df.iterrows():
#         customer_dict = row.to_dict()
#         for key, value in customer_dict.items():
#             if isinstance(value, str):
#                 if value.lower() in ['yes', 'true', '1']:
#                     customer_dict[key] = True
#                 elif value.lower() in ['no', 'false', '0']:
#                     customer_dict[key] = False
#         required_fields = {
#             'age': 35, 'employment_type': 'Salaried', 'kyc_verified': True,
#             'bankruptcy_flag': False, 'fraud_flag': False, 'employment_tenure_months': 24,
#             'business_vintage_years': 0, 'bureau_score': 700, 'dpd_90_count_6m': 0,
#             'dpd_30_count_6m': 0, 'credit_utilization_pct': 30, 'recent_inquiries_3m': 0,
#             'active_loans_count': 0, 'existing_emi': 0, 'avg_salary_6m': 50000,
#             'AMT_INCOME_TOTAL': 600000, 'net_cash_surplus_6m': 20000, 'salary_stability_flag': 'STABLE',
#             'loan_amount': 180000, 'loan_tenure_months': 24, 'interest_rate': 10.5, 'AMT_ANNUITY': 8500,
#         }
#         for field, default in required_fields.items():
#             if field not in customer_dict or pd.isna(customer_dict[field]):
#                 customer_dict[field] = default
#         try:
#             decision_data = make_hybrid_decision_enhanced(customer_dict)
#             reasons = generate_reason_codes(
#                 decision=decision_data.get('decision', 'ERROR'),
#                 customer_data=customer_dict,
#                 affordability_data=decision_data.get('affordability_data', {}),
#                 policy_checks=decision_data.get('policy_checks', {})
#             )
#             app_id = f"BATCH_{idx+1:04d}"
#             affordability = decision_data.get('affordability_data', {})
#             result = {
#                 'application_id': app_id,
#                 'decision': decision_data.get('decision', 'ERROR'),
#                 'risk_score': decision_data.get('risk_score', 0),
#                 'pd_percentage': decision_data.get('pd_percentage', 0),
#                 'confidence': round(decision_data.get('confidence', 0), 2),
#                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 'reason_1': reasons[0] if len(reasons) > 0 else '',
#                 'reason_2': reasons[1] if len(reasons) > 1 else '',
#                 'reason_3': reasons[2] if len(reasons) > 2 else '',
#                 'age': customer_dict.get('age', ''),
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
#                 'recent_inquiries': customer_dict.get('recent_inquiries_3m', 0),
#                 'active_loans': customer_dict.get('active_loans_count', 0),
#                 'employment_tenure': customer_dict.get('employment_tenure_months', 0),
#                 'business_vintage': customer_dict.get('business_vintage_years', 0),
#                 'salary_stability': customer_dict.get('salary_stability_flag', ''),
#                 'kyc_status': 'Verified' if customer_dict.get('kyc_verified', True) else 'Not Verified',
#                 'bankruptcy': 'Yes' if customer_dict.get('bankruptcy_flag', False) else 'No',
#                 'fraud': 'Yes' if customer_dict.get('fraud_flag', False) else 'No',
#                 'prob_approve': round(decision_data.get('class_probs', {}).get('APPROVE', 0), 2),
#                 'prob_review': round(decision_data.get('class_probs', {}).get('REVIEW', 0), 2),
#                 'prob_reject': round(decision_data.get('class_probs', {}).get('REJECT', 0), 2),
#             }
#         except Exception as e:
#             result = {
#                 'application_id': f"BATCH_{idx+1:04d}",
#                 'decision': 'ERROR',
#                 'risk_score': 0, 'pd_percentage': 0, 'confidence': 0,
#                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 'reason_1': '', 'reason_2': '', 'reason_3': '',
#                 'age': customer_dict.get('age', ''),
#                 'employment_type': customer_dict.get('employment_type', ''),
#                 'bureau_score': customer_dict.get('bureau_score', ''),
#                 'monthly_income': customer_dict.get('avg_salary_6m', ''),
#                 'loan_amount': customer_dict.get('loan_amount', ''),
#                 'error_message': str(e)
#             }
#         results.append(result)
#     return pd.DataFrame(results)

# def create_download_link(df, filename="batch_results.csv"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;">📥 Download CSV</a>'

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
#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1:
#         st.markdown(f'<div class="stat-card"><div class="stat-number">{risk_score}</div><div class="stat-label">Risk Score</div></div>', unsafe_allow_html=True)
#     with col2:
#         st.markdown(f'<div class="stat-card"><div class="stat-number">{pd_score}%</div><div class="stat-label">PD Score</div></div>', unsafe_allow_html=True)
#     with col3:
#         st.markdown(f'<div class="stat-card"><div class="stat-number">Rs.{approved_amount:,.0f}</div><div class="stat-label">Loan Amount</div></div>', unsafe_allow_html=True)
#     with col4:
#         st.markdown(f'<div class="stat-card"><div class="stat-number">{tenure}</div><div class="stat-label">Tenure (Months)</div></div>', unsafe_allow_html=True)
#     with col5:
#         st.markdown(f'<div class="stat-card"><div class="stat-number">{decision_data.get("confidence", 0):.0f}%</div><div class="stat-label">Confidence</div></div>', unsafe_allow_html=True)
#     st.markdown("<br>", unsafe_allow_html=True)
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown(f'<div class="info-box"><strong>📋 Application ID:</strong> {app_id}</div>', unsafe_allow_html=True)
#     with col2:
#         st.markdown(f'<div class="info-box"><strong>🕐 Decision Timestamp:</strong> {timestamp}</div>', unsafe_allow_html=True)

# def render_info_card(title, icon, data_dict, status_dict=None):
#     st.markdown(f'<div class="info-card"><div class="info-card-title"><span class="icon">{icon}</span><span>{title}</span></div><div class="info-card-content">', unsafe_allow_html=True)
#     for label, value in data_dict.items():
#         status = ""
#         if status_dict and label in status_dict:
#             if status_dict[label] == "pass":
#                 status = '<span class="status-badge badge-pass">✓ Passed</span>'
#             elif status_dict[label] == "fail":
#                 status = '<span class="status-badge badge-fail">✗ Failed</span>'
#             elif status_dict[label] == "warning":
#                 status = '<span class="status-badge badge-warning">⚠ Warning</span>'
#         st.markdown(f'<div class="data-row"><span class="data-label">{label}</span><span class="data-value">{value} {status}</span></div>', unsafe_allow_html=True)
#     st.markdown('</div></div>', unsafe_allow_html=True)

# def render_reason_codes(reasons):
#     st.markdown('<div class="info-card"><div class="info-card-title"><span class="icon">📝</span><span>Decision Reasons</span></div><div class="info-card-content">', unsafe_allow_html=True)
#     for i, reason in enumerate(reasons, 1):
#         st.markdown(f'<div class="reason-item"><span class="reason-icon">{i}.</span><span>{reason}</span></div>', unsafe_allow_html=True)
#     st.markdown('</div></div>', unsafe_allow_html=True)

# def create_modern_gauge(value, title, max_value=100):
#     if value <= 50: color = "#f56565"
#     elif value <= 75: color = "#ed8936"
#     else: color = "#48bb78"
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=value,
#         title={'text': title, 'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}},
#         number={'suffix': "%", 'font': {'size': 36, 'color': '#2d3748', 'family': 'Inter'}},
#         gauge={'axis': {'range': [0, max_value], 'tickfont': {'size': 12, 'color': '#718096'}},
#                'bar': {'color': color, 'thickness': 0.75}, 'bgcolor': 'white', 'borderwidth': 0,
#                'steps': [{'range': [0, 50], 'color': '#fed7d7'},
#                          {'range': [50, 75], 'color': '#feebc8'},
#                          {'range': [75, 100], 'color': '#c6f6d5'}]}
#     ))
#     fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white',
#                       font={'family': 'Inter', 'color': '#2d3748'})
#     return fig

# def create_modern_bar_chart(class_probs):
#     df = pd.DataFrame({'Decision': list(class_probs.keys()), 'Probability': list(class_probs.values())})
#     colors = {'REVIEW': '#ed8936', 'APPROVE': '#48bb78', 'REJECT': '#f56565'}
#     fig = px.bar(df, x='Decision', y='Probability', title='Decision Probabilities', color='Decision',
#                  color_discrete_map=colors, text='Probability')
#     fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
#     fig.update_layout(showlegend=False, yaxis_title='Probability (%)', xaxis_title='', height=300,
#                       margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white', plot_bgcolor='white',
#                       font={'family': 'Inter', 'color': '#2d3748'},
#                       yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]},
#                       xaxis={'tickfont': {'size': 14, 'color': '#2d3748'}})
#     return fig

# # =============================================================================
# # STAGE 2 RESULTS DISPLAY FUNCTION (ENHANCED WITH COMPREHENSIVE PDF DATA)
# # =============================================================================
# def display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data):
#     """Display comprehensive Stage 2 results with decision report and download options"""
    
#     st.markdown("---")
#     st.markdown('<p class="main-header">🎯 Stage 2 Final Results</p>', unsafe_allow_html=True)
    
#     # Extract key results
#     final_decision = stage2_result.get('final_decision', 'ERROR')
#     risk_tier = stage2_result.get('tier', 'UNKNOWN')
#     interest_range = stage2_result.get('interest_rate_range', 'N/A')
#     stage2_tier = stage2_result.get('stage2_tier', 'N/A')
#     stage2_confidence = stage2_result.get('stage2_confidence', 0)
#     combined_risk_score = stage2_result.get('combined_risk_score', 0)
    
#     # Decision header card
#     if final_decision == "APPROVE":
#         card_class = "decision-card decision-card-approved"
#         icon = "✓"
#         subtitle = "Application Approved - Proceed to Disbursement"
#     elif final_decision in ["REVIEW", "MANUAL_REVIEW"]:
#         card_class = "decision-card decision-card-review"
#         icon = "⚠"
#         subtitle = "Requires Manual Review"
#     else:
#         card_class = "decision-card decision-card-rejected"
#         icon = "✗"
#         subtitle = "Application Rejected"
    
#     st.markdown(f"""
#         <div class="{card_class}">
#             <div class="decision-title">
#                 <span>{icon}</span>
#                 <span>{final_decision}</span>
#             </div>
#             <div class="decision-subtitle">{subtitle}</div>
#         </div>
#     """, unsafe_allow_html=True)
    
#     # Key metrics
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Risk Tier", stage2_tier)
#     with col2:
#         st.metric("Interest Rate", interest_range)
#     with col3:
#         st.metric("Combined Risk Score", combined_risk_score)
#     with col4:
#         st.metric("Stage 2 Confidence", f"{stage2_confidence:.1f}%")
    
#     st.markdown("<br>", unsafe_allow_html=True)
    
#     # Detailed tabs
#     tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔬 Analysis", "📋 Data", "📥 Download"])
    
#     with tab1:
#         st.markdown("### 📊 Decision Comparison")
#         comparison_df = pd.DataFrame([
#             {
#                 'Stage': 'Stage 1 (Basic)',
#                 'Decision': st.session_state.get('stage1_decision'),
#                 'Risk Score': stage1_data.get('risk_score', 'N/A'),
#                 'Tier': 'N/A'
#             },
#             {
#                 'Stage': 'Stage 2 (CIBIL Deep)',
#                 'Decision': final_decision,
#                 'Risk Score': combined_risk_score,
#                 'Tier': f"{stage2_tier} | {interest_range}"
#             }
#         ])
#         st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
#         # Risk tier explanation
#         st.markdown("### 🎯 Risk Tier Details")
#         tier_info = {
#             'P1': {'name': 'Premium', 'color': '#10B981', 'desc': 'Excellent credit profile'},
#             'P2': {'name': 'Standard', 'color': '#3B82F6', 'desc': 'Good credit profile'},
#             'P3': {'name': 'Subprime', 'color': '#F59E0B', 'desc': 'Fair credit with concerns'},
#             'P4': {'name': 'High Risk', 'color': '#EF4444', 'desc': 'High risk profile'},
#         }
#         if stage2_tier in tier_info:
#             tier_data = tier_info[stage2_tier]
#             st.markdown(f"""
#                 <div style="background: {tier_data['color']}; color: white; padding: 1rem; border-radius: 0.5rem;">
#                     <h3 style="margin: 0; color: white;">{stage2_tier}: {tier_data['name']}</h3>
#                     <p style="margin: 0.5rem 0;">Interest Rate: {interest_range}</p>
#                     <p style="margin: 0;">{tier_data['desc']}</p>
#                 </div>
#             """, unsafe_allow_html=True)
        
#         # Decision reasoning
#         st.markdown("### 📝 Decision Reasoning")
#         st.info(stage2_result.get('reason', 'N/A'))
    
#     with tab2:
#         st.markdown("### 🔬 Detailed Analysis")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("**Tier Probabilities**")
#             if 'tier_probabilities' in stage2_result:
#                 tier_probs = stage2_result['tier_probabilities']
#                 for tier, prob in tier_probs.items():
#                     st.metric(tier, f"{prob:.1f}%")
#         with col2:
#             st.markdown("**Stage Scores**")
#             st.metric("Stage 1 Risk Score", stage1_data.get('risk_score', 'N/A'))
#             st.metric("Stage 2 Risk Score", stage2_result.get('stage2_risk_score', 'N/A'))
#             st.metric("Combined Score", combined_risk_score)
#         with st.expander("📋 Complete Stage 2 Result"):
#             st.json(stage2_result)
    
#     with tab3:
#         st.markdown("### 📋 Input Data")
#         col1, col2 = st.columns(2)
#         with col1:
#             with st.expander("Stage 1 Customer Data"):
#                 st.json(stage1_customer)
#         with col2:
#             with st.expander("Enhanced CIBIL Data"):
#                 st.json(enhanced_customer_data)
    
#     with tab4:
#         st.markdown("### 📥 Download Reports")
        
#         # Extract Stage 1 reasons from customer data
#         stage1_reasons = stage1_customer.get('reason_codes', [])
        
#         # =============================================================================
#         # Compute PD calculation factors from Stage 1 data (for PDF audit trail)
#         # =============================================================================
#         bureau_score = stage1_customer.get('bureau_score', 0)
#         dpd_90 = stage1_customer.get('dpd_90_count_6m', 0)
#         dpd_30 = stage1_customer.get('dpd_30_count_6m', 0)
#         foir = stage1_data.get('affordability_data', {}).get('foir_percentage', 0)
#         employment_type = stage1_customer.get('employment_type', 'Salaried')
#         employment_tenure = stage1_customer.get('employment_tenure_months', 0)
#         business_vintage = stage1_customer.get('business_vintage_years', 0)
#         ml_decision = stage1_data.get('decision', 'ERROR')
#         confidence = stage1_data.get('confidence', 0)
        
#         pd_factors = {
#             'bureau_score': bureau_score,
#             'base_pd': bureau_score_to_pd(bureau_score),
#             'dpd_90': dpd_90,
#             'dpd_30': dpd_30,
#             'delinquency_multiplier': delinquency_to_pd_multiplier(dpd_90, dpd_30),
#             'foir': foir,
#             'foir_adjustment': foir_to_pd_adjustment(foir),
#             'employment_adjustment': employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage),
#             'ml_adjustment': ml_confidence_to_pd_adjustment(confidence, ml_decision),
#             'final_pd': stage1_data.get('pd_percentage', 0)
#         }
#         # =============================================================================
        
#         # Create comprehensive report data that matches expected keys for PDF generator
#         report_data = {
#             # Stage 1 top-level keys (expected by generate_audit_pdf)
#             'application_id': stage1_customer.get('application_id'),
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'decision': stage1_data.get('decision'),
#             'risk_score': stage1_data.get('risk_score'),
#             'pd_percentage': stage1_data.get('pd_percentage'),
#             'confidence': stage1_data.get('confidence'),
#             'policy_checks': stage1_data.get('policy_checks', {}),
#             'affordability_data': stage1_data.get('affordability_data', {}),
#             'customer_data': stage1_customer,
#             'reason_codes': stage1_reasons,
#             'pd_calculation_factors': pd_factors,          # <-- ADDED
#             # Stage 2 data
#             'stage2_final_decision': final_decision,
#             'stage2_tier': stage2_tier,
#             'stage2_interest_range': interest_range,
#             'stage2_combined_risk_score': combined_risk_score,
#             'stage2_confidence': stage2_confidence,
#             'stage2_reason': stage2_result.get('reason'),
#             'stage2_tier_probabilities': stage2_result.get('tier_probabilities'),
#             'stage2_complete_analysis': stage2_result,
#             'stage1_data': stage1_data,
#             'enhanced_customer_data': enhanced_customer_data
#         }
        
#         # PDF download only (JSON and CSV removed as requested)
#         if PDF_AVAILABLE and generate_audit_pdf is not None:
#             try:
#                 pdf_buffer = generate_audit_pdf(report_data)
#                 st.download_button(
#                     "📥 Download PDF Report",
#                     data=pdf_buffer,
#                     file_name=f"stage2_report_{stage1_customer.get('application_id', 'unknown')}.pdf",
#                     mime="application/pdf",
#                     use_container_width=True
#                 )
#             except Exception as e:
#                 st.error(f"PDF generation failed: {str(e)}")
#         else:
#             st.warning("PDF generation is not available. Please install the required PDF generator module.")
    
#     st.markdown("---")
    
#     # Navigation buttons
#     col1, col2, col3 = st.columns(3)
#     app_id_key = stage1_customer.get('application_id', 'unknown')  # use for unique keys

#     with col1:
#         if st.button("🔄 New Assessment", use_container_width=True, key=f"new_assessment_{app_id_key}"):
#             st.session_state.stage1_complete = False
#             st.session_state.stage1_decision = None
#             st.session_state.stage1_data = None
#             st.session_state.current_customer_data = None
#             st.session_state.page_navigation = "👤 Assessment"
#             st.rerun()
#     with col2:
#         if st.button("← Back to Stage 1", use_container_width=True, key=f"back_stage1_{app_id_key}"):
#             st.session_state.page_navigation = "👤 Assessment"
#             st.rerun()
#     with col3:
#         if st.button("🏠 Home", use_container_width=True, key=f"home_{app_id_key}"):
#             st.session_state.page_navigation = "🏠 Home"
#             st.rerun()

# # =============================================================================
# # SIDEBAR
# # =============================================================================
# with st.sidebar:
#     st.markdown("# 🏦 Credit Risk Engine")
#     st.markdown("---")
    
#     # DYNAMIC NAVIGATION BASED ON WORKFLOW STATE
#     navigation_options = ["🏠 Home", "👤 Assessment", "📊 Batch Process", "📈 Model Info", "ℹ️ About"]
    
#     # Add Stage 2 only if Stage 1 complete with APPROVE/REVIEW
#     if (st.session_state.stage1_complete and 
#         st.session_state.stage1_decision in ['APPROVE', 'REVIEW']):
#         navigation_options.insert(2, "🔬 Stage 2 Analysis")
#         st.success(f"✅ Stage 1: {st.session_state.stage1_decision}")
#         st.info("🔬 Stage 2 Analysis unlocked!")
#     elif st.session_state.stage1_complete:
#         st.warning(f"⚠️ Stage 1: {st.session_state.stage1_decision}")
#         st.caption("Stage 2 only for APPROVE/REVIEW")
    
#     # Use radio with key bound to session state for programmatic navigation
#     page = st.radio(
#         "**Navigation**",
#         navigation_options,
#         label_visibility="collapsed",
#         key="page_navigation"
#     )
    
#     st.markdown("---")
    
#     # Enhanced system status
#     stage2_indicator = '✅ Active' if STAGE2_AVAILABLE and is_stage2_available() else '❌ Inactive'
#     ocr_indicator = '✅ Ready' if OCR_AVAILABLE else '❌ Not Installed'
#     pdf_indicator = '✅ Ready' if PDF_AVAILABLE else '❌ Not Installed'
    
#     st.markdown(f"""
#     <div class="info-card">
#         <div class="info-card-title">System Status</div>
#         <div class="info-card-content">
#             <div class="data-row"><span class="data-label">Model</span><span class="data-value">✅ Loaded</span></div>
#             <div class="data-row"><span class="data-label">Version</span><span class="data-value">8.3</span></div>
#             <div class="data-row"><span class="data-label">Stage 2</span><span class="data-value">{stage2_indicator}</span></div>
#             <div class="data-row"><span class="data-label">OCR</span><span class="data-value">{ocr_indicator}</span></div>
#             <div class="data-row"><span class="data-label">PDF Gen</span><span class="data-value">{pdf_indicator}</span></div>
#             <div class="data-row"><span class="data-label">Features</span><span class="data-value">{len(TOP_FEATURES)}</span></div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     with st.expander("🎯 **Top Features**"):
#         for i, feat in enumerate(TOP_FEATURES[:5], 1):
#             st.markdown(f"`{i}.` {feat}")
    
#     # Quick Actions
#     if st.session_state.stage1_complete:
#         st.markdown("---")
#         st.markdown("### 🚀 Quick Actions")
#         if st.button("🔄 New Assessment", use_container_width=True):
#             st.session_state.stage1_complete = False
#             st.session_state.stage1_decision = None
#             st.session_state.stage1_data = None
#             st.session_state.current_customer_data = None
#             st.session_state.extracted_cibil_data = None
#             st.rerun()

# # =============================================================================
# # HOME PAGE
# # =============================================================================
# if page == "🏠 Home":
#     st.markdown('<p class="main-header">Credit Risk Assessment Platform</p>', unsafe_allow_html=True)
#     st.markdown("""
#         <div class="info-box">
#             <h3 style="margin-top: 0;">🎯 AI-Powered Lending Decisions</h3>
#             <p style="margin-bottom: 0;">Comprehensive credit risk evaluation combining hard policy rules, 
#             machine learning models, and affordability analysis for accurate lending decisions.</p>
#         </div>
#     """, unsafe_allow_html=True)
#     st.markdown("<br>", unsafe_allow_html=True)
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.markdown("""
#             <div class="info-card"><div class="info-card-title"><span class="icon">🛡️</span><span>Policy Gates</span></div>
#             <div class="info-card-content"><ul><li>Age & KYC verification</li><li>Employment stability</li>
#             <li>Minimum income checks</li><li>Credit bureau thresholds</li><li>Bankruptcy & fraud detection</li></ul></div></div>
#         """, unsafe_allow_html=True)
#     with col2:
#         st.markdown("""
#             <div class="info-card"><div class="info-card-title"><span class="icon">🤖</span><span>ML Assessment</span></div>
#             <div class="info-card-content"><ul><li>Random Forest classifier</li><li>60K+ training samples</li>
#             <li>Confidence scoring</li><li>Multi-class prediction</li><li>Feature importance</li></ul></div></div>
#         """, unsafe_allow_html=True)
#     with col3:
#         st.markdown("""
#             <div class="info-card"><div class="info-card-title"><span class="icon">💰</span><span>Affordability</span></div>
#             <div class="info-card-content"><ul><li>EMI calculation</li><li>FOIR analysis (max 50%)</li>
#             <li>Net disposable income</li><li>Debt burden assessment</li><li>Affordability scoring</li></ul></div></div>
#         """, unsafe_allow_html=True)
#     st.markdown("<br>", unsafe_allow_html=True)
#     col1, col2, col3, col4 = st.columns(4)
#     with col1: st.metric("🎯 Accuracy", "85%", "+2%")
#     with col2: st.metric("⚡ Avg Response", "1.2s", "-0.3s")
#     with col3: st.metric("📊 Features", len(TOP_FEATURES))
#     with col4: st.metric("🔄 Version", "8.2", "Latest")
#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown("""
#         <div class="warning-box">
#             <strong>🆕 New in Version 8.2:</strong><br>
#             • Completely Fixed PD Calculation<br>
#             • Industry-Standard PD Methodology<br>
#             • Audit Trail as PDF (not JSON)<br>
#             • Age Validation Consistency<br>
#             • FOIR & Policy Gate Integration<br>
#             • Professional UI/UX Enhancements
#         </div>
#     """, unsafe_allow_html=True)

# # =============================================================================
# # ASSESSMENT PAGE (COMPLETE)
# # =============================================================================
# elif page == "👤 Assessment":
#     st.markdown('<p class="main-header">Credit Assessment</p>', unsafe_allow_html=True)
#     st.markdown('<div class="info-box">💡 Complete the form below to assess credit risk. All fields are required for accurate evaluation.</div>', unsafe_allow_html=True)
#     with st.form("assessment_form"):
#         st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             age = st.number_input("Age", 24, 70, 35, help="Customer's age in years (Minimum: 24, Maximum: 70)")
#             employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business'])
#         with col2:
#             # ============================================================
#             # DEPENDENTS INPUT FIELD - ADDED HERE
#             # ============================================================
#             dependents = st.number_input("Number of Dependents", 0, 20, 2,
#                                          help="1-5: Approve eligible | >5: Review required")
#             kyc_verified = st.selectbox("KYC Verified", ['Yes', 'No']) == 'Yes'
#         with col3:
#             bankruptcy_flag = st.selectbox("Bankruptcy Flag", ['No', 'Yes']) == 'Yes'
#             fraud_flag = st.selectbox("Fraud Flag", ['No', 'Yes']) == 'Yes'
#             if employment_type == 'Salaried':
#                 employment_tenure = st.number_input("Employment Tenure (months)", 0, 600, 24)
#                 business_vintage = 0
#             else:
#                 business_vintage = st.number_input("Business Vintage (years)", 0, 50, 3)
#                 employment_tenure = 0
#         st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             bureau_score = st.number_input("Bureau Score", 300, 900, 720, 10)
#             dpd_90_6m = st.number_input("DPD 90+ (Last 6M)", 0, 20, 0)
#             dpd_30_6m = st.number_input("DPD 30+ (Last 6M)", 0, 20, 0)
#         with col2:
#             credit_utilization = st.number_input("Credit Utilization (%)", 0, 100, 30)
#             recent_inquiries = st.number_input("Recent Inquiries (3M)", 0, 20, 2)
#         with col3:
#             active_loans = st.number_input("Active Loans", 0, 10, 1)
#             existing_emi = st.number_input("Existing Total EMI (Rs.)", 0, 200000, 15000, 1000)
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
#         timestamp = datetime.now()
#         app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
#         customer_data = {
#             'age': age, 'employment_type': employment_type,
#             # ============================================================
#             # DEPENDENTS ADDED TO CUSTOMER DATA DICTIONARY
#             # ============================================================
#             'dependents': dependents,
#             'kyc_verified': kyc_verified,
#             'bankruptcy_flag': bankruptcy_flag, 'fraud_flag': fraud_flag,
#             'employment_tenure_months': employment_tenure, 'business_vintage_years': business_vintage,
#             'bureau_score': bureau_score, 'dpd_90_count_6m': dpd_90_6m, 'dpd_30_count_6m': dpd_30_6m,
#             'credit_utilization_pct': credit_utilization, 'recent_inquiries_3m': recent_inquiries,
#             'active_loans_count': active_loans, 'avg_salary_6m': avg_salary, 'AMT_INCOME_TOTAL': amt_income,
#             'net_cash_surplus_6m': net_surplus, 'salary_stability_flag': salary_stability,
#             'loan_amount': loan_amount, 'loan_tenure_months': loan_tenure, 'interest_rate': interest_rate,
#             'existing_emi': existing_emi, 'AMT_ANNUITY': amt_annuity,
#             'application_id': app_id, 'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S")
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

#         # Store in session state for Stage 2
#         st.session_state.stage1_complete = True
#         st.session_state.stage1_decision = decision_data.get('decision', 'ERROR')
#         st.session_state.stage1_data = decision_data
#         st.session_state.current_customer_data = customer_data

#         # Display results (tabs)
#         tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])

#         with tab1:
#             st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
#             col1, col2 = st.columns(2)
#             with col1:
#                 render_info_card("👤 Identity", "👤",
#                                  {"Age": age,
#                                   "Employment": employment_type,
#                                   # ============================================================
#                                   # DEPENDENTS DISPLAYED IN APPLICATION SUMMARY
#                                   # ============================================================
#                                   "Dependents": dependents,
#                                   "KYC Status": "Verified" if kyc_verified else "Not Verified",
#                                   "Tenure": f"{employment_tenure} months" if employment_type == 'Salaried' else f"{business_vintage} years"})
#                 render_info_card("💰 Financial", "💰",
#                                  {"Monthly Income": f"Rs.{avg_salary:,}", "Annual Income": f"Rs.{amt_income:,}",
#                                   "Net Surplus": f"Rs.{net_surplus:,}", "Stability": salary_stability})
#             with col2:
#                 render_info_card("🏦 Credit Bureau", "🏦",
#                                  {"Bureau Score": bureau_score, "DPD 90+": dpd_90_6m, "DPD 30+": dpd_30_6m,
#                                   "Utilization": f"{credit_utilization}%", "Recent Inquiries": recent_inquiries,
#                                   "Existing EMI": f"Rs.{existing_emi:,}"})
#                 render_info_card("📋 Loan Request", "📋",
#                                  {"Amount": f"Rs.{loan_amount:,}", "Tenure": f"{loan_tenure} months",
#                                   "Interest Rate": f"{interest_rate}%", "Requested EMI": f"Rs.{amt_annuity:,}"})

#         with tab2:
#             st.markdown('<p class="section-header">Decision Summary</p>', unsafe_allow_html=True)
#             render_decision_header(decision_data, customer_data)
#             st.markdown("<br>", unsafe_allow_html=True)
            
#             # STAGE 2 BUTTONS - now three options under the eligibility box
#             final_decision = decision_data.get('decision', 'ERROR')
            
#             if final_decision in ['APPROVE', 'REVIEW']:
#                 st.markdown("---")
#                 st.markdown("""
#                     <div class="info-box" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white; text-align: center;">
#                         <h3 style="margin: 0; color: white;">✅ Eligible for Stage 2 Deep Dive</h3>
#                         <p style="margin: 0.5rem 0 0 0;">Choose an input method to proceed:</p>
#                     </div>
#                 """, unsafe_allow_html=True)
                
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     if st.button("📝 Manual Entry", use_container_width=True, type="primary"):
#                         st.session_state.stage2_selected_tab = "Manual Entry"
#                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
#                         st.rerun()
#                 with col2:
#                     if st.button("📄 PDF Upload", use_container_width=True, type="primary"):
#                         st.session_state.stage2_selected_tab = "PDF Upload"
#                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
#                         st.rerun()
#                 with col3:
#                     if st.button("📊 Batch Analysis", use_container_width=True, type="primary"):
#                         st.session_state.stage2_selected_tab = "Batch Analysis"
#                         st.session_state.page_navigation = "🔬 Stage 2 Analysis"
#                         st.rerun()
#             elif final_decision == 'REJECT':
#                 st.markdown("---")
#                 st.markdown("""
#                     <div class="warning-box" style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; text-align: center;">
#                         <h3 style="margin: 0; color: white;">❌ Stage 2 Not Available</h3>
#                         <p style="margin: 0.5rem 0 0 0;">Application rejected. Stage 2 only for APPROVE/REVIEW.</p>
#                     </div>
#                 """, unsafe_allow_html=True)
            
#             # Continue with policy checks
#             st.markdown("<br>", unsafe_allow_html=True)
#             affordability = decision_data.get('affordability_data', {})
#             foir = affordability.get('foir_percentage', 0)
#             total_emi = affordability.get('total_emi', 0)
#             net_disp = affordability.get('net_disposable', 0)
            
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 render_info_card("Identity & Eligibility", "👤",
#                                 {f"Age: {age}": "",
#                                  f"Employment: {employment_type}": "",
#                                  # ============================================================
#                                  # DEPENDENTS ADDED TO POLICY CHECKS CARD
#                                  # ============================================================
#                                  f"Dependents: {dependents}": "",
#                                  f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""},
#                                 {f"Age: {age}": "pass" if 24 <= age <= 70 else "fail",
#                                  f"Employment: {employment_type}": "pass",
#                                  f"Dependents: {dependents}": "pass" if dependents <= 5 else "warning",
#                                  f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_verified else "fail"})
#             with col2:
#                 bureau_pass = bureau_score >= 550
#                 dpd_pass = dpd_90_6m == 0
#                 render_info_card("Credit Bureau", "🏦",
#                                 {f"Bureau Score: {bureau_score}": "", f"DPD 90+: {dpd_90_6m}": "",
#                                  f"Utilization: {credit_utilization}%": ""},
#                                 {f"Bureau Score: {bureau_score}": "pass" if bureau_pass else "fail",
#                                  f"DPD 90+: {dpd_90_6m}": "pass" if dpd_pass else "fail",
#                                  f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"})
#             with col3:
#                 render_info_card("Affordability", "💰",
#                                 {f"Monthly Income: Rs.{avg_salary:,}": "", f"FOIR: {foir:.1f}%": "",
#                                  f"Total EMI: Rs.{total_emi:,}": "", f"Net Disposable: Rs.{net_disp:,}": ""},
#                                 {f"Monthly Income: Rs.{avg_salary:,}": "pass",
#                                  f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
#                                  f"Total EMI: Rs.{total_emi:,}": "pass",
#                                  f"Net Disposable: Rs.{net_disp:,}": "pass" if net_disp >= 10000 else "warning"})
            
#             st.markdown("<br>", unsafe_allow_html=True)
#             render_reason_codes(reasons)
#             st.markdown("<br>", unsafe_allow_html=True)
            
#             # PDF download buttons
#             col1, col2, col3 = st.columns([1, 1, 2])
#             with col1:
#                 if PDF_AVAILABLE and generate_decision_pdf is not None:
#                     try:
#                         pdf_buffer = generate_decision_pdf(
#                             decision_data=decision_data, customer_data=customer_data,
#                             affordability_data=decision_data.get('affordability_data', {}), reasons=reasons)
#                         st.download_button("📥 Decision Report (PDF)", data=pdf_buffer,
#                                            file_name=f"credit_decision_{app_id}.pdf", mime="application/pdf", use_container_width=True)
#                     except Exception as e:
#                         st.error(f"Error generating PDF: {str(e)}")
#                 else:
#                     st.warning("PDF generation not available.")
#             with col2:
#                 if st.button("🔄 Re-Evaluate", use_container_width=True):
#                     st.rerun()

#         with tab3:
#             st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
#             col1, col2 = st.columns(2)
#             with col1:
#                 fig1 = create_modern_gauge(decision_data.get('confidence', 0), "Model Confidence")
#                 st.plotly_chart(fig1, use_container_width=True)
#             with col2:
#                 final_decision = decision_data.get('decision', 'ERROR')
#                 if final_decision == "REVIEW":
#                     class_probs = {"APPROVE": 0, "REVIEW": 100, "REJECT": 0}
#                 elif final_decision == "REJECT":
#                     class_probs = {"APPROVE": 0, "REVIEW": 0, "REJECT": 100}
#                 else:
#                     class_probs = decision_data.get('class_probs', {"APPROVE": 0, "REVIEW": 0, "REJECT": 0})
#                 fig2 = create_modern_bar_chart(class_probs)
#                 st.plotly_chart(fig2, use_container_width=True)

#             st.markdown("<br>", unsafe_allow_html=True)
#             st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
#             policy_df = pd.DataFrame([{'Check': k, 'Result': v} for k, v in decision_data.get('policy_checks', {}).items()])
#             st.dataframe(policy_df, use_container_width=True, hide_index=True)
#             st.markdown('<p class="section-header">PD Calculation Breakdown</p>', unsafe_allow_html=True)
#             pd_factors_display = {
#                 'Bureau Score': f"{bureau_score} → Base PD: {bureau_score_to_pd(bureau_score):.1f}%",
#                 'Delinquency': f"DPD 90+: {dpd_90_6m}, DPD 30+: {dpd_30_6m} → Multiplier: {delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x",
#                 'FOIR Impact': f"{foir:.1f}% → Adjustment: {foir_to_pd_adjustment(foir):.1f}%",
#                 'Employment Stability': f"{employment_type}, {employment_tenure if employment_type == 'Salaried' else business_vintage}{' months' if employment_type == 'Salaried' else ' years'} → Adjustment: {employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage):.1f}%",
#                 'ML Confidence': f"{decision_data.get('confidence', 0):.1f}% → Adjustment: {ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')):.1f}%",
#                 'Final PD': f"{decision_data.get('pd_percentage', 0)}%"
#             }
#             for factor, value in pd_factors_display.items():
#                 st.markdown(f"**{factor}:** {value}")

#         with tab4:
#             st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
#             audit_log_raw = {
#                 'application_id': app_id,
#                 'timestamp': timestamp.isoformat(),
#                 'decision': decision_data.get('decision', 'ERROR'),
#                 'risk_score': decision_data.get('risk_score', 0),
#                 'pd_percentage': decision_data.get('pd_percentage', 0),
#                 'confidence': round(decision_data.get('confidence', 0), 2),
#                 'model_version': '8.2',
#                 'reason_codes': reasons,
#                 'policy_checks': decision_data.get('policy_checks', {}),
#                 'affordability': decision_data.get('affordability_data', {}),
#                 'customer_data': {k: v for k, v in customer_data.items() if k not in ['application_id', 'timestamp', 'reason_codes']},
#                 'pd_calculation_factors': {
#                     'bureau_score': bureau_score,
#                     'base_pd': bureau_score_to_pd(bureau_score),
#                     'dpd_90': dpd_90_6m, 'dpd_30': dpd_30_6m,
#                     'delinquency_multiplier': delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m),
#                     'foir': foir,
#                     'foir_adjustment': foir_to_pd_adjustment(foir),
#                     'employment_adjustment': employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage),
#                     'ml_adjustment': ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')),
#                     'final_pd': decision_data.get('pd_percentage', 0)
#                 }
#             }
#             audit_log = sanitize_for_json(audit_log_raw)

#             with st.expander("📋 View Audit Log (JSON)"):
#                 st.json(audit_log)

#             col1, col2 = st.columns(2)
#             with col1:
#                 if PDF_AVAILABLE and generate_audit_pdf is not None:
#                     try:
#                         audit_pdf_buffer = generate_audit_pdf(audit_log)
#                         st.download_button("📥 Download Audit Trail (PDF)",
#                                            data=audit_pdf_buffer,
#                                            file_name=f"audit_trail_{app_id}.pdf",
#                                            mime="application/pdf",
#                                            use_container_width=True)
#                     except Exception as e:
#                         st.error(f"Error generating audit PDF: {str(e)}")
#                 else:
#                     st.warning("Audit PDF generation is not available.")
#             with col2:
#                 audit_json = json.dumps(audit_log, indent=2)
#                 st.download_button("📥 Download Audit Log (JSON)",
#                                    data=audit_json,
#                                    file_name=f"audit_{app_id}.json",
#                                    mime="application/json",
#                                    use_container_width=True)

#             st.markdown('<p class="section-header">PD Calculation Summary</p>', unsafe_allow_html=True)
#             pd_table = pd.DataFrame([
#                 {"Factor": "Bureau Score", "Value": f"{bureau_score}", "Impact": f"{bureau_score_to_pd(bureau_score):.1f}% base PD"},
#                 {"Factor": "Delinquency (DPD 90+)", "Value": f"{dpd_90_6m} times", "Impact": f"{delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x multiplier"},
#                 {"Factor": "FOIR", "Value": f"{foir:.1f}%", "Impact": f"{foir_to_pd_adjustment(foir):.1f}% adjustment"},
#                 {"Factor": "Employment Stability", "Value": f"{employment_type} ({employment_tenure if employment_type == 'Salaried' else business_vintage}{' months' if employment_type == 'Salaried' else ' years'})", "Impact": f"{employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage):.1f}% adjustment"},
#                 {"Factor": "ML Decision Confidence", "Value": f"{decision_data.get('confidence', 0):.1f}% ({decision_data.get('decision', 'ERROR')})", "Impact": f"{ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')):.1f}% adjustment"},
#                 {"Factor": "Final PD", "Value": f"{decision_data.get('pd_percentage', 0)}%", "Impact": "Industry-standard calculation"}
#             ])
#             st.dataframe(pd_table, use_container_width=True, hide_index=True)


# # =============================================================================
# # STAGE 2 ANALYSIS PAGE (FULL WITH RADIO TAB SELECTION)
# # =============================================================================
# elif page == "🔬 Stage 2 Analysis":
#     st.markdown('<p class="main-header">Stage 2: CIBIL Deep Dive</p>', unsafe_allow_html=True)
    
#     # SECURITY CHECK
#     if not st.session_state.get('stage1_complete', False):
#         st.error("❌ You must complete Stage 1 Assessment first!")
#         st.info("Please go to the 👤 Assessment page and submit an application.")
#         if st.button("← Go to Assessment", use_container_width=True):
#             st.session_state.page_navigation = "👤 Assessment"
#             st.rerun()
#         st.stop()
    
#     if st.session_state.get('stage1_decision') not in ['APPROVE', 'REVIEW']:
#         st.error("❌ Stage 2 is only available for APPROVED or REVIEW applications!")
#         st.warning(f"Your Stage 1 decision: {st.session_state.get('stage1_decision', 'Unknown')}")
#         st.info("Only APPROVE and REVIEW decisions can proceed to Stage 2 CIBIL deep dive.")
#         if st.button("← Go Back", use_container_width=True):
#             st.session_state.page_navigation = "👤 Assessment"
#             st.rerun()
#         st.stop()
    
#     if not (STAGE2_AVAILABLE and is_stage2_available()):
#         st.error("❌ Stage 2 model not available!")
#         st.info("Please ensure `stage2_cibil_model.pkl` is in the project directory.")
#         if st.button("← Go Back", use_container_width=True):
#             st.session_state.page_navigation = "👤 Assessment"
#             st.rerun()
#         st.stop()
    
#     # STAGE 1 SUMMARY BANNER
#     stage1_data = st.session_state.get('stage1_data', {})
#     stage1_customer = st.session_state.get('current_customer_data', {})
    
#     st.markdown(f"""
#         <div class="info-box" style="background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%); color: white;">
#             <h3 style="margin: 0; color: white;">📊 Stage 1 Results</h3>
#             <p style="margin: 0.5rem 0 0 0;">
#                 <strong>Decision:</strong> {st.session_state.get('stage1_decision', 'N/A')} | 
#                 <strong>Risk Score:</strong> {stage1_data.get('risk_score', 'N/A')} | 
#                 <strong>Application ID:</strong> {stage1_customer.get('application_id', 'N/A')}
#             </p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("<br>", unsafe_allow_html=True)
    
#     # RADIO BUTTONS AS TABS
#     tab_options = ["Manual Entry", "PDF Upload", "Batch Analysis"]
#     default_tab = st.session_state.get('stage2_selected_tab', 'Manual Entry')
#     if default_tab not in tab_options:
#         default_tab = "Manual Entry"
#     selected_tab = st.radio(
#         "Select input method",
#         tab_options,
#         index=tab_options.index(default_tab),
#         horizontal=True,
#         label_visibility="collapsed"
#     )
    
#     # =========================================================================
#     # TAB 1: MANUAL ENTRY
#     # =========================================================================
#     if selected_tab == "Manual Entry":
#         st.markdown('<p class="section-header">Manual CIBIL Data Entry</p>', unsafe_allow_html=True)
#         st.markdown("""
#             <div class="info-box">
#                 📝 <strong>Manual Data Entry</strong><br>
#                 Enter CIBIL bureau data to enhance Stage 1 customer profile.<br>
#                 The Stage 2 model will use this data to predict risk tier (P1/P2/P3/P4).
#             </div>
#         """, unsafe_allow_html=True)
        
#         with st.form("stage2_manual_form"):
#             st.markdown("### 📋 Application Reference")
#             col1, col2 = st.columns(2)
#             with col1:
#                 ref_app_id = st.text_input(
#                     "Application ID", 
#                     value=stage1_customer.get('application_id', 'N/A'),
#                     disabled=True
#                 )
#                 stage1_decision_display = st.text_input(
#                     "Stage 1 Decision",
#                     value=st.session_state.get('stage1_decision', 'N/A'),
#                     disabled=True
#                 )
#             with col2:
#                 customer_name = st.text_input("Customer Name (Optional)", "")
#                 stage1_risk_display = st.number_input(
#                     "Stage 1 Risk Score", 
#                     value=int(stage1_data.get('risk_score', 750)),
#                     disabled=True
#                 )
            
#             st.markdown("---")
#             st.markdown("### 🏦 CIBIL Bureau Data")
            
#             # Credit Score & Delinquency
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.markdown("**Credit Score & History**")
#                 cibil_score = st.number_input("Credit Score", 300, 900, 720, 10)
#                 max_delinquency = st.number_input("Max Delinquency Level", 0, 100, 0)
#                 num_times_30dpd = st.number_input("Times 30+ DPD", 0, 50, 0)
#                 num_times_60dpd = st.number_input("Times 60+ DPD", 0, 50, 0)
#                 num_times_delinquent = st.number_input("Total Delinquent", 0, 50, 0)
            
#             with col2:
#                 st.markdown("**Recent Behavior (6-12M)**")
#                 num_deliq_6m = st.number_input("Delinquencies (6M)", 0, 20, 0)
#                 num_deliq_12m = st.number_input("Delinquencies (12M)", 0, 20, 0)
#                 max_deliq_6m = st.number_input("Max Delinq (6M)", 0, 100, 0)
#                 max_deliq_12m = st.number_input("Max Delinq (12M)", 0, 100, 0)
#                 enq_L3m = st.number_input("Inquiries (3M)", 0, 20, 2)
#                 enq_L6m = st.number_input("Inquiries (6M)", 0, 30, 4)
#                 enq_L12m = st.number_input("Inquiries (12M)", 0, 50, 6)
            
#             with col3:
#                 st.markdown("**Account Quality**")
#                 num_std = st.number_input("Standard Accounts", 0, 50, 3)
#                 num_std_6m = st.number_input("Standard (6M)", 0, 50, 3)
#                 num_std_12m = st.number_input("Standard (12M)", 0, 50, 3)
#                 num_sub = st.number_input("Sub-standard", 0, 20, 0)
#                 num_sub_6m = st.number_input("Sub-standard (6M)", 0, 20, 0)
#                 num_dbt = st.number_input("Doubtful", 0, 10, 0)
#                 num_lss = st.number_input("Loss", 0, 10, 0)
            
#             # Utilization & Demographics
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.markdown("**Utilization**")
#                 pct_active_tls = st.number_input("% Active TLs", 0.0, 1.0, 0.60, 0.01)
#                 pct_current_bal = st.number_input("Current Balance %", 0.0, 1.0, 0.30, 0.01)
#                 cc_utilization = st.number_input("CC Utilization", 0.0, 1.0, 0.35, 0.01)
#                 pl_utilization = st.number_input("PL Utilization", 0.0, 1.0, 0.25, 0.01)
#                 max_unsec_exposure = st.number_input("Max Unsec Exposure %", 0, 100, 30)
            
#             with col2:
#                 st.markdown("**Demographics**")
#                 age_cibil = st.number_input(
#                     "Age", 
#                     24, 70, 
#                     stage1_customer.get('age', 35)
#                 )
#                 net_monthly_income = st.number_input(
#                     "Net Monthly Income", 
#                     0, 1000000, 
#                     stage1_customer.get('avg_salary_6m', 50000), 
#                     5000
#                 )
#                 time_curr_employer = st.number_input(
#                     "Employment Tenure (months)", 
#                     0, 600, 
#                     stage1_customer.get('employment_tenure_months', 24)
#                 )
            
#             with col3:
#                 st.markdown("**Product Flags**")
#                 cc_flag = st.selectbox("Credit Card", ["Yes", "No"]) == "Yes"
#                 pl_flag = st.selectbox("Personal Loan", ["Yes", "No"]) == "No"
#                 hl_flag = st.selectbox("Home Loan", ["Yes", "No"]) == "No"
#                 gl_flag = st.selectbox("Gold Loan", ["Yes", "No"]) == "No"
            
#             st.markdown("<br>", unsafe_allow_html=True)
#             submitted = st.form_submit_button("🔬 Run Stage 2 Analysis", use_container_width=True, type="primary")
        
#         if submitted:
#             with st.spinner("🔬 Running Stage 2 CIBIL Deep Analysis..."):
#                 enhanced_customer_data = stage1_customer.copy()
#                 enhanced_customer_data.update({
#                     'bureau_score': cibil_score,
#                     'age': age_cibil,
#                     'avg_salary_6m': net_monthly_income,
#                     'employment_tenure_months': time_curr_employer,
#                     'dpd_30_count_6m': num_times_30dpd,
#                     'dpd_90_count_6m': num_times_60dpd,
#                     'max_delinquency_level': max_delinquency,
#                     'num_times_delinquent': num_times_delinquent,
#                     'num_deliq_6mts': num_deliq_6m,
#                     'num_deliq_12mts': num_deliq_12m,
#                     'max_deliq_6mts': max_deliq_6m,
#                     'max_deliq_12mts': max_deliq_12m,
#                     'recent_inquiries_3m': enq_L3m,
#                     'enq_L6m': enq_L6m,
#                     'enq_L12m': enq_L12m,
#                     'active_loans_count': num_std,
#                     'num_std_6mts': num_std_6m,
#                     'num_std_12mts': num_std_12m,
#                     'num_sub': num_sub,
#                     'num_sub_6mts': num_sub_6m,
#                     'num_dbt': num_dbt,
#                     'num_lss': num_lss,
#                     'credit_utilization_pct': cc_utilization * 100,
#                     'pct_of_active_TLs_ever': pct_active_tls,
#                     'pct_currentBal_all_TL': pct_current_bal,
#                     'CC_utilization': cc_utilization,
#                     'PL_utilization': pl_utilization,
#                     'max_unsec_exposure_inPct': max_unsec_exposure,
#                     'CC_Flag': 1 if cc_flag else 0,
#                     'PL_Flag': 1 if pl_flag else 0,
#                     'HL_Flag': 1 if hl_flag else 0,
#                     'GL_Flag': 1 if gl_flag else 0,
#                 })
#                 try:
#                     stage2_result = make_two_stage_decision(
#                         enhanced_customer_data,
#                         stage1_function=make_hybrid_decision_enhanced
#                     )
#                     display_stage2_results(
#                         stage2_result, 
#                         stage1_data, 
#                         stage1_customer, 
#                         enhanced_customer_data
#                     )
#                 except Exception as e:
#                     st.error(f"❌ Stage 2 analysis failed: {str(e)}")
#                     st.exception(e)
#                     st.info("Please verify the data and try again.")
    
#     # =========================================================================
#     # TAB 2: PDF UPLOAD
#     # =========================================================================
#     elif selected_tab == "PDF Upload":
#         st.markdown('<p class="section-header">📄 CIBIL PDF Upload</p>', unsafe_allow_html=True)
#         if not OCR_AVAILABLE:
#             st.error("❌ OCR libraries not installed!")
#             st.info("Install with: `pip install pytesseract pdf2image opencv-python pillow`")
#             st.warning("⚠️ For now, please use the **Manual Entry** tab.")
#         else:
#             st.markdown("""
#                 <div class="info-box">
#                     📄 <strong>CIBIL PDF Extraction</strong><br>
#                     Upload a CIBIL bureau report PDF for automatic extraction and analysis.
#                 </div>
#             """, unsafe_allow_html=True)
#             uploaded_pdf = st.file_uploader(
#                 "Upload CIBIL Report (PDF)", 
#                 type=['pdf'], 
#                 key="stage2_pdf"
#             )
#             if uploaded_pdf is not None:
#                 st.success(f"✅ File uploaded: {uploaded_pdf.name} ({uploaded_pdf.size / 1024:.1f} KB)")
#                 if st.button("🔬 Extract & Analyze", type="primary", use_container_width=True):
#                     with st.spinner("🔄 Extracting data from PDF..."):
#                         extraction_result = extract_cibil_from_pdf(uploaded_pdf)
#                         if extraction_result.get('success', False):
#                             st.success("✅ PDF extraction successful!")
#                             with st.expander("📋 View Extracted Data"):
#                                 st.json({
#                                     'Credit Score': extraction_result.get('Credit_Score'),
#                                     'Delinquency Level': extraction_result.get('max_delinquency_level'),
#                                     'Monthly Income': extraction_result.get('NETMONTHLYINCOME'),
#                                 })
#                             enhanced_customer_data = stage1_customer.copy()
#                             enhanced_customer_data.update({
#                                 'bureau_score': extraction_result.get('Credit_Score', 720),
#                                 'age': extraction_result.get('AGE', stage1_customer.get('age', 35)),
#                                 'avg_salary_6m': extraction_result.get('NETMONTHLYINCOME', stage1_customer.get('avg_salary_6m')),
#                                 'employment_tenure_months': extraction_result.get('Time_With_Curr_Empr', stage1_customer.get('employment_tenure_months')),
#                                 'dpd_30_count_6m': extraction_result.get('num_times_30p_dpd', 0),
#                                 'dpd_90_count_6m': extraction_result.get('num_times_60p_dpd', 0),
#                                 'max_delinquency_level': extraction_result.get('max_delinquency_level', 0),
#                                 'num_times_delinquent': extraction_result.get('num_times_delinquent', 0),
#                                 'num_deliq_6mts': extraction_result.get('num_deliq_6mts', 0),
#                                 'num_deliq_12mts': extraction_result.get('num_deliq_12mts', 0),
#                                 'max_deliq_6mts': extraction_result.get('max_deliq_6mts', 0),
#                                 'max_deliq_12mts': extraction_result.get('max_deliq_12mts', 0),
#                                 'recent_inquiries_3m': extraction_result.get('enq_L3m', 2),
#                                 'enq_L6m': extraction_result.get('enq_L6m', 4),
#                                 'enq_L12m': extraction_result.get('enq_L12m', 6),
#                                 'active_loans_count': extraction_result.get('num_std', 1),
#                                 'num_std_6mts': extraction_result.get('num_std_6mts', 0),
#                                 'num_std_12mts': extraction_result.get('num_std_12mts', 0),
#                                 'num_sub': extraction_result.get('num_sub', 0),
#                                 'num_sub_6mts': extraction_result.get('num_sub_6mts', 0),
#                                 'num_dbt': extraction_result.get('num_dbt', 0),
#                                 'num_lss': extraction_result.get('num_lss', 0),
#                                 'credit_utilization_pct': extraction_result.get('CC_utilization', 0.35) * 100,
#                                 'pct_of_active_TLs_ever': extraction_result.get('pct_of_active_TLs_ever', 0.6),
#                                 'pct_currentBal_all_TL': extraction_result.get('pct_currentBal_all_TL', 0.3),
#                                 'CC_utilization': extraction_result.get('CC_utilization', 0.35),
#                                 'PL_utilization': extraction_result.get('PL_utilization', 0.25),
#                                 'max_unsec_exposure_inPct': extraction_result.get('max_unsec_exposure_inPct', 30),
#                                 'CC_Flag': extraction_result.get('CC_Flag', 0),
#                                 'PL_Flag': extraction_result.get('PL_Flag', 0),
#                                 'HL_Flag': extraction_result.get('HL_Flag', 0),
#                                 'GL_Flag': extraction_result.get('GL_Flag', 0),
#                             })
#                             with st.spinner("🔬 Running Stage 2 analysis..."):
#                                 try:
#                                     stage2_result = make_two_stage_decision(
#                                         enhanced_customer_data,
#                                         stage1_function=make_hybrid_decision_enhanced
#                                     )
#                                     display_stage2_results(
#                                         stage2_result, 
#                                         stage1_data, 
#                                         stage1_customer, 
#                                         enhanced_customer_data
#                                     )
#                                 except Exception as e:
#                                     st.error(f"❌ Analysis failed: {str(e)}")
#                         else:
#                             st.error("❌ PDF extraction failed!")
#                             st.warning(f"Error: {extraction_result.get('error')}")
    
#     # =========================================================================
#     # TAB 3: BATCH ANALYSIS (PLACEHOLDER)
#     # =========================================================================
#     elif selected_tab == "Batch Analysis":
#         st.markdown('<p class="section-header">📊 Batch CIBIL Analysis</p>', unsafe_allow_html=True)
#         st.info("📊 Batch analysis feature coming soon!")

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
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#             st.success(f"✅ Successfully loaded {len(df)} records")
#             with st.expander("📄 Preview Uploaded Data"):
#                 st.dataframe(df.head(), use_container_width=True)
#                 st.write(f"**Total Records:** {len(df)}")
#                 st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
#             required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
#             missing_cols = [col for col in required_cols if col not in df.columns]
#             if missing_cols:
#                 st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
#                 st.info("Please ensure your CSV includes at least these columns: age, employment_type, avg_salary_6m, bureau_score, loan_amount")
#             else:
#                 if st.button("🚀 Process Batch Predictions", type="primary", use_container_width=True):
#                     with st.spinner(f"🔍 Processing {len(df)} records..."):
#                         progress_bar = st.progress(0)
#                         results_df = process_batch_predictions(df)
#                         progress_bar.progress(100)
#                         st.success(f"✅ Completed processing {len(results_df)} records!")
#                         tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
#                         with tab1:
#                             st.dataframe(results_df, use_container_width=True)
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
#                             col1, col2 = st.columns(2)
#                             with col1:
#                                 decision_counts = results_df['decision'].value_counts()
#                                 fig1 = px.pie(values=decision_counts.values, names=decision_counts.index,
#                                               title="Decision Distribution", color=decision_counts.index,
#                                               color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
#                                 st.plotly_chart(fig1, use_container_width=True)
#                             with col2:
#                                 fig2 = px.histogram(results_df, x='risk_score', title="Risk Score Distribution",
#                                                     nbins=20, color_discrete_sequence=['#587042'])
#                                 st.plotly_chart(fig2, use_container_width=True)
#                             fig3 = px.scatter(results_df, x='monthly_income', y='loan_amount', color='decision',
#                                               size='risk_score', title="Income vs Loan Amount (Colored by Decision)",
#                                               hover_data=['application_id', 'foir_percentage'],
#                                               color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
#                             st.plotly_chart(fig3, use_container_width=True)
#                             fig4 = px.box(results_df, x='decision', y='pd_percentage',
#                                           title="PD Distribution by Decision", color='decision',
#                                           color_discrete_map={'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'})
#                             st.plotly_chart(fig4, use_container_width=True)
#                         with tab3:
#                             st.markdown("### Download Results")
#                             col1, col2 = st.columns(2)
#                             with col1:
#                                 csv = results_df.to_csv(index=False)
#                                 st.download_button("📥 Download as CSV", data=csv,
#                                                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                                    mime="text/csv", use_container_width=True)
#                             with col2:
#                                 json_data = results_df.to_json(orient='records', indent=2)
#                                 st.download_button("📥 Download as JSON", data=json_data,
#                                                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                                                    mime="application/json", use_container_width=True)
#                             st.markdown("---")
#                             st.markdown("#### Filtered Downloads")
#                             col1, col2, col3 = st.columns(3)
#                             with col1:
#                                 approved_df = results_df[results_df['decision'] == 'APPROVE']
#                                 if len(approved_df) > 0:
#                                     st.download_button(f"✅ Approved Only ({len(approved_df)})",
#                                                        data=approved_df.to_csv(index=False),
#                                                        file_name=f"approved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                                        mime="text/csv", use_container_width=True)
#                             with col2:
#                                 rejected_df = results_df[results_df['decision'] == 'REJECT']
#                                 if len(rejected_df) > 0:
#                                     st.download_button(f"❌ Rejected Only ({len(rejected_df)})",
#                                                        data=rejected_df.to_csv(index=False),
#                                                        file_name=f"rejected_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                                        mime="text/csv", use_container_width=True)
#                             with col3:
#                                 review_df = results_df[results_df['decision'] == 'REVIEW']
#                                 if len(review_df) > 0:
#                                     st.download_button(f"⚠️ Review Only ({len(review_df)})",
#                                                        data=review_df.to_csv(index=False),
#                                                        file_name=f"review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                                                        mime="text/csv", use_container_width=True)
#         except Exception as e:
#             st.error(f"❌ Error processing file: {str(e)}")
#             st.info("Please ensure the CSV file is properly formatted and contains the required columns.")
#     else:
#         st.markdown("---")
#         st.markdown("### 📋 CSV Template")
#         template_data = {
#             'age': [35, 42, 28],
#             'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
#             # ============================================================
#             # DEPENDENTS COLUMN ADDED TO CSV TEMPLATE
#             # ============================================================
#             'dependents': [2, 3, 6],
#             'kyc_verified': ['Yes', 'Yes', 'No'],
#             'bankruptcy_flag': ['No', 'No', 'No'],
#             'fraud_flag': ['No', 'No', 'No'],
#             'employment_tenure_months': [24, 0, 18],
#             'business_vintage_years': [0, 5, 0],
#             'bureau_score': [720, 680, 580],
#             'dpd_90_count_6m': [0, 1, 2],
#             'dpd_30_count_6m': [0, 2, 1],
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
#         csv_template = template_df.to_csv(index=False)
#         st.download_button("📥 Download CSV Template", data=csv_template,
#                            file_name="credit_assessment_template.csv", mime="text/csv", use_container_width=True)

# # =============================================================================
# # MODEL INFO PAGE
# # =============================================================================
# elif page == "📈 Model Info":
#     st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.markdown('<div class="stat-card"><div class="stat-number">RF</div><div class="stat-label">Model Type</div></div>', unsafe_allow_html=True)
#     with col2:
#         st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TOP_FEATURES)}</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)
#     with col3:
#         st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TARGET_LE.classes_)}</div><div class="stat-label">Classes</div></div>', unsafe_allow_html=True)
#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
#     feature_df = pd.DataFrame({'Rank': range(1, min(21, len(TOP_FEATURES) + 1)), 'Feature': TOP_FEATURES[:20]})
#     st.dataframe(feature_df, use_container_width=True, hide_index=True)

# # =============================================================================
# # ABOUT PAGE
# # =============================================================================
# elif page == "ℹ️ About":
#     st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
#     st.markdown("""
#         <div class="info-card">
#             <div class="info-card-title"><span class="icon">🏦</span><span>Credit Risk Assessment Platform</span></div>
#             <div class="info-card-content">
#                 <p><strong>Version:</strong> 8.2 - COMPLETELY FIXED PD CALCULATION</p>
#                 <p><strong>Developer:</strong> Zen Meraki</p>
#                 <p><strong>Date:</strong> January 2026</p>
#                 <br>
#                 <p>A comprehensive credit risk evaluation system combining hard policy rules,
#                 machine learning models, and affordability analysis for accurate and compliant lending decisions.</p>
#             </div>
#         </div>
#     """, unsafe_allow_html=True)
#     st.markdown("<br>", unsafe_allow_html=True)
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("""
#             <div class="info-card">
#                 <div class="info-card-title"><span class="icon">🎯</span><span>Key Features</span></div>
#                 <div class="info-card-content">
#                     <ul style="margin: 0; padding-left: 1.25rem;">
#                         <li>Three-layer decision engine</li>
#                         <li>Real-time risk assessment</li>
#                         <li>Industry-standard PD calculation</li>
#                         <li>FOIR calculation & validation</li>
#                         <li>Automated reason generation</li>
#                         <li>Complete audit trail (PDF)</li>
#                         <li>Professional UI/UX</li>
#                     </ul>
#                 </div>
#             </div>
#         """, unsafe_allow_html=True)
#     with col2:
#         st.markdown("""
#             <div class="info-card">
#                 <div class="info-card-title"><span class="icon">🛠️</span><span>Technology Stack</span></div>
#                 <div class="info-card-content">
#                     <ul style="margin: 0; padding-left: 1.25rem;">
#                         <li>Streamlit (UI Framework)</li>
#                         <li>Scikit-learn (ML)</li>
#                         <li>Plotly (Visualizations)</li>
#                         <li>Pandas (Data Processing)</li>
#                         <li>ReportLab (PDF Generation)</li>
#                         <li>Python 3.8+</li>
#                     </ul>
#                 </div>
#             </div>
#         """, unsafe_allow_html=True)









# CORRECTED test.py - VERSION 8.2 (FIXED: use_two_stage session state, tab4 indentation, page fallback)
"""
Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
Enhanced with Modern UI/UX Design
Run with: streamlit run test.py (from inside the notebooks folder)
Author: Zen Meraki
Date: January 2026
VERSION: 8.2 - COMPLETELY FIXED PD CALCULATION & AUDIT PDF
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
from typing import Dict, List, Any, Union
import json
import sys
import os
from pathlib import Path

import css_styles
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# IMPORT CSS – ONLY EXTERNAL IMPORT
# =============================================================================
from css_styles import CSS

warnings.filterwarnings('ignore')

# =============================================================================
# STAGE 2 ENGINE – ROBUST FALLBACK
# =============================================================================
try:
    import stage2_engine
    from stage2_engine import make_two_stage_decision, is_stage2_available, get_stage2_status
    STAGE2_AVAILABLE = True
        
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
# OCR IMPORTS FOR CIBIL PDF EXTRACTION
# =============================================================================
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    import cv2
    from PIL import Image
    import re
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# =============================================================================
# PDF GENERATION – SAFE FALLBACK
# =============================================================================
PDF_AVAILABLE = False
generate_decision_pdf = None
generate_audit_pdf = None
try:
    from utils.pdf_generator import generate_decision_pdf, generate_audit_pdf
    PDF_AVAILABLE = True
except ImportError as e:
    PDF_AVAILABLE = False
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
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize all session state variables"""
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

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(CSS, unsafe_allow_html=True)
init_session_state() 


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
        return -0.5
    elif foir_percentage <= 40:
        return 0.0
    elif foir_percentage <= 50:
        return 1.0
    elif foir_percentage <= 60:
        return 2.5
    else:
        return 5.0

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
def calculate_final_risk_score(bureau_score, ml_confidence, foir):
    bureau_points = (bureau_score / 900) * 400
    ml_points = (ml_confidence / 100) * 400
    foir_points = max(0, (1 - foir/50) * 200)
    total_score = int(bureau_points + ml_points + foir_points)
    return min(max(total_score, 0), 1000)

# =============================================================================
# CIBIL PDF EXTRACTION ENGINE (OCR + PATTERN MATCHING)
# =============================================================================
def extract_cibil_from_pdf(uploaded_file):
    """Extract CIBIL data using OCR – relies on system PATH."""
    if not OCR_AVAILABLE:
        return {'success': False, 'error': 'OCR libraries not installed'}

    try:
        pdf_bytes = uploaded_file.read()
        images = convert_from_bytes(pdf_bytes, dpi=300)          # no poppler_path
        full_text = ""
        for image in images:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            full_text += pytesseract.image_to_string(binary) + "\n"

        # ----- 1. Credit Score -----
        # CIBIL PDFs often have "612 SUBPRIME" BEFORE the label "CIBIL Score"
        # So we try multiple patterns in priority order
        credit_score = 720
        # Pattern A: standalone 3-digit number followed by a rating word on same line
        score_match = re.search(
            r'\b(\d{3})\s*(?:EXCELLENT|VERY\s*GOOD|GOOD|FAIR|SUBPRIME|POOR|NH|NA)\b',
            full_text, re.IGNORECASE
        )
        if score_match:
            val = int(score_match.group(1))
            if 300 <= val <= 900:
                credit_score = val
        if credit_score == 720:
            # Pattern B: label then number (standard format)
            score_match2 = re.search(
                r'(?:cibil|credit)\s*score\s*[:\-\(]?\s*(\d{3})',
                full_text, re.IGNORECASE
            )
            if score_match2:
                val = int(score_match2.group(1))
                if 300 <= val <= 900:
                    credit_score = val
        if credit_score == 720:
            # Pattern C: number in parentheses near "score"
            score_match3 = re.search(
                r'score.*?\((\d{3})\)',
                full_text, re.IGNORECASE
            )
            if score_match3:
                val = int(score_match3.group(1))
                if 300 <= val <= 900:
                    credit_score = val

        # ----- 2. Monthly Income -----
        # Handles: "Rs. 38,000" / "Rs.38000" / "INR 38,000" / "₹38000"
        monthly_income = 50000
        income_match = re.search(
            r'(?:net\s+monthly\s+income|monthly\s+income|net\s+income|salary)[^\n\r]{0,30}?'
            r'(?:rs\.?\s*|inr\s*|₹\s*)([\d,]+)',
            full_text, re.IGNORECASE
        )
        if income_match:
            val = int(income_match.group(1).replace(',', ''))
            if val > 1000:          # sanity check — not a DPD value
                monthly_income = val
        if monthly_income == 50000:
            # Fallback: "Rs. 38,000" anywhere near "income"
            income_match2 = re.search(
                r'(?:rs\.?\s*|₹\s*)([\d,]{4,})',
                full_text, re.IGNORECASE
            )
            if income_match2:
                val = int(income_match2.group(1).replace(',', ''))
                if 5000 <= val <= 1000000:
                    monthly_income = val

        # ----- 3. CC Utilization -----
        # Handles: "utilization (55%)" / "utilization: 55%" / "utilization 55%"
        cc_util_pct = 35          # default 35%
        util_match = re.search(
            r'utilization\s*[\(:\-]?\s*(\d{1,3})\s*%',
            full_text, re.IGNORECASE
        )
        if util_match:
            cc_util_pct = int(util_match.group(1))
        cc_util = cc_util_pct / 100.0
        high_util = 1 if cc_util_pct > 75 else 0

        # ----- 4. Age from Date of Birth -----
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
                    except:
                        continue
            except:
                pass

        # ----- 5. Business Vintage -----
        biz_vintage = 3
        biz_match = re.search(r'business\s+vintage.*?(\d+)', full_text, re.IGNORECASE)
        if biz_match:
            biz_vintage = int(biz_match.group(1))

        # ----- 6. Parse the ACCOUNT DETAILS table -----
        # Strategy: find section between ACCOUNT DETAILS and ENQUIRY DETAILS
        # then extract DPD column (3-digit number like 000, 030, 060, 090) and Status
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
                # Skip pure header rows
                if re.search(r'\bLender\b|\bAccount\s*No\b|\bOpen\s*Date\b|\bDPD\b|\bStatus\b', line, re.IGNORECASE):
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                # Look for a 3-digit DPD value (000, 030, 060, 090) in the line
                dpd_match = re.search(r'\b(0\d0|0\d\d|\d{3})\b', stripped)
                # Look for status at end of line
                status_match = re.search(
                    r'\b(Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss)\s*$',
                    stripped, re.IGNORECASE
                )
                # Only add if line looks like an account row (has INR or a lender-like start)
                if (re.search(r'\bINR\b', stripped, re.IGNORECASE) or
                        re.match(r'^[A-Z][a-zA-Z\s]+(?:Bank|Finance|Capital|Fincorp|SBI|ICICI|HDFC|Axis|Bajaj|Tata|Kotak)', stripped)):
                    dpd_val = int(dpd_match.group(1)) if dpd_match else 0
                    status_str = status_match.group(1) if status_match else 'Active'
                    accounts.append({'dpd': dpd_val, 'status': status_str.lower()})

            if in_enquiry:
                # Only count lines with a proper enquiry date (dd-Mon-yyyy) NOT account open dates
                enq_date = re.match(r'^\s*(\d{2}-[A-Za-z]{3}-\d{4})', line)
                if enq_date:
                    enquiry_dates.append(enq_date.group(1))

        # ----- 7. Derive counts from parsed accounts -----
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
            # Table parse failed — fall back to full-text regex counts
            written_off_count = len(re.findall(r'\bwritten[-\s]?off\b', full_text, re.IGNORECASE))
            settled_count     = len(re.findall(r'\bsettled\b', full_text, re.IGNORECASE))
            dpd_90_count      = len(re.findall(r'\b090\b|90\+?\s*dpd', full_text, re.IGNORECASE))
            dpd_60_count      = len(re.findall(r'\b060\b|60\+?\s*dpd', full_text, re.IGNORECASE))
            dpd_30_count      = len(re.findall(r'\b030\b|30\+?\s*dpd', full_text, re.IGNORECASE))
            # Try summary line: "Total Accounts  Active  Closed ..."
            active_sum = re.search(r'Total\s+Accounts\s+Active.*?(\d+)\s+(\d+)', full_text, re.IGNORECASE)
            if active_sum:
                active_count = int(active_sum.group(2))

        # ----- 8. Active accounts from summary line if table parse missed -----
        if active_count == 0:
            # "5  4  0  299,000  4" → try to get the "Active" count from summary
            summary_match = re.search(
                r'Total\s+Accounts\s+Active[^\n]*\n\s*(\d+)\s+(\d+)',
                full_text, re.IGNORECASE
            )
            if summary_match:
                active_count = int(summary_match.group(2))
            else:
                # Inline: "5 4 0 299,000 4"
                inline = re.search(
                    r'(?:Total\s+Accounts.*?Active.*?Closed.*?\n|'
                    r'(\d+)\s+(\d+)\s+(\d+)\s+[\d,]+\s+\d+)',
                    full_text, re.IGNORECASE
                )
                if inline and inline.group(2):
                    active_count = int(inline.group(2))

        # ----- 9. Enquiry counts (ONLY from ENQUIRY DETAILS section) -----
        # enquiry_dates already filtered to only lines starting with date in enquiry section
        # Also try to get total from summary: "Enquiries (12M)  4"
        enq_12m_total = len(enquiry_dates)
        enq_sum_match = re.search(r'Enquiries?\s*\(?12M\)?\s*[:\s]+(\d+)', full_text, re.IGNORECASE)
        if enq_sum_match:
            enq_12m_total = max(enq_12m_total, int(enq_sum_match.group(1)))

        # For 3M/6M we approximate from enquiry dates within range
        # Use enquiry_dates list (already only from ENQUIRY DETAILS section)
        enq_L3m = min(len(enquiry_dates), enq_12m_total)  # conservative: use actual parsed count
        enq_L6m = enq_12m_total
        enq_L12m = enq_12m_total

        # ----- 10. Credit score override ONLY for truly bad profiles (750+ with hard negatives) -----
        # Do NOT override a legitimate subprime score like 612
        if credit_score >= 750 and (written_off_count > 0 or dpd_90_count > 0):
            credit_score = 550

        # ----- 11. pct fields -----
        total_accounts = max(len(accounts), active_count + settled_count + written_off_count)
        pct_active = active_count / total_accounts if total_accounts > 0 else 0.6

        # ----- 12. Build the result dictionary -----
        extracted_data = {
            'Credit_Score': credit_score,
            'max_delinquency_level': max(dpd_90_count * 90, dpd_60_count * 60, dpd_30_count * 30),
            'num_times_30p_dpd': dpd_30_count,
            'num_times_60p_dpd': dpd_60_count,
            'num_times_delinquent': dpd_30_count + dpd_60_count + dpd_90_count,
            'num_deliq_6mts': dpd_30_count + dpd_60_count + dpd_90_count,
            'num_deliq_12mts': dpd_30_count + dpd_60_count + dpd_90_count,
            'max_deliq_6mts': dpd_90_count,
            'max_deliq_12mts': dpd_90_count,
            'enq_L3m': enq_L3m,
            'enq_L6m': enq_L6m,
            'enq_L12m': enq_L12m,
            'num_std': active_count,
            'num_std_6mts': active_count,
            'num_std_12mts': active_count,
            'num_sub': sub_standard_count,
            'num_sub_6mts': sub_standard_count,
            'num_dbt': dpd_90_count,
            'num_lss': written_off_count,
            'pct_of_active_TLs_ever': round(pct_active, 2),
            'pct_currentBal_all_TL': 0.3,
            'CC_utilization': round(cc_util, 2),
            'PL_utilization': 0.25,
            'max_unsec_exposure_inPct': cc_util_pct,
            'AGE': age_extracted,
            'NETMONTHLYINCOME': monthly_income,
            'Time_With_Curr_Empr': biz_vintage * 12,   # convert years → months for consistency
            'CC_Flag': 1 if re.search(r'credit card', full_text, re.IGNORECASE) else 0,
            'PL_Flag': 1 if re.search(r'personal loan', full_text, re.IGNORECASE) else 0,
            'HL_Flag': 1 if re.search(r'home loan', full_text, re.IGNORECASE) else 0,
            'GL_Flag': 1 if re.search(r'gold loan', full_text, re.IGNORECASE) else 0,
            'raw_text': full_text,
            'success': True,
            'extraction_method': 'OCR+robust',
            'written_off_count': written_off_count,
            'settled_count': settled_count,
            'high_util_flag': high_util,
            'dpd_90_count_6m': dpd_90_count,
            'recent_deliq_flag': 1 if (dpd_90_count > 0 or dpd_60_count > 0) else 0,
            'account_quality_score': max(0, 100 - (written_off_count * 20) - (settled_count * 10) - (dpd_90_count * 15) - (dpd_30_count * 5))
        }
        return extracted_data
    except Exception as e:
        return {'error': str(e), 'message': f'Error extracting CIBIL data: {str(e)}', 'success': False}


# =============================================================================
# HYBRID DECISION ENGINE
# =============================================================================
def make_hybrid_decision_enhanced(customer_dict):
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

    # DEPENDENTS CHECK
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
        policy_checks['income'] = f"❌ Income Rs.{monthly_income:,.0f} (Min: Rs.15,000)"
        return {'decision': "REJECT", 'reason': "Policy Gate: Income below minimum", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 100.0, 'affordability_data': {}}
    policy_checks['income'] = f"✅ Income Rs.{monthly_income:,.0f}"
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
    if dpd_90 > 0:
        policy_checks['dpd'] = f"❌ {dpd_90} instances of 90+ DPD"
        return {'decision': "REJECT", 'reason': "Policy Gate: Severe delinquency", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 100.0, 'affordability_data': {}}
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
        class_probs = {cls: prob * 100 for cls, prob in zip(TARGET_LE.classes_, pred_proba)}
    except:
        confidence = 75.0
        class_probs = {ml_decision: 100.0}

    # Affordability
    loan_amount = customer_dict.get('loan_amount', 0)
    loan_tenure = customer_dict.get('loan_tenure_months', 12)
    interest_rate = customer_dict.get('interest_rate', 10.5)
    existing_emi = customer_dict.get('existing_emi', 0)
    affordability_data = calculate_affordability(monthly_income, loan_amount, interest_rate, loan_tenure, existing_emi)
    foir = affordability_data['foir_percentage']
    if ml_decision == "APPROVE" and foir > 45:
        ml_decision = "REVIEW"

    # Apply dependents rule
    if dependents_flag_review and ml_decision == "APPROVE":
        ml_decision = "REVIEW"

    risk_score = calculate_final_risk_score(bureau_score, confidence, foir)
    pd_percentage = calculate_final_pd(
        bureau_score=bureau_score, foir=foir, confidence=confidence,
        dpd_90_count=dpd_90, dpd_30_count=customer_dict.get('dpd_30_count_6m', 0),
        employment_type=employment_type, employment_tenure=employment_tenure,
        business_vintage=business_vintage, recent_inquiries=recent_inquiries,
        ml_decision=ml_decision
    )
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
            'dependents': 2,
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
            <div class="decision-title">
                <span>{icon}</span>
                <span>{decision}</span>
            </div>
            <div class="decision-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{risk_score}</div><div class="stat-label">Risk Score</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{pd_score}%</div><div class="stat-label">PD Score</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">Rs.{approved_amount:,.0f}</div><div class="stat-label">Loan Amount</div></div>', unsafe_allow_html=True)
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
    if value <= 50: color = "#f56565"
    elif value <= 75: color = "#ed8936"
    else: color = "#48bb78"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}},
        number={'suffix': "%", 'font': {'size': 36, 'color': '#2d3748', 'family': 'Inter'}},
        gauge={'axis': {'range': [0, max_value], 'tickfont': {'size': 12, 'color': '#718096'}},
               'bar': {'color': color, 'thickness': 0.75}, 'bgcolor': 'white', 'borderwidth': 0,
               'steps': [{'range': [0, 50], 'color': '#fed7d7'},
                         {'range': [50, 75], 'color': '#feebc8'},
                         {'range': [75, 100], 'color': '#c6f6d5'}]}
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
    fig.update_layout(showlegend=False, yaxis_title='Probability (%)', xaxis_title='', height=300,
                      margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white', plot_bgcolor='white',
                      font={'family': 'Inter', 'color': '#2d3748'},
                      yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]},
                      xaxis={'tickfont': {'size': 14, 'color': '#2d3748'}})
    return fig

# =============================================================================
# STAGE 2 RESULTS DISPLAY FUNCTION
# =============================================================================
def display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data):
    """Display comprehensive Stage 2 results with decision report and download options"""

    st.markdown("---")
    st.markdown('<p class="main-header">🎯 Stage 2 Final Results</p>', unsafe_allow_html=True)

    final_decision = stage2_result.get('final_decision', 'ERROR')
    risk_tier = stage2_result.get('tier', 'UNKNOWN')
    interest_range = stage2_result.get('interest_rate_range', 'N/A')
    stage2_tier = stage2_result.get('stage2_tier', 'N/A')
    stage2_confidence = stage2_result.get('stage2_confidence', 0)
    combined_risk_score = stage2_result.get('combined_risk_score', 0)

    if final_decision == "APPROVE":
        card_class = "decision-card decision-card-approved"
        icon = "✓"
        subtitle = "Application Approved - Proceed to Disbursement"
    elif final_decision in ["REVIEW", "MANUAL_REVIEW"]:
        card_class = "decision-card decision-card-review"
        icon = "⚠"
        subtitle = "Requires Manual Review"
    else:
        card_class = "decision-card decision-card-rejected"
        icon = "✗"
        subtitle = "Application Rejected"

    st.markdown(f"""
        <div class="{card_class}">
            <div class="decision-title">
                <span>{icon}</span>
                <span>{final_decision}</span>
            </div>
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
        st.metric("Stage 2 Confidence", f"{stage2_confidence:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔬 Analysis", "📋 Data", "📥 Download"])

    with tab1:
        st.markdown("### 📊 Decision Comparison")
        comparison_df = pd.DataFrame([
            {
                'Stage': 'Stage 1 (Basic)',
                'Decision': st.session_state.get('stage1_decision'),
                'Risk Score': stage1_data.get('risk_score', 'N/A'),
                'Tier': 'N/A'
            },
            {
                'Stage': 'Stage 2 (CIBIL Deep)',
                'Decision': final_decision,
                'Risk Score': combined_risk_score,
                'Tier': f"{stage2_tier} | {interest_range}"
            }
        ])
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("### 🎯 Risk Tier Details")
        tier_info = {
            'P1': {'name': 'Premium', 'color': '#10B981', 'desc': 'Excellent credit profile'},
            'P2': {'name': 'Standard', 'color': '#3B82F6', 'desc': 'Good credit profile'},
            'P3': {'name': 'Subprime', 'color': '#F59E0B', 'desc': 'Fair credit with concerns'},
            'P4': {'name': 'High Risk', 'color': '#EF4444', 'desc': 'High risk profile'},
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
                tier_probs = stage2_result['tier_probabilities']
                for tier, prob in tier_probs.items():
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

        stage1_reasons = stage1_customer.get('reason_codes', [])

        # Compute PD factors for PDF
        bureau_score = stage1_customer.get('bureau_score', 0)
        dpd_90 = stage1_customer.get('dpd_90_count_6m', 0)
        dpd_30 = stage1_customer.get('dpd_30_count_6m', 0)
        foir = stage1_data.get('affordability_data', {}).get('foir_percentage', 0)
        employment_type = stage1_customer.get('employment_type', 'Salaried')
        employment_tenure = stage1_customer.get('employment_tenure_months', 0)
        business_vintage = stage1_customer.get('business_vintage_years', 0)
        ml_decision = stage1_data.get('decision', 'ERROR')
        confidence = stage1_data.get('confidence', 0)

        pd_factors = {
            'bureau_score': bureau_score,
            'base_pd': bureau_score_to_pd(bureau_score),
            'dpd_90': dpd_90,
            'dpd_30': dpd_30,
            'delinquency_multiplier': delinquency_to_pd_multiplier(dpd_90, dpd_30),
            'foir': foir,
            'foir_adjustment': foir_to_pd_adjustment(foir),
            'employment_adjustment': employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage),
            'ml_adjustment': ml_confidence_to_pd_adjustment(confidence, ml_decision),
            'final_pd': stage1_data.get('pd_percentage', 0)
        }

        report_data = {
            'application_id': stage1_customer.get('application_id'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'decision': stage1_data.get('decision'),
            'risk_score': stage1_data.get('risk_score'),
            'pd_percentage': stage1_data.get('pd_percentage'),
            'confidence': stage1_data.get('confidence'),
            'policy_checks': stage1_data.get('policy_checks', {}),
            'affordability_data': stage1_data.get('affordability_data', {}),
            'customer_data': stage1_customer,
            'reason_codes': stage1_reasons,
            'pd_calculation_factors': pd_factors,
            'stage2_final_decision': final_decision,
            'stage2_tier': stage2_tier,
            'stage2_interest_range': interest_range,
            'stage2_combined_risk_score': combined_risk_score,
            'stage2_confidence': stage2_confidence,
            'stage2_reason': stage2_result.get('reason'),
            'stage2_tier_probabilities': stage2_result.get('tier_probabilities'),
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
        if st.button("🔄 New Assessment", use_container_width=True):
            st.session_state.stage1_complete = False
            st.session_state.stage1_decision = None
            st.session_state.stage1_data = None
            st.session_state.current_customer_data = None
            st.session_state.page_navigation = "👤 Assessment"
            st.rerun()
    with col2:
        if st.button("← Back to Stage 1", use_container_width=True):
            st.session_state.page_navigation = "👤 Assessment"
            st.rerun()
    with col3:
        if st.button("🏠 Home", use_container_width=True):
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
    pdf_indicator = '✅ Ready' if PDF_AVAILABLE else '❌ Not Installed'

    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">System Status</div>
        <div class="info-card-content">
            <div class="data-row"><span class="data-label">Model</span><span class="data-value">✅ Loaded</span></div>
            <div class="data-row"><span class="data-label">Version</span><span class="data-value">8.3</span></div>
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
        if st.button("🔄 New Assessment", use_container_width=True):
            st.session_state.stage1_complete = False
            st.session_state.stage1_decision = None
            st.session_state.stage1_data = None
            st.session_state.current_customer_data = None
            st.session_state.extracted_cibil_data = None
            st.rerun()

# =============================================================================
# HOME PAGE
# =============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">Credit Risk Assessment Platform</p>', unsafe_allow_html=True)
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
    with col4: st.metric("🔄 Version", "8.3", "Latest")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="warning-box">
            <strong>🆕 New in Version 8.3:</strong><br>
            • Fixed Mixed Numeric Types Error<br>
            • Fixed Missing Submit Button<br>
            • Dependents field properly integrated<br>
            • PDF auto-fill from CIBIL report<br>
            • Industry-Standard PD Methodology<br>
            • Professional UI/UX Enhancements
        </div>
    """, unsafe_allow_html=True)

# =============================================================================
# ASSESSMENT PAGE
# =============================================================================
elif page == "👤 Assessment":
    st.markdown('<p class="main-header">Credit Assessment</p>', unsafe_allow_html=True)

    # Track whether PDF data was freshly extracted (so we show the preview banner)
    pdf_just_extracted = st.session_state.get('pdf_just_extracted', False)

    # ----- PDF Upload for Auto-fill (outside the form) -----
    # Auto-open expander if extraction just happened so user sees the result
    with st.expander("📄 Upload CIBIL PDF to auto‑fill bureau fields",
                     expanded=pdf_just_extracted or not st.session_state.get('pdf_bureau_score')):

        if pdf_just_extracted:
            # Show a preview card of what was extracted
            ex = st.session_state.get('_last_extraction', {})
            st.success("✅ CIBIL data extracted — form fields below have been updated automatically.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Credit Score",    ex.get('Credit_Score', '—'))
            c2.metric("Monthly Income",  f"₹{ex.get('NETMONTHLYINCOME', 0):,}")
            c3.metric("DPD 90+ Count",   ex.get('dpd_90_count_6m', 0))
            c4.metric("CC Utilization",  f"{ex.get('CC_utilization', 0)*100:.0f}%")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("DPD 30+ Count",  ex.get('num_times_30p_dpd', 0))
            c2.metric("Inquiries (3M)", ex.get('enq_L3m', 0))
            c3.metric("Active Accounts",ex.get('num_std', 0))
            c4.metric("Written-Off",    ex.get('written_off_count', 0))
            if ex.get('written_off_count', 0) > 0 or ex.get('settled_count', 0) > 0:
                st.warning(f"⚠️ Severe negatives detected: "
                           f"{ex.get('written_off_count',0)} written-off, "
                           f"{ex.get('settled_count',0)} settled accounts. "
                           f"Score overridden to {ex.get('Credit_Score','?')}.")
            with st.expander("📋 Full extracted JSON"):
                st.json({k: v for k, v in ex.items() if k != 'raw_text'})
            st.markdown("---")
            if st.button("🔄 Upload a different PDF", key="reset_pdf"):
                st.session_state.pdf_just_extracted = False
                st.session_state.pop('_last_extraction', None)
                st.rerun()
        else:
            st.markdown('<div class="info-box">💡 Complete the form below or upload a CIBIL PDF to auto‑fill bureau data.</div>', unsafe_allow_html=True)
            uploaded_pdf = st.file_uploader(
                "Upload CIBIL Report (PDF)",
                type=['pdf'],
                key="assessment_pdf"
            )
            if uploaded_pdf is not None:
                st.info(f"📄 File ready: **{uploaded_pdf.name}** ({uploaded_pdf.size/1024:.1f} KB)")
                if st.button("🔍 Extract & Auto-fill Form", key="extract_assessment", type="primary",
                             use_container_width=True):
                    with st.spinner("🔄 Running OCR on CIBIL PDF — this takes 10-30 seconds..."):
                        extraction_result = extract_cibil_from_pdf(uploaded_pdf)
                    if extraction_result.get('success', False):
                        # ── Store ALL extracted values into session_state (all int/float typed) ──
                        st.session_state.pdf_age              = int(extraction_result.get('AGE', 35))
                        st.session_state.pdf_employment_type  = 'Salaried'
                        st.session_state.pdf_kyc              = True
                        st.session_state.pdf_bankruptcy       = False
                        st.session_state.pdf_fraud            = False
                        st.session_state.pdf_bureau_score     = int(extraction_result.get('Credit_Score', 720))
                        st.session_state.pdf_dpd_90           = int(extraction_result.get('dpd_90_count_6m', 0))
                        st.session_state.pdf_dpd_30           = int(extraction_result.get('num_times_30p_dpd', 0))
                        st.session_state.pdf_credit_util      = int(float(extraction_result.get('CC_utilization', 0.35)) * 100)
                        st.session_state.pdf_inquiries        = int(extraction_result.get('enq_L3m', 2))
                        st.session_state.pdf_active_loans     = int(extraction_result.get('num_std', 1))
                        st.session_state.pdf_existing_emi     = int(extraction_result.get('existing_emi', 15000))
                        st.session_state.pdf_monthly_income   = int(extraction_result.get('NETMONTHLYINCOME', 50000))
                        st.session_state.pdf_annual_income    = int(extraction_result.get('NETMONTHLYINCOME', 50000)) * 12
                        st.session_state.pdf_net_surplus      = int(extraction_result.get('net_surplus', 20000))
                        st.session_state.pdf_salary_stability = 'STABLE'
                        st.session_state.pdf_loan_amount      = int(extraction_result.get('loan_amount', 180000))
                        st.session_state.pdf_loan_tenure      = int(extraction_result.get('loan_tenure', 24))
                        st.session_state.pdf_interest_rate    = float(extraction_result.get('interest_rate', 10.5))
                        st.session_state.pdf_amt_annuity      = int(extraction_result.get('amt_annuity', 8500))
                        st.session_state.pdf_employment_tenure = int(extraction_result.get('Time_With_Curr_Empr', 24))
                        st.session_state.pdf_business_vintage  = int(extraction_result.get('business_vintage', 3))
                        st.session_state.pdf_dependents        = int(extraction_result.get('dependents', 2))
                        # ── Flags for display ──
                        st.session_state.pdf_just_extracted    = True
                        st.session_state._last_extraction      = extraction_result
                        st.rerun()
                    else:
                        st.error(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")
                        st.info("Tip: Make sure Tesseract and Poppler are installed and paths are set correctly.")

    # ----- Assessment Form -----
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
            # DEPENDENTS field — required by make_hybrid_decision_enhanced
            dependents = st.number_input(
                "Number of Dependents", 0, 20,
                value=int(st.session_state.get('pdf_dependents', 2)),
                help="1-5: Approve eligible | >5: Review required"
            )
            kyc_verified = st.selectbox(
                "KYC Verified",
                ['Yes', 'No'],
                index=0 if st.session_state.get('pdf_kyc', True) else 1
            ) == 'Yes'
        with col3:
            bankruptcy_flag = st.selectbox(
                "Bankruptcy Flag",
                ['No', 'Yes'],
                index=0 if not st.session_state.get('pdf_bankruptcy', False) else 1
            ) == 'Yes'
            fraud_flag = st.selectbox(
                "Fraud Flag",
                ['No', 'Yes'],
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
            # FIX: ensure value is int to match max_value=100 (int)
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
                "Existing Total EMI (Rs.)", 0, 200000,
                value=int(st.session_state.get('pdf_existing_emi', 15000)), step=1000
            )

        st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_salary = st.number_input(
                "Monthly Income (Rs.)", 0, 1000000,
                value=int(st.session_state.get('pdf_monthly_income', 50000)), step=5000
            )
            amt_income = st.number_input(
                "Annual Income (Rs.)", 0, 10000000,
                value=int(st.session_state.get('pdf_annual_income', 600000)), step=10000
            )
        with col2:
            net_surplus = st.number_input(
                "Net Cash Surplus (Rs.)", -100000, 500000,
                value=int(st.session_state.get('pdf_net_surplus', 20000)), step=5000
            )
            salary_stability = st.selectbox(
                "Salary Stability",
                ['STABLE', 'MODERATE', 'UNSTABLE'],
                index=['STABLE', 'MODERATE', 'UNSTABLE'].index(
                    st.session_state.get('pdf_salary_stability', 'STABLE')
                )
            )
        with col3:
            loan_amount = st.number_input(
                "Loan Amount (Rs.)", 0, 5000000,
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
                "Requested EMI (Rs.)", 0, 200000,
                value=int(st.session_state.get('pdf_amt_annuity', 8500)), step=500
            )

        st.markdown("<br>", unsafe_allow_html=True)
        # FIX: submit button present inside the form
        submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)

    # ----- Form Submission Handling -----
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

        # Clear PDF session state after submission
        for key in list(st.session_state.keys()):
            if key.startswith('pdf_') or key in ('_last_extraction',):
                del st.session_state[key]

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])

        with tab1:
            st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                render_info_card("👤 Identity", "👤",
                                 {"Age": age,
                                  "Employment": employment_type,
                                  "Dependents": dependents,
                                  "KYC Status": "Verified" if kyc_verified else "Not Verified",
                                  "Tenure": f"{employment_tenure} months" if employment_type == 'Salaried' else f"{business_vintage} years"})
                render_info_card("💰 Financial", "💰",
                                 {"Monthly Income": f"Rs.{avg_salary:,}", "Annual Income": f"Rs.{amt_income:,}",
                                  "Net Surplus": f"Rs.{net_surplus:,}", "Stability": salary_stability})
            with col2:
                render_info_card("🏦 Credit Bureau", "🏦",
                                 {"Bureau Score": bureau_score, "DPD 90+": dpd_90_6m, "DPD 30+": dpd_30_6m,
                                  "Utilization": f"{credit_utilization}%", "Recent Inquiries": recent_inquiries,
                                  "Existing EMI": f"Rs.{existing_emi:,}"})
                render_info_card("📋 Loan Request", "📋",
                                 {"Amount": f"Rs.{loan_amount:,}", "Tenure": f"{loan_tenure} months",
                                  "Interest Rate": f"{interest_rate}%", "Requested EMI": f"Rs.{amt_annuity:,}"})

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
                    if st.button("📝 Manual Entry", use_container_width=True, type="primary"):
                        st.session_state.stage2_selected_tab = "Manual Entry"
                        st.session_state.page_navigation = "🔬 Stage 2 Analysis"
                        st.rerun()
                with col2:
                    if st.button("📄 PDF Upload", use_container_width=True, type="primary"):
                        st.session_state.stage2_selected_tab = "PDF Upload"
                        st.session_state.page_navigation = "🔬 Stage 2 Analysis"
                        st.rerun()
                with col3:
                    if st.button("📊 Batch Analysis", use_container_width=True, type="primary"):
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
                                {f"Age: {age}": "",
                                 f"Employment: {employment_type}": "",
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
                                {f"Monthly Income: Rs.{avg_salary:,}": "", f"FOIR: {foir:.1f}%": "",
                                 f"Total EMI: Rs.{total_emi:,}": "", f"Net Disposable: Rs.{net_disp:,}": ""},
                                {f"Monthly Income: Rs.{avg_salary:,}": "pass",
                                 f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
                                 f"Total EMI: Rs.{total_emi:,}": "pass",
                                 f"Net Disposable: Rs.{net_disp:,}": "pass" if net_disp >= 10000 else "warning"})

            st.markdown("<br>", unsafe_allow_html=True)
            render_reason_codes(reasons)
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 2])
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
                if st.button("🔄 Re-Evaluate", use_container_width=True):
                    st.rerun()

        with tab3:
            st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_modern_gauge(decision_data.get('confidence', 0), "Model Confidence")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                final_decision_tab3 = decision_data.get('decision', 'ERROR')
                if final_decision_tab3 == "REVIEW":
                    class_probs = {"APPROVE": 0, "REVIEW": 100, "REJECT": 0}
                elif final_decision_tab3 == "REJECT":
                    class_probs = {"APPROVE": 0, "REVIEW": 0, "REJECT": 100}
                else:
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
                'model_version': '8.3',
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
                {"Factor": "Employment Stability", "Value": f"{employment_type} ({employment_tenure if employment_type == 'Salaried' else business_vintage}{' months' if employment_type == 'Salaried' else ' years'})", "Impact": f"{employment_stability_to_pd_adjustment(employment_type, employment_tenure, business_vintage):.1f}% adjustment"},
                {"Factor": "ML Decision Confidence", "Value": f"{decision_data.get('confidence', 0):.1f}% ({decision_data.get('decision', 'ERROR')})", "Impact": f"{ml_confidence_to_pd_adjustment(decision_data.get('confidence', 0), decision_data.get('decision', 'ERROR')):.1f}% adjustment"},
                {"Factor": "Final PD", "Value": f"{decision_data.get('pd_percentage', 0)}%", "Impact": "Industry-standard calculation"}
            ])
            st.dataframe(pd_table, use_container_width=True, hide_index=True)

# =============================================================================
# STAGE 2 ANALYSIS PAGE
# =============================================================================
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

    # =========================================================================
    # TAB 1: MANUAL ENTRY
    # =========================================================================
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
                ref_app_id = st.text_input(
                    "Application ID",
                    value=stage1_customer.get('application_id', 'N/A'),
                    disabled=True
                )
                stage1_decision_display = st.text_input(
                    "Stage 1 Decision",
                    value=st.session_state.get('stage1_decision', 'N/A'),
                    disabled=True
                )
            with col2:
                customer_name = st.text_input("Customer Name (Optional)", "")
                stage1_risk_display = st.number_input(
                    "Stage 1 Risk Score",
                    value=int(stage1_data.get('risk_score', 750)),
                    disabled=True
                )

            st.markdown("---")
            st.markdown("### 🏦 CIBIL Bureau Data")

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
                age_cibil = st.number_input(
                    "Age", 24, 70,
                    int(stage1_customer.get('age', 35))
                )
                net_monthly_income = st.number_input(
                    "Net Monthly Income", 0, 1000000,
                    int(stage1_customer.get('avg_salary_6m', 50000)), 5000
                )
                time_curr_employer = st.number_input(
                    "Employment Tenure (months)", 0, 600,
                    int(stage1_customer.get('employment_tenure_months', 24))
                )

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
                enhanced_customer_data.update({
                    'bureau_score': cibil_score,
                    'age': age_cibil,
                    'avg_salary_6m': net_monthly_income,
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
                })
                try:
                    stage2_result = make_two_stage_decision(
                        enhanced_customer_data,
                        stage1_function=make_hybrid_decision_enhanced
                    )
                    display_stage2_results(
                        stage2_result, stage1_data, stage1_customer, enhanced_customer_data
                    )
                except Exception as e:
                    st.error(f"❌ Stage 2 analysis failed: {str(e)}")
                    st.exception(e)

    # =========================================================================
    # TAB 2: PDF UPLOAD
    # =========================================================================
    elif selected_tab == "PDF Upload":
        st.markdown('<p class="section-header">📄 CIBIL PDF Upload</p>', unsafe_allow_html=True)
        if not OCR_AVAILABLE:
            st.error("❌ OCR libraries not installed!")
            st.info("Install with: `pip install pytesseract pdf2image opencv-python pillow`")
            st.warning("⚠️ For now, please use the **Manual Entry** tab.")
        else:
            st.markdown("""
                <div class="info-box">
                    📄 <strong>CIBIL PDF Extraction</strong><br>
                    Upload a CIBIL bureau report PDF for automatic extraction and analysis.
                </div>
            """, unsafe_allow_html=True)
            uploaded_pdf = st.file_uploader(
                "Upload CIBIL Report (PDF)",
                type=['pdf'],
                key="stage2_pdf"
            )
            if uploaded_pdf is not None:
                st.success(f"✅ File uploaded: {uploaded_pdf.name} ({uploaded_pdf.size / 1024:.1f} KB)")
                if st.button("🔬 Extract & Analyze", type="primary", use_container_width=True):
                    with st.spinner("🔄 Extracting data from PDF..."):
                        extraction_result = extract_cibil_from_pdf(uploaded_pdf)
                        if extraction_result.get('success', False):
                            st.success("✅ PDF extraction successful!")

                            st.markdown("### 📋 Extracted CIBIL Data")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("**Credit Score & History**")
                                st.metric("Credit Score", extraction_result.get('Credit_Score', 'N/A'))
                                st.metric("Max Delinquency Level", extraction_result.get('max_delinquency_level', 0))
                            with col2:
                                st.metric("Times 30+ DPD", extraction_result.get('num_times_30p_dpd', 0))
                                st.metric("Times 60+ DPD", extraction_result.get('num_times_60p_dpd', 0))
                            with col3:
                                st.metric("Total Delinquent", extraction_result.get('num_times_delinquent', 0))

                            st.markdown("---")
                            st.markdown("**Recent Behavior (6-12M)**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Delinquencies (6M)", extraction_result.get('num_deliq_6mts', 0))
                                st.metric("Max Delinq (6M)", extraction_result.get('max_deliq_6mts', 0))
                            with col2:
                                st.metric("Delinquencies (12M)", extraction_result.get('num_deliq_12mts', 0))
                                st.metric("Max Delinq (12M)", extraction_result.get('max_deliq_12mts', 0))
                            with col3:
                                st.metric("Inquiries (3M)", extraction_result.get('enq_L3m', 0))
                                st.metric("Inquiries (6M)", extraction_result.get('enq_L6m', 0))
                                st.metric("Inquiries (12M)", extraction_result.get('enq_L12m', 0))

                            st.markdown("---")
                            st.markdown("**Account Quality**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Standard Accounts", extraction_result.get('num_std', 0))
                                st.metric("Standard (6M)", extraction_result.get('num_std_6mts', 0))
                            with col2:
                                st.metric("Sub-standard", extraction_result.get('num_sub', 0))
                                st.metric("Sub-standard (6M)", extraction_result.get('num_sub_6mts', 0))
                            with col3:
                                st.metric("Doubtful", extraction_result.get('num_dbt', 0))
                                st.metric("Loss", extraction_result.get('num_lss', 0))

                            st.markdown("---")
                            st.markdown("**Utilization**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("% Active TLs", f"{extraction_result.get('pct_of_active_TLs_ever', 0.6):.2f}")
                                st.metric("Current Balance %", f"{extraction_result.get('pct_currentBal_all_TL', 0.3):.2f}")
                            with col2:
                                cc_util = extraction_result.get('CC_utilization', 0.35)
                                st.metric("CC Utilization", f"{cc_util*100:.1f}%")
                                pl_util = extraction_result.get('PL_utilization', 0.25)
                                st.metric("PL Utilization", f"{pl_util*100:.1f}%")
                            with col3:
                                st.metric("Max Unsec Exposure %", extraction_result.get('max_unsec_exposure_inPct', 30))

                            st.markdown("---")
                            st.markdown("**Demographics**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Age", extraction_result.get('AGE', 35))
                            with col2:
                                st.metric("Net Monthly Income", f"₹{extraction_result.get('NETMONTHLYINCOME', 50000):,}")
                            with col3:
                                st.metric("Employment Tenure (months)", extraction_result.get('Time_With_Curr_Empr', 24))

                            st.markdown("---")
                            st.markdown("**Product Flags**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Credit Card", "Yes" if extraction_result.get('CC_Flag', 0) else "No")
                            with col2:
                                st.metric("Personal Loan", "Yes" if extraction_result.get('PL_Flag', 0) else "No")
                            with col3:
                                st.metric("Home Loan", "Yes" if extraction_result.get('HL_Flag', 0) else "No")
                            with col4:
                                st.metric("Gold Loan", "Yes" if extraction_result.get('GL_Flag', 0) else "No")

                            with st.expander("📋 View Full JSON Data"):
                                st.json(extraction_result)

                            enhanced_customer_data = stage1_customer.copy()
                            enhanced_customer_data.update({
                                'bureau_score': extraction_result.get('Credit_Score', 720),
                                'age': extraction_result.get('AGE', stage1_customer.get('age', 35)),
                                'avg_salary_6m': extraction_result.get('NETMONTHLYINCOME', stage1_customer.get('avg_salary_6m', 50000)),
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
                                'credit_utilization_pct': extraction_result.get('CC_utilization', 0.35) * 100,
                                'pct_of_active_TLs_ever': extraction_result.get('pct_of_active_TLs_ever', 0.6),
                                'pct_currentBal_all_TL': extraction_result.get('pct_currentBal_all_TL', 0.3),
                                'CC_utilization': extraction_result.get('CC_utilization', 0.35),
                                'PL_utilization': extraction_result.get('PL_utilization', 0.25),
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

                            with st.spinner("🔬 Running Stage 2 analysis..."):
                                try:
                                    stage2_result = make_two_stage_decision(
                                        enhanced_customer_data,
                                        stage1_function=make_hybrid_decision_enhanced
                                    )
                                    display_stage2_results(
                                        stage2_result, stage1_data, stage1_customer, enhanced_customer_data
                                    )
                                except Exception as e:
                                    st.error(f"❌ Analysis failed: {str(e)}")
                        else:
                            st.error("❌ PDF extraction failed!")
                            st.warning(f"Error: {extraction_result.get('error')}")

    # =========================================================================
    # TAB 3: BATCH ANALYSIS (PLACEHOLDER)
    # =========================================================================
    elif selected_tab == "Batch Analysis":
        st.markdown('<p class="section-header">📊 Batch CIBIL Analysis</p>', unsafe_allow_html=True)
        st.info("📊 Batch analysis feature coming soon!")

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
                if st.button("🚀 Process Batch Predictions", type="primary", use_container_width=True):
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
                                csv = results_df.to_csv(index=False)
                                st.download_button("📥 Download as CSV", data=csv,
                                                   file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                   mime="text/csv", use_container_width=True)
                            with col2:
                                json_data = results_df.to_json(orient='records', indent=2)
                                st.download_button("📥 Download as JSON", data=json_data,
                                                   file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                                   mime="application/json", use_container_width=True)
                            st.markdown("---")
                            st.markdown("#### Filtered Downloads")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                approved_df = results_df[results_df['decision'] == 'APPROVE']
                                if len(approved_df) > 0:
                                    st.download_button(f"✅ Approved Only ({len(approved_df)})",
                                                       data=approved_df.to_csv(index=False),
                                                       file_name=f"approved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                       mime="text/csv", use_container_width=True)
                            with col2:
                                rejected_df = results_df[results_df['decision'] == 'REJECT']
                                if len(rejected_df) > 0:
                                    st.download_button(f"❌ Rejected Only ({len(rejected_df)})",
                                                       data=rejected_df.to_csv(index=False),
                                                       file_name=f"rejected_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                       mime="text/csv", use_container_width=True)
                            with col3:
                                review_df = results_df[results_df['decision'] == 'REVIEW']
                                if len(review_df) > 0:
                                    st.download_button(f"⚠️ Review Only ({len(review_df)})",
                                                       data=review_df.to_csv(index=False),
                                                       file_name=f"review_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                       mime="text/csv", use_container_width=True)
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
            'AMT_ANNUITY': [8500, 9500, 4500]
        }
        template_df = pd.DataFrame(template_data)
        st.dataframe(template_df, use_container_width=True)
        csv_template = template_df.to_csv(index=False)
        st.download_button("📥 Download CSV Template", data=csv_template,
                           file_name="credit_assessment_template.csv", mime="text/csv", use_container_width=True)

# =============================================================================
# MODEL INFO PAGE
# =============================================================================
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

# =============================================================================
# ABOUT PAGE
# =============================================================================
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-card">
            <div class="info-card-title"><span class="icon">🏦</span><span>Credit Risk Assessment Platform</span></div>
            <div class="info-card-content">
                <p><strong>Version:</strong> 8.3 - FIXED NUMERIC TYPES & SUBMIT BUTTON</p>
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
