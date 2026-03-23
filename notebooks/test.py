# # # """
# # # Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
# # # Enhanced with Modern UI/UX Design
# # # Run with: streamlit run app.py (from inside the notebooks folder)
# # # Author: Zen Meraki
# # # Date: March 2026
# # # VERSION: 8.7 - Renamed from test.py, dead code removed, all audit fixes applied (C1/H1/H2/M1/M2/M3/L1/L2/L3)
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
# # # import base64
# # # from typing import List, Any
# # # import json
# # # import sys
# # # import os
# # # from pathlib import Path
# # # import re

# # # # =============================================================================
# # # # SUPPRESS SCIKIT-LEARN VERSION WARNINGS
# # # # =============================================================================
# # # warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# # # # =============================================================================
# # # # DYNAMIC PATH RESOLUTION
# # # # =============================================================================
# # # CURRENT_DIR = Path(__file__).resolve().parent
# # # PROJECT_ROOT = CURRENT_DIR.parent
# # # POSSIBLE_LOCATIONS = [
# # #     # FIX A-2: CURRENT_DIR is the notebooks/ folder where stage2_engine.py lives.
# # #     # It was already present but listed alongside PROJECT_ROOT without emphasis.
# # #     # Adding it first and also adding CURRENT_DIR / "utils" ensures both
# # #     # stage2_engine.py and utils/pdf_generator.py are importable on Streamlit Cloud
# # #     # regardless of the working directory at launch time.
# # #     CURRENT_DIR,                          # notebooks/  ← stage2_engine.py lives here
# # #     CURRENT_DIR / "utils",               # notebooks/utils/  (if utils is nested)
# # #     PROJECT_ROOT,
# # #     PROJECT_ROOT / "loan",
# # #     PROJECT_ROOT / "utils",              # credit_risk_engine/utils/  ← pdf_generator etc.
# # #     PROJECT_ROOT / "notebooks",
# # # ]
# # # for loc in POSSIBLE_LOCATIONS:
# # #     if loc.exists() and str(loc) not in sys.path:
# # #         sys.path.insert(0, str(loc))

# # # # =============================================================================
# # # # OPTIONAL OCR DEPENDENCIES – GRACEFUL FALLBACK
# # # # =============================================================================
# # # OCR_AVAILABLE = False
# # # OCR_ERROR_MSG = ""
# # # try:
# # #     import pytesseract
# # #     from pdf2image import convert_from_bytes
# # #     import cv2
# # #     from PIL import Image
# # #     import shutil as _shutil
# # #     _tess_cmd = (
# # #         _shutil.which("tesseract")
# # #         or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# # #     )
# # #     if _tess_cmd:
# # #         pytesseract.pytesseract.tesseract_cmd = _tess_cmd
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
# # #         .stat-card { background: white; padding: 1rem; border-radius: 0.5rem;
# # #                      box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
# # #         .stat-number { font-size: 1.8rem; font-weight: bold; color: #2d3748; }
# # #         .stat-label { font-size: 0.875rem; color: #718096; }
# # #         .info-card { background: white; border-radius: 0.5rem; padding: 1rem;
# # #                      margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
# # #         .info-card-title { font-weight: 600; margin-bottom: 0.5rem; color: #2d3748; }
# # #         .info-card-content { font-size: 0.875rem; }
# # #         .data-row { display: flex; justify-content: space-between;
# # #                     padding: 0.25rem 0; border-bottom: 1px solid #e2e8f0; }
# # #         .data-label { color: #4a5568; }
# # #         .data-value { font-weight: 500; }
# # #         .status-badge { padding: 0.25rem 0.5rem; border-radius: 0.25rem;
# # #                         font-size: 0.75rem; margin-left: 0.5rem; }
# # #         .badge-pass { background: #c6f6d5; color: #22543d; }
# # #         .badge-fail { background: #fed7d7; color: #742a2a; }
# # #         .badge-warning { background: #feebc8; color: #744210; }
# # #         .reason-item { padding: 0.25rem 0; }
# # #         .reason-icon { color: #587042; font-weight: bold; margin-right: 0.5rem; }
# # #     </style>
# # #     """
# # # st.markdown(CSS, unsafe_allow_html=True)

# # # # =============================================================================
# # # # CITY TIER MAPPING
# # # # =============================================================================
# # # CITY_TIERS = {
# # #     "Tier 1 – Metro (Mumbai, Delhi, Bengaluru, Chennai, Hyderabad, Kolkata, Pune, Ahmedabad)": "Tier 1",
# # #     "Tier 2 – Large City (Jaipur, Lucknow, Kochi, Nagpur, Indore, Bhopal, Patna, Vadodara…)": "Tier 2",
# # #     "Tier 3 – Small City / Town": "Tier 3",
# # #     "Rural / Village": "Rural",
# # # }

# # # # =============================================================================
# # # # SESSION STATE INITIALIZATION
# # # # =============================================================================
# # # def init_session_state():
# # #     defaults = {
# # #         'stage1_complete':       False,
# # #         'stage1_decision':       None,
# # #         'stage1_data':           None,
# # #         'current_customer_data': None,
# # #         'page_navigation':       "🏠 Home",
# # #         'use_two_stage':         False,
# # #         'stage2_selected_tab':   "Manual Entry",
# # #         # Fairness log — persists across sessions in memory
# # #         'fairness_log':          [],
# # #     }
# # #     for k, v in defaults.items():
# # #         if k not in st.session_state:
# # #             st.session_state[k] = v

# # # init_session_state()

# # # # =============================================================================
# # # # IMPORT BUSINESS LOGIC MODULES
# # # # =============================================================================
# # # try:
# # #     from affordability_engine import calculate_emi, calculate_affordability
# # #     from reason_codes import generate_reason_codes
# # #     from risk_engine import (
# # #         calculate_final_risk_score, fill_missing_ml_fields,
# # #         clean_sentinel_values
# # #     )
# # #     from affordability_engine import check_net_disposable
# # # except ImportError as e:
# # #     st.error(f"❌ Failed to import required modules: {e}")
# # #     st.info("""
# # #     Required files (place in notebooks/, loan/, utils/, or project root):
# # #     - affordability_engine.py  |  reason_codes.py  |  risk_engine.py
# # #     - utils/__init__.py  |  utils/pdf_generator.py
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
# # #     def is_stage2_available(): return False
# # #     def get_stage2_status(): return {"error": "Stage 2 engine module not found", "available": False}

# # # # =============================================================================
# # # # PDF GENERATION – SAFE FALLBACK
# # # # FIX A-1: Use explicit try/except import blocks instead of a single-path import.
# # # # Tries utils.pdf_generator first (standard install), then bare pdf_generator
# # # # (notebooks/ deployment). Sets PDF_AVAILABLE=False and shows a visible warning
# # # # in the UI if neither path works, so users know PDF download will be disabled.
# # # # =============================================================================
# # # PDF_AVAILABLE = False
# # # generate_decision_pdf = None
# # # generate_audit_pdf = None
# # # try:
# # #     from utils.pdf_generator import generate_decision_pdf, generate_audit_pdf
# # #     PDF_AVAILABLE = True
# # # except ImportError:
# # #     try:
# # #         from pdf_generator import generate_decision_pdf, generate_audit_pdf
# # #         PDF_AVAILABLE = True
# # #     except ImportError:
# # #         PDF_AVAILABLE = False  # UI will show warning — see A-4 note in pdf download buttons

# # # # =============================================================================
# # # # JSON SANITIZER
# # # # =============================================================================
# # # def sanitize_for_json(obj: Any) -> Any:
# # #     if obj is None or isinstance(obj, (str, int, float, bool)): return obj
# # #     if isinstance(obj, set): return list(obj)
# # #     if isinstance(obj, datetime): return obj.isoformat()
# # #     if isinstance(obj, np.integer): return int(obj)
# # #     if isinstance(obj, np.floating): return float(obj)
# # #     if isinstance(obj, np.ndarray): return obj.tolist()
# # #     if isinstance(obj, dict): return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
# # #     if isinstance(obj, (list, tuple)): return [sanitize_for_json(item) for item in obj]
# # #     try:
# # #         json.dumps(obj); return obj
# # #     except (TypeError, ValueError): return str(obj)

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
# # #             try: assets = joblib.load(path); break
# # #             except FileNotFoundError: continue
# # #         if assets is None:
# # #             raise FileNotFoundError("Could not find credit_risk_assets.pkl")
# # #         return {
# # #             'model': assets['model'], 'features': assets['features'],
# # #             'le_map': assets['le_map'], 'target_le': assets['target_le'],
# # #             'loaded': True, 'error': None
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

# # # MODEL      = ASSETS['model']
# # # TOP_FEATURES = ASSETS['features']
# # # LE_MAP     = ASSETS['le_map']
# # # TARGET_LE  = ASSETS['target_le']

# # # # =============================================================================
# # # # PD CALCULATION FUNCTIONS
# # # # NOTE: calculate_emi, calculate_affordability, generate_reason_codes,
# # # #       calculate_final_risk_score are imported from their respective modules.
# # # #       The PD functions below are NOT in any module so are kept here.
# # # # =============================================================================
# # # def bureau_score_to_pd(bureau_score):
# # #     if bureau_score >= 800: return 0.5 + (900 - bureau_score) / 200 * 0.5
# # #     elif bureau_score >= 750: return 1.0 + (800 - bureau_score) / 50 * 1.0
# # #     elif bureau_score >= 700: return 2.0 + (750 - bureau_score) / 50 * 1.5
# # #     elif bureau_score >= 650: return 3.5 + (700 - bureau_score) / 50 * 2.5
# # #     elif bureau_score >= 600: return 6.0 + (650 - bureau_score) / 50 * 4.0
# # #     elif bureau_score >= 550: return 10.0 + (600 - bureau_score) / 50 * 5.0
# # #     else: return min(25.0, 15.0 + (550 - bureau_score) / 50 * 10.0)

# # # def foir_to_pd_adjustment(foir_percentage):
# # #     if foir_percentage <= 30: return -0.75
# # #     elif foir_percentage <= 40: return 0.00
# # #     elif foir_percentage <= 45: return 0.75
# # #     elif foir_percentage <= 50: return 1.50
# # #     elif foir_percentage <= 55: return 2.25
# # #     elif foir_percentage <= 60: return 3.50
# # #     else: return 6.00

# # # def delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count=0):
# # #     if dpd_90_count >= 3: return 5.0
# # #     elif dpd_90_count == 2: return 3.0
# # #     elif dpd_90_count == 1: return 2.0
# # #     elif dpd_30_count >= 3: return 1.6
# # #     elif dpd_30_count >= 1: return 1.3
# # #     else: return 1.0

# # # def employment_stability_to_pd_adjustment(employment_type, tenure_months, business_vintage_years=0):
# # #     if employment_type == 'Salaried':
# # #         if tenure_months >= 36: return -0.5
# # #         elif tenure_months >= 12: return 0.0
# # #         elif tenure_months >= 6: return 0.5
# # #         else: return 2.0
# # #     elif employment_type in ['Self-Employed', 'Business']:
# # #         if business_vintage_years >= 5: return -0.5
# # #         elif business_vintage_years >= 2: return 0.0
# # #         else: return 1.5
# # #     else: return 1.0

# # # def inquiry_pattern_to_pd_adjustment(recent_inquiries_3m):
# # #     if recent_inquiries_3m <= 1: return -0.3
# # #     elif recent_inquiries_3m <= 3: return 0.0
# # #     elif recent_inquiries_3m <= 5: return 0.8
# # #     elif recent_inquiries_3m <= 8: return 1.5
# # #     else: return 3.0

# # # def ml_confidence_to_pd_adjustment(ml_confidence, ml_decision):
# # #     if ml_decision == "APPROVE":
# # #         if ml_confidence >= 90: return -0.5
# # #         elif ml_confidence >= 70: return 0.0
# # #         else: return 0.5
# # #     elif ml_decision == "REVIEW": return 1.0
# # #     else: return 5.0

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
# # #     return round(max(0.5, min(final_pd, 25.0)), 2)

# # # # =============================================================================
# # # # CATEGORICAL FLAG INFERENCE (v8.5 dual-dataset)
# # # # =============================================================================
# # # def _infer_surplus_from_cibil(score: int, dpd_60: int, dpd_30: int, income: float) -> float:
# # #     if dpd_60 >= 3: return income * -0.5
# # #     elif score < 650 or dpd_60 >= 1: return income * -0.2
# # #     elif score < 700: return income * 0.1
# # #     else: return income * 0.3

# # # def infer_categorical_flags(extraction_result: dict) -> dict:
# # #     score       = int(extraction_result.get('Credit_Score', 700) or 700)
# # #     dpd_30      = int(extraction_result.get('num_times_30p_dpd', 0) or 0)
# # #     dpd_60      = int(extraction_result.get('num_times_60p_dpd', 0) or 0)
# # #     written_off = int(extraction_result.get('num_lss', 0) or extraction_result.get('written_off_count', 0) or 0)
# # #     doubtful    = int(extraction_result.get('num_dbt', 0) or 0)
# # #     cc_util_raw = extraction_result.get('CC_utilization', 0) or 0
# # #     cc_util     = float(cc_util_raw) if cc_util_raw > 0 else 0.0
# # #     income      = float(extraction_result.get('NETMONTHLYINCOME', 0) or
# # #                         extraction_result.get('avg_salary_6m', 50_000) or 50_000)
# # #     tenure      = int(extraction_result.get('Time_With_Curr_Empr', 24) or 24)

# # #     is_bureau_only = (
# # #         'NETMONTHLYINCOME' in extraction_result
# # #         and 'net_cash_surplus_6m' not in extraction_result
# # #         and 'net_surplus' not in extraction_result
# # #     )

# # #     if is_bureau_only:
# # #         dpd_90_proxy = dpd_60
# # #         surplus = _infer_surplus_from_cibil(score, dpd_60, dpd_30, income)
# # #         payment_discipline = 'POOR' if (dpd_60 >= 1 or dpd_30 >= 3) else ('MODERATE' if dpd_30 >= 1 else 'GOOD')
# # #         cashflow_health = ('HEALTHY' if surplus >= 14_000 else 'STABLE' if surplus >= 600 else 'STRESSED' if surplus < -1_000 else 'MODERATE')
# # #         liquidity_flag  = ('ADEQUATE' if surplus > 14_000 else 'LOW' if surplus < -32_000 else 'MODERATE')
# # #         bureau_risk     = ('HIGH' if (written_off >= 1 or doubtful >= 1 or dpd_60 >= 3 or score < 580)
# # #                            else 'MEDIUM' if (score < 650 or (dpd_30 >= 2 and cc_util > 0.60)) else 'LOW')
# # #         salary_stability = ('UNSTABLE' if tenure < 6 else 'STABLE' if (tenure >= 24 and score >= 700 and dpd_30 == 0) else 'MODERATE')
# # #         surplus_for_return = surplus  # FIX L2: assign in both branches — was missing here, causing latent bug if bureau_only path is extended
# # #     else:
# # #         dpd_90      = int(extraction_result.get('dpd_90_count_6m', 0) or 0)
# # #         bounces     = int(extraction_result.get('inward_bounce_count_3m', 0) or 0)
# # #         missing     = int(extraction_result.get('salary_missing_months', 0) or 0)
# # #         hard_reject = int(extraction_result.get('hard_reject_flag', 0) or 0)
# # #         surplus     = float(extraction_result.get('net_cash_surplus_6m') or extraction_result.get('net_surplus') or -50_000)
# # #         payment_discipline = ('POOR' if (dpd_90 >= 1 or bounces >= 2)
# # #                                else 'MODERATE' if (bounces == 1 or dpd_30 >= 3) else 'GOOD')
# # #         cashflow_health = ('HEALTHY' if surplus >= 14_000 else 'STABLE' if 600 <= surplus < 14_000
# # #                             else 'STRESSED' if surplus < -1_000 else 'MODERATE')
# # #         liquidity_flag  = 'ADEQUATE' if surplus > 14_000 else 'LOW' if surplus < -32_000 else 'MODERATE'
# # #         bureau_risk     = ('HIGH' if (hard_reject or dpd_90 >= 3 or written_off >= 1 or (dpd_90 >= 1 and dpd_30 >= 2))
# # #                            else 'MEDIUM' if (score < 580 or (dpd_30 >= 2 and cc_util > 0.60)) else 'LOW')
# # #         salary_stability = ('UNSTABLE' if missing >= 1
# # #                              else 'STABLE' if (missing == 0 and score >= 700 and dpd_30 == 0 and bounces == 0)
# # #                              else 'MODERATE')
# # #         surplus_for_return = surplus

# # #     return {
# # #         'payment_discipline_flag': payment_discipline,
# # #         'cashflow_health':         cashflow_health,
# # #         'liquidity_flag':          liquidity_flag,
# # #         'bureau_risk_flag':        bureau_risk,
# # #         'salary_stability_flag':   salary_stability,
# # #         '_inference_path':         'bureau_only' if is_bureau_only else 'bank_statement',
# # #     }

# # # # =============================================================================
# # # # CIBIL PDF EXTRACTION ENGINE (OCR + PATTERN MATCHING)
# # # # =============================================================================
# # # def _re_int(pattern, text, default, lo=None, hi=None):
# # #     """Safe regex → int extraction with optional range clamp."""
# # #     m = re.search(pattern, text, re.IGNORECASE)
# # #     if m:
# # #         try:
# # #             v = int(str(m.group(1)).replace(',', '').replace(' ', ''))
# # #             if lo is not None and v < lo: return default
# # #             if hi is not None and v > hi: return default
# # #             return v
# # #         except Exception: pass
# # #     return default

# # # def _re_float(pattern, text, default, lo=None, hi=None):
# # #     """Safe regex → float extraction with optional range clamp."""
# # #     m = re.search(pattern, text, re.IGNORECASE)
# # #     if m:
# # #         try:
# # #             v = float(str(m.group(1)).replace(',', '').replace(' ', ''))
# # #             if lo is not None and v < lo: return default
# # #             if hi is not None and v > hi: return default
# # #             return v
# # #         except Exception: pass
# # #     return default

# # # def extract_cibil_from_pdf(uploaded_file):
# # #     if not OCR_AVAILABLE:
# # #         return {'success': False, 'error': OCR_ERROR_MSG or 'OCR libraries not installed.'}
# # #     try:
# # #         # ── 1. OCR: PDF → full text ──────────────────────────────────────────
# # #         pdf_bytes = uploaded_file.read()
# # #         images    = convert_from_bytes(pdf_bytes, dpi=300)
# # #         full_text = ""
# # #         for image in images:
# # #             gray        = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
# # #             _, binary   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # #             full_text  += pytesseract.image_to_string(binary) + "\n"
# # #         txt = full_text   # shorthand

# # #         # ── 2. CREDIT SCORE (Bureau / CIBIL score) ───────────────────────────
# # #         credit_score = 720
# # #         for pat in [
# # #             r'\b(8[0-9]{2}|7[0-9]{2}|6[0-9]{2}|[3-5][0-9]{2})\s*(?:EXCELLENT|VERY\s*GOOD|GOOD|FAIR|SUBPRIME|POOR|NH|NA)\b',
# # #             r'(?:cibil|credit|bureau)\s*score\s*[:\-\(]?\s*(\d{3})',
# # #             r'score[^\n\r]{0,40}?(\d{3})',
# # #         ]:
# # #             m = re.search(pat, txt, re.IGNORECASE)
# # #             if m:
# # #                 v = int(m.group(1))
# # #                 if 300 <= v <= 900:
# # #                     credit_score = v; break

# # #         # ── 3. PERSONAL INFO ────────────────────────────────────────────────
# # #         # Age via DOB
# # #         age_extracted = 35
# # #         for dob_pat in [
# # #             r'(?:date\s+of\s+birth|dob|d\.o\.b)[\s:\-]+(\d{2}[-/]\w{3,9}[-/]\d{2,4})',
# # #             r'(?:date\s+of\s+birth|dob)[\s:\-]+(\d{2}[-/]\d{2}[-/]\d{4})',
# # #             r'born[\s:]+(\d{2}[-/]\w{3,9}[-/]\d{4})',
# # #         ]:
# # #             m = re.search(dob_pat, txt, re.IGNORECASE)
# # #             if m:
# # #                 for fmt in ('%d-%b-%Y','%d/%b/%Y','%d-%b-%y','%d-%m-%Y','%d/%m/%Y'):
# # #                     try:
# # #                         dob = datetime.strptime(m.group(1), fmt)
# # #                         age_extracted = int((datetime.now() - dob).days / 365.25)
# # #                         break
# # #                     except Exception: continue
# # #                 if age_extracted != 35: break
# # #         # fallback: age stated directly
# # #         if age_extracted == 35:
# # #             age_extracted = _re_int(r'(?:^|\s)age[\s:\-]+(\d{2})\b', txt, 35, lo=18, hi=75)

# # #         # Gender
# # #         if re.search(r'\bfemale\b|\bF\b', txt, re.IGNORECASE):
# # #             gender = 'F'
# # #         elif re.search(r'\bmale\b|\bM\b', txt, re.IGNORECASE):
# # #             gender = 'M'
# # #         else:
# # #             gender = 'M'

# # #         # Marital status
# # #         if re.search(r'\bsingle\b|\bunmarried\b', txt, re.IGNORECASE):
# # #             marital_status = 'Single'
# # #         else:
# # #             marital_status = 'Married'

# # #         # Education
# # #         education = 'GRADUATE'
# # #         for pat, val in [
# # #             (r'post.?grad(uate)?|m\.?tech|mba|mca',    'POST-GRADUATE'),
# # #             (r'professional|ca\b|cs\b|icai',             'PROFESSIONAL'),
# # #             (r'\b12th\b|\bhsc\b|\binter(mediate)?\b',   '12TH'),
# # #             (r'\bssc\b|\b10th\b|\bmatric',               'SSC'),
# # #             (r'under.?grad(uate)?',                      'UNDER GRADUATE'),
# # #             (r'\bgrad(uate)?\b|\bb\.?tech\b|\bb\.?e\b|\bb\.?sc\b|\bb\.?com\b', 'GRADUATE'),
# # #         ]:
# # #             if re.search(pat, txt, re.IGNORECASE): education = val; break

# # #         # ── 4. INCOME & EMPLOYMENT ──────────────────────────────────────────
# # #         monthly_income = 50000
# # #         for inc_pat in [
# # #             r'net\s+monthly\s+income[\s:\-₹Rs\.]*([0-9,]+)',
# # #             r'monthly\s+(?:take.?home|salary|income)[\s:\-₹Rs\.]*([0-9,]+)',
# # #             r'(?:salary|income)\s+per\s+month[\s:\-₹Rs\.]*([0-9,]+)',
# # #             r'₹\s*([0-9,]+)\s+(?:per\s+month|p\.?m\.?|monthly)',
# # #         ]:
# # #             m = re.search(inc_pat, txt, re.IGNORECASE)
# # #             if m:
# # #                 v = int(m.group(1).replace(',',''))
# # #                 if 5000 < v < 5_000_000:
# # #                     monthly_income = v; break

# # #         # Employment type
# # #         employment_type = 'Salaried'
# # #         if re.search(r'self.?employed|self employ|proprietor|freelance', txt, re.IGNORECASE):
# # #             employment_type = 'Self-Employed'
# # #         elif re.search(r'\bbusiness\b|\bfirm\b|\bpartner(ship)?\b', txt, re.IGNORECASE):
# # #             employment_type = 'Business'

# # #         # Employment tenure (months)
# # #         employment_tenure_months = 36
# # #         m = re.search(r'(?:with\s+current\s+employer|employment\s+tenure|employed\s+(?:since|for))[^\d]{0,20}(\d+)\s*(?:year|yr)', txt, re.IGNORECASE)
# # #         if m:
# # #             employment_tenure_months = int(m.group(1)) * 12
# # #         else:
# # #             m = re.search(r'(?:with\s+current\s+employer|tenure)[^\d]{0,20}(\d+)\s*(?:month|mth)', txt, re.IGNORECASE)
# # #             if m: employment_tenure_months = int(m.group(1))

# # #         # Existing EMI
# # #         existing_emi = 0
# # #         for emi_pat in [
# # #             r'(?:total\s+emi|existing\s+emi|current\s+emi|monthly\s+emi)[^\d₹]{0,20}[₹Rs\.]*\s*([0-9,]+)',
# # #             r'emi\s+(?:outflow|obligation)[^\d]{0,20}([0-9,]+)',
# # #             r'amt_annuity[\s:\-]+([0-9,]+)',
# # #         ]:
# # #             m = re.search(emi_pat, txt, re.IGNORECASE)
# # #             if m:
# # #                 v = int(m.group(1).replace(',',''))
# # #                 if 500 < v < 500_000:
# # #                     existing_emi = v; break

# # #         # Business vintage
# # #         business_vintage = 0
# # #         m = re.search(r'(?:business\s+(?:since|established|vintage|age|started))[^\d]{0,20}(\d+)\s*(?:year|yr)', txt, re.IGNORECASE)
# # #         if m: business_vintage = int(m.group(1))

# # #         # ── 5. CREDIT UTILISATION ───────────────────────────────────────────
# # #         cc_util_pct = -99999   # -99999 = no CC (like CIBIL dataset convention)
# # #         m = re.search(r'(?:credit\s+card\s+utiliz[ao]tion|cc\s+utiliz[ao]tion|utiliz[ao]tion\s+ratio)[^\d]{0,20}(\d{1,3})\s*%?', txt, re.IGNORECASE)
# # #         if m:
# # #             cc_util_pct = int(m.group(1))
# # #         pl_util = _re_float(r'(?:personal\s+loan\s+utiliz[ao]tion|pl\s+utiliz[ao]tion)[^\d]{0,20}([\d\.]+)', txt, 0.25, lo=0, hi=5)

# # #         # ── 6. ENQUIRIES ─────────────────────────────────────────────────────
# # #         # Parse enquiry section for product-wise breakdown
# # #         enq_section = ""
# # #         m = re.search(r'enquir(?:y|ies)\s+details(.*?)(?:account\s+summary|$)', txt, re.IGNORECASE | re.DOTALL)
# # #         if m: enq_section = m.group(1)

# # #         tot_enq    = _re_int(r'total\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, 0)
# # #         enq_L12m   = _re_int(r'enquir(?:y|ies)\s*(?:\(?12\s*(?:m(?:on)?(?:th)?s?|M)\)?)?[\s:\-]+(\d+)', txt, 0)
# # #         enq_L6m    = _re_int(r'enquir(?:y|ies)\s*\(?6\s*(?:m(?:on)?(?:th)?s?|M)\)?[\s:\-]+(\d+)', txt, 0)
# # #         enq_L3m    = _re_int(r'enquir(?:y|ies)\s*\(?3\s*(?:m(?:on)?(?:th)?s?|M)\)?[\s:\-]+(\d+)', txt, 0)

# # #         # Count enquiry dates in section as fallback
# # #         enq_dates = re.findall(r'\b\d{2}-[A-Za-z]{3}-\d{4}\b', enq_section)
# # #         tot_enq  = max(tot_enq, len(enq_dates))
# # #         enq_L12m = max(enq_L12m, len(enq_dates))

# # #         # Product-wise enquiries (CC / PL)
# # #         CC_enq     = _re_int(r'credit\s+card\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, -99999)
# # #         CC_enq_L6m = _re_int(r'cc\s+enq(?:uiry|uiries)?\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0 if CC_enq >= 0 else -99999)
# # #         CC_enq_L12m= _re_int(r'cc\s+enq(?:uiry|uiries)?\s*\(?12m\)?[\s:\-]+(\d+)', txt, 0 if CC_enq >= 0 else -99999)
# # #         PL_enq     = _re_int(r'personal\s+loan\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, -99999)
# # #         PL_enq_L6m = _re_int(r'pl\s+enq(?:uiry|uiries)?\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0 if PL_enq >= 0 else -99999)
# # #         PL_enq_L12m= _re_int(r'pl\s+enq(?:uiry|uiries)?\s*\(?12m\)?[\s:\-]+(\d+)', txt, 0 if PL_enq >= 0 else -99999)

# # #         # Time since most recent enquiry (days)
# # #         time_since_recent_enq = _re_int(r'(?:last|recent)\s+enquiry[\s:\-]+(\d+)\s*days?', txt, -99999)
# # #         if time_since_recent_enq == -99999 and enq_dates:
# # #             try:
# # #                 most_recent = max(datetime.strptime(d, '%d-%b-%Y') for d in enq_dates)
# # #                 time_since_recent_enq = (datetime.now() - most_recent).days
# # #             except Exception: pass

# # #         # ── 7. ACCOUNT / DPD PARSING ─────────────────────────────────────────
# # #         accounts, dpd_all = [], []
# # #         in_accounts = False
# # #         for line in txt.split('\n'):
# # #             lu = line.upper()
# # #             if 'ACCOUNT DETAILS' in lu or 'LOAN DETAILS' in lu:
# # #                 in_accounts = True; continue
# # #             if re.search(r'ENQUIRY\s+DETAILS|SUMMARY|PERSONAL\s+INFO', lu):
# # #                 in_accounts = False; continue
# # #             if not in_accounts: continue
# # #             stripped = line.strip()
# # #             if not stripped: continue
# # #             stat_m = re.search(r'\b(Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss)\b', stripped, re.IGNORECASE)
# # #             dpd_m  = re.search(r'\b(0\d0|0\d\d|\d{3})\b', stripped)
# # #             if re.search(r'\bINR\b|\bBank\b|\bFinance\b|\bCapital\b|\bNBFC\b', stripped, re.IGNORECASE) or stat_m:
# # #                 dpd_val = int(dpd_m.group(1)) if dpd_m else 0
# # #                 status  = (stat_m.group(1) if stat_m else 'Active').lower()
# # #                 accounts.append({'dpd': dpd_val, 'status': status})
# # #                 dpd_all.append(dpd_val)

# # #         # Aggregate DPD counts
# # #         dpd_90_count = dpd_60_count = dpd_30_count = 0
# # #         written_off_count = settled_count = active_count = sub_std = 0
# # #         if accounts:
# # #             for acc in accounts:
# # #                 d, s = acc['dpd'], acc['status']
# # #                 if d >= 90: dpd_90_count += 1
# # #                 elif d >= 60: dpd_60_count += 1
# # #                 elif d >= 30: dpd_30_count += 1
# # #                 if 'written' in s:  written_off_count += 1
# # #                 elif 'settled' in s: settled_count += 1
# # #                 elif 'active'  in s: active_count += 1
# # #                 if d >= 30: sub_std += 1
# # #         else:
# # #             # Fallback: keyword scan
# # #             written_off_count = len(re.findall(r'\bwritten[-\s]?off\b',       txt, re.IGNORECASE))
# # #             settled_count     = len(re.findall(r'\bsettled\b',                txt, re.IGNORECASE))
# # #             dpd_90_count      = len(re.findall(r'\b090\b|90\+?\s*dpd',        txt, re.IGNORECASE))
# # #             dpd_60_count      = len(re.findall(r'\b060\b|60\+?\s*dpd',        txt, re.IGNORECASE))
# # #             dpd_30_count      = len(re.findall(r'\b030\b|30\+?\s*dpd',        txt, re.IGNORECASE))
# # #             active_count      = len(re.findall(r'\bactive\b',                 txt, re.IGNORECASE))
# # #             active_count      = min(active_count, 10)  # cap noise

# # #         # Standard (num_std) = active performing accounts
# # #         total_accounts = max(len(accounts), active_count + settled_count + written_off_count, 1)
# # #         num_std        = active_count
# # #         pct_active     = active_count / total_accounts

# # #         # Substandard / doubtful / loss (CIBIL classification)
# # #         num_sub = sub_std
# # #         num_dbt = dpd_90_count
# # #         num_lss = written_off_count

# # #         # ── 8. DELINQUENCY TIMINGS ───────────────────────────────────────────
# # #         # CIBIL PDF usually shows months-ago; we convert to days
# # #         # time_since_recent_payment
# # #         time_since_recent_payment = _re_int(
# # #             r'(?:last|recent)\s+payment[\s:\-]+(\d+)\s*days?', txt, -99999)
# # #         if time_since_recent_payment == -99999:
# # #             # try "X months ago"
# # #             m = re.search(r'(?:last|recent)\s+payment[\s:\-]+(\d+)\s*(?:month|mth)', txt, re.IGNORECASE)
# # #             if m: time_since_recent_payment = int(m.group(1)) * 30

# # #         time_since_first_deliq  = -99999 if (dpd_30_count + dpd_60_count + dpd_90_count) == 0 else \
# # #             _re_int(r'first\s+delinquency[\s:\-]+(\d+)\s*days?', txt, 365)
# # #         time_since_recent_deliq = -99999 if (dpd_30_count + dpd_60_count + dpd_90_count) == 0 else \
# # #             _re_int(r'(?:last|recent)\s+delinquency[\s:\-]+(\d+)\s*days?', txt, 90)
# # #         recent_level_of_deliq   = max(
# # #             dpd_90_count * 90, dpd_60_count * 60, dpd_30_count * 30)

# # #         # 6-month vs 12-month split
# # #         num_deliq_6mts   = dpd_30_count + dpd_60_count + dpd_90_count
# # #         num_deliq_12mts  = num_deliq_6mts   # single source; 12m ≥ 6m
# # #         num_deliq_6_12mts = 0               # can't distinguish without dates
# # #         max_deliq_6mts   = -99999 if num_deliq_6mts  == 0 else recent_level_of_deliq
# # #         max_deliq_12mts  = -99999 if num_deliq_12mts == 0 else recent_level_of_deliq

# # #         # num_std time splits
# # #         num_std_6mts  = min(num_std, _re_int(r'standard\s+accounts?\s*\(?6m\)?[\s:\-]+(\d+)', txt, num_std))
# # #         num_std_12mts = _re_int(r'standard\s+accounts?\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_std)
# # #         num_sub_6mts  = _re_int(r'sub.?standard\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
# # #         num_sub_12mts = _re_int(r'sub.?standard\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_sub)
# # #         num_dbt_6mts  = _re_int(r'doubtful\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
# # #         num_dbt_12mts = _re_int(r'doubtful\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_dbt)
# # #         num_lss_6mts  = _re_int(r'loss\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
# # #         num_lss_12mts = _re_int(r'loss\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_lss)
# # #         num_times_delinquent = dpd_30_count + dpd_60_count + dpd_90_count
# # #         num_times_30p_dpd    = dpd_30_count + dpd_60_count + dpd_90_count
# # #         num_times_60p_dpd    = dpd_60_count + dpd_90_count
# # #         max_delinquency_level = max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)

# # #         # ── 9. TRADE-LINE RATIOS (pct_ fields) ──────────────────────────────
# # #         pct_of_active_TLs_ever     = round(pct_active, 3)
# # #         pct_opened_TLs_L6m_of_L12m = _re_float(
# # #             r'(?:opened|new)\s+accounts?\s*\(?6m\s*/\s*12m\)?[\s:\-]+([\d\.]+)', txt, 0.3, lo=0, hi=1)
# # #         pct_currentBal_all_TL      = _re_float(
# # #             r'current\s+balance\s+(?:ratio|pct|%)[\s:\-]+([\d\.]+)', txt, 0.3, lo=0, hi=10)
# # #         pct_PL_enq_L6m_of_L12m    = round(PL_enq_L6m / max(PL_enq_L12m, 1), 2) if PL_enq_L6m >= 0 else 0
# # #         pct_CC_enq_L6m_of_L12m    = round(CC_enq_L6m / max(CC_enq_L12m, 1), 2) if CC_enq_L6m >= 0 else 0
# # #         pct_PL_enq_L6m_of_ever    = round(PL_enq_L6m / max(PL_enq if PL_enq >= 0 else 1, 1), 2)
# # #         pct_CC_enq_L6m_of_ever    = round(CC_enq_L6m / max(CC_enq if CC_enq >= 0 else 1, 1), 2)

# # #         # ── 10. PRODUCT FLAGS ────────────────────────────────────────────────
# # #         CC_Flag = 1 if re.search(r'credit\s+card', txt, re.IGNORECASE) else 0
# # #         PL_Flag = 1 if re.search(r'personal\s+loan', txt, re.IGNORECASE) else 0
# # #         HL_Flag = 1 if re.search(r'home\s+loan|housing\s+loan', txt, re.IGNORECASE) else 0
# # #         GL_Flag = 1 if re.search(r'gold\s+loan', txt, re.IGNORECASE) else 0

# # #         prod_map = {r'personal\s+loan':'PL', r'credit\s+card':'CC',
# # #                     r'home\s+loan|housing':'HL', r'auto\s+loan|car\s+loan':'AL',
# # #                     r'gold\s+loan':'GL', r'business\s+loan':'BL'}
# # #         last_prod = first_prod = 'others'
# # #         for pat, label in prod_map.items():
# # #             if re.search(pat, txt, re.IGNORECASE):
# # #                 last_prod = first_prod = label; break

# # #         # ── 11. SANITY CHECK: high score vs bad history ──────────────────────
# # #         if credit_score >= 750 and (written_off_count > 0 or dpd_90_count > 0):
# # #             credit_score = min(credit_score, 550)

# # #         # ── 12. NET CASH SURPLUS PROXY ───────────────────────────────────────
# # #         # Try to extract if stated, else infer
# # #         net_cash_surplus = _re_int(
# # #             r'(?:net\s+(?:cash\s+)?surplus|disposable\s+income)[^\d₹]{0,20}[₹Rs\.]*\s*([0-9,]+)', txt, 0)
# # #         if net_cash_surplus == 0:
# # #             net_cash_surplus = int(_infer_surplus_from_cibil(credit_score, dpd_60_count, dpd_30_count, float(monthly_income)))

# # #         # ── 13. INWARD BOUNCE & SALARY STABILITY (60k-specific fields) ───────
# # #         # These are bank-statement fields; CIBIL PDF won't have them directly.
# # #         # We infer them from available signals.
# # #         inward_bounce_count_3m  = dpd_90_count + dpd_60_count      # proxy: each severe DPD → bounce
# # #         salary_missing_months   = 0                                  # can't determine from CIBIL
# # #         total_credit_6m         = _re_int(r'total\s+credits?\s*\(?6m\)?[\s:\-₹]*([0-9,]+)', txt, 0)
# # #         total_debit_6m          = _re_int(r'total\s+debits?\s*\(?6m\)?[\s:\-₹]*([0-9,]+)', txt, 0)

# # #         # ── 14. STAGE-1 60K DATASET FIELD MAPPING ────────────────────────────
# # #         # All columns from train_60k_rule_accepted.csv mapped from OCR data
# # #         s1 = {
# # #             # Income / salary
# # #             'AMT_INCOME_TOTAL':          monthly_income * 12,
# # #             'AMT_ANNUITY':               existing_emi if existing_emi > 0 else int(monthly_income * 0.25),
# # #             'avg_salary_6m':             float(monthly_income),
# # #             'salary_txn_count_6m':       6.0,       # assume regular salary
# # #             'salary_amount_cv':          0.05 if employment_type == 'Salaried' else 0.25,
# # #             'salary_date_std':           2.0,
# # #             'salary_creditor_consistent': 1.0 if employment_type == 'Salaried' else 0.7,
# # #             'salary_missing_months':     float(salary_missing_months),
# # #             # Delinquency
# # #             'dpd_15_count_6m':           0.0,
# # #             'dpd_30_count_6m':           float(dpd_30_count),
# # #             'dpd_90_count_6m':           float(dpd_90_count),
# # #             'max_dpd_6m':                float(max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)),
# # #             'dpd_30_count_3m':           float(dpd_30_count),
# # #             'total_payments_6m':         0.0,
# # #             'total_late_15_6m':          0.0,
# # #             'total_late_30_6m':          float(dpd_30_count),
# # #             'total_late_60_6m':          float(dpd_60_count),
# # #             'total_late_90_6m':          float(dpd_90_count),
# # #             'max_days_late_6m':          float(max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)),
# # #             'avg_days_late_6m':          float(dpd_30_count * 10 + dpd_60_count * 20 + dpd_90_count * 40) / max(total_accounts, 1),
# # #             'total_late_30_3m':          float(dpd_30_count),
# # #             'total_late_90_3m':          float(dpd_90_count),
# # #             # Credit card
# # #             'avg_balance_cc':            0.0,
# # #             'total_drawings_cc':         0.0,
# # #             'avg_credit_limit':          0.0,
# # #             'max_utilization':           (cc_util_pct / 100) if cc_util_pct > 0 else 0.0,
# # #             'total_payments_cc':         0.0,
# # #             'dpd_count_cc':              0.0,
# # #             # POS / installment
# # #             'avg_balance_pos':           0.0,
# # #             'dpd_count_pos':             0.0,
# # #             # Aggregate
# # #             'total_credit_activity':     float(total_accounts),
# # #             'total_dpd_count':           float(dpd_30_count + dpd_60_count + dpd_90_count),
# # #             'avg_monthly_balance_6m':    float(net_cash_surplus),
# # #             'total_emi_monthly':         float(existing_emi if existing_emi > 0 else int(monthly_income * 0.25)),
# # #             'net_cash_surplus_6m':       float(net_cash_surplus),
# # #             'total_credit_6m':           float(total_credit_6m),
# # #             'total_debit_6m':            float(total_debit_6m),
# # #             # Cashflow
# # #             'inward_bounce_count_3m':    float(inward_bounce_count_3m),
# # #             'recent_payment_stress':     float(dpd_30_count + dpd_60_count),
# # #             # Active loans
# # #             'active_loans_count':        float(active_count),
# # #             # Bureau
# # #             'bureau_score':              float(credit_score),
# # #             'hard_reject_flag':          1 if (dpd_90_count > 5 or written_off_count > 0 or credit_score < 550) else 0  # DPD90 1-5 = REVIEW not hard reject,
# # #         }

# # #         # ── 15. INFERRED CATEGORICAL FLAGS (60k) ─────────────────────────────
# # #         _inferred = infer_categorical_flags({
# # #             'Credit_Score': credit_score, 'num_times_30p_dpd': dpd_30_count,
# # #             'num_times_60p_dpd': dpd_60_count, 'num_lss': num_lss,
# # #             'num_dbt': num_dbt, 'CC_utilization': cc_util_pct / 100 if cc_util_pct > 0 else 0,
# # #             'NETMONTHLYINCOME': monthly_income, 'Time_With_Curr_Empr': employment_tenure_months,
# # #             'dpd_90_count_6m': dpd_90_count, 'inward_bounce_count_3m': inward_bounce_count_3m,
# # #             'salary_missing_months': salary_missing_months,
# # #             'net_cash_surplus_6m': net_cash_surplus,
# # #         })

# # #         # ── 16. STAGE-2 EXTERNAL CIBIL DATASET FIELD MAPPING ─────────────────
# # #         # All 62 columns from External_Cibil_Dataset.xlsx
# # #         s2 = {
# # #             'Credit_Score':               credit_score,
# # #             'AGE':                        age_extracted,
# # #             'GENDER':                     gender,
# # #             'MARITALSTATUS':              marital_status,
# # #             'EDUCATION':                  education,
# # #             'NETMONTHLYINCOME':           monthly_income,
# # #             'Time_With_Curr_Empr':        employment_tenure_months,
# # #             # Delinquency counts
# # #             'num_times_delinquent':       num_times_delinquent,
# # #             'max_delinquency_level':      max_delinquency_level,
# # #             'max_recent_level_of_deliq':  max(dpd_60_count*60, dpd_30_count*30),
# # #             'num_deliq_6mts':             num_deliq_6mts,
# # #             'num_deliq_12mts':            num_deliq_12mts,
# # #             'num_deliq_6_12mts':          num_deliq_6_12mts,
# # #             'max_deliq_6mts':             max_deliq_6mts,
# # #             'max_deliq_12mts':            max_deliq_12mts,
# # #             'num_times_30p_dpd':          num_times_30p_dpd,
# # #             'num_times_60p_dpd':          num_times_60p_dpd,
# # #             'recent_level_of_deliq':      recent_level_of_deliq,
# # #             # Standard / substandard / doubtful / loss
# # #             'num_std':                    num_std,
# # #             'num_std_6mts':               num_std_6mts,
# # #             'num_std_12mts':              num_std_12mts,
# # #             'num_sub':                    num_sub,
# # #             'num_sub_6mts':               num_sub_6mts,
# # #             'num_sub_12mts':              num_sub_12mts,
# # #             'num_dbt':                    num_dbt,
# # #             'num_dbt_6mts':               num_dbt_6mts,
# # #             'num_dbt_12mts':              num_dbt_12mts,
# # #             'num_lss':                    num_lss,
# # #             'num_lss_6mts':               num_lss_6mts,
# # #             'num_lss_12mts':              num_lss_12mts,
# # #             # Timings
# # #             'time_since_recent_payment':  time_since_recent_payment,
# # #             'time_since_first_deliquency': time_since_first_deliq,
# # #             'time_since_recent_deliquency': time_since_recent_deliq,
# # #             # Enquiries
# # #             'tot_enq':                    tot_enq,
# # #             'enq_L3m':                    enq_L3m,
# # #             'enq_L6m':                    enq_L6m,
# # #             'enq_L12m':                   enq_L12m,
# # #             'time_since_recent_enq':      time_since_recent_enq,
# # #             'CC_enq':                     CC_enq,
# # #             'CC_enq_L6m':                 CC_enq_L6m,
# # #             'CC_enq_L12m':                CC_enq_L12m,
# # #             'PL_enq':                     PL_enq,
# # #             'PL_enq_L6m':                 PL_enq_L6m,
# # #             'PL_enq_L12m':                PL_enq_L12m,
# # #             # Ratios / pct fields
# # #             'pct_of_active_TLs_ever':     pct_of_active_TLs_ever,
# # #             'pct_opened_TLs_L6m_of_L12m': pct_opened_TLs_L6m_of_L12m,
# # #             'pct_currentBal_all_TL':      pct_currentBal_all_TL,
# # #             'pct_PL_enq_L6m_of_L12m':     pct_PL_enq_L6m_of_L12m,
# # #             'pct_CC_enq_L6m_of_L12m':     pct_CC_enq_L6m_of_L12m,
# # #             'pct_PL_enq_L6m_of_ever':     pct_PL_enq_L6m_of_ever,
# # #             'pct_CC_enq_L6m_of_ever':     pct_CC_enq_L6m_of_ever,
# # #             # Utilisation
# # #             'CC_utilization':             cc_util_pct / 100 if cc_util_pct > 0 else -99999,
# # #             'PL_utilization':             pl_util,
# # #             'CC_Flag':                    CC_Flag,
# # #             'PL_Flag':                    PL_Flag,
# # #             'HL_Flag':                    HL_Flag,
# # #             'GL_Flag':                    GL_Flag,
# # #             'max_unsec_exposure_inPct':   cc_util_pct if cc_util_pct > 0 else 0,
# # #             'last_prod_enq2':             last_prod,
# # #             'first_prod_enq2':            first_prod,
# # #         }

# # #         # ── 17. MERGE AND RETURN ─────────────────────────────────────────────
# # #         return {
# # #             **s1, **s2,
# # #             # Stage-1 form-specific fields
# # #             'existing_emi':              existing_emi if existing_emi > 0 else s1['total_emi_monthly'],
# # #             'employment_type':           employment_type,
# # #             'business_vintage_years':    business_vintage,
# # #             'credit_utilization_pct':    cc_util_pct if cc_util_pct > 0 else 0,
# # #             # Inferred categoricals for Stage 1 form dropdowns
# # #             'salary_stability_flag':     _inferred['salary_stability_flag'],
# # #             'payment_discipline_flag':   _inferred['payment_discipline_flag'],
# # #             'cashflow_health':           _inferred['cashflow_health'],
# # #             'liquidity_flag':            _inferred['liquidity_flag'],
# # #             'bureau_risk_flag':          _inferred['bureau_risk_flag'],
# # #             # Computed extra signals
# # #             'written_off_count':         written_off_count,
# # #             'settled_count':             settled_count,
# # #             'high_util_flag':            1 if cc_util_pct > 75 else 0,
# # #             'recent_deliq_flag':         1 if (dpd_90_count > 0 or dpd_60_count > 0) else 0,
# # #             'account_quality_score':     max(0, 100 - written_off_count*20 - settled_count*10 - dpd_90_count*15 - dpd_30_count*5),
# # #             '_surplus_proxy':            int(net_cash_surplus),
# # #             # Passthrough for UI display / audit
# # #             'raw_text':                  full_text,
# # #             'success':                   True,
# # #             'extraction_method':         'OCR+FullDatasetMapping_v2',
# # #         }

# # #     except Exception as e:
# # #         return {'error': str(e), 'message': f'Error extracting CIBIL data: {str(e)}', 'success': False}

# # # # =============================================================================
# # # # FAIRNESS LOG HELPER
# # # # =============================================================================
# # # def log_decision_for_fairness(customer_data: dict, decision: str, risk_score: int, pd_pct: float,
# # #                                application_id: str = None, source: str = 'stage1'):
# # #     """
# # #     Append a minimal record to the in-session fairness log.
# # #     source = 'stage1' | 'stage2' | 'batch'
# # #     When Stage 2 completes, it REPLACES the Stage 1 record for the same application_id,
# # #     so the fairness dashboard always shows the FINAL binding decision.

# # #     NOTE A-3 — risk_score scale:
# # #       source='stage1' or 'batch' → risk_score is on 0-100 (Stage 1 engine output).
# # #       source='stage2'            → risk_score is the combined_risk_score on 0-1000
# # #                                    (Stage 1 normalised + Stage 2 tier, see stage2_engine.py).
# # #     The fairness dashboard currently uses risk_score only for the 'Avg Risk Score' summary
# # #     column. If cross-source comparisons are needed, normalise to a common scale first.
# # #     """
# # #     record = {
# # #         'ts':              datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# # #         'application_id':  application_id or customer_data.get('application_id', ''),
# # #         'source':          source,
# # #         'decision':        decision,
# # #         'risk_score':      risk_score,
# # #         'pd_pct':          pd_pct,
# # #         'gender':          customer_data.get('gender', 'Unknown'),
# # #         'city_tier':       customer_data.get('city_tier', 'Unknown'),
# # #         'employment_type': customer_data.get('employment_type', 'Unknown'),
# # #         'bureau_score':    customer_data.get('bureau_score', 0),
# # #         'age_band':        (
# # #             '24-30' if customer_data.get('age', 0) < 31 else
# # #             '31-40' if customer_data.get('age', 0) < 41 else
# # #             '41-50' if customer_data.get('age', 0) < 51 else '51+'
# # #         ),
# # #     }
# # #     st.session_state.fairness_log.append(record)

# # # # =============================================================================
# # # # STAGE 2 BINARY RESOLVER  (defined early — called from page routing below)
# # # # =============================================================================
# # # def resolve_stage2_to_binary(stage2_result: dict) -> dict:
# # #     """
# # #     Normalise Stage 2 result to a binary APPROVE / REJECT decision.
# # #     REVIEW outcomes are resolved via tier mapping; score is used as tie-breaker.
# # #     Defined here (before page routing) so it is always in scope regardless of
# # #     which section of the file Streamlit is executing.
# # #     """
# # #     result = stage2_result.copy()
# # #     tier  = result.get('stage2_tier', '')
# # #     raw   = result.get('final_decision', '')
# # #     score = result.get('combined_risk_score', 0) or 0
# # #     TIER_MAP = {'P1': 'APPROVE', 'P2': 'APPROVE', 'P3': 'REJECT', 'P4': 'REJECT'}
# # #     if raw == 'REJECT':
# # #         result['final_decision'] = 'REJECT'
# # #     elif raw == 'APPROVE':
# # #         result['final_decision'] = TIER_MAP.get(tier, 'APPROVE')
# # #     else:
# # #         if tier in TIER_MAP:
# # #             result['final_decision'] = TIER_MAP[tier]
# # #             result['reason'] = result.get('reason', '') + f" [REVIEW resolved to {TIER_MAP[tier]} via tier {tier}]"
# # #         else:
# # #             resolved = 'APPROVE' if score >= 600 else 'REJECT'
# # #             result['final_decision'] = resolved
# # #             result['reason'] = result.get('reason', '') + f" [REVIEW resolved to {resolved} via score {score}]"
# # #     if result['final_decision'] == 'APPROVE':
# # #         result.setdefault('interest_rate_range', {'P1': '9.5%–11%', 'P2': '11%–13%'}.get(tier, '11%–14%'))
# # #     else:
# # #         result['interest_rate_range'] = 'N/A — Rejected'
# # #     return result


# # # # =============================================================================
# # # # HYBRID DECISION ENGINE
# # # # =============================================================================
# # # def make_hybrid_decision_enhanced(customer_dict):
# # #     fill_missing_ml_fields(customer_dict)
# # #     policy_checks = {}
# # #     age = customer_dict.get('age', 0)
# # #     employment_type = customer_dict.get('employment_type', 'Salaried')
# # #     kyc_verified = customer_dict.get('kyc_verified', True)
# # #     bankruptcy_flag = customer_dict.get('bankruptcy_flag', False)
# # #     fraud_flag = customer_dict.get('fraud_flag', False)

# # #     # AGE POLICY GATE — split by employment type per spec
# # #     # UI allows 18–70 for input flexibility, but policy enforces:
# # #     #   - All types:       age must be > 24  (≤ 24 → too young)
# # #     #   - Salaried:        age must be ≤ 65  (retirement risk)
# # #     #   - Self-Employed / Business: age must be ≤ 70
# # #     _is_salaried = employment_type == 'Salaried'
# # #     _max_age     = 65 if _is_salaried else 70
# # #     _age_label   = "24–65 for Salaried" if _is_salaried else "24–70 for Self-Employed/Business"
# # #     if age <= 24:
# # #         policy_checks['age'] = f"❌ Age {age} — Too young (Min: 25)"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: Applicant too young (minimum age 25)", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     if age > _max_age:
# # #         policy_checks['age'] = f"❌ Age {age} — Exceeds max ({_age_label})"
# # #         return {'decision': "REJECT", 'reason': f"Policy Gate: Age exceeds maximum for {employment_type} ({_max_age})", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     policy_checks['age'] = f"✅ Age {age} (Valid — {_age_label})"

# # #     if not kyc_verified:
# # #         policy_checks['kyc'] = "❌ KYC Not Verified"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: KYC verification required", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     policy_checks['kyc'] = "✅ KYC Verified"

# # #     if not customer_dict.get('rbi_consent', False):
# # #         policy_checks['rbi_consent'] = "❌ RBI Consent not obtained"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: Customer consent not obtained", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     policy_checks['rbi_consent'] = "✅ Consent Obtained"

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
# # #     dependents_flag_review = dependents > 5
# # #     policy_checks['dependents'] = (f"⚠️ Dependents {dependents} (>5: Review Required)"
# # #                                    if dependents_flag_review else f"✅ Dependents {dependents} (Acceptable)")

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
# # #     policy_checks['tenure'] = (f"✅ Tenure {employment_tenure} months" if employment_type == 'Salaried'
# # #                                 else f"✅ Business Vintage {business_vintage} years")

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

# # #     # DPD90 TIERED GATE:
# # #     #   0     -> PASS (clean)
# # #     #   1-5   -> REVIEW flag (elevated risk, underwriter required)
# # #     #   > 5   -> REJECT (severe delinquency, hard stop)
# # #     dpd_90_review_flag = False
# # #     # DESIGN NOTE (M2): DPD90 gate is tiered — >5 = hard REJECT, 1-5 = REVIEW flag.
# # #     # Legacy calculate_risk_score() (fallback-only) uses softer penalty for DPD90=1;
# # #     # that path is NEVER reached in production. This gate is the intended behavior.
# # #     if dpd_90 > 5:
# # #         policy_checks['dpd'] = f"❌ {dpd_90} instance(s) of 90+ DPD — Hard Reject (Max: 5)"
# # #         return {'decision': "REJECT", 'reason': "Policy Gate: Severe delinquency (90+ DPD > 5)", 'confidence': 0,
# # #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# # #                 'pd_percentage': 100.0, 'affordability_data': {}}
# # #     elif dpd_90 >= 1:
# # #         dpd_90_review_flag = True
# # #         policy_checks['dpd'] = f"⚠️ {dpd_90} instance(s) of 90+ DPD — Underwriter Review Required"
# # #     else:
# # #         policy_checks['dpd'] = "✅ No 90+ DPD (Clean)"
# # #     policy_checks['utilization'] = (f"⚠️ High utilization {credit_utilization}%" if credit_utilization > 80
# # #                                     else f"✅ Utilization {credit_utilization}%")
# # #     policy_checks['inquiries'] = (f"⚠️ {recent_inquiries} recent inquiries" if recent_inquiries > 5
# # #                                   else f"✅ {recent_inquiries} inquiries")

# # #     active_loans = customer_dict.get('active_loans_count', 0)
# # #     active_loans_flag = active_loans >= 5
# # #     policy_checks['active_loans'] = (f"⚠️ High active loans ({int(active_loans)}) — Review"
# # #                                      if active_loans_flag else f"✅ Active loans: {int(active_loans)}")

# # #     salary_stability = customer_dict.get('salary_stability_flag', 'STABLE')
# # #     salary_flag = salary_stability == 'UNSTABLE'
# # #     policy_checks['salary'] = (
# # #         "⚠️ Unstable salary — Review required" if salary_stability == 'UNSTABLE' else
# # #         "⚠️ Moderate salary stability" if salary_stability == 'MODERATE' else "✅ Stable salary"
# # #     )

# # #     input_df = pd.DataFrame([customer_dict])
# # #     for col in TOP_FEATURES:
# # #         if col not in input_df.columns:
# # #             input_df[col] = "Unknown" if col in LE_MAP else 0
# # #     for col, le in LE_MAP.items():
# # #         if col in input_df.columns:
# # #             val = str(input_df[col].values[0])
# # #             try: input_df[col] = le.transform([val])[0]
# # #             except ValueError: input_df[col] = 0
# # #     final_input = input_df[TOP_FEATURES]
# # #     pred_idx = MODEL.predict(final_input)[0]
# # #     ml_decision = TARGET_LE.inverse_transform([pred_idx])[0]
# # #     try:
# # #         pred_proba = MODEL.predict_proba(final_input)[0]
# # #         confidence = max(pred_proba) * 100
# # #         class_probs = {cls: prob * 100 for cls, prob in zip(TARGET_LE.classes_, pred_proba)}
# # #     except Exception:
# # #         confidence = 75.0
# # #         class_probs = {ml_decision: 100.0}

# # #     loan_amount   = customer_dict.get('loan_amount', 0)
# # #     loan_tenure   = customer_dict.get('loan_tenure_months', 12)
# # #     interest_rate = customer_dict.get('interest_rate', 10.5)
# # #     existing_emi  = customer_dict.get('existing_emi', 0)
# # #     affordability_data = calculate_affordability(monthly_income, loan_amount, interest_rate, loan_tenure, existing_emi)
# # #     foir = affordability_data['foir_percentage']

# # #     if foir > 50:
# # #         ml_decision = "REJECT"
# # #         policy_checks['foir'] = f"❌ FOIR {foir:.1f}% exceeds maximum allowed (50%)"

# # #     if dependents_flag_review and ml_decision == "APPROVE": ml_decision = "REVIEW"
# # #     if active_loans_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"
# # #     if salary_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"
# # #     if dpd_90_review_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"  # DPD90 1-5 forces review

# # #     risk_score = calculate_final_risk_score(
# # #         bureau_score=bureau_score, ml_confidence=confidence, foir=foir,
# # #         dpd_90=dpd_90, dpd_30=customer_dict.get('dpd_30_count_6m', 0),
# # #         net_surplus=customer_dict.get('net_cash_surplus_6m', 0),
# # #         bounces=customer_dict.get('inward_bounce_count_3m', 0),
# # #         missing_months=customer_dict.get('salary_missing_months', 0),
# # #         active_loans=active_loans
# # #     )
# # #     pd_percentage = calculate_final_pd(
# # #         bureau_score=bureau_score, foir=foir, confidence=confidence,
# # #         dpd_90_count=dpd_90, dpd_30_count=customer_dict.get('dpd_30_count_6m', 0),
# # #         employment_type=employment_type, employment_tenure=employment_tenure,
# # #         business_vintage=business_vintage, recent_inquiries=recent_inquiries,
# # #         ml_decision=ml_decision
# # #     )
# # #     return {
# # #         'decision': ml_decision, 'ml_raw_decision': ml_decision,
# # #         'reason': "Decision based on comprehensive assessment",
# # #         'confidence': confidence, 'class_probs': class_probs,
# # #         'policy_checks': policy_checks, 'risk_score': risk_score,
# # #         'pd_percentage': round(pd_percentage, 2), 'affordability_data': affordability_data
# # #     }

# # # # =============================================================================
# # # # BATCH PREDICTION ENGINE
# # # # =============================================================================
# # # def process_batch_predictions(df):
# # #     results = []
# # #     required_fields = {
# # #         'age': 35, 'employment_type': 'Salaried', 'kyc_verified': True,
# # #         'bankruptcy_flag': False, 'fraud_flag': False, 'rbi_consent': True,
# # #         'employment_tenure_months': 24, 'business_vintage_years': 0,
# # #         'bureau_score': 700, 'dpd_90_count_6m': 0, 'dpd_30_count_6m': 0,
# # #         'credit_utilization_pct': 30, 'recent_inquiries_3m': 0,
# # #         'active_loans_count': 0, 'existing_emi': 0, 'avg_salary_6m': 50000,
# # #         'AMT_INCOME_TOTAL': 600000, 'net_cash_surplus_6m': 20000,
# # #         'salary_stability_flag': 'STABLE', 'loan_amount': 180000,
# # #         'loan_tenure_months': 24, 'interest_rate': 10.5, 'AMT_ANNUITY': 8500,
# # #         'dependents': 0, 'payment_discipline_flag': 'GOOD',
# # #         'liquidity_flag': 'LOW', 'cashflow_health': 'MODERATE',
# # #         'bureau_risk_flag': 'LOW', 'inward_bounce_count_3m': 0,
# # #         'salary_missing_months': 0, 'gender': 'Unknown', 'city_tier': 'Unknown',
# # #     }
# # #     for idx, row in df.iterrows():
# # #         customer_dict = row.to_dict()
# # #         for k, v in customer_dict.items():
# # #             if isinstance(v, str):
# # #                 if v.lower() in ['yes', 'true', '1']: customer_dict[k] = True
# # #                 elif v.lower() in ['no', 'false', '0']: customer_dict[k] = False
# # #         for field, default in required_fields.items():
# # #             if field not in customer_dict or pd.isna(customer_dict.get(field, None)):
# # #                 customer_dict[field] = default
# # #         try:
# # #             decision_data = make_hybrid_decision_enhanced(customer_dict)
# # #             customer_dict['ml_confidence'] = decision_data.get('confidence', 0)
# # #             reasons = generate_reason_codes(
# # #                 decision=decision_data.get('decision', 'ERROR'),
# # #                 customer_data=customer_dict,
# # #                 affordability_data=decision_data.get('affordability_data', {}),
# # #                 policy_checks=decision_data.get('policy_checks', {})
# # #             )
# # #             affordability = decision_data.get('affordability_data', {})
# # #             result = {
# # #                 'application_id': f"BATCH_{idx+1:04d}",
# # #                 'decision': decision_data.get('decision', 'ERROR'),
# # #                 'risk_score': decision_data.get('risk_score', 0),
# # #                 'pd_percentage': decision_data.get('pd_percentage', 0),
# # #                 'confidence': round(decision_data.get('confidence', 0), 2),
# # #                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# # #                 'reason_1': reasons[0] if len(reasons) > 0 else '',
# # #                 'reason_2': reasons[1] if len(reasons) > 1 else '',
# # #                 'reason_3': reasons[2] if len(reasons) > 2 else '',
# # #                 'age': customer_dict.get('age', ''),
# # #                 'gender': customer_dict.get('gender', ''),
# # #                 'city_tier': customer_dict.get('city_tier', ''),
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
# # #                 'prob_approve': round(decision_data.get('class_probs', {}).get('APPROVE', 0), 2),
# # #                 'prob_review': round(decision_data.get('class_probs', {}).get('REVIEW', 0), 2),
# # #                 'prob_reject': round(decision_data.get('class_probs', {}).get('REJECT', 0), 2),
# # #             }
# # #         except Exception as e:
# # #             result = {
# # #                 'application_id': f"BATCH_{idx+1:04d}", 'decision': 'ERROR',
# # #                 'risk_score': 0, 'pd_percentage': 0, 'confidence': 0,
# # #                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# # #                 'reason_1': '', 'reason_2': '', 'reason_3': '',
# # #                 'age': customer_dict.get('age', ''), 'gender': customer_dict.get('gender', ''),
# # #                 'city_tier': customer_dict.get('city_tier', ''),
# # #                 'employment_type': customer_dict.get('employment_type', ''),
# # #                 'bureau_score': customer_dict.get('bureau_score', ''),
# # #                 'monthly_income': customer_dict.get('avg_salary_6m', ''),
# # #                 'loan_amount': customer_dict.get('loan_amount', ''),
# # #                 'error_message': str(e)
# # #             }
# # #         else:
# # #             # Log to fairness monitor (success path only)
# # #             log_decision_for_fairness(
# # #                 customer_dict,
# # #                 result['decision'],
# # #                 result['risk_score'],
# # #                 result['pd_percentage']
# # #             )
# # #         results.append(result)
# # #     return pd.DataFrame(results)

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
# # #         card_class = "decision-card decision-card-approved"; icon = "✓"; subtitle = "Application Approved Successfully"
# # #     elif decision == "REJECT":
# # #         card_class = "decision-card decision-card-rejected"; icon = "✗"; subtitle = "Application Not Approved"
# # #     else:
# # #         card_class = "decision-card decision-card-review"; icon = "⚠"; subtitle = "Requires Manual Review"
# # #     st.markdown(f'<div class="{card_class}"><div class="decision-title">{icon} {decision}</div><div class="decision-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)
# # #     col1, col2, col3, col4, col5 = st.columns(5)
# # #     with col1: st.markdown(f'<div class="stat-card"><div class="stat-number">{risk_score}</div><div class="stat-label">Risk Score</div></div>', unsafe_allow_html=True)
# # #     _pd_color = '#48bb78' if pd_score < 5 else ('#ed8936' if pd_score < 10 else '#f56565')
# # #     _pd_label = 'Low Risk' if pd_score < 5 else ('Moderate Risk' if pd_score < 10 else 'High Risk')
# # #     with col2: st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{_pd_color}">{pd_score}%</div><div class="stat-label">PD Score</div><div style="font-size:11px;color:{_pd_color};font-weight:600">{_pd_label}</div></div>', unsafe_allow_html=True)
# # #     with col3: st.markdown(f'<div class="stat-card"><div class="stat-number">₹{approved_amount:,.0f}</div><div class="stat-label">Loan Amount</div></div>', unsafe_allow_html=True)
# # #     with col4: st.markdown(f'<div class="stat-card"><div class="stat-number">{tenure}</div><div class="stat-label">Tenure (Months)</div></div>', unsafe_allow_html=True)
# # #     with col5: st.markdown(f'<div class="stat-card"><div class="stat-number">{decision_data.get("confidence", 0):.0f}%</div><div class="stat-label">Confidence</div></div>', unsafe_allow_html=True)
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     col1, col2 = st.columns(2)
# # #     with col1: st.markdown(f'<div class="info-box"><strong>📋 Application ID:</strong> {app_id}</div>', unsafe_allow_html=True)
# # #     with col2: st.markdown(f'<div class="info-box"><strong>🕐 Decision Timestamp:</strong> {timestamp}</div>', unsafe_allow_html=True)

# # # def render_info_card(title, icon, data_dict, status_dict=None):
# # #     st.markdown(f'<div class="info-card"><div class="info-card-title">{icon} {title}</div><div class="info-card-content">', unsafe_allow_html=True)
# # #     for label, value in data_dict.items():
# # #         status = ""
# # #         if status_dict and label in status_dict:
# # #             if status_dict[label] == "pass": status = '<span class="status-badge badge-pass">✓</span>'
# # #             elif status_dict[label] == "fail": status = '<span class="status-badge badge-fail">✗</span>'
# # #             elif status_dict[label] == "warning": status = '<span class="status-badge badge-warning">⚠</span>'
# # #         st.markdown(f'<div class="data-row"><span class="data-label">{label}</span><span class="data-value">{value} {status}</span></div>', unsafe_allow_html=True)
# # #     st.markdown('</div></div>', unsafe_allow_html=True)

# # # def render_reason_codes(reasons):
# # #     st.markdown('<div class="info-card"><div class="info-card-title">📝 Decision Reasons</div><div class="info-card-content">', unsafe_allow_html=True)
# # #     for i, reason in enumerate(reasons, 1):
# # #         st.markdown(f'<div class="reason-item"><span class="reason-icon">{i}.</span>{reason}</div>', unsafe_allow_html=True)
# # #     st.markdown('</div></div>', unsafe_allow_html=True)

# # # def create_modern_gauge(value, title, max_value=100):
# # #     color = "#f56565" if value <= 50 else "#ed8936" if value <= 75 else "#48bb78"
# # #     fig = go.Figure(go.Indicator(
# # #         mode="gauge+number", value=value,
# # #         title={'text': title, 'font': {'size': 18, 'color': '#2d3748'}},
# # #         number={'suffix': "%", 'font': {'size': 36, 'color': '#2d3748'}},
# # #         gauge={
# # #             'axis': {'range': [0, max_value]},
# # #             'bar': {'color': color, 'thickness': 0.75},
# # #             'bgcolor': 'white', 'borderwidth': 0,
# # #             'steps': [{'range': [0, 50], 'color': '#fed7d7'},
# # #                       {'range': [50, 75], 'color': '#feebc8'},
# # #                       {'range': [75, 100], 'color': '#c6f6d5'}]
# # #         }
# # #     ))
# # #     fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white')
# # #     return fig

# # # def create_modern_bar_chart(class_probs):
# # #     df = pd.DataFrame({'Decision': list(class_probs.keys()), 'Probability': list(class_probs.values())})
# # #     colors = {'REVIEW': '#ed8936', 'APPROVE': '#48bb78', 'REJECT': '#f56565'}
# # #     fig = px.bar(df, x='Decision', y='Probability', title='Decision Probabilities',
# # #                  color='Decision', color_discrete_map=colors, text='Probability')
# # #     fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
# # #     fig.update_layout(showlegend=False, yaxis_title='Probability (%)', xaxis_title='', height=300,
# # #                       margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white', plot_bgcolor='white',
# # #                       yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]})
# # #     return fig

# # # # =============================================================================
# # # # STAGE 2 BINARY RESOLVER
# # # # =============================================================================
# # # def resolve_stage2_to_binary(stage2_result: dict) -> dict:
# # #     result = stage2_result.copy()
# # #     tier  = result.get('stage2_tier', '')
# # #     raw   = result.get('final_decision', '')
# # #     score = result.get('combined_risk_score', 0) or 0
# # #     TIER_MAP = {'P1': 'APPROVE', 'P2': 'APPROVE', 'P3': 'REJECT', 'P4': 'REJECT'}
# # #     if raw == 'REJECT':
# # #         result['final_decision'] = 'REJECT'
# # #     elif raw == 'APPROVE':
# # #         result['final_decision'] = TIER_MAP.get(tier, 'APPROVE')
# # #     else:
# # #         if tier in TIER_MAP:
# # #             result['final_decision'] = TIER_MAP[tier]
# # #             result['reason'] = result.get('reason', '') + f" [REVIEW resolved to {TIER_MAP[tier]} via tier {tier}]"
# # #         else:
# # #             resolved = 'APPROVE' if score >= 600 else 'REJECT'
# # #             result['final_decision'] = resolved
# # #             result['reason'] = result.get('reason', '') + f" [REVIEW resolved to {resolved} via score {score}]"
# # #     if result['final_decision'] == 'APPROVE':
# # #         result.setdefault('interest_rate_range', {'P1': '9.5%–11%', 'P2': '11%–13%'}.get(tier, '11%–14%'))
# # #     else:
# # #         result['interest_rate_range'] = 'N/A — Rejected'
# # #     return result

# # # # =============================================================================
# # # # STAGE 2 RESULTS DISPLAY
# # # # =============================================================================
# # # def display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data):
# # #     st.markdown("---")
# # #     st.markdown('<p class="main-header">🎯 Stage 2 Final Results</p>', unsafe_allow_html=True)
# # #     final_decision    = stage2_result.get('final_decision', 'ERROR')
# # #     interest_range    = stage2_result.get('interest_rate_range', 'N/A')
# # #     stage2_tier       = stage2_result.get('stage2_tier', 'N/A')
# # #     stage2_confidence = stage2_result.get('stage2_confidence', 0)
# # #     combined_risk     = stage2_result.get('combined_risk_score', 0)

# # #     # ── Fairness log: use Stage 2 FINAL decision, remove the earlier Stage 1 entry ──
# # #     # Stage 1 logged a preliminary decision for this customer. Since Stage 2
# # #     # is the BINDING final decision, we replace that entry so the fairness
# # #     # dashboard always reflects the true outcome.
# # #     app_id = stage1_customer.get('application_id', None)
# # #     if app_id and 'fairness_log' in st.session_state:
# # #         st.session_state.fairness_log = [
# # #             r for r in st.session_state.fairness_log
# # #             if r.get('application_id') != app_id
# # #         ]
# # #     log_decision_for_fairness(
# # #         enhanced_customer_data,
# # #         final_decision,
# # #         combined_risk,
# # #         stage2_result.get('pd_percentage', stage1_data.get('pd_percentage', 0)),
# # #         application_id=app_id,
# # #         source='stage2'
# # #     )

# # #     # Update session state — Stage 2 is the binding final decision
# # #     st.session_state['stage2_final_decision'] = final_decision

# # #     if final_decision == "APPROVE":
# # #         st.markdown(
# # #             '<div class="decision-card decision-card-approved" style="padding:2.5rem;">'
# # #             '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">✔  APPROVED</div>'
# # #             '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">✅ STAGE 2 FINAL DECISION — Proceed to Disbursement</div>'
# # #             '</div>', unsafe_allow_html=True)
# # #     elif final_decision == "REJECT":
# # #         st.markdown(
# # #             '<div class="decision-card decision-card-rejected" style="padding:2.5rem;">'
# # #             '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">✘  REJECTED</div>'
# # #             '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">❌ STAGE 2 FINAL DECISION — Application Declined</div>'
# # #             '</div>', unsafe_allow_html=True)
# # #     else:
# # #         st.markdown(
# # #             '<div class="decision-card decision-card-review" style="padding:2.5rem;">'
# # #             '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">⚑  REVIEW</div>'
# # #             '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">⚠️ STAGE 2 FINAL DECISION — Requires Manual Credit Officer Review</div>'
# # #             '</div>', unsafe_allow_html=True)

# # #     col1, col2, col3, col4 = st.columns(4)
# # #     with col1: st.metric("Risk Tier", stage2_tier)
# # #     with col2: st.metric("Interest Rate", interest_range)
# # #     with col3: st.metric("Combined Risk Score", combined_risk)
# # #     with col4: st.metric("Stage 2 Confidence", f"{stage2_confidence:.1f}%" if stage2_confidence else "N/A")

# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔬 Analysis", "📋 Data", "📥 Download"])

# # #     with tab1:
# # #         s1_dec = st.session_state.get('stage1_decision', 'N/A')
# # #         s2_label = "✅ APPROVE" if final_decision == "APPROVE" else "❌ REJECT"
# # #         comparison_df = pd.DataFrame([
# # #             {'Stage': 'Stage 1 (Screening)', 'Decision': s1_dec, 'Risk Score': stage1_data.get('risk_score', 'N/A'), 'Tier': 'N/A', 'Note': 'APPROVE/REVIEW → proceed to Stage 2'},
# # #             {'Stage': 'Stage 2 — FINAL', 'Decision': s2_label, 'Risk Score': combined_risk, 'Tier': f"{stage2_tier} | {interest_range}", 'Note': 'Binding final decision'}
# # #         ])
# # #         st.dataframe(comparison_df, use_container_width=True, hide_index=True)
# # #         tier_info = {
# # #             'P1': {'name': 'Premium → APPROVED', 'color': '#10B981', 'desc': 'Excellent credit profile — lowest interest rate band'},
# # #             'P2': {'name': 'Standard → APPROVED', 'color': '#3B82F6', 'desc': 'Good credit profile — standard interest rate band'},
# # #             'P3': {'name': 'Subprime → REJECTED', 'color': '#F59E0B', 'desc': 'Fair credit with elevated risk — application declined'},
# # #             'P4': {'name': 'High Risk → REJECTED', 'color': '#EF4444', 'desc': 'High risk profile — application declined'},
# # #         }
# # #         if stage2_tier in tier_info:
# # #             td = tier_info[stage2_tier]
# # #             st.markdown(f'<div style="background:{td["color"]};color:white;padding:1rem;border-radius:0.5rem;"><h3 style="margin:0;color:white;">{stage2_tier}: {td["name"]}</h3><p style="margin:0.5rem 0 0 0;">{td["desc"]}</p></div>', unsafe_allow_html=True)
# # #         st.info(stage2_result.get('reason', 'N/A'))

# # #     with tab2:
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
# # #             st.metric("Combined Score", combined_risk)
# # #         with st.expander("Complete Stage 2 Result (JSON)"):
# # #             st.json(stage2_result)

# # #     with tab3:
# # #         col1, col2 = st.columns(2)
# # #         with col1:
# # #             with st.expander("Stage 1 Customer Data"): st.json(stage1_customer)
# # #         with col2:
# # #             with st.expander("Enhanced CIBIL Data"): st.json(enhanced_customer_data)

# # #     with tab4:
# # #         if PDF_AVAILABLE and generate_audit_pdf is not None:
# # #             try:
# # #                 _safe = lambda v, d='N/A': v if v is not None else d
# # #                 # Build full pd_calculation_factors from enhanced customer data
# # #                 _bs  = enhanced_customer_data.get('bureau_score', stage1_customer.get('bureau_score', 0))
# # #                 _foir = stage1_data.get('affordability_data', {}).get('foir_percentage', 0)
# # #                 _conf = stage1_data.get('confidence', 0)
# # #                 _dpd90 = enhanced_customer_data.get('dpd_90_count_6m', stage1_customer.get('dpd_90_count_6m', 0))
# # #                 _dpd30 = enhanced_customer_data.get('dpd_30_count_6m', stage1_customer.get('dpd_30_count_6m', 0))
# # #                 _emp_type = enhanced_customer_data.get('employment_type', stage1_customer.get('employment_type', 'Salaried'))
# # #                 _emp_ten  = enhanced_customer_data.get('employment_tenure_months', stage1_customer.get('employment_tenure_months', 24))
# # #                 _biz_vin  = enhanced_customer_data.get('business_vintage_years', stage1_customer.get('business_vintage_years', 0))
# # #                 _inq      = enhanced_customer_data.get('recent_inquiries_3m', stage1_customer.get('recent_inquiries_3m', 0))
# # #                 _base_pd   = bureau_score_to_pd(_bs)
# # #                 _foir_adj  = foir_to_pd_adjustment(_foir)
# # #                 _deliq_mul = delinquency_to_pd_multiplier(_dpd90, _dpd30)
# # #                 _emp_adj   = employment_stability_to_pd_adjustment(_emp_type, _emp_ten, _biz_vin)
# # #                 _inq_adj   = inquiry_pattern_to_pd_adjustment(_inq)
# # #                 _ml_adj    = ml_confidence_to_pd_adjustment(_conf, stage1_data.get('decision','REVIEW'))
# # #                 _final_pd  = stage1_data.get('pd_percentage', round(max(0.5, min(
# # #                     _base_pd * _deliq_mul + _foir_adj + _emp_adj + _inq_adj + _ml_adj, 25.0)), 2))

# # #                 report_data = {
# # #                     'application_id':  _safe(stage1_customer.get('application_id')),
# # #                     'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# # #                     'model_version':   '8.7',
# # #                     'decision':        _safe(stage1_data.get('decision')),
# # #                     'stage2_final_decision':      _safe(final_decision),
# # #                     'stage2_tier':                _safe(stage2_tier),
# # #                     'stage2_interest_range':      _safe(interest_range),
# # #                     'stage2_combined_risk_score': _safe(combined_risk, 0),
# # #                     'stage2_confidence':          _safe(stage2_confidence, 0),
# # #                     'stage2_reason':              _safe(stage2_result.get('reason')),
# # #                     'stage2_tier_probabilities':  stage2_result.get('tier_probabilities') or {},
# # #                     'stage2_complete_analysis':   stage2_result,
# # #                     # Top-level PD — used by audit header (must match pd_calculation_factors.final_pd)
# # #                     'pd_percentage':              _final_pd,
# # #                     'risk_score':                 _safe(combined_risk, 0),
# # #                     'confidence':                 _safe(stage2_confidence, 0),
# # #                     # Policy gate results
# # #                     'policy_checks': stage1_data.get('policy_checks', {}),
# # #                     # Full PD calculation breakdown
# # #                     'pd_calculation_factors': {
# # #                         'bureau_score':           _bs,
# # #                         'base_pd':                round(_base_pd, 2),
# # #                         'dpd_90':                 _dpd90,
# # #                         'dpd_30':                 _dpd30,
# # #                         'delinquency_multiplier': round(_deliq_mul, 2),
# # #                         'foir':                   round(_foir, 2),
# # #                         'foir_adjustment':        round(_foir_adj, 2),
# # #                         'employment_adjustment':  round(_emp_adj, 2),
# # #                         'inquiry_adjustment':     round(_inq_adj, 2),
# # #                         'ml_adjustment':          round(_ml_adj, 2),
# # #                         'final_pd':               _final_pd,
# # #                     },
# # #                     # Reason codes from Stage 1
# # #                     'reason_codes': stage1_customer.get('reason_codes', []),
# # #                     # Raw data refs
# # #                     'customer_data':          stage1_customer,
# # #                     'enhanced_customer_data': enhanced_customer_data,
# # #                 }
# # #                 pdf_buffer = generate_audit_pdf(report_data)
# # #                 st.download_button("📥 Download PDF Report", data=pdf_buffer,
# # #                                    file_name=f"stage2_report_{stage1_customer.get('application_id','X')}.pdf",
# # #                                    mime="application/pdf", use_container_width=True)
# # #             except Exception as e:
# # #                 st.error(f"PDF generation failed: {str(e)}")
# # #         else:
# # #             st.warning("⚠️ PDF generation is not available. Ensure utils/pdf_generator.py is present and `reportlab` is installed (add to requirements.txt).")

# # #     st.markdown("---")
# # #     col1, col2, col3 = st.columns(3)
# # #     with col1:
# # #         if st.button("🔄 New Assessment", key="new_assessment_stage2", use_container_width=True):
# # #             for k in ['stage1_complete','stage1_decision','stage1_data','current_customer_data']:
# # #                 st.session_state[k] = (False if k == 'stage1_complete' else None)
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
# # # # FAIRNESS MONITORING DASHBOARD
# # # # =============================================================================
# # # def render_fairness_dashboard():
# # #     st.markdown('<p class="main-header">⚖️ Fairness Monitoring</p>', unsafe_allow_html=True)
# # #     st.markdown("""
# # #         <div class="info-box">
# # #             <strong>RBI Fair Lending Compliance Dashboard</strong><br>
# # #             Tracks approval rates across demographic groups to detect potential disparate impact.
# # #             <strong>Fairness is measured on the FINAL binding decision</strong> — Stage 2 outcome
# # #             is used when available; Stage 1 (screening) entries are automatically replaced once
# # #             Stage 2 completes for the same application.
# # #             Data is session-based — decisions accumulate as applications are processed.
# # #         </div>
# # #     """, unsafe_allow_html=True)

# # #     log = st.session_state.get('fairness_log', [])

# # #     col1, col2 = st.columns([3, 1])
# # #     with col2:
# # #         if st.button("🗑️ Clear Log", use_container_width=True):
# # #             st.session_state.fairness_log = []
# # #             st.rerun()

# # #     if not log:
# # #         st.info("ℹ️ No decisions logged yet. Process some applications from the Assessment page to see fairness metrics here.")
# # #         st.markdown("### 📊 What will appear here:")
# # #         st.markdown("""
# # #         - **Approval rate by Gender** — tracks if male/female/other applicants are treated equitably
# # #         - **Approval rate by City Tier** — checks for geographic bias (Tier 1 vs Tier 3 vs Rural)
# # #         - **Approval rate by Age Band** — identifies potential age discrimination
# # #         - **Approval rate by Employment Type** — salaried vs self-employed equity check
# # #         - **Average Risk Score & PD by group** — confirms scoring is not systematically biased
# # #         """)
# # #         return

# # #     df = pd.DataFrame(log)
# # #     df['approved'] = (df['decision'] == 'APPROVE').astype(int)
# # #     n = len(df)

# # #     # Source breakdown
# # #     if 'source' in df.columns:
# # #         n_s2    = int((df['source'] == 'stage2').sum())
# # #         n_s1    = int((df['source'] == 'stage1').sum())
# # #         n_batch = int((df['source'] == 'batch').sum())
# # #         src_note = f"📌 {n_s2} Stage 2 (final) · {n_s1} Stage 1 (screening) · {n_batch} Batch"
# # #         st.caption(src_note)

# # #     st.markdown("---")
# # #     c1, c2, c3, c4 = st.columns(4)
# # #     with c1: st.metric("Total Decisions", n)
# # #     with c2: st.metric("Approvals", int(df['approved'].sum()), f"{df['approved'].mean()*100:.1f}%")
# # #     with c3: st.metric("Reviews", int((df['decision']=='REVIEW').sum()))
# # #     with c4: st.metric("Rejections", int((df['decision']=='REJECT').sum()))

# # #     st.markdown("---")
# # #     tab1, tab2, tab3, tab4 = st.tabs(["👥 Gender", "🏙️ City Tier", "📅 Age Band", "💼 Employment"])

# # #     COLOR_MAP = {'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}

# # #     def _approval_bar(group_col, title):
# # #         grp = df.groupby(group_col).agg(
# # #             Total=('decision', 'count'),
# # #             Approved=('approved', 'sum'),
# # #             Avg_Risk=('risk_score', 'mean'),
# # #             Avg_PD=('pd_pct', 'mean'),
# # #         ).reset_index()
# # #         grp['Approval Rate %'] = (grp['Approved'] / grp['Total'] * 100).round(1)
# # #         grp['Avg Risk Score'] = grp['Avg_Risk'].round(1)
# # #         grp['Avg PD %'] = grp['Avg_PD'].round(2)

# # #         col1, col2 = st.columns([2, 1])
# # #         with col1:
# # #             fig = px.bar(grp, x=group_col, y='Approval Rate %',
# # #                          title=title, text='Approval Rate %',
# # #                          color='Approval Rate %',
# # #                          color_continuous_scale=['#f56565', '#ed8936', '#48bb78'],
# # #                          range_color=[0, 100])
# # #             fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
# # #             fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10),
# # #                               coloraxis_showscale=False, paper_bgcolor='white', plot_bgcolor='white',
# # #                               yaxis={'range': [0, 110], 'gridcolor': '#e2e8f0'})
# # #             st.plotly_chart(fig, use_container_width=True)
# # #         with col2:
# # #             st.markdown("**Summary Table**")
# # #             display_df = grp[[group_col, 'Total', 'Approval Rate %', 'Avg Risk Score', 'Avg PD %']].copy()
# # #             # Flag groups with approval rate deviation > 15pp from overall
# # #             overall_rate = df['approved'].mean() * 100
# # #             def _flag(rate):
# # #                 diff = rate - overall_rate
# # #                 if abs(diff) > 15: return f"{'🔴' if diff < 0 else '🟢'} {rate:.1f}%"
# # #                 return f"✅ {rate:.1f}%"
# # #             display_df['Approval Rate %'] = display_df['Approval Rate %'].apply(_flag)
# # #             st.dataframe(display_df, use_container_width=True, hide_index=True)
# # #             overall_str = f"{overall_rate:.1f}%"
# # #             st.caption(f"Overall approval rate: **{overall_str}**. 🔴 = >15pp below average (potential bias). 🟢 = >15pp above average.")

# # #     with tab1:
# # #         if df['gender'].nunique() > 1:
# # #             _approval_bar('gender', 'Approval Rate by Gender')
# # #             # Decision mix donut per gender
# # #             fig2 = px.pie(df, names='decision', color='decision', color_discrete_map=COLOR_MAP,
# # #                           title='Decision Mix (all)', hole=0.5)
# # #             fig2.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
# # #             st.plotly_chart(fig2, use_container_width=True)
# # #         else:
# # #             st.info("Need 2+ gender values in decisions to show chart. Ensure Gender field is filled on the form.")

# # #     with tab2:
# # #         if df['city_tier'].nunique() > 1:
# # #             _approval_bar('city_tier', 'Approval Rate by City Tier')
# # #         else:
# # #             st.info("Need 2+ city tier values. Ensure City Tier field is filled on the form.")

# # #     with tab3:
# # #         if df['age_band'].nunique() > 1:
# # #             _approval_bar('age_band', 'Approval Rate by Age Band')
# # #         else:
# # #             st.info("Need decisions across multiple age bands (24-30, 31-40, 41-50, 51+).")

# # #     with tab4:
# # #         if df['employment_type'].nunique() > 1:
# # #             _approval_bar('employment_type', 'Approval Rate by Employment Type')
# # #         else:
# # #             st.info("Need 2+ employment types in decisions.")

# # #     st.markdown("---")
# # #     st.markdown("### 📥 Export Fairness Report")
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         csv_data = df.to_csv(index=False)
# # #         st.download_button("📥 Download Decision Log (CSV)", data=csv_data,
# # #                            file_name=f"fairness_log_{datetime.now().strftime('%Y%m%d')}.csv",
# # #                            mime="text/csv", use_container_width=True)
# # #     with col2:
# # #         st.caption("⚠️ **Note:** This log is session-based and resets when the app restarts. "
# # #                    "For persistent fairness monitoring, connect to a database or export regularly.")


# # # # =============================================================================
# # # # SIDEBAR
# # # # =============================================================================
# # # with st.sidebar:
# # #     st.markdown("# 🏦 Credit Risk Engine")
# # #     st.markdown("---")

# # #     navigation_options = ["🏠 Home", "👤 Assessment", "📊 Batch Process", "⚖️ Fairness", "📈 Model Info", "ℹ️ About"]

# # #     if (st.session_state.stage1_complete and st.session_state.stage1_decision in ['APPROVE', 'REVIEW']):
# # #         navigation_options.insert(2, "🔬 Stage 2 Analysis")
# # #         st.success(f"✅ Stage 1: {st.session_state.stage1_decision}")
# # #         st.info("🔬 Stage 2 Analysis unlocked!")
# # #     elif st.session_state.stage1_complete:
# # #         st.warning(f"⚠️ Stage 1: {st.session_state.stage1_decision}")
# # #         st.caption("Stage 2 only for APPROVE/REVIEW")

# # #     page = st.radio("**Navigation**", navigation_options,
# # #                     label_visibility="collapsed", key="page_navigation")

# # #     st.markdown("---")
# # #     stage2_indicator = '✅ Active' if STAGE2_AVAILABLE and is_stage2_available() else '❌ Inactive'
# # #     ocr_indicator = '✅ Ready' if OCR_AVAILABLE else '❌ Not Installed'
# # #     pdf_indicator = '✅ Ready' if PDF_AVAILABLE else '❌ Not Installed'
# # #     fairness_count = len(st.session_state.fairness_log)

# # #     st.markdown(f"""
# # #     <div class="info-card">
# # #         <div class="info-card-title">System Status</div>
# # #         <div class="info-card-content">
# # #             <div class="data-row"><span class="data-label">Model</span><span class="data-value">✅ Loaded</span></div>
# # #             <div class="data-row"><span class="data-label">Version</span><span class="data-value">8.6</span></div>
# # #             <div class="data-row"><span class="data-label">Stage 2</span><span class="data-value">{stage2_indicator}</span></div>
# # #             <div class="data-row"><span class="data-label">OCR</span><span class="data-value">{ocr_indicator}</span></div>
# # #             <div class="data-row"><span class="data-label">PDF Gen</span><span class="data-value">{pdf_indicator}</span></div>
# # #             <div class="data-row"><span class="data-label">Fairness Log</span><span class="data-value">{fairness_count} decisions</span></div>
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
# # #             for k in ['stage1_complete','stage1_decision','stage1_data','current_customer_data','extracted_cibil_data']:
# # #                 st.session_state[k] = False if k == 'stage1_complete' else None
# # #             st.rerun()

# # # # =============================================================================
# # # # PAGE ROUTING
# # # # =============================================================================
# # # if page == "🏠 Home":
# # #     st.markdown('<p class="main-header">Credit Risk Engine</p>', unsafe_allow_html=True)
# # #     st.markdown('<div class="info-box"><h3 style="margin-top:0;">🎯 AI-Powered Lending Decisions</h3><p style="margin-bottom:0;">Comprehensive credit risk evaluation combining hard policy rules, machine learning models, and affordability analysis.</p></div>', unsafe_allow_html=True)
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     col1, col2, col3 = st.columns(3)
# # #     with col1:
# # #         st.markdown('<div class="info-card"><div class="info-card-title">🛡️ Policy Gates</div><div class="info-card-content"><ul><li>Age & KYC verification</li><li>RBI consent check</li><li>Employment stability</li><li>Minimum income checks</li><li>Credit bureau thresholds</li><li>Bankruptcy & fraud detection</li></ul></div></div>', unsafe_allow_html=True)
# # #     with col2:
# # #         st.markdown('<div class="info-card"><div class="info-card-title">🤖 ML Assessment</div><div class="info-card-content"><ul><li>Random Forest classifier</li><li>60K+ training samples</li><li>Confidence scoring</li><li>Multi-class prediction</li><li>Feature importance</li></ul></div></div>', unsafe_allow_html=True)
# # #     with col3:
# # #         st.markdown('<div class="info-card"><div class="info-card-title">⚖️ Fairness Monitoring</div><div class="info-card-content"><ul><li>Approval rate by gender</li><li>Approval rate by city tier</li><li>Age band equity check</li><li>Employment type parity</li><li>RBI compliance ready</li></ul></div></div>', unsafe_allow_html=True)
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     col1, col2, col3, col4 = st.columns(4)
# # #     with col1: st.metric("🎯 Accuracy", "85%", "+2%")
# # #     with col2: st.metric("⚡ Avg Response", "1.2s", "-0.3s")
# # #     with col3: st.metric("📊 Features", len(TOP_FEATURES))
# # #     with col4: st.metric("🔄 Version", "8.6", "Latest")
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     st.markdown("""
# # #         <div class="warning-box" style="background:#f0fff4;border:1px solid #9ae6b4;padding:1rem;border-radius:0.5rem;">
# # #             <strong>🆕 New in Version 8.6:</strong><br>
# # #             • <strong>Cleaned codebase</strong> — removed ~210 lines of duplicate function definitions<br>
# # #             • <strong>City Tier field</strong> — Tier 1/2/3/Rural captured on every application<br>
# # #             • <strong>Gender field</strong> — explicit gender capture for fairness logging<br>
# # #             • <strong>RBI Consent checkbox</strong> — required policy gate before assessment<br>
# # #             • <strong>Fairness Monitoring dashboard</strong> — approval rates by gender, city tier, age band, employment type<br>
# # #             • <strong>v8.5 features retained</strong> — dual-dataset OCR inference, categorical flag auto-fill
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
# # #             c1.metric("Credit Score", ex.get('Credit_Score', '—'))
# # #             c2.metric("Monthly Income", f"₹{ex.get('NETMONTHLYINCOME') or ex.get('avg_salary_6m', 0):,}")
# # #             c3.metric("DPD 60+ Count", ex.get('num_times_60p_dpd', 0))
# # #             c4.metric("CC Utilization", f"{max(0, float(ex.get('CC_utilization', 0) or 0))*100:.0f}%")
# # #             _inf = st.session_state.get('_last_inferred_flags', {})
# # #             if _inf:
# # #                 st.markdown("**📊 Inferred Categorical Flags:**")
# # #                 fc1, fc2, fc3, fc4, fc5 = st.columns(5)
# # #                 fc1.metric("Payment Discipline", _inf.get('payment_discipline_flag', '—'))
# # #                 fc2.metric("Cashflow Health", _inf.get('cashflow_health', '—'))
# # #                 fc3.metric("Liquidity", _inf.get('liquidity_flag', '—'))
# # #                 fc4.metric("Bureau Risk", _inf.get('bureau_risk_flag', '—'))
# # #                 fc5.metric("Salary Stability", _inf.get('salary_stability_flag', '—'))
# # #             if st.button("🔄 Upload a different PDF", key="reset_pdf"):
# # #                 st.session_state.pdf_just_extracted = False
# # #                 st.session_state.pop('_last_extraction', None)
# # #                 st.session_state.pop('_last_inferred_flags', None)
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
# # #                         # ── Stage 1: 60k dataset field autofill ──────────────
# # #                         st.session_state.pdf_age               = int(extraction_result.get('AGE', 35))
# # #                         st.session_state.pdf_bureau_score      = int(extraction_result.get('Credit_Score', 720))
# # #                         st.session_state.pdf_dpd_90            = int(extraction_result.get('dpd_90_count_6m', 0))
# # #                         st.session_state.pdf_dpd_30            = int(extraction_result.get('dpd_30_count_6m', 0))
# # #                         _cc_util_raw = extraction_result.get('CC_utilization', 0) or 0
# # #                         st.session_state.pdf_credit_util       = int(max(0, float(_cc_util_raw)) * 100) if _cc_util_raw > 0 else 0
# # #                         st.session_state.pdf_inquiries         = int(extraction_result.get('enq_L3m', 2))
# # #                         st.session_state.pdf_active_loans      = int(extraction_result.get('num_std', 1))
# # #                         _emi = int(extraction_result.get('existing_emi') or extraction_result.get('total_emi_monthly') or 0)
# # #                         st.session_state.pdf_existing_emi      = _emi
# # #                         _income = int(extraction_result.get('NETMONTHLYINCOME') or extraction_result.get('avg_salary_6m') or 50000)
# # #                         st.session_state.pdf_monthly_income    = _income
# # #                         st.session_state.pdf_annual_income     = int(extraction_result.get('AMT_INCOME_TOTAL') or _income * 12)
# # #                         _surplus = int(extraction_result.get('net_cash_surplus_6m') or extraction_result.get('_surplus_proxy') or 0)
# # #                         st.session_state.pdf_net_surplus       = _surplus
# # #                         st.session_state.pdf_employment_tenure = int(extraction_result.get('Time_With_Curr_Empr', 24))
# # #                         # Employment type (new — was never filled before)
# # #                         _emp = extraction_result.get('employment_type', 'Salaried')
# # #                         if _emp in ['Salaried', 'Self-Employed', 'Business']:
# # #                             st.session_state.pdf_employment_type = _emp
# # #                         # Business vintage (new)
# # #                         st.session_state.pdf_business_vintage  = int(extraction_result.get('business_vintage_years', 0))
# # #                         # Gender (new — was extracted but never applied to form)
# # #                         _g = extraction_result.get('GENDER', 'M')
# # #                         st.session_state.pdf_gender = 'Male' if _g == 'M' else 'Female'
# # #                         # Dependents: CIBIL PDFs rarely state this; leave at form default
# # #                         # Inward bounce & missing salary (inferred from delinquency)
# # #                         st.session_state.pdf_inward_bounce     = int(extraction_result.get('inward_bounce_count_3m', 0))
# # #                         st.session_state.pdf_salary_missing    = int(extraction_result.get('salary_missing_months', 0))
# # #                         # Categorical flags (now come directly from extraction, no second infer needed)
# # #                         st.session_state.pdf_salary_stability   = extraction_result.get('salary_stability_flag', 'MODERATE')
# # #                         st.session_state.pdf_payment_discipline = extraction_result.get('payment_discipline_flag', 'GOOD')
# # #                         st.session_state.pdf_cashflow_health    = extraction_result.get('cashflow_health', 'MODERATE')
# # #                         st.session_state.pdf_liquidity_flag     = extraction_result.get('liquidity_flag', 'MODERATE')
# # #                         st.session_state.pdf_bureau_risk_flag   = extraction_result.get('bureau_risk_flag', 'MODERATE')
# # #                         st.session_state.pdf_just_extracted     = True
# # #                         st.session_state._last_extraction       = extraction_result
# # #                         st.rerun()
# # #                     else:
# # #                         st.error(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")

# # #     with st.form("assessment_form"):
# # #         # ── Identity & Eligibility ─────────────────────────────────────────
# # #         st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
# # #         col_name1, col_name2 = st.columns([2, 2])
# # #         with col_name1:
# # #             customer_name = st.text_input("Customer Name (Optional)", value="", placeholder="e.g. Ramesh Kumar")
# # #         col1, col2, col3, col4 = st.columns(4)
# # #         with col1:
# # #             age = st.number_input("Age", 18, 70, value=int(st.session_state.get('pdf_age', 35)))
# # #             employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business'],
# # #                 index=['Salaried','Self-Employed','Business'].index(st.session_state.get('pdf_employment_type','Salaried')))
# # #         with col2:
# # #             _gender_opts = ['Male', 'Female', 'Non-binary / Other', 'Prefer not to say']
# # #             _gender_default = st.session_state.get('pdf_gender', 'Male')
# # #             _gender_idx = _gender_opts.index(_gender_default) if _gender_default in _gender_opts else 0
# # #             gender = st.selectbox("Gender", _gender_opts, index=_gender_idx)
# # #             dependents = st.number_input("Number of Dependents", 0, 20, value=int(st.session_state.get('pdf_dependents', 2)))
# # #         with col3:
# # #             # City Tier — field for fairness monitoring.
# # #             # FIX A-6: Use format_func so the selectbox displays the full label to the user
# # #             # but city_tier is derived immediately from CITY_TIERS at render time —
# # #             # no deferred lookup needed. A caption confirms the stored code.
# # #             _city_keys = list(CITY_TIERS.keys())
# # #             city_tier_label = st.selectbox(
# # #                 "City Tier", _city_keys, index=0,
# # #                 format_func=lambda k: k  # full descriptive label shown to user
# # #             )
# # #             city_tier = CITY_TIERS[city_tier_label]   # short code: 'Tier 1' / 'Tier 2' / etc.
# # #             st.caption(f"Stored as: **{city_tier}**")
# # #             kyc_verified = st.selectbox("KYC Verified", ['Yes', 'No'],
# # #                 index=0 if st.session_state.get('pdf_kyc', True) else 1) == 'Yes'
# # #         with col4:
# # #             bankruptcy_flag = st.selectbox("Bankruptcy Flag", ['No', 'Yes'],
# # #                 index=0 if not st.session_state.get('pdf_bankruptcy', False) else 1) == 'Yes'
# # #             fraud_flag = st.selectbox("Fraud Flag", ['No', 'Yes'],
# # #                 index=0 if not st.session_state.get('pdf_fraud', False) else 1) == 'Yes'

# # #         # RBI Consent — REQUIRED
# # #         st.markdown('<p class="section-header">📜 RBI Compliance</p>', unsafe_allow_html=True)
# # #         col1, col2 = st.columns([2, 1])
# # #         with col1:
# # #             rbi_consent = st.checkbox(
# # #                 "✅ I confirm the customer has been informed of and consented to: (a) credit bureau enquiry, "
# # #                 "(b) data usage for credit assessment, (c) Key Fact Statement (KFS) terms, and "
# # #                 "(d) grievance redressal process. **(Required — RBI Digital Lending Guidelines)**",
# # #                 value=False
# # #             )
# # #         with col2:
# # #             st.markdown("""
# # #                 <div style="background:#fff3cd;border:1px solid #ffc107;padding:0.5rem;border-radius:0.4rem;font-size:0.82rem;">
# # #                     ⚠️ Without consent, the application cannot proceed per RBI DLG 2022.
# # #                 </div>
# # #             """, unsafe_allow_html=True)

# # #         # Employment tenure
# # #         st.markdown('<p class="section-header">💼 Employment</p>', unsafe_allow_html=True)
# # #         col1, col2 = st.columns(2)
# # #         with col1:
# # #             if employment_type == 'Salaried':
# # #                 employment_tenure = st.number_input("Employment Tenure (months)", 0, 600,
# # #                     value=int(st.session_state.get('pdf_employment_tenure', 24)))
# # #                 business_vintage = 0
# # #             else:
# # #                 business_vintage = st.number_input("Business Vintage (years)", 0, 50,
# # #                     value=int(st.session_state.get('pdf_business_vintage', 3)))
# # #                 employment_tenure = 0
# # #         with col2:
# # #             st.markdown("""
# # #                 <div class="info-box" style="margin-top:1rem;">
# # #                     <strong>Policy thresholds:</strong><br>
# # #                     Salaried: min 6 months<br>
# # #                     Self-Employed/Business: min 2 years
# # #                 </div>
# # #             """, unsafe_allow_html=True)

# # #         # Credit Bureau
# # #         st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
# # #         col1, col2, col3 = st.columns(3)
# # #         with col1:
# # #             bureau_score = st.number_input("Bureau Score", 300, 900,
# # #                 value=int(st.session_state.get('pdf_bureau_score', 720)), step=10)
# # #             dpd_90_6m = st.number_input("DPD 90+ (Last 6M)", 0, 20,
# # #                 value=int(st.session_state.get('pdf_dpd_90', 0)))
# # #             dpd_30_6m = st.number_input("DPD 30+ (Last 6M)", 0, 20,
# # #                 value=int(st.session_state.get('pdf_dpd_30', 0)))
# # #         with col2:
# # #             credit_utilization = st.number_input("Credit Utilization (%)", 0, 100,
# # #                 value=int(st.session_state.get('pdf_credit_util', 30)))
# # #             recent_inquiries = st.number_input("Recent Inquiries (3M)", 0, 20,
# # #                 value=int(st.session_state.get('pdf_inquiries', 2)))
# # #         with col3:
# # #             active_loans = st.number_input("Active Loans", 0, 10,
# # #                 value=int(st.session_state.get('pdf_active_loans', 1)))
# # #             existing_emi = st.number_input("Existing Total EMI (₹)", 0, 200000,
# # #                 value=int(st.session_state.get('pdf_existing_emi', 15000)), step=1000)

# # #         # Income & Financial
# # #         st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
# # #         col1, col2, col3, col4 = st.columns(4)
# # #         with col1:
# # #             avg_salary = st.number_input("Monthly Income (₹)", 0, 1000000,
# # #                 value=int(st.session_state.get('pdf_monthly_income', 50000)), step=5000)
# # #             amt_income = st.number_input("Annual Income (₹)", 0, 10000000,
# # #                 value=int(st.session_state.get('pdf_annual_income', 600000)), step=10000)
# # #         with col2:
# # #             net_surplus = st.number_input("Net Cash Surplus (₹)", -100000, 500000,
# # #                 value=int(st.session_state.get('pdf_net_surplus', 20000)), step=5000)
# # #             _ss_opts = ['STABLE', 'MODERATE', 'UNSTABLE']
# # #             salary_stability = st.selectbox("Salary Stability", _ss_opts,
# # #                 index=_ss_opts.index(st.session_state.get('pdf_salary_stability', 'STABLE')))
# # #         with col3:
# # #             loan_amount = st.number_input("Loan Amount (₹)", 0, 5000000,
# # #                 value=int(st.session_state.get('pdf_loan_amount', 180000)), step=10000)
# # #             loan_tenure = st.number_input("Tenure (months)", 3, 360,
# # #                 value=int(st.session_state.get('pdf_loan_tenure', 24)))
# # #         with col4:
# # #             interest_rate = st.number_input("Interest Rate (%)", 8.0, 20.0,
# # #                 value=float(st.session_state.get('pdf_interest_rate', 10.5)), step=0.5)
# # #             amt_annuity = st.number_input("Requested EMI (₹)", 0, 200000,
# # #                 value=int(st.session_state.get('pdf_amt_annuity', 8500)), step=500)

# # #         # Additional Credit Behaviour
# # #         st.markdown('<p class="section-header">📋 Additional Credit Behaviour</p>', unsafe_allow_html=True)
# # #         col1, col2, col3 = st.columns(3)
# # #         with col1:
# # #             _pd_opts = ['GOOD', 'MODERATE', 'POOR']
# # #             payment_discipline = st.selectbox("Payment Discipline", _pd_opts,
# # #                 index=_pd_opts.index(st.session_state.get('pdf_payment_discipline', 'GOOD')))
# # #             _lq_opts = ['LOW', 'ADEQUATE', 'MODERATE']
# # #             liquidity_flag = st.selectbox("Liquidity", _lq_opts,
# # #                 index=_lq_opts.index(st.session_state.get('pdf_liquidity_flag', 'LOW')))
# # #         with col2:
# # #             _cf_opts = ['MODERATE', 'HEALTHY', 'STRESSED', 'STABLE']
# # #             cashflow_health = st.selectbox("Cashflow Health", _cf_opts,
# # #                 index=_cf_opts.index(st.session_state.get('pdf_cashflow_health', 'MODERATE')))
# # #             _br_opts = ['LOW', 'MEDIUM', 'HIGH']
# # #             bureau_risk_flag = st.selectbox("Bureau Risk", _br_opts,
# # #                 index=_br_opts.index(st.session_state.get('pdf_bureau_risk_flag', 'LOW')))
# # #         with col3:
# # #             inward_bounce_count   = st.number_input("Inward Bounce Count (3M)", 0, 10, value=int(st.session_state.get('pdf_inward_bounce', 0)))
# # #             salary_missing_months = st.number_input("Missing Salary Months (6M)", 0, 6, value=int(st.session_state.get('pdf_salary_missing', 0)))

# # #         st.markdown("<br>", unsafe_allow_html=True)
# # #         submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)

# # #     if submitted:
# # #         timestamp = datetime.now()
# # #         app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
# # #         customer_data = {
# # #             'name': customer_name.strip() if customer_name.strip() else 'N/A',
# # #             'age': age, 'employment_type': employment_type,
# # #             'gender': gender, 'city_tier': city_tier,
# # #             'dependents': dependents, 'kyc_verified': kyc_verified,
# # #             'rbi_consent': rbi_consent,
# # #             'bankruptcy_flag': bankruptcy_flag, 'fraud_flag': fraud_flag,
# # #             'employment_tenure_months': employment_tenure,
# # #             'business_vintage_years': business_vintage,
# # #             'bureau_score': bureau_score,
# # #             'dpd_90_count_6m': dpd_90_6m, 'dpd_30_count_6m': dpd_30_6m,
# # #             'credit_utilization_pct': credit_utilization, 'max_utilization': credit_utilization,
# # #             'recent_inquiries_3m': recent_inquiries, 'active_loans_count': active_loans,
# # #             'avg_salary_6m': avg_salary, 'AMT_INCOME_TOTAL': amt_income,
# # #             'net_cash_surplus_6m': net_surplus, 'salary_stability_flag': salary_stability,
# # #             'loan_amount': loan_amount, 'loan_tenure_months': loan_tenure,
# # #             'interest_rate': interest_rate, 'existing_emi': existing_emi,
# # #             'AMT_ANNUITY': amt_annuity, 'application_id': app_id,
# # #             'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
# # #             'payment_discipline_flag': payment_discipline,
# # #             'liquidity_flag': liquidity_flag, 'cashflow_health': cashflow_health,
# # #             'bureau_risk_flag': bureau_risk_flag,
# # #             'inward_bounce_count_3m': inward_bounce_count,
# # #             'salary_missing_months': salary_missing_months,
# # #         }

# # #         with st.spinner("🔄 Processing Stage 1 assessment..."):
# # #             decision_data = make_hybrid_decision_enhanced(customer_data)

# # #         # Inject ML confidence so reason_codes.py can distinguish ML-driven REVIEW
# # #         customer_data['ml_confidence'] = decision_data.get('confidence', 0)
# # #         reasons = generate_reason_codes(
# # #             decision=decision_data.get('decision', 'ERROR'),
# # #             customer_data=customer_data,
# # #             affordability_data=decision_data.get('affordability_data', {}),
# # #             policy_checks=decision_data.get('policy_checks', {})
# # #         )
# # #         customer_data['reason_codes'] = reasons

# # #         # Log to fairness monitor (Stage 1 — may be replaced by Stage 2 final decision)
# # #         log_decision_for_fairness(customer_data, decision_data.get('decision','ERROR'),
# # #                                   decision_data.get('risk_score', 0), decision_data.get('pd_percentage', 0),
# # #                                   application_id=customer_data.get('application_id'),
# # #                                   source='stage1')

# # #         st.session_state.stage1_complete = True
# # #         st.session_state.stage1_decision = decision_data.get('decision', 'ERROR')
# # #         st.session_state.stage1_data = decision_data
# # #         st.session_state.current_customer_data = customer_data

# # #         for key in list(st.session_state.keys()):
# # #             if key.startswith('pdf_') or key in ('_last_extraction', '_last_inferred_flags'):
# # #                 del st.session_state[key]

# # #         tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])

# # #         with tab1:
# # #             st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
# # #             col1, col2 = st.columns(2)
# # #             with col1:
# # #                 render_info_card("👤 Identity", "👤",
# # #                                  {"Age": age, "Gender": gender, "City Tier": city_tier,
# # #                                   "Employment": employment_type, "Dependents": dependents,
# # #                                   "KYC Status": "Verified" if kyc_verified else "Not Verified",
# # #                                   "RBI Consent": "✅ Obtained" if rbi_consent else "❌ Not obtained"})
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
# # #                 st.markdown('<div class="info-box" style="background:linear-gradient(135deg,#10B981,#059669);color:white;text-align:center;"><h3 style="margin:0;color:white;">✅ Eligible for Stage 2 Deep Dive</h3></div>', unsafe_allow_html=True)
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
# # #                 st.markdown('<div style="background:linear-gradient(135deg,#EF4444,#DC2626);color:white;padding:1rem;border-radius:0.5rem;text-align:center;"><h3 style="margin:0;color:white;">❌ Stage 2 Not Available</h3><p style="margin:0.5rem 0 0 0;">Application rejected. Stage 2 only for APPROVE/REVIEW.</p></div>', unsafe_allow_html=True)

# # #             st.markdown("<br>", unsafe_allow_html=True)
# # #             affordability = decision_data.get('affordability_data', {})
# # #             foir      = affordability.get('foir_percentage', 0)
# # #             total_emi = int(round(affordability.get('total_emi', 0)))
# # #             net_disp  = int(round(affordability.get('net_disposable', 0)))

# # #             col1, col2, col3 = st.columns(3)
# # #             with col1:
# # #                 render_info_card("Identity & Eligibility", "👤",
# # #                     {f"Age: {age}": "", f"Employment: {employment_type}": "",
# # #                      f"City Tier: {city_tier}": "", f"Dependents: {dependents}": "",
# # #                      f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""},
# # #                     {f"Age: {age}": "pass" if (age > 24 and age <= (65 if employment_type == 'Salaried' else 70)) else "fail",
# # #                      f"Employment: {employment_type}": "pass",
# # #                      f"City Tier: {city_tier}": "pass",
# # #                      f"Dependents: {dependents}": "pass" if dependents <= 5 else "warning",
# # #                      f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_verified else "fail"})
# # #             with col2:
# # #                 render_info_card("Credit Bureau", "🏦",
# # #                     {f"Bureau Score: {bureau_score}": "", f"DPD 90+: {dpd_90_6m}": "",
# # #                      f"Utilization: {credit_utilization}%": ""},
# # #                     {f"Bureau Score: {bureau_score}": "pass" if bureau_score >= 550 else "fail",
# # #                      f"DPD 90+: {dpd_90_6m}": "pass" if dpd_90_6m == 0 else ("warning" if dpd_90_6m <= 5 else "fail"),
# # #                      f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"})
# # #             with col3:
# # #                 render_info_card("Affordability", "💰",
# # #                     {f"Monthly Income: ₹{avg_salary:,}": "", f"FOIR: {foir:.1f}%": "",
# # #                      f"Total EMI: ₹{total_emi:,}": "", f"Net Disposable: ₹{net_disp:,}": ""},
# # #                     {f"Monthly Income: ₹{avg_salary:,}": "pass",
# # #                      f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
# # #                      f"Total EMI: ₹{total_emi:,}": "pass",
# # #                      f"Net Disposable: ₹{net_disp:,}": "pass" if net_disp >= 10000 else "warning"})

# # #             st.markdown("<br>", unsafe_allow_html=True)
# # #             render_reason_codes(reasons)
# # #             st.markdown("<br>", unsafe_allow_html=True)
# # #             col1, col2 = st.columns([1, 1])
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
# # #                     st.warning("⚠️ PDF download unavailable — `reportlab` or `pdf_generator.py` not found. Check requirements.txt.")
# # #             with col2:
# # #                 if st.button("🔄 Re-Evaluate", key="reevaluate_btn", use_container_width=True):
# # #                     st.rerun()

# # #         with tab3:
# # #             st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
# # #             col1, col2 = st.columns(2)
# # #             with col1:
# # #                 st.plotly_chart(create_modern_gauge(decision_data.get('confidence', 0), "Model Confidence"), use_container_width=True)
# # #             with col2:
# # #                 st.plotly_chart(create_modern_bar_chart(decision_data.get('class_probs', {"APPROVE": 0, "REVIEW": 0, "REJECT": 0})), use_container_width=True)
# # #             st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
# # #             policy_df = pd.DataFrame([{'Check': k, 'Result': v} for k, v in decision_data.get('policy_checks', {}).items()])
# # #             st.dataframe(policy_df, use_container_width=True, hide_index=True)
# # #             st.markdown('<p class="section-header">PD Calculation Breakdown</p>', unsafe_allow_html=True)
# # #             for factor, value in {
# # #                 'Bureau Score': f"{bureau_score} → Base PD: {bureau_score_to_pd(bureau_score):.1f}%",
# # #                 'Delinquency': f"DPD 90+: {dpd_90_6m}, DPD 30+: {dpd_30_6m} → Multiplier: {delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x",
# # #                 'FOIR Impact': f"{foir:.1f}% → Adjustment: {foir_to_pd_adjustment(foir):.1f}%",
# # #                 'Final PD': f"{decision_data.get('pd_percentage', 0)}%"
# # #             }.items():
# # #                 st.markdown(f"**{factor}:** {value}")

# # #         with tab4:
# # #             st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
# # #             audit_log = sanitize_for_json({
# # #                 'application_id': app_id,
# # #                 'timestamp': timestamp.isoformat(),
# # #                 'decision': decision_data.get('decision', 'ERROR'),
# # #                 'risk_score': decision_data.get('risk_score', 0),
# # #                 'pd_percentage': decision_data.get('pd_percentage', 0),
# # #                 'confidence': round(decision_data.get('confidence', 0), 2),
# # #                 'model_version': '8.6',
# # #                 'gender': gender, 'city_tier': city_tier,
# # #                 'rbi_consent': rbi_consent,
# # #                 'reason_codes': reasons,
# # #                 'policy_checks': decision_data.get('policy_checks', {}),
# # #                 'affordability': decision_data.get('affordability_data', {}),
# # #                 'customer_data': {k: v for k, v in customer_data.items() if k not in ['application_id','timestamp','reason_codes']},
# # #             })
# # #             with st.expander("📋 View Audit Log (JSON)"):
# # #                 st.json(audit_log)
# # #             col1, col2 = st.columns(2)
# # #             with col1:
# # #                 if PDF_AVAILABLE and generate_audit_pdf is not None:
# # #                     try:
# # #                         audit_pdf_buffer = generate_audit_pdf(audit_log)
# # #                         st.download_button("📥 Download Audit Trail (PDF)", data=audit_pdf_buffer,
# # #                                            file_name=f"audit_trail_{app_id}.pdf", mime="application/pdf",
# # #                                            use_container_width=True)
# # #                     except Exception as e:
# # #                         st.error(f"Error generating audit PDF: {str(e)}")
# # #                 else:
# # #                     st.warning("⚠️ Audit PDF download unavailable — `reportlab` or `pdf_generator.py` not found. Check requirements.txt.")
# # #             with col2:
# # #                 st.download_button("📥 Download Audit Log (JSON)",
# # #                                    data=json.dumps(audit_log, indent=2),
# # #                                    file_name=f"audit_{app_id}.json", mime="application/json",
# # #                                    use_container_width=True)

# # # elif page == "🔬 Stage 2 Analysis":
# # #     st.markdown('<p class="main-header">Stage 2: CIBIL Deep Dive</p>', unsafe_allow_html=True)

# # #     if not st.session_state.get('stage1_complete', False):
# # #         st.error("❌ You must complete Stage 1 Assessment first!")
# # #         if st.button("← Go to Assessment", use_container_width=True):
# # #             st.session_state.page_navigation = "👤 Assessment"
# # #             st.rerun()
# # #         st.stop()

# # #     if st.session_state.get('stage1_decision') not in ['APPROVE', 'REVIEW']:
# # #         st.error("❌ Stage 2 is only available for APPROVED or REVIEW applications!")
# # #         if st.button("← Go Back", use_container_width=True):
# # #             st.session_state.page_navigation = "👤 Assessment"
# # #             st.rerun()
# # #         st.stop()

# # #     if not (STAGE2_AVAILABLE and is_stage2_available()):
# # #         st.error("❌ Stage 2 model not available! Please ensure `stage2_cibil_model.pkl` is in the project directory.")
# # #         if st.button("← Go Back", use_container_width=True):
# # #             st.session_state.page_navigation = "👤 Assessment"
# # #             st.rerun()
# # #         st.stop()

# # #     stage1_data = st.session_state.get('stage1_data', {})
# # #     stage1_customer = st.session_state.get('current_customer_data', {})

# # #     st.markdown(f'<div class="info-box" style="background:linear-gradient(135deg,#3B82F6,#2563EB);color:white;"><h3 style="margin:0;color:white;">📊 Stage 1 Results</h3><p style="margin:0.5rem 0 0 0;"><strong>Decision:</strong> {st.session_state.get("stage1_decision","N/A")} | <strong>Risk Score:</strong> {stage1_data.get("risk_score","N/A")} | <strong>App ID:</strong> {stage1_customer.get("application_id","N/A")}</p></div>', unsafe_allow_html=True)
# # #     st.markdown("<br>", unsafe_allow_html=True)

# # #     tab_options = ["Manual Entry", "PDF Upload", "Batch Analysis"]
# # #     default_tab = st.session_state.get('stage2_selected_tab', 'Manual Entry')
# # #     selected_tab = st.radio("Select input method", tab_options,
# # #                             index=tab_options.index(default_tab) if default_tab in tab_options else 0,
# # #                             horizontal=True, label_visibility="collapsed")

# # #     if selected_tab == "Manual Entry":
# # #         st.markdown('<p class="section-header">Manual CIBIL Data Entry</p>', unsafe_allow_html=True)
# # #         with st.form("stage2_manual_form"):
# # #             st.markdown("### 👤 Demographics & Product Enquiries")
# # #             col1, col2, col3 = st.columns(3)
# # #             with col1:
# # #                 gender_s2 = st.selectbox("Gender", ["Male", "Female", "Others"])
# # #                 marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Others"])
# # #                 education = st.selectbox("Education", ["Graduate", "Post Graduate", "Under Graduate", "Professional", "Others"])
# # #             with col2:
# # #                 st.markdown("**Credit Score & History**")
# # #                 cibil_score = st.number_input("Credit Score", 300, 900, 720, 10)
# # #                 max_delinquency = st.number_input("Max Delinquency Level", 0, 100, 0)
# # #                 num_times_30dpd = st.number_input("Times 30+ DPD", 0, 50, 0)
# # #                 num_times_60dpd = st.number_input("Times 60+ DPD", 0, 50, 0)
# # #                 num_times_delinquent = st.number_input("Total Delinquent", 0, 50, 0)
# # #             with col3:
# # #                 st.markdown("**Recent Behavior**")
# # #                 num_deliq_6m = st.number_input("Delinquencies (6M)", 0, 20, 0)
# # #                 num_deliq_12m = st.number_input("Delinquencies (12M)", 0, 20, 0)
# # #                 max_deliq_6m = st.number_input("Max Delinq (6M)", 0, 100, 0)
# # #                 max_deliq_12m = st.number_input("Max Delinq (12M)", 0, 100, 0)
# # #                 enq_L3m = st.number_input("Inquiries (3M)", 0, 20, 2)
# # #                 enq_L6m = st.number_input("Inquiries (6M)", 0, 30, 4)
# # #                 enq_L12m = st.number_input("Inquiries (12M)", 0, 50, 6)

# # #             col1, col2, col3 = st.columns(3)
# # #             with col1:
# # #                 st.markdown("**Account Quality**")
# # #                 num_std = st.number_input("Standard Accounts", 0, 50, 3)
# # #                 num_std_6m = st.number_input("Standard (6M)", 0, 50, 3)
# # #                 num_std_12m = st.number_input("Standard (12M)", 0, 50, 3)
# # #                 num_sub = st.number_input("Sub-standard", 0, 20, 0)
# # #                 num_sub_6m = st.number_input("Sub-standard (6M)", 0, 20, 0)
# # #                 num_dbt = st.number_input("Doubtful", 0, 10, 0)
# # #                 num_lss = st.number_input("Loss", 0, 10, 0)
# # #             with col2:
# # #                 st.markdown("**Utilization**")
# # #                 pct_active_tls = st.number_input("% Active TLs", 0.0, 1.0, 0.60, 0.01)
# # #                 pct_current_bal = st.number_input("Current Balance %", 0.0, 1.0, 0.30, 0.01)
# # #                 cc_utilization = st.number_input("CC Utilization", 0.0, 1.0, 0.35, 0.01)
# # #                 pl_utilization = st.number_input("PL Utilization", 0.0, 1.0, 0.25, 0.01)
# # #                 max_unsec_exposure = st.number_input("Max Unsec Exposure %", 0, 100, 30)
# # #             with col3:
# # #                 st.markdown("**Demographics & Products**")
# # #                 age_cibil = st.number_input("Age", 18, 70, int(stage1_customer.get('age', 35)))
# # #                 net_monthly_income = st.number_input("Net Monthly Income", 0, 1000000, int(stage1_customer.get('avg_salary_6m', 50000)), 5000)
# # #                 time_curr_employer = st.number_input("Employment Tenure (months)", 0, 600, int(stage1_customer.get('employment_tenure_months', 24)))
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
# # #                     st.warning(f'⚠️ CIBIL income ₹{_s2_inc:,} much lower than application income ₹{_s1_inc:,}. Using application income.')
# # #                 enhanced_customer_data.update({
# # #                     'bureau_score': cibil_score, 'age': age_cibil,
# # #                     'avg_salary_6m': _final_income, 'employment_tenure_months': time_curr_employer,
# # #                     'dpd_30_count_6m': num_times_30dpd, 'dpd_90_count_6m': num_times_60dpd,
# # #                     'max_delinquency_level': max_delinquency, 'num_times_delinquent': num_times_delinquent,
# # #                     'num_deliq_6mts': num_deliq_6m, 'num_deliq_12mts': num_deliq_12m,
# # #                     'max_deliq_6mts': max_deliq_6m, 'max_deliq_12mts': max_deliq_12m,
# # #                     'recent_inquiries_3m': enq_L3m, 'enq_L6m': enq_L6m, 'enq_L12m': enq_L12m,
# # #                     'active_loans_count': num_std, 'num_std_6mts': num_std_6m, 'num_std_12mts': num_std_12m,
# # #                     'num_sub': num_sub, 'num_sub_6mts': num_sub_6m,
# # #                     'num_dbt': num_dbt, 'num_lss': num_lss,
# # #                     'credit_utilization_pct': cc_utilization * 100,
# # #                     'pct_of_active_TLs_ever': pct_active_tls, 'pct_currentBal_all_TL': pct_current_bal,
# # #                     'CC_utilization': cc_utilization, 'PL_utilization': pl_utilization,
# # #                     'max_unsec_exposure_inPct': max_unsec_exposure,
# # #                     'CC_Flag': 1 if cc_flag else 0, 'PL_Flag': 1 if pl_flag else 0,
# # #                     'HL_Flag': 1 if hl_flag else 0, 'GL_Flag': 1 if gl_flag else 0,
# # #                     'GENDER': gender_s2, 'MARITALSTATUS': marital_status, 'EDUCATION': education,
# # #                 })
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
# # #             st.warning("Please use the **Manual Entry** tab.")
# # #         else:
# # #             uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="stage2_pdf")
# # #             if uploaded_pdf is not None:
# # #                 st.success(f"✅ File uploaded: {uploaded_pdf.name} ({uploaded_pdf.size/1024:.1f} KB)")
# # #                 if st.button("🔬 Extract & Analyze", key="extract_analyze_stage2", type="primary", use_container_width=True):
# # #                     with st.spinner("🔄 Extracting data from PDF..."):
# # #                         extraction_result = extract_cibil_from_pdf(uploaded_pdf)
# # #                     if extraction_result.get('success', False):
# # #                         st.success("✅ PDF extraction successful!")

# # #                         # ── Summary metrics ──────────────────────────────────
# # #                         c1, c2, c3, c4 = st.columns(4)
# # #                         c1.metric("Credit Score",    extraction_result.get('Credit_Score', 'N/A'))
# # #                         c2.metric("DPD 30+ Count",   extraction_result.get('num_times_30p_dpd', 0))
# # #                         c3.metric("DPD 60+ Count",   extraction_result.get('num_times_60p_dpd', 0))
# # #                         c4.metric("Active Accounts", extraction_result.get('num_std', 0))
# # #                         c1, c2, c3, c4 = st.columns(4)
# # #                         c1.metric("Monthly Income", f"₹{extraction_result.get('NETMONTHLYINCOME', 0):,}")
# # #                         c2.metric("Employment Tenure", f"{extraction_result.get('Time_With_Curr_Empr',0)} mo")
# # #                         c3.metric("Written Off",    extraction_result.get('num_lss', 0))
# # #                         c4.metric("Enquiries (3M)", extraction_result.get('enq_L3m', 0))
# # #                         c1, c2, c3, c4 = st.columns(4)
# # #                         c1.metric("Payment Discipline", extraction_result.get('payment_discipline_flag','—'))
# # #                         c2.metric("Cashflow Health",    extraction_result.get('cashflow_health','—'))
# # #                         c3.metric("Bureau Risk",        extraction_result.get('bureau_risk_flag','—'))
# # #                         c4.metric("Salary Stability",   extraction_result.get('salary_stability_flag','—'))

# # #                         if extraction_result.get('written_off_count', 0) > 0:
# # #                             st.warning(f"⚠️ {extraction_result['written_off_count']} written-off accounts detected — score may be overridden.")

# # #                         _surplus_proxy = extraction_result.get('_surplus_proxy', 0)
# # #                         if _surplus_proxy:
# # #                             st.info(f"💡 Bureau-only PDF — net surplus estimated from income: ₹{_surplus_proxy:,}")

# # #                         with st.expander("📋 View all extracted fields"):
# # #                             _display = {k: v for k, v in extraction_result.items() if k not in ('raw_text','success','extraction_method')}
# # #                             st.json(_display)

# # #                         # ── Build enhanced_customer_data ─────────────────────
# # #                         # Start from Stage 1 customer (has gender, city_tier, rbi_consent, loan details)
# # #                         enhanced_customer_data = stage1_customer.copy()

# # #                         # Apply ALL extracted fields directly — the new extractor maps every column
# # #                         _skip = {'raw_text', 'success', 'extraction_method',
# # #                                  'loan_amount', 'loan_tenure_months', 'interest_rate',
# # #                                  'rbi_consent', 'kyc_verified', 'bankruptcy_flag', 'fraud_flag'}
# # #                         for k, v in extraction_result.items():
# # #                             if k not in _skip and v is not None:
# # #                                 enhanced_customer_data[k] = v

# # #                         # Income safety: if CIBIL income << Stage 1 application income, keep Stage 1
# # #                         _s1_inc = stage1_customer.get('avg_salary_6m', 50000)
# # #                         _s2_inc = extraction_result.get('NETMONTHLYINCOME', 0) or 0
# # #                         if 0 < _s2_inc < _s1_inc * 0.4:
# # #                             enhanced_customer_data['avg_salary_6m'] = _s1_inc
# # #                             enhanced_customer_data['AMT_INCOME_TOTAL'] = _s1_inc * 12
# # #                             st.warning(f"⚠️ CIBIL income ₹{_s2_inc:,} << application income ₹{_s1_inc:,} — using application income for FOIR.")

# # #                         # Sentinel cleanup
# # #                         enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)

# # #                         with st.spinner("🔬 Running Stage 2 analysis..."):
# # #                             try:
# # #                                 stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
# # #                                 stage2_result = resolve_stage2_to_binary(stage2_result)
# # #                                 display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
# # #                             except Exception as e:
# # #                                 st.error(f"❌ Analysis failed: {str(e)}")
# # #                                 st.exception(e)
# # #                     else:
# # #                         st.error("❌ PDF extraction failed: " + extraction_result.get('error', 'Unknown'))

# # #     elif selected_tab == "Batch Analysis":
# # #         st.info("📊 Stage 2 Batch analysis coming soon.")

# # # elif page == "⚖️ Fairness":
# # #     render_fairness_dashboard()

# # # elif page == "📊 Batch Process":
# # #     st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
# # #     st.markdown('<div class="info-box">📤 Upload a CSV file with customer data for bulk credit assessment.</div>', unsafe_allow_html=True)
# # #     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# # #     if uploaded_file is not None:
# # #         try:
# # #             df = pd.read_csv(uploaded_file)
# # #             st.success(f"✅ Successfully loaded {len(df)} records")
# # #             with st.expander("📄 Preview Uploaded Data"):
# # #                 st.dataframe(df.head(), use_container_width=True)
# # #             required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
# # #             missing_cols = [col for col in required_cols if col not in df.columns]
# # #             if missing_cols:
# # #                 st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
# # #             else:
# # #                 if st.button("🚀 Process Batch Predictions", key="process_batch_btn", type="primary", use_container_width=True):
# # #                     with st.spinner(f"🔍 Processing {len(df)} records..."):
# # #                         results_df = process_batch_predictions(df)
# # #                     st.success(f"✅ Completed {len(results_df)} records!")
# # #                     tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
# # #                     with tab1:
# # #                         st.dataframe(results_df, use_container_width=True)
# # #                         c1, c2, c3, c4 = st.columns(4)
# # #                         with c1: st.metric("✅ Approved", len(results_df[results_df['decision'] == 'APPROVE']))
# # #                         with c2: st.metric("❌ Rejected", len(results_df[results_df['decision'] == 'REJECT']))
# # #                         with c3: st.metric("⚠️ Review", len(results_df[results_df['decision'] == 'REVIEW']))
# # #                         with c4: st.metric("📊 Avg Risk Score", f"{results_df['risk_score'].mean():.0f}")
# # #                     with tab2:
# # #                         col1, col2 = st.columns(2)
# # #                         with col1:
# # #                             dc = results_df['decision'].value_counts()
# # #                             fig1 = px.pie(values=dc.values, names=dc.index, title="Decision Distribution",
# # #                                           color=dc.index, color_discrete_map={'APPROVE':'#48bb78','REVIEW':'#ed8936','REJECT':'#f56565'})
# # #                             st.plotly_chart(fig1, use_container_width=True)
# # #                         with col2:
# # #                             fig2 = px.histogram(results_df, x='risk_score', title="Risk Score Distribution",
# # #                                                 nbins=20, color_discrete_sequence=['#587042'])
# # #                             st.plotly_chart(fig2, use_container_width=True)
# # #                         # Fairness charts from batch
# # #                         if 'gender' in results_df.columns and results_df['gender'].nunique() > 1:
# # #                             results_df['approved_num'] = (results_df['decision'] == 'APPROVE').astype(int)
# # #                             grp = results_df.groupby('gender')['approved_num'].mean().reset_index()
# # #                             grp['Approval Rate %'] = (grp['approved_num'] * 100).round(1)
# # #                             fig3 = px.bar(grp, x='gender', y='Approval Rate %', title='Approval Rate by Gender (Batch)',
# # #                                           color='Approval Rate %', color_continuous_scale=['#f56565','#48bb78'], range_color=[0,100])
# # #                             st.plotly_chart(fig3, use_container_width=True)
# # #                         if 'city_tier' in results_df.columns and results_df['city_tier'].nunique() > 1:
# # #                             results_df['approved_num'] = (results_df['decision'] == 'APPROVE').astype(int)
# # #                             grp2 = results_df.groupby('city_tier')['approved_num'].mean().reset_index()
# # #                             grp2['Approval Rate %'] = (grp2['approved_num'] * 100).round(1)
# # #                             fig4 = px.bar(grp2, x='city_tier', y='Approval Rate %', title='Approval Rate by City Tier (Batch)',
# # #                                           color='Approval Rate %', color_continuous_scale=['#f56565','#48bb78'], range_color=[0,100])
# # #                             st.plotly_chart(fig4, use_container_width=True)
# # #                     with tab3:
# # #                         col1, col2 = st.columns(2)
# # #                         with col1:
# # #                             st.download_button("📥 Download as CSV", data=results_df.to_csv(index=False),
# # #                                                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# # #                                                mime="text/csv", use_container_width=True)
# # #                         with col2:
# # #                             st.download_button("📥 Download as JSON", data=results_df.to_json(orient='records', indent=2),
# # #                                                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
# # #                                                mime="application/json", use_container_width=True)
# # #         except Exception as e:
# # #             st.error(f"❌ Error processing file: {str(e)}")
# # #     else:
# # #         st.markdown("---")
# # #         st.markdown("### 📋 CSV Template")
# # #         template_data = {
# # #             'age': [35, 42, 28], 'gender': ['Male', 'Female', 'Male'],
# # #             'city_tier': ['Tier 1', 'Tier 2', 'Tier 3'],
# # #             'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
# # #             'dependents': [2, 3, 6], 'kyc_verified': ['Yes', 'Yes', 'No'],
# # #             'bankruptcy_flag': ['No', 'No', 'No'], 'fraud_flag': ['No', 'No', 'No'],
# # #             'rbi_consent': ['Yes', 'Yes', 'Yes'],
# # #             'employment_tenure_months': [24, 0, 18], 'business_vintage_years': [0, 5, 0],
# # #             'bureau_score': [720, 680, 580], 'dpd_90_count_6m': [0, 1, 2],
# # #             'dpd_30_count_6m': [0, 2, 1], 'credit_utilization_pct': [30, 45, 75],
# # #             'recent_inquiries_3m': [2, 1, 5], 'active_loans_count': [1, 2, 3],
# # #             'avg_salary_6m': [50000, 75000, 35000], 'AMT_INCOME_TOTAL': [600000, 900000, 420000],
# # #             'net_cash_surplus_6m': [20000, 35000, 10000],
# # #             'salary_stability_flag': ['STABLE', 'MODERATE', 'UNSTABLE'],
# # #             'loan_amount': [180000, 250000, 100000], 'loan_tenure_months': [24, 36, 12],
# # #             'interest_rate': [10.5, 11.0, 12.0], 'existing_emi': [15000, 20000, 8000],
# # #             'AMT_ANNUITY': [8500, 9500, 4500],
# # #             'payment_discipline_flag': ['GOOD', 'MODERATE', 'POOR'],
# # #             'liquidity_flag': ['LOW', 'ADEQUATE', 'LOW'],
# # #             'cashflow_health': ['HEALTHY', 'MODERATE', 'STRESSED'],
# # #             'bureau_risk_flag': ['LOW', 'MEDIUM', 'HIGH'],
# # #             'inward_bounce_count_3m': [0, 1, 3], 'salary_missing_months': [0, 0, 2],
# # #         }
# # #         template_df = pd.DataFrame(template_data)
# # #         st.dataframe(template_df, use_container_width=True)
# # #         st.caption("📝 New columns: `gender`, `city_tier`, `rbi_consent` — required for fairness monitoring and compliance.")
# # #         st.download_button("📥 Download CSV Template", data=template_df.to_csv(index=False),
# # #                            file_name="credit_assessment_template_v8.6.csv",
# # #                            mime="text/csv", use_container_width=True)

# # # elif page == "📈 Model Info":
# # #     st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
# # #     col1, col2, col3 = st.columns(3)
# # #     with col1: st.markdown(f'<div class="stat-card"><div class="stat-number">RF</div><div class="stat-label">Model Type</div></div>', unsafe_allow_html=True)
# # #     with col2: st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TOP_FEATURES)}</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)
# # #     with col3: st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TARGET_LE.classes_)}</div><div class="stat-label">Classes</div></div>', unsafe_allow_html=True)
# # #     st.markdown("<br>", unsafe_allow_html=True)
# # #     st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
# # #     feature_df = pd.DataFrame({'Rank': range(1, min(21, len(TOP_FEATURES)+1)), 'Feature': TOP_FEATURES[:20]})
# # #     st.dataframe(feature_df, use_container_width=True, hide_index=True)

# # # elif page == "ℹ️ About":
# # #     st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
# # #     st.markdown("""
# # #         <div class="info-card">
# # #             <div class="info-card-title">🏦 Credit Risk Assessment Platform</div>
# # #             <div class="info-card-content">
# # #                 <p><strong>Version:</strong> 8.6 — Cleaned codebase + Fairness Monitoring + City Tier + RBI Consent</p>
# # #                 <p><strong>Developer:</strong> Zen Meraki</p>
# # #                 <p><strong>Date:</strong> January 2026</p>
# # #                 <br>
# # #                 <p>A comprehensive credit risk evaluation system combining hard policy rules,
# # #                 machine learning, and affordability analysis for accurate and RBI-compliant lending decisions.</p>
# # #             </div>
# # #         </div>
# # #     """, unsafe_allow_html=True)
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         st.markdown("""
# # #             <div class="info-card">
# # #                 <div class="info-card-title">🎯 Key Features</div>
# # #                 <div class="info-card-content"><ul style="margin:0;padding-left:1.25rem;">
# # #                     <li>Three-layer decision engine</li>
# # #                     <li>Real-time risk assessment</li>
# # #                     <li>Industry-standard PD calculation</li>
# # #                     <li>FOIR calculation & validation</li>
# # #                     <li>Automated reason generation</li>
# # #                     <li>Complete audit trail (PDF)</li>
# # #                     <li>OCR auto-fill with categorical inference</li>
# # #                     <li>⚖️ Fairness monitoring dashboard</li>
# # #                     <li>🏙️ City Tier field for geographic equity</li>
# # #                     <li>📜 RBI consent gate (DLG 2022)</li>
# # #                 </ul></div>
# # #             </div>
# # #         """, unsafe_allow_html=True)
# # #     with col2:
# # #         st.markdown("""
# # #             <div class="info-card">
# # #                 <div class="info-card-title">🛠️ Technology Stack</div>
# # #                 <div class="info-card-content"><ul style="margin:0;padding-left:1.25rem;">
# # #                     <li>Streamlit (UI Framework)</li>
# # #                     <li>Scikit-learn (ML)</li>
# # #                     <li>Plotly (Visualizations)</li>
# # #                     <li>Pandas (Data Processing)</li>
# # #                     <li>ReportLab (PDF Generation)</li>
# # #                     <li>Tesseract OCR + pdf2image</li>
# # #                     <li>Python 3.8+</li>
# # #                 </ul></div>
# # #             </div>
# # #         """, unsafe_allow_html=True)





# # """
# # Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
# # Enhanced with Modern UI/UX Design
# # Run with: streamlit run app.py (from inside the notebooks folder)
# # Author: Zen Meraki
# # Date: March 2026
# # VERSION: 8.7 - Renamed from test.py, dead code removed, all audit fixes applied (C1/H1/H2/M1/M2/M3/L1/L2/L3)
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
# # import base64
# # from typing import List, Any
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
# #     # FIX A-2: CURRENT_DIR is the notebooks/ folder where stage2_engine.py lives.
# #     # It was already present but listed alongside PROJECT_ROOT without emphasis.
# #     # Adding it first and also adding CURRENT_DIR / "utils" ensures both
# #     # stage2_engine.py and utils/pdf_generator.py are importable on Streamlit Cloud
# #     # regardless of the working directory at launch time.
# #     CURRENT_DIR,                          # notebooks/  ← stage2_engine.py lives here
# #     CURRENT_DIR / "utils",               # notebooks/utils/  (if utils is nested)
# #     PROJECT_ROOT,
# #     PROJECT_ROOT / "loan",
# #     PROJECT_ROOT / "utils",              # credit_risk_engine/utils/  ← pdf_generator etc.
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
# #         .stat-card { background: white; padding: 1rem; border-radius: 0.5rem;
# #                      box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
# #         .stat-number { font-size: 1.8rem; font-weight: bold; color: #2d3748; }
# #         .stat-label { font-size: 0.875rem; color: #718096; }
# #         .info-card { background: white; border-radius: 0.5rem; padding: 1rem;
# #                      margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
# #         .info-card-title { font-weight: 600; margin-bottom: 0.5rem; color: #2d3748; }
# #         .info-card-content { font-size: 0.875rem; }
# #         .data-row { display: flex; justify-content: space-between;
# #                     padding: 0.25rem 0; border-bottom: 1px solid #e2e8f0; }
# #         .data-label { color: #4a5568; }
# #         .data-value { font-weight: 500; }
# #         .status-badge { padding: 0.25rem 0.5rem; border-radius: 0.25rem;
# #                         font-size: 0.75rem; margin-left: 0.5rem; }
# #         .badge-pass { background: #c6f6d5; color: #22543d; }
# #         .badge-fail { background: #fed7d7; color: #742a2a; }
# #         .badge-warning { background: #feebc8; color: #744210; }
# #         .reason-item { padding: 0.25rem 0; }
# #         .reason-icon { color: #587042; font-weight: bold; margin-right: 0.5rem; }
# #     </style>
# #     """
# # st.markdown(CSS, unsafe_allow_html=True)

# # # =============================================================================
# # # CITY TIER MAPPING
# # # =============================================================================
# # CITY_TIERS = {
# #     "Tier 1 – Metro (Mumbai, Delhi, Bengaluru, Chennai, Hyderabad, Kolkata, Pune, Ahmedabad)": "Tier 1",
# #     "Tier 2 – Large City (Jaipur, Lucknow, Kochi, Nagpur, Indore, Bhopal, Patna, Vadodara…)": "Tier 2",
# #     "Tier 3 – Small City / Town": "Tier 3",
# #     "Rural / Village": "Rural",
# # }

# # # =============================================================================
# # # SESSION STATE INITIALIZATION
# # # =============================================================================
# # def init_session_state():
# #     defaults = {
# #         'stage1_complete':       False,
# #         'stage1_decision':       None,
# #         'stage1_data':           None,
# #         'current_customer_data': None,
# #         'page_navigation':       "🏠 Home",
# #         'use_two_stage':         False,
# #         'stage2_selected_tab':   "Manual Entry",
# #         # Fairness log — persists across sessions in memory
# #         'fairness_log':          [],
# #     }
# #     for k, v in defaults.items():
# #         if k not in st.session_state:
# #             st.session_state[k] = v

# # init_session_state()

# # # =============================================================================
# # # IMPORT BUSINESS LOGIC MODULES
# # # =============================================================================
# # try:
# #     from affordability_engine import calculate_emi, calculate_affordability
# #     from reason_codes import generate_reason_codes
# #     from risk_engine import (
# #         calculate_final_risk_score, fill_missing_ml_fields,
# #         clean_sentinel_values
# #     )
# #     from affordability_engine import check_net_disposable
# # except ImportError as e:
# #     st.error(f"❌ Failed to import required modules: {e}")
# #     st.info("""
# #     Required files (place in notebooks/, loan/, utils/, or project root):
# #     - affordability_engine.py  |  reason_codes.py  |  risk_engine.py
# #     - utils/__init__.py  |  utils/pdf_generator.py
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
# #     def is_stage2_available(): return False
# #     def get_stage2_status(): return {"error": "Stage 2 engine module not found", "available": False}

# # # =============================================================================
# # # PDF GENERATION – SAFE FALLBACK
# # # FIX A-1: Use explicit try/except import blocks instead of a single-path import.
# # # Tries utils.pdf_generator first (standard install), then bare pdf_generator
# # # (notebooks/ deployment). Sets PDF_AVAILABLE=False and shows a visible warning
# # # in the UI if neither path works, so users know PDF download will be disabled.
# # # =============================================================================
# # PDF_AVAILABLE = False
# # generate_decision_pdf = None
# # generate_audit_pdf = None
# # try:
# #     from utils.pdf_generator import generate_decision_pdf, generate_audit_pdf
# #     PDF_AVAILABLE = True
# # except ImportError:
# #     try:
# #         from pdf_generator import generate_decision_pdf, generate_audit_pdf
# #         PDF_AVAILABLE = True
# #     except ImportError:
# #         PDF_AVAILABLE = False  # UI will show warning — see A-4 note in pdf download buttons

# # # =============================================================================
# # # JSON SANITIZER
# # # =============================================================================
# # def sanitize_for_json(obj: Any) -> Any:
# #     if obj is None or isinstance(obj, (str, int, float, bool)): return obj
# #     if isinstance(obj, set): return list(obj)
# #     if isinstance(obj, datetime): return obj.isoformat()
# #     if isinstance(obj, np.integer): return int(obj)
# #     if isinstance(obj, np.floating): return float(obj)
# #     if isinstance(obj, np.ndarray): return obj.tolist()
# #     if isinstance(obj, dict): return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
# #     if isinstance(obj, (list, tuple)): return [sanitize_for_json(item) for item in obj]
# #     try:
# #         json.dumps(obj); return obj
# #     except (TypeError, ValueError): return str(obj)

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
# #             try: assets = joblib.load(path); break
# #             except FileNotFoundError: continue
# #         if assets is None:
# #             raise FileNotFoundError("Could not find credit_risk_assets.pkl")
# #         return {
# #             'model': assets['model'], 'features': assets['features'],
# #             'le_map': assets['le_map'], 'target_le': assets['target_le'],
# #             'loaded': True, 'error': None
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

# # MODEL      = ASSETS['model']
# # TOP_FEATURES = ASSETS['features']
# # LE_MAP     = ASSETS['le_map']
# # TARGET_LE  = ASSETS['target_le']

# # # =============================================================================
# # # PD CALCULATION FUNCTIONS
# # # NOTE: calculate_emi, calculate_affordability, generate_reason_codes,
# # #       calculate_final_risk_score are imported from their respective modules.
# # #       The PD functions below are NOT in any module so are kept here.
# # # =============================================================================
# # def bureau_score_to_pd(bureau_score):
# #     if bureau_score >= 800: return 0.5 + (900 - bureau_score) / 200 * 0.5
# #     elif bureau_score >= 750: return 1.0 + (800 - bureau_score) / 50 * 1.0
# #     elif bureau_score >= 700: return 2.0 + (750 - bureau_score) / 50 * 1.5
# #     elif bureau_score >= 650: return 3.5 + (700 - bureau_score) / 50 * 2.5
# #     elif bureau_score >= 600: return 6.0 + (650 - bureau_score) / 50 * 4.0
# #     elif bureau_score >= 550: return 10.0 + (600 - bureau_score) / 50 * 5.0
# #     else: return min(25.0, 15.0 + (550 - bureau_score) / 50 * 10.0)

# # def foir_to_pd_adjustment(foir_percentage):
# #     if foir_percentage <= 30: return -0.75
# #     elif foir_percentage <= 40: return 0.00
# #     elif foir_percentage <= 45: return 0.75
# #     elif foir_percentage <= 50: return 1.50
# #     elif foir_percentage <= 55: return 2.25
# #     elif foir_percentage <= 60: return 3.50
# #     else: return 6.00

# # def delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count=0):
# #     if dpd_90_count >= 3: return 5.0
# #     elif dpd_90_count == 2: return 3.0
# #     elif dpd_90_count == 1: return 2.0
# #     elif dpd_30_count >= 3: return 1.6
# #     elif dpd_30_count >= 1: return 1.3
# #     else: return 1.0

# # def employment_stability_to_pd_adjustment(employment_type, tenure_months, business_vintage_years=0):
# #     if employment_type == 'Salaried':
# #         if tenure_months >= 36: return -0.5
# #         elif tenure_months >= 12: return 0.0
# #         elif tenure_months >= 6: return 0.5
# #         else: return 2.0
# #     elif employment_type in ['Self-Employed', 'Business']:
# #         if business_vintage_years >= 5: return -0.5
# #         elif business_vintage_years >= 2: return 0.0
# #         else: return 1.5
# #     else: return 1.0

# # def inquiry_pattern_to_pd_adjustment(recent_inquiries_3m):
# #     if recent_inquiries_3m <= 1: return -0.3
# #     elif recent_inquiries_3m <= 3: return 0.0
# #     elif recent_inquiries_3m <= 5: return 0.8
# #     elif recent_inquiries_3m <= 8: return 1.5
# #     else: return 3.0

# # def ml_confidence_to_pd_adjustment(ml_confidence, ml_decision):
# #     if ml_decision == "APPROVE":
# #         if ml_confidence >= 90: return -0.5
# #         elif ml_confidence >= 70: return 0.0
# #         else: return 0.5
# #     elif ml_decision == "REVIEW": return 1.0
# #     else: return 5.0

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
# #     return round(max(0.5, min(final_pd, 25.0)), 2)

# # # =============================================================================
# # # CATEGORICAL FLAG INFERENCE (v8.5 dual-dataset)
# # # =============================================================================
# # def _infer_surplus_from_cibil(score: int, dpd_60: int, dpd_30: int, income: float) -> float:
# #     if dpd_60 >= 3: return income * -0.5
# #     elif score < 650 or dpd_60 >= 1: return income * -0.2
# #     elif score < 700: return income * 0.1
# #     else: return income * 0.3

# # def infer_categorical_flags(extraction_result: dict) -> dict:
# #     score       = int(extraction_result.get('Credit_Score', 700) or 700)
# #     dpd_30      = int(extraction_result.get('num_times_30p_dpd', 0) or 0)
# #     dpd_60      = int(extraction_result.get('num_times_60p_dpd', 0) or 0)
# #     written_off = int(extraction_result.get('num_lss', 0) or extraction_result.get('written_off_count', 0) or 0)
# #     doubtful    = int(extraction_result.get('num_dbt', 0) or 0)
# #     cc_util_raw = extraction_result.get('CC_utilization', 0) or 0
# #     cc_util     = float(cc_util_raw) if cc_util_raw > 0 else 0.0
# #     income      = float(extraction_result.get('NETMONTHLYINCOME', 0) or
# #                         extraction_result.get('avg_salary_6m', 50_000) or 50_000)
# #     tenure      = int(extraction_result.get('Time_With_Curr_Empr', 24) or 24)

# #     is_bureau_only = (
# #         'NETMONTHLYINCOME' in extraction_result
# #         and 'net_cash_surplus_6m' not in extraction_result
# #         and 'net_surplus' not in extraction_result
# #     )

# #     if is_bureau_only:
# #         dpd_90_proxy = dpd_60
# #         surplus = _infer_surplus_from_cibil(score, dpd_60, dpd_30, income)
# #         payment_discipline = 'POOR' if (dpd_60 >= 1 or dpd_30 >= 3) else ('MODERATE' if dpd_30 >= 1 else 'GOOD')
# #         cashflow_health = ('HEALTHY' if surplus >= 14_000 else 'STABLE' if surplus >= 600 else 'STRESSED' if surplus < -1_000 else 'MODERATE')
# #         liquidity_flag  = ('ADEQUATE' if surplus > 14_000 else 'LOW' if surplus < -32_000 else 'MODERATE')
# #         bureau_risk     = ('HIGH' if (written_off >= 1 or doubtful >= 1 or dpd_60 >= 3 or score < 580)
# #                            else 'MEDIUM' if (score < 650 or (dpd_30 >= 2 and cc_util > 0.60)) else 'LOW')
# #         salary_stability = ('UNSTABLE' if tenure < 6 else 'STABLE' if (tenure >= 24 and score >= 700 and dpd_30 == 0) else 'MODERATE')
# #         surplus_for_return = surplus  # FIX L2: assign in both branches — was missing here, causing latent bug if bureau_only path is extended
# #     else:
# #         dpd_90      = int(extraction_result.get('dpd_90_count_6m', 0) or 0)
# #         bounces     = int(extraction_result.get('inward_bounce_count_3m', 0) or 0)
# #         missing     = int(extraction_result.get('salary_missing_months', 0) or 0)
# #         hard_reject = int(extraction_result.get('hard_reject_flag', 0) or 0)
# #         surplus     = float(extraction_result.get('net_cash_surplus_6m') or extraction_result.get('net_surplus') or -50_000)
# #         payment_discipline = ('POOR' if (dpd_90 >= 1 or bounces >= 2)
# #                                else 'MODERATE' if (bounces == 1 or dpd_30 >= 3) else 'GOOD')
# #         cashflow_health = ('HEALTHY' if surplus >= 14_000 else 'STABLE' if 600 <= surplus < 14_000
# #                             else 'STRESSED' if surplus < -1_000 else 'MODERATE')
# #         liquidity_flag  = 'ADEQUATE' if surplus > 14_000 else 'LOW' if surplus < -32_000 else 'MODERATE'
# #         bureau_risk     = ('HIGH' if (hard_reject or dpd_90 >= 3 or written_off >= 1 or (dpd_90 >= 1 and dpd_30 >= 2))
# #                            else 'MEDIUM' if (score < 580 or (dpd_30 >= 2 and cc_util > 0.60)) else 'LOW')
# #         salary_stability = ('UNSTABLE' if missing >= 1
# #                              else 'STABLE' if (missing == 0 and score >= 700 and dpd_30 == 0 and bounces == 0)
# #                              else 'MODERATE')
# #         surplus_for_return = surplus

# #     return {
# #         'payment_discipline_flag': payment_discipline,
# #         'cashflow_health':         cashflow_health,
# #         'liquidity_flag':          liquidity_flag,
# #         'bureau_risk_flag':        bureau_risk,
# #         'salary_stability_flag':   salary_stability,
# #         '_inference_path':         'bureau_only' if is_bureau_only else 'bank_statement',
# #     }

# # # =============================================================================
# # # CIBIL PDF EXTRACTION ENGINE (OCR + PATTERN MATCHING)
# # # =============================================================================
# # def _re_int(pattern, text, default, lo=None, hi=None):
# #     """Safe regex → int extraction with optional range clamp."""
# #     m = re.search(pattern, text, re.IGNORECASE)
# #     if m:
# #         try:
# #             v = int(str(m.group(1)).replace(',', '').replace(' ', ''))
# #             if lo is not None and v < lo: return default
# #             if hi is not None and v > hi: return default
# #             return v
# #         except Exception: pass
# #     return default

# # def _re_float(pattern, text, default, lo=None, hi=None):
# #     """Safe regex → float extraction with optional range clamp."""
# #     m = re.search(pattern, text, re.IGNORECASE)
# #     if m:
# #         try:
# #             v = float(str(m.group(1)).replace(',', '').replace(' ', ''))
# #             if lo is not None and v < lo: return default
# #             if hi is not None and v > hi: return default
# #             return v
# #         except Exception: pass
# #     return default

# # def extract_cibil_from_pdf(uploaded_file):
# #     if not OCR_AVAILABLE:
# #         return {'success': False, 'error': OCR_ERROR_MSG or 'OCR libraries not installed.'}
# #     try:
# #         # ── 1. OCR: PDF → full text ──────────────────────────────────────────
# #         pdf_bytes = uploaded_file.read()
# #         images    = convert_from_bytes(pdf_bytes, dpi=300)
# #         full_text = ""
# #         for image in images:
# #             gray        = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
# #             _, binary   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# #             full_text  += pytesseract.image_to_string(binary) + "\n"
# #         txt = full_text   # shorthand

# #         # ── 2. CREDIT SCORE (Bureau / CIBIL score) ───────────────────────────
# #         credit_score = 720
# #         for pat in [
# #             r'\b(8[0-9]{2}|7[0-9]{2}|6[0-9]{2}|[3-5][0-9]{2})\s*(?:EXCELLENT|VERY\s*GOOD|GOOD|FAIR|SUBPRIME|POOR|NH|NA)\b',
# #             r'(?:cibil|credit|bureau)\s*score\s*[:\-\(]?\s*(\d{3})',
# #             r'score[^\n\r]{0,40}?(\d{3})',
# #         ]:
# #             m = re.search(pat, txt, re.IGNORECASE)
# #             if m:
# #                 v = int(m.group(1))
# #                 if 300 <= v <= 900:
# #                     credit_score = v; break

# #         # ── 3. PERSONAL INFO ────────────────────────────────────────────────
# #         # Age via DOB
# #         age_extracted = 35
# #         for dob_pat in [
# #             r'(?:date\s+of\s+birth|dob|d\.o\.b)[\s:\-]+(\d{2}[-/]\w{3,9}[-/]\d{2,4})',
# #             r'(?:date\s+of\s+birth|dob)[\s:\-]+(\d{2}[-/]\d{2}[-/]\d{4})',
# #             r'born[\s:]+(\d{2}[-/]\w{3,9}[-/]\d{4})',
# #         ]:
# #             m = re.search(dob_pat, txt, re.IGNORECASE)
# #             if m:
# #                 for fmt in ('%d-%b-%Y','%d/%b/%Y','%d-%b-%y','%d-%m-%Y','%d/%m/%Y'):
# #                     try:
# #                         dob = datetime.strptime(m.group(1), fmt)
# #                         age_extracted = int((datetime.now() - dob).days / 365.25)
# #                         break
# #                     except Exception: continue
# #                 if age_extracted != 35: break
# #         # fallback: age stated directly
# #         if age_extracted == 35:
# #             age_extracted = _re_int(r'(?:^|\s)age[\s:\-]+(\d{2})\b', txt, 35, lo=18, hi=75)

# #         # Gender
# #         if re.search(r'\bfemale\b|\bF\b', txt, re.IGNORECASE):
# #             gender = 'F'
# #         elif re.search(r'\bmale\b|\bM\b', txt, re.IGNORECASE):
# #             gender = 'M'
# #         else:
# #             gender = 'M'

# #         # Marital status
# #         if re.search(r'\bsingle\b|\bunmarried\b', txt, re.IGNORECASE):
# #             marital_status = 'Single'
# #         else:
# #             marital_status = 'Married'

# #         # Education
# #         education = 'GRADUATE'
# #         for pat, val in [
# #             (r'post.?grad(uate)?|m\.?tech|mba|mca',    'POST-GRADUATE'),
# #             (r'professional|ca\b|cs\b|icai',             'PROFESSIONAL'),
# #             (r'\b12th\b|\bhsc\b|\binter(mediate)?\b',   '12TH'),
# #             (r'\bssc\b|\b10th\b|\bmatric',               'SSC'),
# #             (r'under.?grad(uate)?',                      'UNDER GRADUATE'),
# #             (r'\bgrad(uate)?\b|\bb\.?tech\b|\bb\.?e\b|\bb\.?sc\b|\bb\.?com\b', 'GRADUATE'),
# #         ]:
# #             if re.search(pat, txt, re.IGNORECASE): education = val; break

# #         # ── 4. INCOME & EMPLOYMENT ──────────────────────────────────────────
# #         monthly_income = 50000
# #         for inc_pat in [
# #             r'net\s+monthly\s+income[\s:\-₹Rs\.]*([0-9,]+)',
# #             r'monthly\s+(?:take.?home|salary|income)[\s:\-₹Rs\.]*([0-9,]+)',
# #             r'(?:salary|income)\s+per\s+month[\s:\-₹Rs\.]*([0-9,]+)',
# #             r'₹\s*([0-9,]+)\s+(?:per\s+month|p\.?m\.?|monthly)',
# #         ]:
# #             m = re.search(inc_pat, txt, re.IGNORECASE)
# #             if m:
# #                 v = int(m.group(1).replace(',',''))
# #                 if 5000 < v < 5_000_000:
# #                     monthly_income = v; break

# #         # Employment type
# #         employment_type = 'Salaried'
# #         if re.search(r'self.?employed|self employ|proprietor|freelance', txt, re.IGNORECASE):
# #             employment_type = 'Self-Employed'
# #         elif re.search(r'\bbusiness\b|\bfirm\b|\bpartner(ship)?\b', txt, re.IGNORECASE):
# #             employment_type = 'Business'

# #         # Employment tenure (months)
# #         employment_tenure_months = 36
# #         m = re.search(r'(?:with\s+current\s+employer|employment\s+tenure|employed\s+(?:since|for))[^\d]{0,20}(\d+)\s*(?:year|yr)', txt, re.IGNORECASE)
# #         if m:
# #             employment_tenure_months = int(m.group(1)) * 12
# #         else:
# #             m = re.search(r'(?:with\s+current\s+employer|tenure)[^\d]{0,20}(\d+)\s*(?:month|mth)', txt, re.IGNORECASE)
# #             if m: employment_tenure_months = int(m.group(1))

# #         # Existing EMI
# #         existing_emi = 0
# #         for emi_pat in [
# #             r'(?:total\s+emi|existing\s+emi|current\s+emi|monthly\s+emi)[^\d₹]{0,20}[₹Rs\.]*\s*([0-9,]+)',
# #             r'emi\s+(?:outflow|obligation)[^\d]{0,20}([0-9,]+)',
# #             r'amt_annuity[\s:\-]+([0-9,]+)',
# #         ]:
# #             m = re.search(emi_pat, txt, re.IGNORECASE)
# #             if m:
# #                 v = int(m.group(1).replace(',',''))
# #                 if 500 < v < 500_000:
# #                     existing_emi = v; break

# #         # Business vintage
# #         business_vintage = 0
# #         m = re.search(r'(?:business\s+(?:since|established|vintage|age|started))[^\d]{0,20}(\d+)\s*(?:year|yr)', txt, re.IGNORECASE)
# #         if m: business_vintage = int(m.group(1))

# #         # ── 5. CREDIT UTILISATION ───────────────────────────────────────────
# #         cc_util_pct = -99999   # -99999 = no CC (like CIBIL dataset convention)
# #         m = re.search(r'(?:credit\s+card\s+utiliz[ao]tion|cc\s+utiliz[ao]tion|utiliz[ao]tion\s+ratio)[^\d]{0,20}(\d{1,3})\s*%?', txt, re.IGNORECASE)
# #         if m:
# #             cc_util_pct = int(m.group(1))
# #         pl_util = _re_float(r'(?:personal\s+loan\s+utiliz[ao]tion|pl\s+utiliz[ao]tion)[^\d]{0,20}([\d\.]+)', txt, 0.25, lo=0, hi=5)

# #         # ── 6. ENQUIRIES ─────────────────────────────────────────────────────
# #         # Parse enquiry section for product-wise breakdown
# #         enq_section = ""
# #         m = re.search(r'enquir(?:y|ies)\s+details(.*?)(?:account\s+summary|$)', txt, re.IGNORECASE | re.DOTALL)
# #         if m: enq_section = m.group(1)

# #         tot_enq    = _re_int(r'total\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, 0)
# #         enq_L12m   = _re_int(r'enquir(?:y|ies)\s*(?:\(?12\s*(?:m(?:on)?(?:th)?s?|M)\)?)?[\s:\-]+(\d+)', txt, 0)
# #         enq_L6m    = _re_int(r'enquir(?:y|ies)\s*\(?6\s*(?:m(?:on)?(?:th)?s?|M)\)?[\s:\-]+(\d+)', txt, 0)
# #         enq_L3m    = _re_int(r'enquir(?:y|ies)\s*\(?3\s*(?:m(?:on)?(?:th)?s?|M)\)?[\s:\-]+(\d+)', txt, 0)

# #         # Count enquiry dates in section as fallback
# #         enq_dates = re.findall(r'\b\d{2}-[A-Za-z]{3}-\d{4}\b', enq_section)
# #         tot_enq  = max(tot_enq, len(enq_dates))
# #         enq_L12m = max(enq_L12m, len(enq_dates))

# #         # Product-wise enquiries (CC / PL)
# #         CC_enq     = _re_int(r'credit\s+card\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, -99999)
# #         CC_enq_L6m = _re_int(r'cc\s+enq(?:uiry|uiries)?\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0 if CC_enq >= 0 else -99999)
# #         CC_enq_L12m= _re_int(r'cc\s+enq(?:uiry|uiries)?\s*\(?12m\)?[\s:\-]+(\d+)', txt, 0 if CC_enq >= 0 else -99999)
# #         PL_enq     = _re_int(r'personal\s+loan\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, -99999)
# #         PL_enq_L6m = _re_int(r'pl\s+enq(?:uiry|uiries)?\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0 if PL_enq >= 0 else -99999)
# #         PL_enq_L12m= _re_int(r'pl\s+enq(?:uiry|uiries)?\s*\(?12m\)?[\s:\-]+(\d+)', txt, 0 if PL_enq >= 0 else -99999)

# #         # Time since most recent enquiry (days)
# #         time_since_recent_enq = _re_int(r'(?:last|recent)\s+enquiry[\s:\-]+(\d+)\s*days?', txt, -99999)
# #         if time_since_recent_enq == -99999 and enq_dates:
# #             try:
# #                 most_recent = max(datetime.strptime(d, '%d-%b-%Y') for d in enq_dates)
# #                 time_since_recent_enq = (datetime.now() - most_recent).days
# #             except Exception: pass

# #         # ── 7. ACCOUNT / DPD PARSING ─────────────────────────────────────────
# #         accounts, dpd_all = [], []
# #         in_accounts = False
# #         for line in txt.split('\n'):
# #             lu = line.upper()
# #             if 'ACCOUNT DETAILS' in lu or 'LOAN DETAILS' in lu:
# #                 in_accounts = True; continue
# #             if re.search(r'ENQUIRY\s+DETAILS|SUMMARY|PERSONAL\s+INFO', lu):
# #                 in_accounts = False; continue
# #             if not in_accounts: continue
# #             stripped = line.strip()
# #             if not stripped: continue
# #             stat_m = re.search(r'\b(Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss)\b', stripped, re.IGNORECASE)
# #             dpd_m  = re.search(r'\b(0\d0|0\d\d|\d{3})\b', stripped)
# #             if re.search(r'\bINR\b|\bBank\b|\bFinance\b|\bCapital\b|\bNBFC\b', stripped, re.IGNORECASE) or stat_m:
# #                 dpd_val = int(dpd_m.group(1)) if dpd_m else 0
# #                 status  = (stat_m.group(1) if stat_m else 'Active').lower()
# #                 accounts.append({'dpd': dpd_val, 'status': status})
# #                 dpd_all.append(dpd_val)

# #         # Aggregate DPD counts
# #         dpd_90_count = dpd_60_count = dpd_30_count = 0
# #         written_off_count = settled_count = active_count = sub_std = 0
# #         if accounts:
# #             for acc in accounts:
# #                 d, s = acc['dpd'], acc['status']
# #                 if d >= 90: dpd_90_count += 1
# #                 elif d >= 60: dpd_60_count += 1
# #                 elif d >= 30: dpd_30_count += 1
# #                 if 'written' in s:  written_off_count += 1
# #                 elif 'settled' in s: settled_count += 1
# #                 elif 'active'  in s: active_count += 1
# #                 if d >= 30: sub_std += 1
# #         else:
# #             # Fallback: keyword scan
# #             written_off_count = len(re.findall(r'\bwritten[-\s]?off\b',       txt, re.IGNORECASE))
# #             settled_count     = len(re.findall(r'\bsettled\b',                txt, re.IGNORECASE))
# #             dpd_90_count      = len(re.findall(r'\b090\b|90\+?\s*dpd',        txt, re.IGNORECASE))
# #             dpd_60_count      = len(re.findall(r'\b060\b|60\+?\s*dpd',        txt, re.IGNORECASE))
# #             dpd_30_count      = len(re.findall(r'\b030\b|30\+?\s*dpd',        txt, re.IGNORECASE))
# #             active_count      = len(re.findall(r'\bactive\b',                 txt, re.IGNORECASE))
# #             active_count      = min(active_count, 10)  # cap noise

# #         # Standard (num_std) = active performing accounts
# #         total_accounts = max(len(accounts), active_count + settled_count + written_off_count, 1)
# #         num_std        = active_count
# #         pct_active     = active_count / total_accounts

# #         # Substandard / doubtful / loss (CIBIL classification)
# #         num_sub = sub_std
# #         num_dbt = dpd_90_count
# #         num_lss = written_off_count

# #         # ── 8. DELINQUENCY TIMINGS ───────────────────────────────────────────
# #         # CIBIL PDF usually shows months-ago; we convert to days
# #         # time_since_recent_payment
# #         time_since_recent_payment = _re_int(
# #             r'(?:last|recent)\s+payment[\s:\-]+(\d+)\s*days?', txt, -99999)
# #         if time_since_recent_payment == -99999:
# #             # try "X months ago"
# #             m = re.search(r'(?:last|recent)\s+payment[\s:\-]+(\d+)\s*(?:month|mth)', txt, re.IGNORECASE)
# #             if m: time_since_recent_payment = int(m.group(1)) * 30

# #         time_since_first_deliq  = -99999 if (dpd_30_count + dpd_60_count + dpd_90_count) == 0 else \
# #             _re_int(r'first\s+delinquency[\s:\-]+(\d+)\s*days?', txt, 365)
# #         time_since_recent_deliq = -99999 if (dpd_30_count + dpd_60_count + dpd_90_count) == 0 else \
# #             _re_int(r'(?:last|recent)\s+delinquency[\s:\-]+(\d+)\s*days?', txt, 90)
# #         recent_level_of_deliq   = max(
# #             dpd_90_count * 90, dpd_60_count * 60, dpd_30_count * 30)

# #         # 6-month vs 12-month split
# #         num_deliq_6mts   = dpd_30_count + dpd_60_count + dpd_90_count
# #         num_deliq_12mts  = num_deliq_6mts   # single source; 12m ≥ 6m
# #         num_deliq_6_12mts = 0               # can't distinguish without dates
# #         max_deliq_6mts   = -99999 if num_deliq_6mts  == 0 else recent_level_of_deliq
# #         max_deliq_12mts  = -99999 if num_deliq_12mts == 0 else recent_level_of_deliq

# #         # num_std time splits
# #         num_std_6mts  = min(num_std, _re_int(r'standard\s+accounts?\s*\(?6m\)?[\s:\-]+(\d+)', txt, num_std))
# #         num_std_12mts = _re_int(r'standard\s+accounts?\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_std)
# #         num_sub_6mts  = _re_int(r'sub.?standard\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
# #         num_sub_12mts = _re_int(r'sub.?standard\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_sub)
# #         num_dbt_6mts  = _re_int(r'doubtful\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
# #         num_dbt_12mts = _re_int(r'doubtful\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_dbt)
# #         num_lss_6mts  = _re_int(r'loss\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
# #         num_lss_12mts = _re_int(r'loss\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_lss)
# #         num_times_delinquent = dpd_30_count + dpd_60_count + dpd_90_count
# #         num_times_30p_dpd    = dpd_30_count + dpd_60_count + dpd_90_count
# #         num_times_60p_dpd    = dpd_60_count + dpd_90_count
# #         max_delinquency_level = max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)

# #         # ── 9. TRADE-LINE RATIOS (pct_ fields) ──────────────────────────────
# #         pct_of_active_TLs_ever     = round(pct_active, 3)
# #         pct_opened_TLs_L6m_of_L12m = _re_float(
# #             r'(?:opened|new)\s+accounts?\s*\(?6m\s*/\s*12m\)?[\s:\-]+([\d\.]+)', txt, 0.3, lo=0, hi=1)
# #         pct_currentBal_all_TL      = _re_float(
# #             r'current\s+balance\s+(?:ratio|pct|%)[\s:\-]+([\d\.]+)', txt, 0.3, lo=0, hi=10)
# #         pct_PL_enq_L6m_of_L12m    = round(PL_enq_L6m / max(PL_enq_L12m, 1), 2) if PL_enq_L6m >= 0 else 0
# #         pct_CC_enq_L6m_of_L12m    = round(CC_enq_L6m / max(CC_enq_L12m, 1), 2) if CC_enq_L6m >= 0 else 0
# #         pct_PL_enq_L6m_of_ever    = round(PL_enq_L6m / max(PL_enq if PL_enq >= 0 else 1, 1), 2)
# #         pct_CC_enq_L6m_of_ever    = round(CC_enq_L6m / max(CC_enq if CC_enq >= 0 else 1, 1), 2)

# #         # ── 10. PRODUCT FLAGS ────────────────────────────────────────────────
# #         CC_Flag = 1 if re.search(r'credit\s+card', txt, re.IGNORECASE) else 0
# #         PL_Flag = 1 if re.search(r'personal\s+loan', txt, re.IGNORECASE) else 0
# #         HL_Flag = 1 if re.search(r'home\s+loan|housing\s+loan', txt, re.IGNORECASE) else 0
# #         GL_Flag = 1 if re.search(r'gold\s+loan', txt, re.IGNORECASE) else 0

# #         prod_map = {r'personal\s+loan':'PL', r'credit\s+card':'CC',
# #                     r'home\s+loan|housing':'HL', r'auto\s+loan|car\s+loan':'AL',
# #                     r'gold\s+loan':'GL', r'business\s+loan':'BL'}
# #         last_prod = first_prod = 'others'
# #         for pat, label in prod_map.items():
# #             if re.search(pat, txt, re.IGNORECASE):
# #                 last_prod = first_prod = label; break

# #         # ── 11. SANITY CHECK: high score vs bad history ──────────────────────
# #         if credit_score >= 750 and (written_off_count > 0 or dpd_90_count > 0):
# #             credit_score = min(credit_score, 550)

# #         # ── 12. NET CASH SURPLUS PROXY ───────────────────────────────────────
# #         # Try to extract if stated, else infer
# #         net_cash_surplus = _re_int(
# #             r'(?:net\s+(?:cash\s+)?surplus|disposable\s+income)[^\d₹]{0,20}[₹Rs\.]*\s*([0-9,]+)', txt, 0)
# #         if net_cash_surplus == 0:
# #             net_cash_surplus = int(_infer_surplus_from_cibil(credit_score, dpd_60_count, dpd_30_count, float(monthly_income)))

# #         # ── 13. INWARD BOUNCE & SALARY STABILITY (60k-specific fields) ───────
# #         # These are bank-statement fields; CIBIL PDF won't have them directly.
# #         # We infer them from available signals.
# #         inward_bounce_count_3m  = dpd_90_count + dpd_60_count      # proxy: each severe DPD → bounce
# #         salary_missing_months   = 0                                  # can't determine from CIBIL
# #         total_credit_6m         = _re_int(r'total\s+credits?\s*\(?6m\)?[\s:\-₹]*([0-9,]+)', txt, 0)
# #         total_debit_6m          = _re_int(r'total\s+debits?\s*\(?6m\)?[\s:\-₹]*([0-9,]+)', txt, 0)

# #         # ── 14. STAGE-1 60K DATASET FIELD MAPPING ────────────────────────────
# #         # All columns from train_60k_rule_accepted.csv mapped from OCR data
# #         s1 = {
# #             # Income / salary
# #             'AMT_INCOME_TOTAL':          monthly_income * 12,
# #             'AMT_ANNUITY':               existing_emi if existing_emi > 0 else int(monthly_income * 0.25),
# #             'avg_salary_6m':             float(monthly_income),
# #             'salary_txn_count_6m':       6.0,       # assume regular salary
# #             'salary_amount_cv':          0.05 if employment_type == 'Salaried' else 0.25,
# #             'salary_date_std':           2.0,
# #             'salary_creditor_consistent': 1.0 if employment_type == 'Salaried' else 0.7,
# #             'salary_missing_months':     float(salary_missing_months),
# #             # Delinquency
# #             'dpd_15_count_6m':           0.0,
# #             'dpd_30_count_6m':           float(dpd_30_count),
# #             'dpd_90_count_6m':           float(dpd_90_count),
# #             'max_dpd_6m':                float(max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)),
# #             'dpd_30_count_3m':           float(dpd_30_count),
# #             'total_payments_6m':         0.0,
# #             'total_late_15_6m':          0.0,
# #             'total_late_30_6m':          float(dpd_30_count),
# #             'total_late_60_6m':          float(dpd_60_count),
# #             'total_late_90_6m':          float(dpd_90_count),
# #             'max_days_late_6m':          float(max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)),
# #             'avg_days_late_6m':          float(dpd_30_count * 10 + dpd_60_count * 20 + dpd_90_count * 40) / max(total_accounts, 1),
# #             'total_late_30_3m':          float(dpd_30_count),
# #             'total_late_90_3m':          float(dpd_90_count),
# #             # Credit card
# #             'avg_balance_cc':            0.0,
# #             'total_drawings_cc':         0.0,
# #             'avg_credit_limit':          0.0,
# #             'max_utilization':           (cc_util_pct / 100) if cc_util_pct > 0 else 0.0,
# #             'total_payments_cc':         0.0,
# #             'dpd_count_cc':              0.0,
# #             # POS / installment
# #             'avg_balance_pos':           0.0,
# #             'dpd_count_pos':             0.0,
# #             # Aggregate
# #             'total_credit_activity':     float(total_accounts),
# #             'total_dpd_count':           float(dpd_30_count + dpd_60_count + dpd_90_count),
# #             'avg_monthly_balance_6m':    float(net_cash_surplus),
# #             'total_emi_monthly':         float(existing_emi if existing_emi > 0 else int(monthly_income * 0.25)),
# #             'net_cash_surplus_6m':       float(net_cash_surplus),
# #             'total_credit_6m':           float(total_credit_6m),
# #             'total_debit_6m':            float(total_debit_6m),
# #             # Cashflow
# #             'inward_bounce_count_3m':    float(inward_bounce_count_3m),
# #             'recent_payment_stress':     float(dpd_30_count + dpd_60_count),
# #             # Active loans
# #             'active_loans_count':        float(active_count),
# #             # Bureau
# #             'bureau_score':              float(credit_score),
# #             'hard_reject_flag':          1 if (dpd_90_count > 5 or written_off_count > 0 or credit_score < 550) else 0  # DPD90 1-5 = REVIEW not hard reject,
# #         }

# #         # ── 15. INFERRED CATEGORICAL FLAGS (60k) ─────────────────────────────
# #         _inferred = infer_categorical_flags({
# #             'Credit_Score': credit_score, 'num_times_30p_dpd': dpd_30_count,
# #             'num_times_60p_dpd': dpd_60_count, 'num_lss': num_lss,
# #             'num_dbt': num_dbt, 'CC_utilization': cc_util_pct / 100 if cc_util_pct > 0 else 0,
# #             'NETMONTHLYINCOME': monthly_income, 'Time_With_Curr_Empr': employment_tenure_months,
# #             'dpd_90_count_6m': dpd_90_count, 'inward_bounce_count_3m': inward_bounce_count_3m,
# #             'salary_missing_months': salary_missing_months,
# #             'net_cash_surplus_6m': net_cash_surplus,
# #         })

# #         # ── 16. STAGE-2 EXTERNAL CIBIL DATASET FIELD MAPPING ─────────────────
# #         # All 62 columns from External_Cibil_Dataset.xlsx
# #         s2 = {
# #             'Credit_Score':               credit_score,
# #             'AGE':                        age_extracted,
# #             'GENDER':                     gender,
# #             'MARITALSTATUS':              marital_status,
# #             'EDUCATION':                  education,
# #             'NETMONTHLYINCOME':           monthly_income,
# #             'Time_With_Curr_Empr':        employment_tenure_months,
# #             # Delinquency counts
# #             'num_times_delinquent':       num_times_delinquent,
# #             'max_delinquency_level':      max_delinquency_level,
# #             'max_recent_level_of_deliq':  max(dpd_60_count*60, dpd_30_count*30),
# #             'num_deliq_6mts':             num_deliq_6mts,
# #             'num_deliq_12mts':            num_deliq_12mts,
# #             'num_deliq_6_12mts':          num_deliq_6_12mts,
# #             'max_deliq_6mts':             max_deliq_6mts,
# #             'max_deliq_12mts':            max_deliq_12mts,
# #             'num_times_30p_dpd':          num_times_30p_dpd,
# #             'num_times_60p_dpd':          num_times_60p_dpd,
# #             'recent_level_of_deliq':      recent_level_of_deliq,
# #             # Standard / substandard / doubtful / loss
# #             'num_std':                    num_std,
# #             'num_std_6mts':               num_std_6mts,
# #             'num_std_12mts':              num_std_12mts,
# #             'num_sub':                    num_sub,
# #             'num_sub_6mts':               num_sub_6mts,
# #             'num_sub_12mts':              num_sub_12mts,
# #             'num_dbt':                    num_dbt,
# #             'num_dbt_6mts':               num_dbt_6mts,
# #             'num_dbt_12mts':              num_dbt_12mts,
# #             'num_lss':                    num_lss,
# #             'num_lss_6mts':               num_lss_6mts,
# #             'num_lss_12mts':              num_lss_12mts,
# #             # Timings
# #             'time_since_recent_payment':  time_since_recent_payment,
# #             'time_since_first_deliquency': time_since_first_deliq,
# #             'time_since_recent_deliquency': time_since_recent_deliq,
# #             # Enquiries
# #             'tot_enq':                    tot_enq,
# #             'enq_L3m':                    enq_L3m,
# #             'enq_L6m':                    enq_L6m,
# #             'enq_L12m':                   enq_L12m,
# #             'time_since_recent_enq':      time_since_recent_enq,
# #             'CC_enq':                     CC_enq,
# #             'CC_enq_L6m':                 CC_enq_L6m,
# #             'CC_enq_L12m':                CC_enq_L12m,
# #             'PL_enq':                     PL_enq,
# #             'PL_enq_L6m':                 PL_enq_L6m,
# #             'PL_enq_L12m':                PL_enq_L12m,
# #             # Ratios / pct fields
# #             'pct_of_active_TLs_ever':     pct_of_active_TLs_ever,
# #             'pct_opened_TLs_L6m_of_L12m': pct_opened_TLs_L6m_of_L12m,
# #             'pct_currentBal_all_TL':      pct_currentBal_all_TL,
# #             'pct_PL_enq_L6m_of_L12m':     pct_PL_enq_L6m_of_L12m,
# #             'pct_CC_enq_L6m_of_L12m':     pct_CC_enq_L6m_of_L12m,
# #             'pct_PL_enq_L6m_of_ever':     pct_PL_enq_L6m_of_ever,
# #             'pct_CC_enq_L6m_of_ever':     pct_CC_enq_L6m_of_ever,
# #             # Utilisation
# #             'CC_utilization':             cc_util_pct / 100 if cc_util_pct > 0 else -99999,
# #             'PL_utilization':             pl_util,
# #             'CC_Flag':                    CC_Flag,
# #             'PL_Flag':                    PL_Flag,
# #             'HL_Flag':                    HL_Flag,
# #             'GL_Flag':                    GL_Flag,
# #             'max_unsec_exposure_inPct':   cc_util_pct if cc_util_pct > 0 else 0,
# #             'last_prod_enq2':             last_prod,
# #             'first_prod_enq2':            first_prod,
# #         }

# #         # ── 17. MERGE AND RETURN ─────────────────────────────────────────────
# #         return {
# #             **s1, **s2,
# #             # Stage-1 form-specific fields
# #             'existing_emi':              existing_emi if existing_emi > 0 else s1['total_emi_monthly'],
# #             'employment_type':           employment_type,
# #             'business_vintage_years':    business_vintage,
# #             'credit_utilization_pct':    cc_util_pct if cc_util_pct > 0 else 0,
# #             # Inferred categoricals for Stage 1 form dropdowns
# #             'salary_stability_flag':     _inferred['salary_stability_flag'],
# #             'payment_discipline_flag':   _inferred['payment_discipline_flag'],
# #             'cashflow_health':           _inferred['cashflow_health'],
# #             'liquidity_flag':            _inferred['liquidity_flag'],
# #             'bureau_risk_flag':          _inferred['bureau_risk_flag'],
# #             # Computed extra signals
# #             'written_off_count':         written_off_count,
# #             'settled_count':             settled_count,
# #             'high_util_flag':            1 if cc_util_pct > 75 else 0,
# #             'recent_deliq_flag':         1 if (dpd_90_count > 0 or dpd_60_count > 0) else 0,
# #             'account_quality_score':     max(0, 100 - written_off_count*20 - settled_count*10 - dpd_90_count*15 - dpd_30_count*5),
# #             '_surplus_proxy':            int(net_cash_surplus),
# #             # Passthrough for UI display / audit
# #             'raw_text':                  full_text,
# #             'success':                   True,
# #             'extraction_method':         'OCR+FullDatasetMapping_v2',
# #         }

# #     except Exception as e:
# #         return {'error': str(e), 'message': f'Error extracting CIBIL data: {str(e)}', 'success': False}

# # # =============================================================================
# # # FAIRNESS LOG HELPER
# # # =============================================================================
# # def log_decision_for_fairness(customer_data: dict, decision: str, risk_score: int, pd_pct: float,
# #                                application_id: str = None, source: str = 'stage1'):
# #     """
# #     Append a minimal record to the in-session fairness log.
# #     source = 'stage1' | 'stage2' | 'batch'
# #     When Stage 2 completes, it REPLACES the Stage 1 record for the same application_id,
# #     so the fairness dashboard always shows the FINAL binding decision.

# #     NOTE A-3 — risk_score scale:
# #       source='stage1' or 'batch' → risk_score is on 0-100 (Stage 1 engine output).
# #       source='stage2'            → risk_score is the combined_risk_score on 0-1000
# #                                    (Stage 1 normalised + Stage 2 tier, see stage2_engine.py).
# #     The fairness dashboard currently uses risk_score only for the 'Avg Risk Score' summary
# #     column. If cross-source comparisons are needed, normalise to a common scale first.
# #     """
# #     record = {
# #         'ts':              datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# #         'application_id':  application_id or customer_data.get('application_id', ''),
# #         'source':          source,
# #         'decision':        decision,
# #         'risk_score':      risk_score,
# #         'pd_pct':          pd_pct,
# #         'gender':          customer_data.get('gender', 'Unknown'),
# #         'city_tier':       customer_data.get('city_tier', 'Unknown'),
# #         'employment_type': customer_data.get('employment_type', 'Unknown'),
# #         'bureau_score':    customer_data.get('bureau_score', 0),
# #         'age_band':        (
# #             '24-30' if customer_data.get('age', 0) < 31 else
# #             '31-40' if customer_data.get('age', 0) < 41 else
# #             '41-50' if customer_data.get('age', 0) < 51 else '51+'
# #         ),
# #     }
# #     st.session_state.fairness_log.append(record)
# #     # D3 FIX: cap at 1000 entries to prevent unbounded memory growth per session
# #     if len(st.session_state.fairness_log) > 1000:
# #         st.session_state.fairness_log = st.session_state.fairness_log[-1000:]

# # # =============================================================================
# # # STAGE 2 BINARY RESOLVER  (defined early — called from page routing below)
# # # =============================================================================
# # def resolve_stage2_to_binary(stage2_result: dict) -> dict:
# #     """
# #     Normalise Stage 2 result to a binary APPROVE / REJECT decision.
# #     REVIEW outcomes are resolved via tier mapping; score is used as tie-breaker.
# #     Defined here (before page routing) so it is always in scope regardless of
# #     which section of the file Streamlit is executing.
# #     """
# #     result = stage2_result.copy()
# #     tier  = result.get('stage2_tier', '')
# #     raw   = result.get('final_decision', '')
# #     score = result.get('combined_risk_score', 0) or 0
# #     TIER_MAP = {'P1': 'APPROVE', 'P2': 'APPROVE', 'P3': 'REJECT', 'P4': 'REJECT'}
# #     if raw == 'REJECT':
# #         result['final_decision'] = 'REJECT'
# #     elif raw == 'APPROVE':
# #         result['final_decision'] = TIER_MAP.get(tier, 'APPROVE')
# #     else:
# #         if tier in TIER_MAP:
# #             result['final_decision'] = TIER_MAP[tier]
# #             result['reason'] = result.get('reason', '') + f" [REVIEW resolved to {TIER_MAP[tier]} via tier {tier}]"
# #         else:
# #             resolved = 'APPROVE' if score >= 600 else 'REJECT'
# #             result['final_decision'] = resolved
# #             result['reason'] = result.get('reason', '') + f" [REVIEW resolved to {resolved} via score {score}]"
# #     if result['final_decision'] == 'APPROVE':
# #         result.setdefault('interest_rate_range', {'P1': '9.5%–11%', 'P2': '11%–13%'}.get(tier, '11%–14%'))
# #     else:
# #         result['interest_rate_range'] = 'N/A — Rejected'
# #     return result


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

# #     # AGE POLICY GATE — split by employment type per spec
# #     # UI allows 18–70 for input flexibility, but policy enforces:
# #     #   - All types:       age must be > 24  (≤ 24 → too young)
# #     #   - Salaried:        age must be ≤ 65  (retirement risk)
# #     #   - Self-Employed / Business: age must be ≤ 70
# #     _is_salaried = employment_type == 'Salaried'
# #     _max_age     = 65 if _is_salaried else 70
# #     _age_label   = "24–65 for Salaried" if _is_salaried else "24–70 for Self-Employed/Business"
# #     if age <= 24:
# #         policy_checks['age'] = f"❌ Age {age} — Too young (Min: 25)"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Applicant too young (minimum age 25)", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 70.0, 'affordability_data': {}}
# #     if age > _max_age:
# #         policy_checks['age'] = f"❌ Age {age} — Exceeds max ({_age_label})"
# #         return {'decision': "REJECT", 'reason': f"Policy Gate: Age exceeds maximum for {employment_type} ({_max_age})", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 70.0, 'affordability_data': {}}
# #     policy_checks['age'] = f"✅ Age {age} (Valid — {_age_label})"

# #     if not kyc_verified:
# #         policy_checks['kyc'] = "❌ KYC Not Verified"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: KYC verification required", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 70.0, 'affordability_data': {}}
# #     policy_checks['kyc'] = "✅ KYC Verified"

# #     if not customer_dict.get('rbi_consent', False):
# #         policy_checks['rbi_consent'] = "❌ RBI Consent not obtained"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Customer consent not obtained", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 70.0, 'affordability_data': {}}
# #     policy_checks['rbi_consent'] = "✅ Consent Obtained"

# #     if bankruptcy_flag:
# #         policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Active bankruptcy", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 95.0, 'affordability_data': {}}
# #     policy_checks['bankruptcy'] = "✅ No Bankruptcy"

# #     if fraud_flag:
# #         policy_checks['fraud'] = "❌ Fraud Flag"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Fraud detected", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 95.0, 'affordability_data': {}}
# #     policy_checks['fraud'] = "✅ No Fraud History"

# #     dependents = customer_dict.get('dependents', 0)
# #     dependents_flag_review = dependents > 5
# #     policy_checks['dependents'] = (f"⚠️ Dependents {dependents} (>5: Review Required)"
# #                                    if dependents_flag_review else f"✅ Dependents {dependents} (Acceptable)")

# #     monthly_income = customer_dict.get('avg_salary_6m', 0)
# #     employment_tenure = customer_dict.get('employment_tenure_months', 0)
# #     business_vintage = customer_dict.get('business_vintage_years', 0)

# #     if monthly_income < 15000:
# #         policy_checks['income'] = f"❌ Income ₹{monthly_income:,.0f} (Min: ₹15,000)"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Income below minimum", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 72.0, 'affordability_data': {}}
# #     policy_checks['income'] = f"✅ Income ₹{monthly_income:,.0f}"

# #     if employment_type == 'Salaried' and employment_tenure < 6:
# #         policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient tenure", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 72.0, 'affordability_data': {}}
# #     elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
# #         policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient business vintage", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 72.0, 'affordability_data': {}}
# #     policy_checks['tenure'] = (f"✅ Tenure {employment_tenure} months" if employment_type == 'Salaried'
# #                                 else f"✅ Business Vintage {business_vintage} years")

# #     bureau_score = customer_dict.get('bureau_score', 0)
# #     dpd_90 = customer_dict.get('dpd_90_count_6m', 0)
# #     credit_utilization = customer_dict.get('credit_utilization_pct', 0)
# #     recent_inquiries = customer_dict.get('recent_inquiries_3m', 0)

# #     if bureau_score < 550:
# #         policy_checks['bureau'] = f"❌ Bureau Score {bureau_score} (Min: 550)"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Bureau score too low", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 82.0, 'affordability_data': {}}
# #     policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"

# #     # DPD90 TIERED GATE:
# #     #   0     -> PASS (clean)
# #     #   1-5   -> REVIEW flag (elevated risk, underwriter required)
# #     #   > 5   -> REJECT (severe delinquency, hard stop)
# #     dpd_90_review_flag = False
# #     # DESIGN NOTE (M2): DPD90 gate is tiered — >5 = hard REJECT, 1-5 = REVIEW flag.
# #     # Legacy calculate_risk_score() (fallback-only) uses softer penalty for DPD90=1;
# #     # that path is NEVER reached in production. This gate is the intended behavior.
# #     if dpd_90 > 5:
# #         policy_checks['dpd'] = f"❌ {dpd_90} instance(s) of 90+ DPD — Hard Reject (Max: 5)"
# #         return {'decision': "REJECT", 'reason': "Policy Gate: Severe delinquency (90+ DPD > 5)", 'confidence': 0,
# #                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
# #                 'pd_percentage': 88.0, 'affordability_data': {}}
# #     elif dpd_90 >= 1:
# #         dpd_90_review_flag = True
# #         policy_checks['dpd'] = f"⚠️ {dpd_90} instance(s) of 90+ DPD — Underwriter Review Required"
# #     else:
# #         policy_checks['dpd'] = "✅ No 90+ DPD (Clean)"
# #     policy_checks['utilization'] = (f"⚠️ High utilization {credit_utilization}%" if credit_utilization > 80
# #                                     else f"✅ Utilization {credit_utilization}%")
# #     policy_checks['inquiries'] = (f"⚠️ {recent_inquiries} recent inquiries" if recent_inquiries > 5
# #                                   else f"✅ {recent_inquiries} inquiries")

# #     active_loans = customer_dict.get('active_loans_count', 0)
# #     active_loans_flag = active_loans >= 5
# #     policy_checks['active_loans'] = (f"⚠️ High active loans ({int(active_loans)}) — Review"
# #                                      if active_loans_flag else f"✅ Active loans: {int(active_loans)}")

# #     salary_stability = customer_dict.get('salary_stability_flag', 'STABLE')
# #     salary_flag = salary_stability == 'UNSTABLE'
# #     policy_checks['salary'] = (
# #         "⚠️ Unstable salary — Review required" if salary_stability == 'UNSTABLE' else
# #         "⚠️ Moderate salary stability" if salary_stability == 'MODERATE' else "✅ Stable salary"
# #     )

# #     input_df = pd.DataFrame([customer_dict])
# #     for col in TOP_FEATURES:
# #         if col not in input_df.columns:
# #             input_df[col] = "Unknown" if col in LE_MAP else 0
# #     for col, le in LE_MAP.items():
# #         if col in input_df.columns:
# #             val = str(input_df[col].values[0])
# #             try: input_df[col] = le.transform([val])[0]
# #             except ValueError: input_df[col] = 0
# #     final_input = input_df[TOP_FEATURES]
# #     pred_idx = MODEL.predict(final_input)[0]
# #     ml_decision = TARGET_LE.inverse_transform([pred_idx])[0]
# #     try:
# #         pred_proba = MODEL.predict_proba(final_input)[0]
# #         confidence = max(pred_proba) * 100
# #         class_probs = {cls: prob * 100 for cls, prob in zip(TARGET_LE.classes_, pred_proba)}
# #     except Exception:
# #         confidence = 75.0
# #         class_probs = {ml_decision: 100.0}

# #     loan_amount   = customer_dict.get('loan_amount', 0)
# #     loan_tenure   = customer_dict.get('loan_tenure_months', 12)
# #     interest_rate = customer_dict.get('interest_rate', 10.5)
# #     existing_emi  = customer_dict.get('existing_emi', 0)
# #     affordability_data = calculate_affordability(monthly_income, loan_amount, interest_rate, loan_tenure, existing_emi)
# #     foir = affordability_data['foir_percentage']

# #     if foir > 50:
# #         ml_decision = "REJECT"
# #         policy_checks['foir'] = f"❌ FOIR {foir:.1f}% exceeds maximum allowed (50%)"

# #     if dependents_flag_review and ml_decision == "APPROVE": ml_decision = "REVIEW"
# #     if active_loans_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"
# #     if salary_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"
# #     if dpd_90_review_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"  # DPD90 1-5 forces review

# #     risk_score = calculate_final_risk_score(
# #         bureau_score=bureau_score, ml_confidence=confidence, foir=foir,
# #         dpd_90=dpd_90, dpd_30=customer_dict.get('dpd_30_count_6m', 0),
# #         net_surplus=customer_dict.get('net_cash_surplus_6m', 0),
# #         bounces=customer_dict.get('inward_bounce_count_3m', 0),
# #         missing_months=customer_dict.get('salary_missing_months', 0),
# #         active_loans=active_loans
# #     )
# #     pd_percentage = calculate_final_pd(
# #         bureau_score=bureau_score, foir=foir, confidence=confidence,
# #         dpd_90_count=dpd_90, dpd_30_count=customer_dict.get('dpd_30_count_6m', 0),
# #         employment_type=employment_type, employment_tenure=employment_tenure,
# #         business_vintage=business_vintage, recent_inquiries=recent_inquiries,
# #         ml_decision=ml_decision
# #     )
# #     return {
# #         'decision': ml_decision, 'ml_raw_decision': ml_decision,
# #         'reason': "Decision based on comprehensive assessment",
# #         'confidence': confidence, 'class_probs': class_probs,
# #         'policy_checks': policy_checks, 'risk_score': risk_score,
# #         'pd_percentage': round(pd_percentage, 2), 'affordability_data': affordability_data
# #     }

# # # =============================================================================
# # # BATCH PREDICTION ENGINE
# # # =============================================================================
# # def process_batch_predictions(df):
# #     results = []
# #     required_fields = {
# #         'age': 35, 'employment_type': 'Salaried', 'kyc_verified': True,
# #         'bankruptcy_flag': False, 'fraud_flag': False, 'rbi_consent': True,
# #         'employment_tenure_months': 24, 'business_vintage_years': 0,
# #         'bureau_score': 700, 'dpd_90_count_6m': 0, 'dpd_30_count_6m': 0,
# #         'credit_utilization_pct': 30, 'recent_inquiries_3m': 0,
# #         'active_loans_count': 0, 'existing_emi': 0, 'avg_salary_6m': 50000,
# #         'AMT_INCOME_TOTAL': 600000, 'net_cash_surplus_6m': 20000,
# #         'salary_stability_flag': 'STABLE', 'loan_amount': 180000,
# #         'loan_tenure_months': 24, 'interest_rate': 10.5, 'AMT_ANNUITY': 8500,
# #         'dependents': 0, 'payment_discipline_flag': 'GOOD',
# #         'liquidity_flag': 'LOW', 'cashflow_health': 'MODERATE',
# #         'bureau_risk_flag': 'LOW', 'inward_bounce_count_3m': 0,
# #         'salary_missing_months': 0, 'gender': 'Unknown', 'city_tier': 'Unknown',
# #     }
# #     for idx, row in df.iterrows():
# #         customer_dict = row.to_dict()
# #         for k, v in customer_dict.items():
# #             if isinstance(v, str):
# #                 if v.lower() in ['yes', 'true', '1']: customer_dict[k] = True
# #                 elif v.lower() in ['no', 'false', '0']: customer_dict[k] = False
# #         for field, default in required_fields.items():
# #             if field not in customer_dict or pd.isna(customer_dict.get(field, None)):
# #                 customer_dict[field] = default
# #         try:
# #             decision_data = make_hybrid_decision_enhanced(customer_dict)
# #             customer_dict['ml_confidence'] = decision_data.get('confidence', 0)
# #             reasons = generate_reason_codes(
# #                 decision=decision_data.get('decision', 'ERROR'),
# #                 customer_data=customer_dict,
# #                 affordability_data=decision_data.get('affordability_data', {}),
# #                 policy_checks=decision_data.get('policy_checks', {})
# #             )
# #             affordability = decision_data.get('affordability_data', {})
# #             result = {
# #                 'application_id': f"BATCH_{idx+1:04d}",
# #                 'decision': decision_data.get('decision', 'ERROR'),
# #                 'risk_score': decision_data.get('risk_score', 0),
# #                 'pd_percentage': decision_data.get('pd_percentage', 0),
# #                 'confidence': round(decision_data.get('confidence', 0), 2),
# #                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# #                 'reason_1': reasons[0] if len(reasons) > 0 else '',
# #                 'reason_2': reasons[1] if len(reasons) > 1 else '',
# #                 'reason_3': reasons[2] if len(reasons) > 2 else '',
# #                 'age': customer_dict.get('age', ''),
# #                 'gender': customer_dict.get('gender', ''),
# #                 'city_tier': customer_dict.get('city_tier', ''),
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
# #                 'prob_approve': round(decision_data.get('class_probs', {}).get('APPROVE', 0), 2),
# #                 'prob_review': round(decision_data.get('class_probs', {}).get('REVIEW', 0), 2),
# #                 'prob_reject': round(decision_data.get('class_probs', {}).get('REJECT', 0), 2),
# #             }
# #         except Exception as e:
# #             result = {
# #                 'application_id': f"BATCH_{idx+1:04d}", 'decision': 'ERROR',
# #                 'risk_score': 0, 'pd_percentage': 0, 'confidence': 0,
# #                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# #                 'reason_1': '', 'reason_2': '', 'reason_3': '',
# #                 'age': customer_dict.get('age', ''), 'gender': customer_dict.get('gender', ''),
# #                 'city_tier': customer_dict.get('city_tier', ''),
# #                 'employment_type': customer_dict.get('employment_type', ''),
# #                 'bureau_score': customer_dict.get('bureau_score', ''),
# #                 'monthly_income': customer_dict.get('avg_salary_6m', ''),
# #                 'loan_amount': customer_dict.get('loan_amount', ''),
# #                 'error_message': str(e)
# #             }
# #         else:
# #             # Log to fairness monitor (success path only)
# #             log_decision_for_fairness(
# #                 customer_dict,
# #                 result['decision'],
# #                 result['risk_score'],
# #                 result['pd_percentage']
# #             )
# #         results.append(result)
# #     return pd.DataFrame(results)

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
# #         card_class = "decision-card decision-card-approved"; icon = "✓"; subtitle = "Application Approved Successfully"
# #     elif decision == "REJECT":
# #         card_class = "decision-card decision-card-rejected"; icon = "✗"; subtitle = "Application Not Approved"
# #     else:
# #         card_class = "decision-card decision-card-review"; icon = "⚠"; subtitle = "Requires Manual Review"
# #     st.markdown(f'<div class="{card_class}"><div class="decision-title">{icon} {decision}</div><div class="decision-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)
# #     col1, col2, col3, col4, col5 = st.columns(5)
# #     with col1: st.markdown(f'<div class="stat-card"><div class="stat-number">{risk_score}</div><div class="stat-label">Risk Score</div></div>', unsafe_allow_html=True)
# #     _pd_color = '#48bb78' if pd_score < 5 else ('#ed8936' if pd_score < 10 else '#f56565')
# #     _pd_label = 'Low Risk' if pd_score < 5 else ('Moderate Risk' if pd_score < 10 else 'High Risk')
# #     with col2: st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{_pd_color}">{pd_score}%</div><div class="stat-label">PD Score</div><div style="font-size:11px;color:{_pd_color};font-weight:600">{_pd_label}</div></div>', unsafe_allow_html=True)
# #     with col3: st.markdown(f'<div class="stat-card"><div class="stat-number">₹{approved_amount:,.0f}</div><div class="stat-label">Loan Amount</div></div>', unsafe_allow_html=True)
# #     with col4: st.markdown(f'<div class="stat-card"><div class="stat-number">{tenure}</div><div class="stat-label">Tenure (Months)</div></div>', unsafe_allow_html=True)
# #     with col5: st.markdown(f'<div class="stat-card"><div class="stat-number">{decision_data.get("confidence", 0):.0f}%</div><div class="stat-label">Confidence</div></div>', unsafe_allow_html=True)
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     col1, col2 = st.columns(2)
# #     with col1: st.markdown(f'<div class="info-box"><strong>📋 Application ID:</strong> {app_id}</div>', unsafe_allow_html=True)
# #     with col2: st.markdown(f'<div class="info-box"><strong>🕐 Decision Timestamp:</strong> {timestamp}</div>', unsafe_allow_html=True)

# # def render_info_card(title, icon, data_dict, status_dict=None):
# #     st.markdown(f'<div class="info-card"><div class="info-card-title">{icon} {title}</div><div class="info-card-content">', unsafe_allow_html=True)
# #     for label, value in data_dict.items():
# #         status = ""
# #         if status_dict and label in status_dict:
# #             if status_dict[label] == "pass": status = '<span class="status-badge badge-pass">✓</span>'
# #             elif status_dict[label] == "fail": status = '<span class="status-badge badge-fail">✗</span>'
# #             elif status_dict[label] == "warning": status = '<span class="status-badge badge-warning">⚠</span>'
# #         st.markdown(f'<div class="data-row"><span class="data-label">{label}</span><span class="data-value">{value} {status}</span></div>', unsafe_allow_html=True)
# #     st.markdown('</div></div>', unsafe_allow_html=True)

# # def render_reason_codes(reasons):
# #     st.markdown('<div class="info-card"><div class="info-card-title">📝 Decision Reasons</div><div class="info-card-content">', unsafe_allow_html=True)
# #     for i, reason in enumerate(reasons, 1):
# #         st.markdown(f'<div class="reason-item"><span class="reason-icon">{i}.</span>{reason}</div>', unsafe_allow_html=True)
# #     st.markdown('</div></div>', unsafe_allow_html=True)

# # def create_modern_gauge(value, title, max_value=100):
# #     color = "#f56565" if value <= 50 else "#ed8936" if value <= 75 else "#48bb78"
# #     fig = go.Figure(go.Indicator(
# #         mode="gauge+number", value=value,
# #         title={'text': title, 'font': {'size': 18, 'color': '#2d3748'}},
# #         number={'suffix': "%", 'font': {'size': 36, 'color': '#2d3748'}},
# #         gauge={
# #             'axis': {'range': [0, max_value]},
# #             'bar': {'color': color, 'thickness': 0.75},
# #             'bgcolor': 'white', 'borderwidth': 0,
# #             'steps': [{'range': [0, 50], 'color': '#fed7d7'},
# #                       {'range': [50, 75], 'color': '#feebc8'},
# #                       {'range': [75, 100], 'color': '#c6f6d5'}]
# #         }
# #     ))
# #     fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white')
# #     return fig

# # def create_modern_bar_chart(class_probs):
# #     df = pd.DataFrame({'Decision': list(class_probs.keys()), 'Probability': list(class_probs.values())})
# #     colors = {'REVIEW': '#ed8936', 'APPROVE': '#48bb78', 'REJECT': '#f56565'}
# #     fig = px.bar(df, x='Decision', y='Probability', title='Decision Probabilities',
# #                  color='Decision', color_discrete_map=colors, text='Probability')
# #     fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
# #     fig.update_layout(showlegend=False, yaxis_title='Probability (%)', xaxis_title='', height=300,
# #                       margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white', plot_bgcolor='white',
# #                       yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]})
# #     return fig

# # # =============================================================================
# # # STAGE 2 BINARY RESOLVER

# # # =============================================================================
# # # STAGE 2 RESULTS DISPLAY
# # # =============================================================================
# # def display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data):
# #     st.markdown("---")
# #     st.markdown('<p class="main-header">🎯 Stage 2 Final Results</p>', unsafe_allow_html=True)
# #     final_decision    = stage2_result.get('final_decision', 'ERROR')
# #     interest_range    = stage2_result.get('interest_rate_range', 'N/A')
# #     stage2_tier       = stage2_result.get('stage2_tier', 'N/A')
# #     stage2_confidence = stage2_result.get('stage2_confidence', 0)
# #     combined_risk     = stage2_result.get('combined_risk_score', 0)

# #     # ── Fairness log: use Stage 2 FINAL decision, remove the earlier Stage 1 entry ──
# #     # Stage 1 logged a preliminary decision for this customer. Since Stage 2
# #     # is the BINDING final decision, we replace that entry so the fairness
# #     # dashboard always reflects the true outcome.
# #     app_id = stage1_customer.get('application_id', None)
# #     if app_id and 'fairness_log' in st.session_state:
# #         st.session_state.fairness_log = [
# #             r for r in st.session_state.fairness_log
# #             if r.get('application_id') != app_id
# #         ]
# #     log_decision_for_fairness(
# #         enhanced_customer_data,
# #         final_decision,
# #         combined_risk,
# #         stage2_result.get('pd_percentage', stage1_data.get('pd_percentage', 0)),
# #         application_id=app_id,
# #         source='stage2'
# #     )

# #     # Update session state — Stage 2 is the binding final decision
# #     st.session_state['stage2_final_decision'] = final_decision

# #     if final_decision == "APPROVE":
# #         st.markdown(
# #             '<div class="decision-card decision-card-approved" style="padding:2.5rem;">'
# #             '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">✔  APPROVED</div>'
# #             '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">✅ STAGE 2 FINAL DECISION — Proceed to Disbursement</div>'
# #             '</div>', unsafe_allow_html=True)
# #     elif final_decision == "REJECT":
# #         st.markdown(
# #             '<div class="decision-card decision-card-rejected" style="padding:2.5rem;">'
# #             '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">✘  REJECTED</div>'
# #             '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">❌ STAGE 2 FINAL DECISION — Application Declined</div>'
# #             '</div>', unsafe_allow_html=True)
# #     else:
# #         st.markdown(
# #             '<div class="decision-card decision-card-review" style="padding:2.5rem;">'
# #             '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">⚑  REVIEW</div>'
# #             '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">⚠️ STAGE 2 FINAL DECISION — Requires Manual Credit Officer Review</div>'
# #             '</div>', unsafe_allow_html=True)

# #     col1, col2, col3, col4 = st.columns(4)
# #     with col1: st.metric("Risk Tier", stage2_tier)
# #     with col2: st.metric("Interest Rate", interest_range)
# #     with col3: st.metric("Combined Risk Score", combined_risk)
# #     with col4: st.metric("Stage 2 Confidence", f"{stage2_confidence:.1f}%" if stage2_confidence else "N/A")

# #     st.markdown("<br>", unsafe_allow_html=True)
# #     tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔬 Analysis", "📋 Data", "📥 Download"])

# #     with tab1:
# #         s1_dec = st.session_state.get('stage1_decision', 'N/A')
# #         s2_label = "✅ APPROVE" if final_decision == "APPROVE" else "❌ REJECT"
# #         comparison_df = pd.DataFrame([
# #             {'Stage': 'Stage 1 (Screening)', 'Decision': s1_dec, 'Risk Score': stage1_data.get('risk_score', 'N/A'), 'Tier': 'N/A', 'Note': 'APPROVE/REVIEW → proceed to Stage 2'},
# #             {'Stage': 'Stage 2 — FINAL', 'Decision': s2_label, 'Risk Score': combined_risk, 'Tier': f"{stage2_tier} | {interest_range}", 'Note': 'Binding final decision'}
# #         ])
# #         st.dataframe(comparison_df, use_container_width=True, hide_index=True)
# #         tier_info = {
# #             'P1': {'name': 'Premium → APPROVED', 'color': '#10B981', 'desc': 'Excellent credit profile — lowest interest rate band'},
# #             'P2': {'name': 'Standard → APPROVED', 'color': '#3B82F6', 'desc': 'Good credit profile — standard interest rate band'},
# #             'P3': {'name': 'Subprime → REJECTED', 'color': '#F59E0B', 'desc': 'Fair credit with elevated risk — application declined'},
# #             'P4': {'name': 'High Risk → REJECTED', 'color': '#EF4444', 'desc': 'High risk profile — application declined'},
# #         }
# #         if stage2_tier in tier_info:
# #             td = tier_info[stage2_tier]
# #             st.markdown(f'<div style="background:{td["color"]};color:white;padding:1rem;border-radius:0.5rem;"><h3 style="margin:0;color:white;">{stage2_tier}: {td["name"]}</h3><p style="margin:0.5rem 0 0 0;">{td["desc"]}</p></div>', unsafe_allow_html=True)
# #         st.info(stage2_result.get('reason', 'N/A'))

# #     with tab2:
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
# #             st.metric("Combined Score", combined_risk)
# #         with st.expander("Complete Stage 2 Result (JSON)"):
# #             st.json(stage2_result)

# #     with tab3:
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             with st.expander("Stage 1 Customer Data"): st.json(stage1_customer)
# #         with col2:
# #             with st.expander("Enhanced CIBIL Data"): st.json(enhanced_customer_data)

# #     with tab4:
# #         if PDF_AVAILABLE and generate_audit_pdf is not None:
# #             try:
# #                 _safe = lambda v, d='N/A': v if v is not None else d
# #                 # Build full pd_calculation_factors from enhanced customer data
# #                 _bs  = enhanced_customer_data.get('bureau_score', stage1_customer.get('bureau_score', 0))
# #                 _foir = stage1_data.get('affordability_data', {}).get('foir_percentage', 0)
# #                 _conf = stage1_data.get('confidence', 0)
# #                 _dpd90 = enhanced_customer_data.get('dpd_90_count_6m', stage1_customer.get('dpd_90_count_6m', 0))
# #                 _dpd30 = enhanced_customer_data.get('dpd_30_count_6m', stage1_customer.get('dpd_30_count_6m', 0))
# #                 _emp_type = enhanced_customer_data.get('employment_type', stage1_customer.get('employment_type', 'Salaried'))
# #                 _emp_ten  = enhanced_customer_data.get('employment_tenure_months', stage1_customer.get('employment_tenure_months', 24))
# #                 _biz_vin  = enhanced_customer_data.get('business_vintage_years', stage1_customer.get('business_vintage_years', 0))
# #                 _inq      = enhanced_customer_data.get('recent_inquiries_3m', stage1_customer.get('recent_inquiries_3m', 0))
# #                 _base_pd   = bureau_score_to_pd(_bs)
# #                 _foir_adj  = foir_to_pd_adjustment(_foir)
# #                 _deliq_mul = delinquency_to_pd_multiplier(_dpd90, _dpd30)
# #                 _emp_adj   = employment_stability_to_pd_adjustment(_emp_type, _emp_ten, _biz_vin)
# #                 _inq_adj   = inquiry_pattern_to_pd_adjustment(_inq)
# #                 _ml_adj    = ml_confidence_to_pd_adjustment(_conf, stage1_data.get('decision','REVIEW'))
# #                 _final_pd  = stage1_data.get('pd_percentage', round(max(0.5, min(
# #                     _base_pd * _deliq_mul + _foir_adj + _emp_adj + _inq_adj + _ml_adj, 25.0)), 2))

# #                 report_data = {
# #                     'application_id':  _safe(stage1_customer.get('application_id')),
# #                     'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
# #                     'model_version':   '8.7',
# #                     'decision':        _safe(stage1_data.get('decision')),
# #                     'stage2_final_decision':      _safe(final_decision),
# #                     'stage2_tier':                _safe(stage2_tier),
# #                     'stage2_interest_range':      _safe(interest_range),
# #                     'stage2_combined_risk_score': _safe(combined_risk, 0),
# #                     'stage2_confidence':          _safe(stage2_confidence, 0),
# #                     'stage2_reason':              _safe(stage2_result.get('reason')),
# #                     'stage2_tier_probabilities':  stage2_result.get('tier_probabilities') or {},
# #                     'stage2_complete_analysis':   stage2_result,
# #                     # Top-level PD — used by audit header (must match pd_calculation_factors.final_pd)
# #                     'pd_percentage':              _final_pd,
# #                     'risk_score':                 _safe(combined_risk, 0),
# #                     'confidence':                 _safe(stage2_confidence, 0),
# #                     # Policy gate results
# #                     'policy_checks': stage1_data.get('policy_checks', {}),
# #                     # Full PD calculation breakdown
# #                     'pd_calculation_factors': {
# #                         'bureau_score':           _bs,
# #                         'base_pd':                round(_base_pd, 2),
# #                         'dpd_90':                 _dpd90,
# #                         'dpd_30':                 _dpd30,
# #                         'delinquency_multiplier': round(_deliq_mul, 2),
# #                         'foir':                   round(_foir, 2),
# #                         'foir_adjustment':        round(_foir_adj, 2),
# #                         'employment_adjustment':  round(_emp_adj, 2),
# #                         'inquiry_adjustment':     round(_inq_adj, 2),
# #                         'ml_adjustment':          round(_ml_adj, 2),
# #                         'final_pd':               _final_pd,
# #                     },
# #                     # Reason codes from Stage 1
# #                     'reason_codes': stage1_customer.get('reason_codes', []),
# #                     # Raw data refs
# #                     'customer_data':          stage1_customer,
# #                     'enhanced_customer_data': enhanced_customer_data,
# #                 }
# #                 pdf_buffer = generate_audit_pdf(report_data)
# #                 st.download_button("📥 Download PDF Report", data=pdf_buffer,
# #                                    file_name=f"stage2_report_{stage1_customer.get('application_id','X')}.pdf",
# #                                    mime="application/pdf", use_container_width=True)
# #             except Exception as e:
# #                 st.error(f"PDF generation failed: {str(e)}")
# #         else:
# #             st.warning("⚠️ PDF generation is not available. Ensure utils/pdf_generator.py is present and `reportlab` is installed (add to requirements.txt).")

# #     st.markdown("---")
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         if st.button("🔄 New Assessment", key="new_assessment_stage2", use_container_width=True):
# #             for k in ['stage1_complete','stage1_decision','stage1_data','current_customer_data']:
# #                 st.session_state[k] = (False if k == 'stage1_complete' else None)
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
# # # FAIRNESS MONITORING DASHBOARD
# # # =============================================================================
# # def render_fairness_dashboard():
# #     st.markdown('<p class="main-header">⚖️ Fairness Monitoring</p>', unsafe_allow_html=True)
# #     st.markdown("""
# #         <div class="info-box">
# #             <strong>RBI Fair Lending Compliance Dashboard</strong><br>
# #             Tracks approval rates across demographic groups to detect potential disparate impact.
# #             <strong>Fairness is measured on the FINAL binding decision</strong> — Stage 2 outcome
# #             is used when available; Stage 1 (screening) entries are automatically replaced once
# #             Stage 2 completes for the same application.
# #             Data is session-based — decisions accumulate as applications are processed.
# #         </div>
# #     """, unsafe_allow_html=True)

# #     log = st.session_state.get('fairness_log', [])

# #     col1, col2 = st.columns([3, 1])
# #     with col2:
# #         if st.button("🗑️ Clear Log", use_container_width=True):
# #             st.session_state.fairness_log = []
# #             st.rerun()

# #     if not log:
# #         st.info("ℹ️ No decisions logged yet. Process some applications from the Assessment page to see fairness metrics here.")
# #         st.markdown("### 📊 What will appear here:")
# #         st.markdown("""
# #         - **Approval rate by Gender** — tracks if male/female/other applicants are treated equitably
# #         - **Approval rate by City Tier** — checks for geographic bias (Tier 1 vs Tier 3 vs Rural)
# #         - **Approval rate by Age Band** — identifies potential age discrimination
# #         - **Approval rate by Employment Type** — salaried vs self-employed equity check
# #         - **Average Risk Score & PD by group** — confirms scoring is not systematically biased
# #         """)
# #         return

# #     df = pd.DataFrame(log)
# #     df['approved'] = (df['decision'] == 'APPROVE').astype(int)
# #     n = len(df)

# #     # Source breakdown
# #     if 'source' in df.columns:
# #         n_s2    = int((df['source'] == 'stage2').sum())
# #         n_s1    = int((df['source'] == 'stage1').sum())
# #         n_batch = int((df['source'] == 'batch').sum())
# #         src_note = f"📌 {n_s2} Stage 2 (final) · {n_s1} Stage 1 (screening) · {n_batch} Batch"
# #         st.caption(src_note)

# #     st.markdown("---")
# #     c1, c2, c3, c4 = st.columns(4)
# #     with c1: st.metric("Total Decisions", n)
# #     with c2: st.metric("Approvals", int(df['approved'].sum()), f"{df['approved'].mean()*100:.1f}%")
# #     with c3: st.metric("Reviews", int((df['decision']=='REVIEW').sum()))
# #     with c4: st.metric("Rejections", int((df['decision']=='REJECT').sum()))

# #     st.markdown("---")
# #     tab1, tab2, tab3, tab4 = st.tabs(["👥 Gender", "🏙️ City Tier", "📅 Age Band", "💼 Employment"])

# #     COLOR_MAP = {'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}

# #     def _approval_bar(group_col, title):
# #         grp = df.groupby(group_col).agg(
# #             Total=('decision', 'count'),
# #             Approved=('approved', 'sum'),
# #             Avg_Risk=('risk_score', 'mean'),
# #             Avg_PD=('pd_pct', 'mean'),
# #         ).reset_index()
# #         grp['Approval Rate %'] = (grp['Approved'] / grp['Total'] * 100).round(1)
# #         grp['Avg Risk Score'] = grp['Avg_Risk'].round(1)
# #         grp['Avg PD %'] = grp['Avg_PD'].round(2)

# #         col1, col2 = st.columns([2, 1])
# #         with col1:
# #             fig = px.bar(grp, x=group_col, y='Approval Rate %',
# #                          title=title, text='Approval Rate %',
# #                          color='Approval Rate %',
# #                          color_continuous_scale=['#f56565', '#ed8936', '#48bb78'],
# #                          range_color=[0, 100])
# #             fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
# #             fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10),
# #                               coloraxis_showscale=False, paper_bgcolor='white', plot_bgcolor='white',
# #                               yaxis={'range': [0, 110], 'gridcolor': '#e2e8f0'})
# #             st.plotly_chart(fig, use_container_width=True)
# #         with col2:
# #             st.markdown("**Summary Table**")
# #             display_df = grp[[group_col, 'Total', 'Approval Rate %', 'Avg Risk Score', 'Avg PD %']].copy()
# #             # Flag groups with approval rate deviation > 15pp from overall
# #             overall_rate = df['approved'].mean() * 100
# #             def _flag(rate):
# #                 diff = rate - overall_rate
# #                 if abs(diff) > 15: return f"{'🔴' if diff < 0 else '🟢'} {rate:.1f}%"
# #                 return f"✅ {rate:.1f}%"
# #             display_df['Approval Rate %'] = display_df['Approval Rate %'].apply(_flag)
# #             st.dataframe(display_df, use_container_width=True, hide_index=True)
# #             overall_str = f"{overall_rate:.1f}%"
# #             st.caption(f"Overall approval rate: **{overall_str}**. 🔴 = >15pp below average (potential bias). 🟢 = >15pp above average.")

# #     with tab1:
# #         if df['gender'].nunique() > 1:
# #             _approval_bar('gender', 'Approval Rate by Gender')
# #             # Decision mix donut per gender
# #             fig2 = px.pie(df, names='decision', color='decision', color_discrete_map=COLOR_MAP,
# #                           title='Decision Mix (all)', hole=0.5)
# #             fig2.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
# #             st.plotly_chart(fig2, use_container_width=True)
# #         else:
# #             st.info("Need 2+ gender values in decisions to show chart. Ensure Gender field is filled on the form.")

# #     with tab2:
# #         if df['city_tier'].nunique() > 1:
# #             _approval_bar('city_tier', 'Approval Rate by City Tier')
# #         else:
# #             st.info("Need 2+ city tier values. Ensure City Tier field is filled on the form.")

# #     with tab3:
# #         if df['age_band'].nunique() > 1:
# #             _approval_bar('age_band', 'Approval Rate by Age Band')
# #         else:
# #             st.info("Need decisions across multiple age bands (24-30, 31-40, 41-50, 51+).")

# #     with tab4:
# #         if df['employment_type'].nunique() > 1:
# #             _approval_bar('employment_type', 'Approval Rate by Employment Type')
# #         else:
# #             st.info("Need 2+ employment types in decisions.")

# #     st.markdown("---")
# #     st.markdown("### 📥 Export Fairness Report")
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         csv_data = df.to_csv(index=False)
# #         st.download_button("📥 Download Decision Log (CSV)", data=csv_data,
# #                            file_name=f"fairness_log_{datetime.now().strftime('%Y%m%d')}.csv",
# #                            mime="text/csv", use_container_width=True)
# #     with col2:
# #         st.caption("⚠️ **Note:** This log is session-based and resets when the app restarts. "
# #                    "For persistent fairness monitoring, connect to a database or export regularly.")


# # # =============================================================================
# # # SIDEBAR
# # # =============================================================================
# # with st.sidebar:
# #     st.markdown("# 🏦 Credit Risk Engine")
# #     st.markdown("---")

# #     navigation_options = ["🏠 Home", "👤 Assessment", "📊 Batch Process", "⚖️ Fairness", "📈 Model Info", "ℹ️ About"]

# #     if (st.session_state.stage1_complete and st.session_state.stage1_decision in ['APPROVE', 'REVIEW']):
# #         navigation_options.insert(2, "🔬 Stage 2 Analysis")
# #         st.success(f"✅ Stage 1: {st.session_state.stage1_decision}")
# #         st.info("🔬 Stage 2 Analysis unlocked!")
# #     elif st.session_state.stage1_complete:
# #         st.warning(f"⚠️ Stage 1: {st.session_state.stage1_decision}")
# #         st.caption("Stage 2 only for APPROVE/REVIEW")

# #     page = st.radio("**Navigation**", navigation_options,
# #                     label_visibility="collapsed", key="page_navigation")

# #     st.markdown("---")
# #     stage2_indicator = '✅ Active' if STAGE2_AVAILABLE and is_stage2_available() else '❌ Inactive'
# #     ocr_indicator = '✅ Ready' if OCR_AVAILABLE else '❌ Not Installed'
# #     pdf_indicator = '✅ Ready' if PDF_AVAILABLE else '❌ Not Installed'
# #     fairness_count = len(st.session_state.fairness_log)

# #     st.markdown(f"""
# #     <div class="info-card">
# #         <div class="info-card-title">System Status</div>
# #         <div class="info-card-content">
# #             <div class="data-row"><span class="data-label">Model</span><span class="data-value">✅ Loaded</span></div>
# #             <div class="data-row"><span class="data-label">Version</span><span class="data-value">8.7</span></div>
# #             <div class="data-row"><span class="data-label">Stage 2</span><span class="data-value">{stage2_indicator}</span></div>
# #             <div class="data-row"><span class="data-label">OCR</span><span class="data-value">{ocr_indicator}</span></div>
# #             <div class="data-row"><span class="data-label">PDF Gen</span><span class="data-value">{pdf_indicator}</span></div>
# #             <div class="data-row"><span class="data-label">Fairness Log</span><span class="data-value">{fairness_count} decisions</span></div>
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
# #             for k in ['stage1_complete','stage1_decision','stage1_data','current_customer_data','extracted_cibil_data']:
# #                 st.session_state[k] = False if k == 'stage1_complete' else None
# #             st.rerun()

# # # =============================================================================
# # # PAGE ROUTING
# # # =============================================================================
# # if page == "🏠 Home":
# #     st.markdown('<p class="main-header">Credit Risk Engine</p>', unsafe_allow_html=True)
# #     st.markdown('<div class="info-box"><h3 style="margin-top:0;">🎯 AI-Powered Lending Decisions</h3><p style="margin-bottom:0;">Comprehensive credit risk evaluation combining hard policy rules, machine learning models, and affordability analysis.</p></div>', unsafe_allow_html=True)
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         st.markdown('<div class="info-card"><div class="info-card-title">🛡️ Policy Gates</div><div class="info-card-content"><ul><li>Age & KYC verification</li><li>RBI consent check</li><li>Employment stability</li><li>Minimum income checks</li><li>Credit bureau thresholds</li><li>Bankruptcy & fraud detection</li></ul></div></div>', unsafe_allow_html=True)
# #     with col2:
# #         st.markdown('<div class="info-card"><div class="info-card-title">🤖 ML Assessment</div><div class="info-card-content"><ul><li>Random Forest classifier</li><li>60K+ training samples</li><li>Confidence scoring</li><li>Multi-class prediction</li><li>Feature importance</li></ul></div></div>', unsafe_allow_html=True)
# #     with col3:
# #         st.markdown('<div class="info-card"><div class="info-card-title">⚖️ Fairness Monitoring</div><div class="info-card-content"><ul><li>Approval rate by gender</li><li>Approval rate by city tier</li><li>Age band equity check</li><li>Employment type parity</li><li>RBI compliance ready</li></ul></div></div>', unsafe_allow_html=True)
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     col1, col2, col3, col4 = st.columns(4)
# #     with col1: st.metric("🎯 Accuracy", "85%", "+2%")
# #     with col2: st.metric("⚡ Avg Response", "1.2s", "-0.3s")
# #     with col3: st.metric("📊 Features", len(TOP_FEATURES))
# #     with col4: st.metric("🔄 Version", "8.7", "Latest")
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     st.markdown("""
# #         <div class="warning-box" style="background:#f0fff4;border:1px solid #9ae6b4;padding:1rem;border-radius:0.5rem;">
# #             <strong>🆕 New in Version 8.7:</strong><br>
# #             • <strong>Cleaned codebase</strong> — removed ~210 lines of duplicate function definitions<br>
# #             • <strong>City Tier field</strong> — Tier 1/2/3/Rural captured on every application<br>
# #             • <strong>Gender field</strong> — explicit gender capture for fairness logging<br>
# #             • <strong>RBI Consent checkbox</strong> — required policy gate before assessment<br>
# #             • <strong>Fairness Monitoring dashboard</strong> — approval rates by gender, city tier, age band, employment type<br>
# #             • <strong>v8.5 features retained</strong> — dual-dataset OCR inference, categorical flag auto-fill
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
# #             c1.metric("Credit Score", ex.get('Credit_Score', '—'))
# #             c2.metric("Monthly Income", f"₹{ex.get('NETMONTHLYINCOME') or ex.get('avg_salary_6m', 0):,}")
# #             c3.metric("DPD 60+ Count", ex.get('num_times_60p_dpd', 0))
# #             c4.metric("CC Utilization", f"{max(0, float(ex.get('CC_utilization', 0) or 0))*100:.0f}%")
# #             _inf = st.session_state.get('_last_inferred_flags', {})
# #             if _inf:
# #                 st.markdown("**📊 Inferred Categorical Flags:**")
# #                 fc1, fc2, fc3, fc4, fc5 = st.columns(5)
# #                 fc1.metric("Payment Discipline", _inf.get('payment_discipline_flag', '—'))
# #                 fc2.metric("Cashflow Health", _inf.get('cashflow_health', '—'))
# #                 fc3.metric("Liquidity", _inf.get('liquidity_flag', '—'))
# #                 fc4.metric("Bureau Risk", _inf.get('bureau_risk_flag', '—'))
# #                 fc5.metric("Salary Stability", _inf.get('salary_stability_flag', '—'))
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
# #                         # ── Stage 1: 60k dataset field autofill ──────────────
# #                         st.session_state.pdf_age               = int(extraction_result.get('AGE', 35))
# #                         st.session_state.pdf_bureau_score      = int(extraction_result.get('Credit_Score', 720))
# #                         st.session_state.pdf_dpd_90            = int(extraction_result.get('dpd_90_count_6m', 0))
# #                         st.session_state.pdf_dpd_30            = int(extraction_result.get('dpd_30_count_6m', 0))
# #                         _cc_util_raw = extraction_result.get('CC_utilization', 0) or 0
# #                         st.session_state.pdf_credit_util       = int(max(0, float(_cc_util_raw)) * 100) if _cc_util_raw > 0 else 0
# #                         st.session_state.pdf_inquiries         = int(extraction_result.get('enq_L3m', 2))
# #                         st.session_state.pdf_active_loans      = int(extraction_result.get('num_std', 1))
# #                         _emi = int(extraction_result.get('existing_emi') or extraction_result.get('total_emi_monthly') or 0)
# #                         st.session_state.pdf_existing_emi      = _emi
# #                         _income = int(extraction_result.get('NETMONTHLYINCOME') or extraction_result.get('avg_salary_6m') or 50000)
# #                         st.session_state.pdf_monthly_income    = _income
# #                         st.session_state.pdf_annual_income     = int(extraction_result.get('AMT_INCOME_TOTAL') or _income * 12)
# #                         _surplus = int(extraction_result.get('net_cash_surplus_6m') or extraction_result.get('_surplus_proxy') or 0)
# #                         st.session_state.pdf_net_surplus       = _surplus
# #                         st.session_state.pdf_employment_tenure = int(extraction_result.get('Time_With_Curr_Empr', 24))
# #                         # Employment type (new — was never filled before)
# #                         _emp = extraction_result.get('employment_type', 'Salaried')
# #                         if _emp in ['Salaried', 'Self-Employed', 'Business']:
# #                             st.session_state.pdf_employment_type = _emp
# #                         # Business vintage (new)
# #                         st.session_state.pdf_business_vintage  = int(extraction_result.get('business_vintage_years', 0))
# #                         # Gender (new — was extracted but never applied to form)
# #                         _g = extraction_result.get('GENDER', 'M')
# #                         st.session_state.pdf_gender = 'Male' if _g == 'M' else 'Female'
# #                         # Dependents: CIBIL PDFs rarely state this; leave at form default
# #                         # Inward bounce & missing salary (inferred from delinquency)
# #                         st.session_state.pdf_inward_bounce     = int(extraction_result.get('inward_bounce_count_3m', 0))
# #                         st.session_state.pdf_salary_missing    = int(extraction_result.get('salary_missing_months', 0))
# #                         # Categorical flags (now come directly from extraction, no second infer needed)
# #                         st.session_state.pdf_salary_stability   = extraction_result.get('salary_stability_flag', 'MODERATE')
# #                         st.session_state.pdf_payment_discipline = extraction_result.get('payment_discipline_flag', 'GOOD')
# #                         st.session_state.pdf_cashflow_health    = extraction_result.get('cashflow_health', 'MODERATE')
# #                         st.session_state.pdf_liquidity_flag     = extraction_result.get('liquidity_flag', 'MODERATE')
# #                         st.session_state.pdf_bureau_risk_flag   = extraction_result.get('bureau_risk_flag', 'MODERATE')
# #                         st.session_state.pdf_just_extracted     = True
# #                         st.session_state._last_extraction       = extraction_result
# #                         st.rerun()
# #                     else:
# #                         st.error(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")

# #     with st.form("assessment_form"):
# #         # ── Identity & Eligibility ─────────────────────────────────────────
# #         st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
# #         col_name1, col_name2 = st.columns([2, 2])
# #         with col_name1:
# #             customer_name = st.text_input("Customer Name (Optional)", value="", placeholder="e.g. Ramesh Kumar")
# #         col1, col2, col3, col4 = st.columns(4)
# #         with col1:
# #             age = st.number_input("Age", 25, 70, value=int(st.session_state.get('pdf_age', 35)), help="Min 25 per RBI lending policy")
# #             employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business'],
# #                 index=['Salaried','Self-Employed','Business'].index(st.session_state.get('pdf_employment_type','Salaried')))
# #         with col2:
# #             _gender_opts = ['Male', 'Female', 'Non-binary / Other', 'Prefer not to say']
# #             _gender_default = st.session_state.get('pdf_gender', 'Male')
# #             _gender_idx = _gender_opts.index(_gender_default) if _gender_default in _gender_opts else 0
# #             gender = st.selectbox("Gender", _gender_opts, index=_gender_idx)
# #             dependents = st.number_input("Number of Dependents", 0, 20, value=int(st.session_state.get('pdf_dependents', 2)))
# #         with col3:
# #             # City Tier — field for fairness monitoring.
# #             # FIX A-6: Use format_func so the selectbox displays the full label to the user
# #             # but city_tier is derived immediately from CITY_TIERS at render time —
# #             # no deferred lookup needed. A caption confirms the stored code.
# #             _city_keys = list(CITY_TIERS.keys())
# #             city_tier_label = st.selectbox(
# #                 "City Tier", _city_keys, index=0,
# #                 format_func=lambda k: k  # full descriptive label shown to user
# #             )
# #             city_tier = CITY_TIERS[city_tier_label]   # short code: 'Tier 1' / 'Tier 2' / etc.
# #             st.caption(f"Stored as: **{city_tier}**")
# #             kyc_verified = st.selectbox("KYC Verified", ['Yes', 'No'],
# #                 index=0 if st.session_state.get('pdf_kyc', True) else 1) == 'Yes'
# #         with col4:
# #             bankruptcy_flag = st.selectbox("Bankruptcy Flag", ['No', 'Yes'],
# #                 index=0 if not st.session_state.get('pdf_bankruptcy', False) else 1) == 'Yes'
# #             fraud_flag = st.selectbox("Fraud Flag", ['No', 'Yes'],
# #                 index=0 if not st.session_state.get('pdf_fraud', False) else 1) == 'Yes'

# #         # RBI Consent — REQUIRED
# #         st.markdown('<p class="section-header">📜 RBI Compliance</p>', unsafe_allow_html=True)
# #         col1, col2 = st.columns([2, 1])
# #         with col1:
# #             rbi_consent = st.checkbox(
# #                 "✅ I confirm the customer has been informed of and consented to: (a) credit bureau enquiry, "
# #                 "(b) data usage for credit assessment, (c) Key Fact Statement (KFS) terms, and "
# #                 "(d) grievance redressal process. **(Required — RBI Digital Lending Guidelines)**",
# #                 value=False
# #             )
# #         with col2:
# #             st.markdown("""
# #                 <div style="background:#fff3cd;border:1px solid #ffc107;padding:0.5rem;border-radius:0.4rem;font-size:0.82rem;">
# #                     ⚠️ Without consent, the application cannot proceed per RBI DLG 2022.
# #                 </div>
# #             """, unsafe_allow_html=True)

# #         # Employment tenure
# #         st.markdown('<p class="section-header">💼 Employment</p>', unsafe_allow_html=True)
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             if employment_type == 'Salaried':
# #                 employment_tenure = st.number_input("Employment Tenure (months)", 0, 600,
# #                     value=int(st.session_state.get('pdf_employment_tenure', 24)))
# #                 business_vintage = 0
# #             else:
# #                 business_vintage = st.number_input("Business Vintage (years)", 0, 50,
# #                     value=int(st.session_state.get('pdf_business_vintage', 3)))
# #                 employment_tenure = 0
# #         with col2:
# #             st.markdown("""
# #                 <div class="info-box" style="margin-top:1rem;">
# #                     <strong>Policy thresholds:</strong><br>
# #                     Salaried: min 6 months<br>
# #                     Self-Employed/Business: min 2 years
# #                 </div>
# #             """, unsafe_allow_html=True)

# #         # Credit Bureau
# #         st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
# #         col1, col2, col3 = st.columns(3)
# #         with col1:
# #             bureau_score = st.number_input("Bureau Score", 300, 900,
# #                 value=int(st.session_state.get('pdf_bureau_score', 720)), step=10)
# #             dpd_90_6m = st.number_input("DPD 90+ (Last 6M)", 0, 20,
# #                 value=int(st.session_state.get('pdf_dpd_90', 0)))
# #             dpd_30_6m = st.number_input("DPD 30+ (Last 6M)", 0, 20,
# #                 value=int(st.session_state.get('pdf_dpd_30', 0)))
# #         with col2:
# #             credit_utilization = st.number_input("Credit Utilization (%)", 0, 100,
# #                 value=int(st.session_state.get('pdf_credit_util', 30)))
# #             recent_inquiries = st.number_input("Recent Inquiries (3M)", 0, 20,
# #                 value=int(st.session_state.get('pdf_inquiries', 2)))
# #         with col3:
# #             active_loans = st.number_input("Active Loans", 0, 10,
# #                 value=int(st.session_state.get('pdf_active_loans', 1)))
# #             existing_emi = st.number_input("Existing Total EMI (₹)", 0, 200000,
# #                 value=int(st.session_state.get('pdf_existing_emi', 15000)), step=1000)

# #         # Income & Financial
# #         st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
# #         col1, col2, col3, col4 = st.columns(4)
# #         with col1:
# #             avg_salary = st.number_input("Monthly Income (₹)", 0, 1000000,
# #                 value=int(st.session_state.get('pdf_monthly_income', 50000)), step=5000)
# #             amt_income = st.number_input("Annual Income (₹)", 0, 10000000,
# #                 value=int(st.session_state.get('pdf_annual_income', 600000)), step=10000)
# #         with col2:
# #             net_surplus = st.number_input("Net Cash Surplus (₹)", -100000, 500000,
# #                 value=int(st.session_state.get('pdf_net_surplus', 20000)), step=5000)
# #             _ss_opts = ['STABLE', 'MODERATE', 'UNSTABLE']
# #             salary_stability = st.selectbox("Salary Stability", _ss_opts,
# #                 index=_ss_opts.index(st.session_state.get('pdf_salary_stability', 'STABLE')))
# #         with col3:
# #             loan_amount = st.number_input("Loan Amount (₹)", 0, 5000000,
# #                 value=int(st.session_state.get('pdf_loan_amount', 180000)), step=10000)
# #             loan_tenure = st.number_input("Tenure (months)", 3, 360,
# #                 value=int(st.session_state.get('pdf_loan_tenure', 24)))
# #         with col4:
# #             interest_rate = st.number_input("Interest Rate (%)", 8.0, 20.0,
# #                 value=float(st.session_state.get('pdf_interest_rate', 10.5)), step=0.5)
# #             amt_annuity = st.number_input("Requested EMI (₹)", 0, 200000,
# #                 value=int(st.session_state.get('pdf_amt_annuity', 8500)), step=500)

# #         # Additional Credit Behaviour
# #         st.markdown('<p class="section-header">📋 Additional Credit Behaviour</p>', unsafe_allow_html=True)
# #         col1, col2, col3 = st.columns(3)
# #         with col1:
# #             _pd_opts = ['GOOD', 'MODERATE', 'POOR']
# #             payment_discipline = st.selectbox("Payment Discipline", _pd_opts,
# #                 index=_pd_opts.index(st.session_state.get('pdf_payment_discipline', 'GOOD')))
# #             _lq_opts = ['LOW', 'ADEQUATE', 'MODERATE']
# #             liquidity_flag = st.selectbox("Liquidity", _lq_opts,
# #                 index=_lq_opts.index(st.session_state.get('pdf_liquidity_flag', 'LOW')))
# #         with col2:
# #             _cf_opts = ['MODERATE', 'HEALTHY', 'STRESSED', 'STABLE']
# #             cashflow_health = st.selectbox("Cashflow Health", _cf_opts,
# #                 index=_cf_opts.index(st.session_state.get('pdf_cashflow_health', 'MODERATE')))
# #             _br_opts = ['LOW', 'MEDIUM', 'HIGH']
# #             bureau_risk_flag = st.selectbox("Bureau Risk", _br_opts,
# #                 index=_br_opts.index(st.session_state.get('pdf_bureau_risk_flag', 'LOW')))
# #         with col3:
# #             inward_bounce_count   = st.number_input("Inward Bounce Count (3M)", 0, 10, value=int(st.session_state.get('pdf_inward_bounce', 0)))
# #             salary_missing_months = st.number_input("Missing Salary Months (6M)", 0, 6, value=int(st.session_state.get('pdf_salary_missing', 0)))

# #         st.markdown("<br>", unsafe_allow_html=True)
# #         submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)

# #     if submitted:
# #         timestamp = datetime.now()
# #         app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
# #         customer_data = {
# #             'name': customer_name.strip() if customer_name.strip() else 'N/A',
# #             'age': age, 'employment_type': employment_type,
# #             'gender': gender, 'city_tier': city_tier,
# #             'dependents': dependents, 'kyc_verified': kyc_verified,
# #             'rbi_consent': rbi_consent,
# #             'bankruptcy_flag': bankruptcy_flag, 'fraud_flag': fraud_flag,
# #             'employment_tenure_months': employment_tenure,
# #             'business_vintage_years': business_vintage,
# #             'bureau_score': bureau_score,
# #             'dpd_90_count_6m': dpd_90_6m, 'dpd_30_count_6m': dpd_30_6m,
# #             'credit_utilization_pct': credit_utilization, 'max_utilization': credit_utilization,
# #             'recent_inquiries_3m': recent_inquiries, 'active_loans_count': active_loans,
# #             'avg_salary_6m': avg_salary, 'AMT_INCOME_TOTAL': amt_income,
# #             'net_cash_surplus_6m': net_surplus, 'salary_stability_flag': salary_stability,
# #             'loan_amount': loan_amount, 'loan_tenure_months': loan_tenure,
# #             'interest_rate': interest_rate, 'existing_emi': existing_emi,
# #             'AMT_ANNUITY': amt_annuity, 'application_id': app_id,
# #             'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
# #             'payment_discipline_flag': payment_discipline,
# #             'liquidity_flag': liquidity_flag, 'cashflow_health': cashflow_health,
# #             'bureau_risk_flag': bureau_risk_flag,
# #             'inward_bounce_count_3m': inward_bounce_count,
# #             'salary_missing_months': salary_missing_months,
# #         }

# #         with st.spinner("🔄 Processing Stage 1 assessment..."):
# #             decision_data = make_hybrid_decision_enhanced(customer_data)

# #         # Inject ML confidence so reason_codes.py can distinguish ML-driven REVIEW
# #         customer_data['ml_confidence'] = decision_data.get('confidence', 0)
# #         reasons = generate_reason_codes(
# #             decision=decision_data.get('decision', 'ERROR'),
# #             customer_data=customer_data,
# #             affordability_data=decision_data.get('affordability_data', {}),
# #             policy_checks=decision_data.get('policy_checks', {})
# #         )
# #         customer_data['reason_codes'] = reasons

# #         # Log to fairness monitor (Stage 1 — may be replaced by Stage 2 final decision)
# #         log_decision_for_fairness(customer_data, decision_data.get('decision','ERROR'),
# #                                   decision_data.get('risk_score', 0), decision_data.get('pd_percentage', 0),
# #                                   application_id=customer_data.get('application_id'),
# #                                   source='stage1')

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
# #                                  {"Age": age, "Gender": gender, "City Tier": city_tier,
# #                                   "Employment": employment_type, "Dependents": dependents,
# #                                   "KYC Status": "Verified" if kyc_verified else "Not Verified",
# #                                   "RBI Consent": "✅ Obtained" if rbi_consent else "❌ Not obtained"})
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
# #                 st.markdown('<div class="info-box" style="background:linear-gradient(135deg,#10B981,#059669);color:white;text-align:center;"><h3 style="margin:0;color:white;">✅ Eligible for Stage 2 Deep Dive</h3></div>', unsafe_allow_html=True)
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
# #                 st.markdown('<div style="background:linear-gradient(135deg,#EF4444,#DC2626);color:white;padding:1rem;border-radius:0.5rem;text-align:center;"><h3 style="margin:0;color:white;">❌ Stage 2 Not Available</h3><p style="margin:0.5rem 0 0 0;">Application rejected. Stage 2 only for APPROVE/REVIEW.</p></div>', unsafe_allow_html=True)

# #             st.markdown("<br>", unsafe_allow_html=True)
# #             affordability = decision_data.get('affordability_data', {})
# #             foir      = affordability.get('foir_percentage', 0)
# #             total_emi = int(round(affordability.get('total_emi', 0)))
# #             net_disp  = int(round(affordability.get('net_disposable', 0)))

# #             col1, col2, col3 = st.columns(3)
# #             with col1:
# #                 render_info_card("Identity & Eligibility", "👤",
# #                     {f"Age: {age}": "", f"Employment: {employment_type}": "",
# #                      f"City Tier: {city_tier}": "", f"Dependents: {dependents}": "",
# #                      f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""},
# #                     {f"Age: {age}": "pass" if (age > 24 and age <= (65 if employment_type == 'Salaried' else 70)) else "fail",
# #                      f"Employment: {employment_type}": "pass",
# #                      f"City Tier: {city_tier}": "pass",
# #                      f"Dependents: {dependents}": "pass" if dependents <= 5 else "warning",
# #                      f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_verified else "fail"})
# #             with col2:
# #                 render_info_card("Credit Bureau", "🏦",
# #                     {f"Bureau Score: {bureau_score}": "", f"DPD 90+: {dpd_90_6m}": "",
# #                      f"Utilization: {credit_utilization}%": ""},
# #                     {f"Bureau Score: {bureau_score}": "pass" if bureau_score >= 550 else "fail",
# #                      f"DPD 90+: {dpd_90_6m}": "pass" if dpd_90_6m == 0 else ("warning" if dpd_90_6m <= 5 else "fail"),
# #                      f"Utilization: {credit_utilization}%": "pass" if credit_utilization <= 40 else "warning"})
# #             with col3:
# #                 render_info_card("Affordability", "💰",
# #                     {f"Monthly Income: ₹{avg_salary:,}": "", f"FOIR: {foir:.1f}%": "",
# #                      f"Total EMI: ₹{total_emi:,}": "", f"Net Disposable: ₹{net_disp:,}": ""},
# #                     {f"Monthly Income: ₹{avg_salary:,}": "pass",
# #                      f"FOIR: {foir:.1f}%": "pass" if foir <= 50 else "fail",
# #                      f"Total EMI: ₹{total_emi:,}": "pass",
# #                      f"Net Disposable: ₹{net_disp:,}": "pass" if net_disp >= 10000 else "warning"})

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
# #                     st.warning("⚠️ PDF download unavailable — `reportlab` or `pdf_generator.py` not found. Check requirements.txt.")
# #             with col2:
# #                 if st.button("🔄 Re-Evaluate", key="reevaluate_btn", use_container_width=True):
# #                     st.rerun()

# #         with tab3:
# #             st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
# #             col1, col2 = st.columns(2)
# #             with col1:
# #                 st.plotly_chart(create_modern_gauge(decision_data.get('confidence', 0), "Model Confidence"), use_container_width=True)
# #             with col2:
# #                 st.plotly_chart(create_modern_bar_chart(decision_data.get('class_probs', {"APPROVE": 0, "REVIEW": 0, "REJECT": 0})), use_container_width=True)
# #             st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
# #             policy_df = pd.DataFrame([{'Check': k, 'Result': v} for k, v in decision_data.get('policy_checks', {}).items()])
# #             st.dataframe(policy_df, use_container_width=True, hide_index=True)
# #             st.markdown('<p class="section-header">PD Calculation Breakdown</p>', unsafe_allow_html=True)
# #             for factor, value in {
# #                 'Bureau Score': f"{bureau_score} → Base PD: {bureau_score_to_pd(bureau_score):.1f}%",
# #                 'Delinquency': f"DPD 90+: {dpd_90_6m}, DPD 30+: {dpd_30_6m} → Multiplier: {delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x",
# #                 'FOIR Impact': f"{foir:.1f}% → Adjustment: {foir_to_pd_adjustment(foir):.1f}%",
# #                 'Final PD': f"{decision_data.get('pd_percentage', 0)}%"
# #             }.items():
# #                 st.markdown(f"**{factor}:** {value}")

# #         with tab4:
# #             st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
# #             audit_log = sanitize_for_json({
# #                 'application_id': app_id,
# #                 'timestamp': timestamp.isoformat(),
# #                 'decision': decision_data.get('decision', 'ERROR'),
# #                 'risk_score': decision_data.get('risk_score', 0),
# #                 'pd_percentage': decision_data.get('pd_percentage', 0),
# #                 'confidence': round(decision_data.get('confidence', 0), 2),
# #                 'model_version': '8.7',
# #                 'gender': gender, 'city_tier': city_tier,
# #                 'rbi_consent': rbi_consent,
# #                 'reason_codes': reasons,
# #                 'policy_checks': decision_data.get('policy_checks', {}),
# #                 'affordability': decision_data.get('affordability_data', {}),
# #                 'customer_data': {k: v for k, v in customer_data.items() if k not in ['application_id','timestamp','reason_codes']},
# #             })
# #             with st.expander("📋 View Audit Log (JSON)"):
# #                 st.json(audit_log)
# #             col1, col2 = st.columns(2)
# #             with col1:
# #                 if PDF_AVAILABLE and generate_audit_pdf is not None:
# #                     try:
# #                         audit_pdf_buffer = generate_audit_pdf(audit_log)
# #                         st.download_button("📥 Download Audit Trail (PDF)", data=audit_pdf_buffer,
# #                                            file_name=f"audit_trail_{app_id}.pdf", mime="application/pdf",
# #                                            use_container_width=True)
# #                     except Exception as e:
# #                         st.error(f"Error generating audit PDF: {str(e)}")
# #                 else:
# #                     st.warning("⚠️ Audit PDF download unavailable — `reportlab` or `pdf_generator.py` not found. Check requirements.txt.")
# #             with col2:
# #                 st.download_button("📥 Download Audit Log (JSON)",
# #                                    data=json.dumps(audit_log, indent=2),
# #                                    file_name=f"audit_{app_id}.json", mime="application/json",
# #                                    use_container_width=True)

# # elif page == "🔬 Stage 2 Analysis":
# #     st.markdown('<p class="main-header">Stage 2: CIBIL Deep Dive</p>', unsafe_allow_html=True)

# #     if not st.session_state.get('stage1_complete', False):
# #         st.error("❌ You must complete Stage 1 Assessment first!")
# #         if st.button("← Go to Assessment", use_container_width=True):
# #             st.session_state.page_navigation = "👤 Assessment"
# #             st.rerun()
# #         st.stop()

# #     if st.session_state.get('stage1_decision') not in ['APPROVE', 'REVIEW']:
# #         st.error("❌ Stage 2 is only available for APPROVED or REVIEW applications!")
# #         if st.button("← Go Back", use_container_width=True):
# #             st.session_state.page_navigation = "👤 Assessment"
# #             st.rerun()
# #         st.stop()

# #     if not (STAGE2_AVAILABLE and is_stage2_available()):
# #         st.error("❌ Stage 2 model not available! Please ensure `stage2_cibil_model.pkl` is in the project directory.")
# #         if st.button("← Go Back", use_container_width=True):
# #             st.session_state.page_navigation = "👤 Assessment"
# #             st.rerun()
# #         st.stop()

# #     stage1_data = st.session_state.get('stage1_data', {})
# #     stage1_customer = st.session_state.get('current_customer_data', {})

# #     st.markdown(f'<div class="info-box" style="background:linear-gradient(135deg,#3B82F6,#2563EB);color:white;"><h3 style="margin:0;color:white;">📊 Stage 1 Results</h3><p style="margin:0.5rem 0 0 0;"><strong>Decision:</strong> {st.session_state.get("stage1_decision","N/A")} | <strong>Risk Score:</strong> {stage1_data.get("risk_score","N/A")} | <strong>App ID:</strong> {stage1_customer.get("application_id","N/A")}</p></div>', unsafe_allow_html=True)
# #     st.markdown("<br>", unsafe_allow_html=True)

# #     tab_options = ["Manual Entry", "PDF Upload", "Batch Analysis"]
# #     default_tab = st.session_state.get('stage2_selected_tab', 'Manual Entry')
# #     selected_tab = st.radio("Select input method", tab_options,
# #                             index=tab_options.index(default_tab) if default_tab in tab_options else 0,
# #                             horizontal=True, label_visibility="collapsed")

# #     if selected_tab == "Manual Entry":
# #         st.markdown('<p class="section-header">Manual CIBIL Data Entry</p>', unsafe_allow_html=True)
# #         with st.form("stage2_manual_form"):
# #             st.markdown("### 👤 Demographics & Product Enquiries")
# #             col1, col2, col3 = st.columns(3)
# #             with col1:
# #                 gender_s2 = st.selectbox("Gender", ["Male", "Female", "Others"])
# #                 marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Others"])
# #                 education = st.selectbox("Education", ["Graduate", "Post Graduate", "Under Graduate", "Professional", "Others"])
# #             with col2:
# #                 st.markdown("**Credit Score & History**")
# #                 cibil_score = st.number_input("Credit Score", 300, 900, 720, 10)
# #                 max_delinquency = st.number_input("Max Delinquency Level", 0, 100, 0)
# #                 num_times_30dpd = st.number_input("Times 30+ DPD", 0, 50, 0)
# #                 num_times_60dpd = st.number_input("Times 60+ DPD", 0, 50, 0)
# #                 num_times_delinquent = st.number_input("Total Delinquent", 0, 50, 0)
# #             with col3:
# #                 st.markdown("**Recent Behavior**")
# #                 num_deliq_6m = st.number_input("Delinquencies (6M)", 0, 20, 0)
# #                 num_deliq_12m = st.number_input("Delinquencies (12M)", 0, 20, 0)
# #                 max_deliq_6m = st.number_input("Max Delinq (6M)", 0, 100, 0)
# #                 max_deliq_12m = st.number_input("Max Delinq (12M)", 0, 100, 0)
# #                 enq_L3m = st.number_input("Inquiries (3M)", 0, 20, 2)
# #                 enq_L6m = st.number_input("Inquiries (6M)", 0, 30, 4)
# #                 enq_L12m = st.number_input("Inquiries (12M)", 0, 50, 6)

# #             col1, col2, col3 = st.columns(3)
# #             with col1:
# #                 st.markdown("**Account Quality**")
# #                 num_std = st.number_input("Standard Accounts", 0, 50, 3)
# #                 num_std_6m = st.number_input("Standard (6M)", 0, 50, 3)
# #                 num_std_12m = st.number_input("Standard (12M)", 0, 50, 3)
# #                 num_sub = st.number_input("Sub-standard", 0, 20, 0)
# #                 num_sub_6m = st.number_input("Sub-standard (6M)", 0, 20, 0)
# #                 num_dbt = st.number_input("Doubtful", 0, 10, 0)
# #                 num_lss = st.number_input("Loss", 0, 10, 0)
# #             with col2:
# #                 st.markdown("**Utilization**")
# #                 pct_active_tls = st.number_input("% Active TLs", 0.0, 1.0, 0.60, 0.01)
# #                 pct_current_bal = st.number_input("Current Balance %", 0.0, 1.0, 0.30, 0.01)
# #                 cc_utilization = st.number_input("CC Utilization", 0.0, 1.0, 0.35, 0.01)
# #                 pl_utilization = st.number_input("PL Utilization", 0.0, 1.0, 0.25, 0.01)
# #                 max_unsec_exposure = st.number_input("Max Unsec Exposure %", 0, 100, 30)
# #             with col3:
# #                 st.markdown("**Demographics & Products**")
# #                 age_cibil = st.number_input("Age", 25, 70, int(stage1_customer.get('age', 35)), help="Min 25 per RBI lending policy")
# #                 net_monthly_income = st.number_input("Net Monthly Income", 0, 1000000, int(stage1_customer.get('avg_salary_6m', 50000)), 5000)
# #                 time_curr_employer = st.number_input("Employment Tenure (months)", 0, 600, int(stage1_customer.get('employment_tenure_months', 24)))
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
# #                     st.warning(f'⚠️ CIBIL income ₹{_s2_inc:,} much lower than application income ₹{_s1_inc:,}. Using application income.')
# #                 enhanced_customer_data.update({
# #                     'bureau_score': cibil_score, 'age': age_cibil,
# #                     'avg_salary_6m': _final_income, 'employment_tenure_months': time_curr_employer,
# #                     'dpd_30_count_6m': num_times_30dpd, 'dpd_90_count_6m': num_times_60dpd,
# #                     'max_delinquency_level': max_delinquency, 'num_times_delinquent': num_times_delinquent,
# #                     'num_deliq_6mts': num_deliq_6m, 'num_deliq_12mts': num_deliq_12m,
# #                     'max_deliq_6mts': max_deliq_6m, 'max_deliq_12mts': max_deliq_12m,
# #                     'recent_inquiries_3m': enq_L3m, 'enq_L6m': enq_L6m, 'enq_L12m': enq_L12m,
# #                     'active_loans_count': num_std, 'num_std_6mts': num_std_6m, 'num_std_12mts': num_std_12m,
# #                     'num_sub': num_sub, 'num_sub_6mts': num_sub_6m,
# #                     'num_dbt': num_dbt, 'num_lss': num_lss,
# #                     'credit_utilization_pct': cc_utilization * 100,
# #                     'pct_of_active_TLs_ever': pct_active_tls, 'pct_currentBal_all_TL': pct_current_bal,
# #                     'CC_utilization': cc_utilization, 'PL_utilization': pl_utilization,
# #                     'max_unsec_exposure_inPct': max_unsec_exposure,
# #                     'CC_Flag': 1 if cc_flag else 0, 'PL_Flag': 1 if pl_flag else 0,
# #                     'HL_Flag': 1 if hl_flag else 0, 'GL_Flag': 1 if gl_flag else 0,
# #                     'GENDER': gender_s2, 'MARITALSTATUS': marital_status, 'EDUCATION': education,
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
# #             st.warning("Please use the **Manual Entry** tab.")
# #         else:
# #             uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="stage2_pdf")
# #             if uploaded_pdf is not None:
# #                 st.success(f"✅ File uploaded: {uploaded_pdf.name} ({uploaded_pdf.size/1024:.1f} KB)")
# #                 if st.button("🔬 Extract & Analyze", key="extract_analyze_stage2", type="primary", use_container_width=True):
# #                     with st.spinner("🔄 Extracting data from PDF..."):
# #                         extraction_result = extract_cibil_from_pdf(uploaded_pdf)
# #                     if extraction_result.get('success', False):
# #                         st.success("✅ PDF extraction successful!")

# #                         # ── Summary metrics ──────────────────────────────────
# #                         c1, c2, c3, c4 = st.columns(4)
# #                         c1.metric("Credit Score",    extraction_result.get('Credit_Score', 'N/A'))
# #                         c2.metric("DPD 30+ Count",   extraction_result.get('num_times_30p_dpd', 0))
# #                         c3.metric("DPD 60+ Count",   extraction_result.get('num_times_60p_dpd', 0))
# #                         c4.metric("Active Accounts", extraction_result.get('num_std', 0))
# #                         c1, c2, c3, c4 = st.columns(4)
# #                         c1.metric("Monthly Income", f"₹{extraction_result.get('NETMONTHLYINCOME', 0):,}")
# #                         c2.metric("Employment Tenure", f"{extraction_result.get('Time_With_Curr_Empr',0)} mo")
# #                         c3.metric("Written Off",    extraction_result.get('num_lss', 0))
# #                         c4.metric("Enquiries (3M)", extraction_result.get('enq_L3m', 0))
# #                         c1, c2, c3, c4 = st.columns(4)
# #                         c1.metric("Payment Discipline", extraction_result.get('payment_discipline_flag','—'))
# #                         c2.metric("Cashflow Health",    extraction_result.get('cashflow_health','—'))
# #                         c3.metric("Bureau Risk",        extraction_result.get('bureau_risk_flag','—'))
# #                         c4.metric("Salary Stability",   extraction_result.get('salary_stability_flag','—'))

# #                         if extraction_result.get('written_off_count', 0) > 0:
# #                             st.warning(f"⚠️ {extraction_result['written_off_count']} written-off accounts detected — score may be overridden.")

# #                         _surplus_proxy = extraction_result.get('_surplus_proxy', 0)
# #                         if _surplus_proxy:
# #                             st.info(f"💡 Bureau-only PDF — net surplus estimated from income: ₹{_surplus_proxy:,}")

# #                         with st.expander("📋 View all extracted fields"):
# #                             _display = {k: v for k, v in extraction_result.items() if k not in ('raw_text','success','extraction_method')}
# #                             st.json(_display)

# #                         # ── Build enhanced_customer_data ─────────────────────
# #                         # Start from Stage 1 customer (has gender, city_tier, rbi_consent, loan details)
# #                         enhanced_customer_data = stage1_customer.copy()

# #                         # Apply ALL extracted fields directly — the new extractor maps every column
# #                         _skip = {'raw_text', 'success', 'extraction_method',
# #                                  'loan_amount', 'loan_tenure_months', 'interest_rate',
# #                                  'rbi_consent', 'kyc_verified', 'bankruptcy_flag', 'fraud_flag'}
# #                         for k, v in extraction_result.items():
# #                             if k not in _skip and v is not None:
# #                                 enhanced_customer_data[k] = v

# #                         # Income safety: if CIBIL income << Stage 1 application income, keep Stage 1
# #                         _s1_inc = stage1_customer.get('avg_salary_6m', 50000)
# #                         _s2_inc = extraction_result.get('NETMONTHLYINCOME', 0) or 0
# #                         if 0 < _s2_inc < _s1_inc * 0.4:
# #                             enhanced_customer_data['avg_salary_6m'] = _s1_inc
# #                             enhanced_customer_data['AMT_INCOME_TOTAL'] = _s1_inc * 12
# #                             st.warning(f"⚠️ CIBIL income ₹{_s2_inc:,} << application income ₹{_s1_inc:,} — using application income for FOIR.")

# #                         # Sentinel cleanup
# #                         enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)

# #                         with st.spinner("🔬 Running Stage 2 analysis..."):
# #                             try:
# #                                 stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
# #                                 stage2_result = resolve_stage2_to_binary(stage2_result)
# #                                 display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
# #                             except Exception as e:
# #                                 st.error(f"❌ Analysis failed: {str(e)}")
# #                                 st.exception(e)
# #                     else:
# #                         st.error("❌ PDF extraction failed: " + extraction_result.get('error', 'Unknown'))

# #     elif selected_tab == "Batch Analysis":
# #         st.info("📊 Stage 2 Batch analysis coming soon.")

# # elif page == "⚖️ Fairness":
# #     render_fairness_dashboard()

# # elif page == "📊 Batch Process":
# #     st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
# #     st.markdown('<div class="info-box">📤 Upload a CSV file with customer data for bulk credit assessment.</div>', unsafe_allow_html=True)
# #     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# #     if uploaded_file is not None:
# #         try:
# #             df = pd.read_csv(uploaded_file)
# #             st.success(f"✅ Successfully loaded {len(df)} records")
# #             with st.expander("📄 Preview Uploaded Data"):
# #                 st.dataframe(df.head(), use_container_width=True)
# #             required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
# #             missing_cols = [col for col in required_cols if col not in df.columns]
# #             if missing_cols:
# #                 st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
# #             else:
# #                 if st.button("🚀 Process Batch Predictions", key="process_batch_btn", type="primary", use_container_width=True):
# #                     with st.spinner(f"🔍 Processing {len(df)} records..."):
# #                         results_df = process_batch_predictions(df)
# #                     st.success(f"✅ Completed {len(results_df)} records!")
# #                     tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
# #                     with tab1:
# #                         st.dataframe(results_df, use_container_width=True)
# #                         c1, c2, c3, c4 = st.columns(4)
# #                         with c1: st.metric("✅ Approved", len(results_df[results_df['decision'] == 'APPROVE']))
# #                         with c2: st.metric("❌ Rejected", len(results_df[results_df['decision'] == 'REJECT']))
# #                         with c3: st.metric("⚠️ Review", len(results_df[results_df['decision'] == 'REVIEW']))
# #                         with c4: st.metric("📊 Avg Risk Score", f"{results_df['risk_score'].mean():.0f}")
# #                     with tab2:
# #                         col1, col2 = st.columns(2)
# #                         with col1:
# #                             dc = results_df['decision'].value_counts()
# #                             fig1 = px.pie(values=dc.values, names=dc.index, title="Decision Distribution",
# #                                           color=dc.index, color_discrete_map={'APPROVE':'#48bb78','REVIEW':'#ed8936','REJECT':'#f56565'})
# #                             st.plotly_chart(fig1, use_container_width=True)
# #                         with col2:
# #                             fig2 = px.histogram(results_df, x='risk_score', title="Risk Score Distribution",
# #                                                 nbins=20, color_discrete_sequence=['#587042'])
# #                             st.plotly_chart(fig2, use_container_width=True)
# #                         # Fairness charts from batch
# #                         if 'gender' in results_df.columns and results_df['gender'].nunique() > 1:
# #                             results_df['approved_num'] = (results_df['decision'] == 'APPROVE').astype(int)
# #                             grp = results_df.groupby('gender')['approved_num'].mean().reset_index()
# #                             grp['Approval Rate %'] = (grp['approved_num'] * 100).round(1)
# #                             fig3 = px.bar(grp, x='gender', y='Approval Rate %', title='Approval Rate by Gender (Batch)',
# #                                           color='Approval Rate %', color_continuous_scale=['#f56565','#48bb78'], range_color=[0,100])
# #                             st.plotly_chart(fig3, use_container_width=True)
# #                         if 'city_tier' in results_df.columns and results_df['city_tier'].nunique() > 1:
# #                             results_df['approved_num'] = (results_df['decision'] == 'APPROVE').astype(int)
# #                             grp2 = results_df.groupby('city_tier')['approved_num'].mean().reset_index()
# #                             grp2['Approval Rate %'] = (grp2['approved_num'] * 100).round(1)
# #                             fig4 = px.bar(grp2, x='city_tier', y='Approval Rate %', title='Approval Rate by City Tier (Batch)',
# #                                           color='Approval Rate %', color_continuous_scale=['#f56565','#48bb78'], range_color=[0,100])
# #                             st.plotly_chart(fig4, use_container_width=True)
# #                     with tab3:
# #                         col1, col2 = st.columns(2)
# #                         with col1:
# #                             st.download_button("📥 Download as CSV", data=results_df.to_csv(index=False),
# #                                                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
# #                                                mime="text/csv", use_container_width=True)
# #                         with col2:
# #                             st.download_button("📥 Download as JSON", data=results_df.to_json(orient='records', indent=2),
# #                                                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
# #                                                mime="application/json", use_container_width=True)
# #         except Exception as e:
# #             st.error(f"❌ Error processing file: {str(e)}")
# #     else:
# #         st.markdown("---")
# #         st.markdown("### 📋 CSV Template")
# #         template_data = {
# #             'age': [35, 42, 28], 'gender': ['Male', 'Female', 'Male'],
# #             'city_tier': ['Tier 1', 'Tier 2', 'Tier 3'],
# #             'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
# #             'dependents': [2, 3, 6], 'kyc_verified': ['Yes', 'Yes', 'No'],
# #             'bankruptcy_flag': ['No', 'No', 'No'], 'fraud_flag': ['No', 'No', 'No'],
# #             'rbi_consent': ['Yes', 'Yes', 'Yes'],
# #             'employment_tenure_months': [24, 0, 18], 'business_vintage_years': [0, 5, 0],
# #             'bureau_score': [720, 680, 580], 'dpd_90_count_6m': [0, 1, 2],
# #             'dpd_30_count_6m': [0, 2, 1], 'credit_utilization_pct': [30, 45, 75],
# #             'recent_inquiries_3m': [2, 1, 5], 'active_loans_count': [1, 2, 3],
# #             'avg_salary_6m': [50000, 75000, 35000], 'AMT_INCOME_TOTAL': [600000, 900000, 420000],
# #             'net_cash_surplus_6m': [20000, 35000, 10000],
# #             'salary_stability_flag': ['STABLE', 'MODERATE', 'UNSTABLE'],
# #             'loan_amount': [180000, 250000, 100000], 'loan_tenure_months': [24, 36, 12],
# #             'interest_rate': [10.5, 11.0, 12.0], 'existing_emi': [15000, 20000, 8000],
# #             'AMT_ANNUITY': [8500, 9500, 4500],
# #             'payment_discipline_flag': ['GOOD', 'MODERATE', 'POOR'],
# #             'liquidity_flag': ['LOW', 'ADEQUATE', 'LOW'],
# #             'cashflow_health': ['HEALTHY', 'MODERATE', 'STRESSED'],
# #             'bureau_risk_flag': ['LOW', 'MEDIUM', 'HIGH'],
# #             'inward_bounce_count_3m': [0, 1, 3], 'salary_missing_months': [0, 0, 2],
# #         }
# #         template_df = pd.DataFrame(template_data)
# #         st.dataframe(template_df, use_container_width=True)
# #         st.caption("📝 New columns: `gender`, `city_tier`, `rbi_consent` — required for fairness monitoring and compliance.")
# #         st.download_button("📥 Download CSV Template", data=template_df.to_csv(index=False),
# #                            file_name="credit_assessment_template_v8.7.csv",
# #                            mime="text/csv", use_container_width=True)

# # elif page == "📈 Model Info":
# #     st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
# #     col1, col2, col3 = st.columns(3)
# #     with col1: st.markdown(f'<div class="stat-card"><div class="stat-number">RF</div><div class="stat-label">Model Type</div></div>', unsafe_allow_html=True)
# #     with col2: st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TOP_FEATURES)}</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)
# #     with col3: st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TARGET_LE.classes_)}</div><div class="stat-label">Classes</div></div>', unsafe_allow_html=True)
# #     st.markdown("<br>", unsafe_allow_html=True)
# #     st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
# #     feature_df = pd.DataFrame({'Rank': range(1, min(21, len(TOP_FEATURES)+1)), 'Feature': TOP_FEATURES[:20]})
# #     st.dataframe(feature_df, use_container_width=True, hide_index=True)

# # elif page == "ℹ️ About":
# #     st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
# #     st.markdown("""
# #         <div class="info-card">
# #             <div class="info-card-title">🏦 Credit Risk Assessment Platform</div>
# #             <div class="info-card-content">
# #                 <p><strong>Version:</strong> 8.7 — Dead code removed, all audit fixes applied (M1–M4, D1–D4)</p>
# #                 <p><strong>Developer:</strong> Zen Meraki</p>
# #                 <p><strong>Date:</strong> January 2026</p>
# #                 <br>
# #                 <p>A comprehensive credit risk evaluation system combining hard policy rules,
# #                 machine learning, and affordability analysis for accurate and RBI-compliant lending decisions.</p>
# #             </div>
# #         </div>
# #     """, unsafe_allow_html=True)
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.markdown("""
# #             <div class="info-card">
# #                 <div class="info-card-title">🎯 Key Features</div>
# #                 <div class="info-card-content"><ul style="margin:0;padding-left:1.25rem;">
# #                     <li>Three-layer decision engine</li>
# #                     <li>Real-time risk assessment</li>
# #                     <li>Industry-standard PD calculation</li>
# #                     <li>FOIR calculation & validation</li>
# #                     <li>Automated reason generation</li>
# #                     <li>Complete audit trail (PDF)</li>
# #                     <li>OCR auto-fill with categorical inference</li>
# #                     <li>⚖️ Fairness monitoring dashboard</li>
# #                     <li>🏙️ City Tier field for geographic equity</li>
# #                     <li>📜 RBI consent gate (DLG 2022)</li>
# #                 </ul></div>
# #             </div>
# #         """, unsafe_allow_html=True)
# #     with col2:
# #         st.markdown("""
# #             <div class="info-card">
# #                 <div class="info-card-title">🛠️ Technology Stack</div>
# #                 <div class="info-card-content"><ul style="margin:0;padding-left:1.25rem;">
# #                     <li>Streamlit (UI Framework)</li>
# #                     <li>Scikit-learn (ML)</li>
# #                     <li>Plotly (Visualizations)</li>
# #                     <li>Pandas (Data Processing)</li>
# #                     <li>ReportLab (PDF Generation)</li>
# #                     <li>Tesseract OCR + pdf2image</li>
# #                     <li>Python 3.8+</li>
# #                 </ul></div>
# #             </div>
# #         """, unsafe_allow_html=True)









# """
# Credit Risk Assessment Dashboard - Sage Green & Yellow Theme
# Enhanced with Modern UI/UX Design
# Run with: streamlit run app.py (from inside the notebooks folder)
# Author: Zen Meraki
# Date: March 2026
# VERSION: 8.7 - Renamed from test.py, dead code removed, all audit fixes applied (C1/H1/H2/M1/M2/M3/L1/L2/L3)
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
#     # FIX A-2: CURRENT_DIR is the notebooks/ folder where stage2_engine.py lives.
#     # It was already present but listed alongside PROJECT_ROOT without emphasis.
#     # Adding it first and also adding CURRENT_DIR / "utils" ensures both
#     # stage2_engine.py and utils/pdf_generator.py are importable on Streamlit Cloud
#     # regardless of the working directory at launch time.
#     CURRENT_DIR,                          # notebooks/  ← stage2_engine.py lives here
#     CURRENT_DIR / "utils",               # notebooks/utils/  (if utils is nested)
#     PROJECT_ROOT,
#     PROJECT_ROOT / "loan",
#     PROJECT_ROOT / "utils",              # credit_risk_engine/utils/  ← pdf_generator etc.
#     PROJECT_ROOT / "notebooks",
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
# # FIX A-1: Use explicit try/except import blocks instead of a single-path import.
# # Tries utils.pdf_generator first (standard install), then bare pdf_generator
# # (notebooks/ deployment). Sets PDF_AVAILABLE=False and shows a visible warning
# # in the UI if neither path works, so users know PDF download will be disabled.
# # =============================================================================
# PDF_AVAILABLE = False
# generate_decision_pdf = None
# generate_audit_pdf = None
# try:
#     from utils.pdf_generator import generate_decision_pdf, generate_audit_pdf
#     PDF_AVAILABLE = True
# except ImportError:
#     try:
#         from pdf_generator import generate_decision_pdf, generate_audit_pdf
#         PDF_AVAILABLE = True
#     except ImportError:
#         PDF_AVAILABLE = False  # UI will show warning — see A-4 note in pdf download buttons

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
#     # FIX 2: raised ceiling from 25% to 50%.
#     # Previous cap of 25% meant fraud+bankruptcy showed identical PD to a clean
#     # 550-score borrower. Raw PDs for REJECT cases reach 124% before clamping;
#     # 4.2% of rejects exceeded the old cap. 50% preserves discrimination in the
#     # high-risk tail while staying within practical underwriting display ranges.
#     return round(max(0.5, min(final_pd, 50.0)), 2)

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
#         surplus_for_return = surplus  # FIX L2: assign in both branches — was missing here, causing latent bug if bureau_only path is extended
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
# def _re_int(pattern, text, default, lo=None, hi=None):
#     """Safe regex → int extraction with optional range clamp."""
#     m = re.search(pattern, text, re.IGNORECASE)
#     if m:
#         try:
#             v = int(str(m.group(1)).replace(',', '').replace(' ', ''))
#             if lo is not None and v < lo: return default
#             if hi is not None and v > hi: return default
#             return v
#         except Exception: pass
#     return default

# def _re_float(pattern, text, default, lo=None, hi=None):
#     """Safe regex → float extraction with optional range clamp."""
#     m = re.search(pattern, text, re.IGNORECASE)
#     if m:
#         try:
#             v = float(str(m.group(1)).replace(',', '').replace(' ', ''))
#             if lo is not None and v < lo: return default
#             if hi is not None and v > hi: return default
#             return v
#         except Exception: pass
#     return default

# def extract_cibil_from_pdf(uploaded_file):
#     if not OCR_AVAILABLE:
#         return {'success': False, 'error': OCR_ERROR_MSG or 'OCR libraries not installed.'}
#     try:
#         # ── 1. OCR: PDF → full text ──────────────────────────────────────────
#         pdf_bytes = uploaded_file.read()
#         images    = convert_from_bytes(pdf_bytes, dpi=300)
#         full_text = ""
#         for image in images:
#             gray        = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
#             _, binary   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             full_text  += pytesseract.image_to_string(binary) + "\n"
#         txt = full_text   # shorthand

#         # ── 2. CREDIT SCORE (Bureau / CIBIL score) ───────────────────────────
#         credit_score = 720
#         for pat in [
#             r'\b(8[0-9]{2}|7[0-9]{2}|6[0-9]{2}|[3-5][0-9]{2})\s*(?:EXCELLENT|VERY\s*GOOD|GOOD|FAIR|SUBPRIME|POOR|NH|NA)\b',
#             r'(?:cibil|credit|bureau)\s*score\s*[:\-\(]?\s*(\d{3})',
#             r'score[^\n\r]{0,40}?(\d{3})',
#         ]:
#             m = re.search(pat, txt, re.IGNORECASE)
#             if m:
#                 v = int(m.group(1))
#                 if 300 <= v <= 900:
#                     credit_score = v; break

#         # ── 3. PERSONAL INFO ────────────────────────────────────────────────
#         # Age via DOB
#         age_extracted = 35
#         for dob_pat in [
#             r'(?:date\s+of\s+birth|dob|d\.o\.b)[\s:\-]+(\d{2}[-/]\w{3,9}[-/]\d{2,4})',
#             r'(?:date\s+of\s+birth|dob)[\s:\-]+(\d{2}[-/]\d{2}[-/]\d{4})',
#             r'born[\s:]+(\d{2}[-/]\w{3,9}[-/]\d{4})',
#         ]:
#             m = re.search(dob_pat, txt, re.IGNORECASE)
#             if m:
#                 for fmt in ('%d-%b-%Y','%d/%b/%Y','%d-%b-%y','%d-%m-%Y','%d/%m/%Y'):
#                     try:
#                         dob = datetime.strptime(m.group(1), fmt)
#                         age_extracted = int((datetime.now() - dob).days / 365.25)
#                         break
#                     except Exception: continue
#                 if age_extracted != 35: break
#         # fallback: age stated directly
#         if age_extracted == 35:
#             age_extracted = _re_int(r'(?:^|\s)age[\s:\-]+(\d{2})\b', txt, 35, lo=18, hi=75)

#         # Gender
#         if re.search(r'\bfemale\b|\bF\b', txt, re.IGNORECASE):
#             gender = 'F'
#         elif re.search(r'\bmale\b|\bM\b', txt, re.IGNORECASE):
#             gender = 'M'
#         else:
#             gender = 'M'

#         # Marital status
#         if re.search(r'\bsingle\b|\bunmarried\b', txt, re.IGNORECASE):
#             marital_status = 'Single'
#         else:
#             marital_status = 'Married'

#         # Education
#         education = 'GRADUATE'
#         for pat, val in [
#             (r'post.?grad(uate)?|m\.?tech|mba|mca',    'POST-GRADUATE'),
#             (r'professional|ca\b|cs\b|icai',             'PROFESSIONAL'),
#             (r'\b12th\b|\bhsc\b|\binter(mediate)?\b',   '12TH'),
#             (r'\bssc\b|\b10th\b|\bmatric',               'SSC'),
#             (r'under.?grad(uate)?',                      'UNDER GRADUATE'),
#             (r'\bgrad(uate)?\b|\bb\.?tech\b|\bb\.?e\b|\bb\.?sc\b|\bb\.?com\b', 'GRADUATE'),
#         ]:
#             if re.search(pat, txt, re.IGNORECASE): education = val; break

#         # ── 4. INCOME & EMPLOYMENT ──────────────────────────────────────────
#         monthly_income = 50000
#         for inc_pat in [
#             r'net\s+monthly\s+income[\s:\-₹Rs\.]*([0-9,]+)',
#             r'monthly\s+(?:take.?home|salary|income)[\s:\-₹Rs\.]*([0-9,]+)',
#             r'(?:salary|income)\s+per\s+month[\s:\-₹Rs\.]*([0-9,]+)',
#             r'₹\s*([0-9,]+)\s+(?:per\s+month|p\.?m\.?|monthly)',
#         ]:
#             m = re.search(inc_pat, txt, re.IGNORECASE)
#             if m:
#                 v = int(m.group(1).replace(',',''))
#                 if 5000 < v < 5_000_000:
#                     monthly_income = v; break

#         # Employment type
#         employment_type = 'Salaried'
#         if re.search(r'self.?employed|self employ|proprietor|freelance', txt, re.IGNORECASE):
#             employment_type = 'Self-Employed'
#         elif re.search(r'\bbusiness\b|\bfirm\b|\bpartner(ship)?\b', txt, re.IGNORECASE):
#             employment_type = 'Business'

#         # Employment tenure (months)
#         employment_tenure_months = 36
#         m = re.search(r'(?:with\s+current\s+employer|employment\s+tenure|employed\s+(?:since|for))[^\d]{0,20}(\d+)\s*(?:year|yr)', txt, re.IGNORECASE)
#         if m:
#             employment_tenure_months = int(m.group(1)) * 12
#         else:
#             m = re.search(r'(?:with\s+current\s+employer|tenure)[^\d]{0,20}(\d+)\s*(?:month|mth)', txt, re.IGNORECASE)
#             if m: employment_tenure_months = int(m.group(1))

#         # Existing EMI
#         existing_emi = 0
#         for emi_pat in [
#             r'(?:total\s+emi|existing\s+emi|current\s+emi|monthly\s+emi)[^\d₹]{0,20}[₹Rs\.]*\s*([0-9,]+)',
#             r'emi\s+(?:outflow|obligation)[^\d]{0,20}([0-9,]+)',
#             r'amt_annuity[\s:\-]+([0-9,]+)',
#         ]:
#             m = re.search(emi_pat, txt, re.IGNORECASE)
#             if m:
#                 v = int(m.group(1).replace(',',''))
#                 if 500 < v < 500_000:
#                     existing_emi = v; break

#         # Business vintage
#         business_vintage = 0
#         m = re.search(r'(?:business\s+(?:since|established|vintage|age|started))[^\d]{0,20}(\d+)\s*(?:year|yr)', txt, re.IGNORECASE)
#         if m: business_vintage = int(m.group(1))

#         # ── 5. CREDIT UTILISATION ───────────────────────────────────────────
#         cc_util_pct = -99999   # -99999 = no CC (like CIBIL dataset convention)
#         m = re.search(r'(?:credit\s+card\s+utiliz[ao]tion|cc\s+utiliz[ao]tion|utiliz[ao]tion\s+ratio)[^\d]{0,20}(\d{1,3})\s*%?', txt, re.IGNORECASE)
#         if m:
#             cc_util_pct = int(m.group(1))
#         pl_util = _re_float(r'(?:personal\s+loan\s+utiliz[ao]tion|pl\s+utiliz[ao]tion)[^\d]{0,20}([\d\.]+)', txt, 0.25, lo=0, hi=5)

#         # ── 6. ENQUIRIES ─────────────────────────────────────────────────────
#         # Parse enquiry section for product-wise breakdown
#         enq_section = ""
#         m = re.search(r'enquir(?:y|ies)\s+details(.*?)(?:account\s+summary|$)', txt, re.IGNORECASE | re.DOTALL)
#         if m: enq_section = m.group(1)

#         tot_enq    = _re_int(r'total\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, 0)
#         enq_L12m   = _re_int(r'enquir(?:y|ies)\s*(?:\(?12\s*(?:m(?:on)?(?:th)?s?|M)\)?)?[\s:\-]+(\d+)', txt, 0)
#         enq_L6m    = _re_int(r'enquir(?:y|ies)\s*\(?6\s*(?:m(?:on)?(?:th)?s?|M)\)?[\s:\-]+(\d+)', txt, 0)
#         enq_L3m    = _re_int(r'enquir(?:y|ies)\s*\(?3\s*(?:m(?:on)?(?:th)?s?|M)\)?[\s:\-]+(\d+)', txt, 0)

#         # Count enquiry dates in section as fallback
#         enq_dates = re.findall(r'\b\d{2}-[A-Za-z]{3}-\d{4}\b', enq_section)
#         tot_enq  = max(tot_enq, len(enq_dates))
#         enq_L12m = max(enq_L12m, len(enq_dates))

#         # Product-wise enquiries (CC / PL)
#         CC_enq     = _re_int(r'credit\s+card\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, -99999)
#         CC_enq_L6m = _re_int(r'cc\s+enq(?:uiry|uiries)?\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0 if CC_enq >= 0 else -99999)
#         CC_enq_L12m= _re_int(r'cc\s+enq(?:uiry|uiries)?\s*\(?12m\)?[\s:\-]+(\d+)', txt, 0 if CC_enq >= 0 else -99999)
#         PL_enq     = _re_int(r'personal\s+loan\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, -99999)
#         PL_enq_L6m = _re_int(r'pl\s+enq(?:uiry|uiries)?\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0 if PL_enq >= 0 else -99999)
#         PL_enq_L12m= _re_int(r'pl\s+enq(?:uiry|uiries)?\s*\(?12m\)?[\s:\-]+(\d+)', txt, 0 if PL_enq >= 0 else -99999)

#         # Time since most recent enquiry (days)
#         time_since_recent_enq = _re_int(r'(?:last|recent)\s+enquiry[\s:\-]+(\d+)\s*days?', txt, -99999)
#         if time_since_recent_enq == -99999 and enq_dates:
#             try:
#                 most_recent = max(datetime.strptime(d, '%d-%b-%Y') for d in enq_dates)
#                 time_since_recent_enq = (datetime.now() - most_recent).days
#             except Exception: pass

#         # ── 7. ACCOUNT / DPD PARSING ─────────────────────────────────────────
#         accounts, dpd_all = [], []
#         in_accounts = False
#         for line in txt.split('\n'):
#             lu = line.upper()
#             if 'ACCOUNT DETAILS' in lu or 'LOAN DETAILS' in lu:
#                 in_accounts = True; continue
#             if re.search(r'ENQUIRY\s+DETAILS|SUMMARY|PERSONAL\s+INFO', lu):
#                 in_accounts = False; continue
#             if not in_accounts: continue
#             stripped = line.strip()
#             if not stripped: continue
#             stat_m = re.search(r'\b(Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss)\b', stripped, re.IGNORECASE)
#             dpd_m  = re.search(r'\b(0\d0|0\d\d|\d{3})\b', stripped)
#             if re.search(r'\bINR\b|\bBank\b|\bFinance\b|\bCapital\b|\bNBFC\b', stripped, re.IGNORECASE) or stat_m:
#                 dpd_val = int(dpd_m.group(1)) if dpd_m else 0
#                 status  = (stat_m.group(1) if stat_m else 'Active').lower()
#                 accounts.append({'dpd': dpd_val, 'status': status})
#                 dpd_all.append(dpd_val)

#         # Aggregate DPD counts
#         dpd_90_count = dpd_60_count = dpd_30_count = 0
#         written_off_count = settled_count = active_count = sub_std = 0
#         if accounts:
#             for acc in accounts:
#                 d, s = acc['dpd'], acc['status']
#                 if d >= 90: dpd_90_count += 1
#                 elif d >= 60: dpd_60_count += 1
#                 elif d >= 30: dpd_30_count += 1
#                 if 'written' in s:  written_off_count += 1
#                 elif 'settled' in s: settled_count += 1
#                 elif 'active'  in s: active_count += 1
#                 if d >= 30: sub_std += 1
#         else:
#             # Fallback: keyword scan
#             written_off_count = len(re.findall(r'\bwritten[-\s]?off\b',       txt, re.IGNORECASE))
#             settled_count     = len(re.findall(r'\bsettled\b',                txt, re.IGNORECASE))
#             dpd_90_count      = len(re.findall(r'\b090\b|90\+?\s*dpd',        txt, re.IGNORECASE))
#             dpd_60_count      = len(re.findall(r'\b060\b|60\+?\s*dpd',        txt, re.IGNORECASE))
#             dpd_30_count      = len(re.findall(r'\b030\b|30\+?\s*dpd',        txt, re.IGNORECASE))
#             active_count      = len(re.findall(r'\bactive\b',                 txt, re.IGNORECASE))
#             active_count      = min(active_count, 10)  # cap noise

#         # Standard (num_std) = active performing accounts
#         total_accounts = max(len(accounts), active_count + settled_count + written_off_count, 1)
#         num_std        = active_count
#         pct_active     = active_count / total_accounts

#         # Substandard / doubtful / loss (CIBIL classification)
#         num_sub = sub_std
#         num_dbt = dpd_90_count
#         num_lss = written_off_count

#         # ── 8. DELINQUENCY TIMINGS ───────────────────────────────────────────
#         # CIBIL PDF usually shows months-ago; we convert to days
#         # time_since_recent_payment
#         time_since_recent_payment = _re_int(
#             r'(?:last|recent)\s+payment[\s:\-]+(\d+)\s*days?', txt, -99999)
#         if time_since_recent_payment == -99999:
#             # try "X months ago"
#             m = re.search(r'(?:last|recent)\s+payment[\s:\-]+(\d+)\s*(?:month|mth)', txt, re.IGNORECASE)
#             if m: time_since_recent_payment = int(m.group(1)) * 30

#         time_since_first_deliq  = -99999 if (dpd_30_count + dpd_60_count + dpd_90_count) == 0 else \
#             _re_int(r'first\s+delinquency[\s:\-]+(\d+)\s*days?', txt, 365)
#         time_since_recent_deliq = -99999 if (dpd_30_count + dpd_60_count + dpd_90_count) == 0 else \
#             _re_int(r'(?:last|recent)\s+delinquency[\s:\-]+(\d+)\s*days?', txt, 90)
#         recent_level_of_deliq   = max(
#             dpd_90_count * 90, dpd_60_count * 60, dpd_30_count * 30)

#         # 6-month vs 12-month split
#         num_deliq_6mts   = dpd_30_count + dpd_60_count + dpd_90_count
#         num_deliq_12mts  = num_deliq_6mts   # single source; 12m ≥ 6m
#         num_deliq_6_12mts = 0               # can't distinguish without dates
#         max_deliq_6mts   = -99999 if num_deliq_6mts  == 0 else recent_level_of_deliq
#         max_deliq_12mts  = -99999 if num_deliq_12mts == 0 else recent_level_of_deliq

#         # num_std time splits
#         num_std_6mts  = min(num_std, _re_int(r'standard\s+accounts?\s*\(?6m\)?[\s:\-]+(\d+)', txt, num_std))
#         num_std_12mts = _re_int(r'standard\s+accounts?\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_std)
#         num_sub_6mts  = _re_int(r'sub.?standard\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
#         num_sub_12mts = _re_int(r'sub.?standard\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_sub)
#         num_dbt_6mts  = _re_int(r'doubtful\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
#         num_dbt_12mts = _re_int(r'doubtful\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_dbt)
#         num_lss_6mts  = _re_int(r'loss\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
#         num_lss_12mts = _re_int(r'loss\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_lss)
#         num_times_delinquent = dpd_30_count + dpd_60_count + dpd_90_count
#         num_times_30p_dpd    = dpd_30_count + dpd_60_count + dpd_90_count
#         num_times_60p_dpd    = dpd_60_count + dpd_90_count
#         max_delinquency_level = max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)

#         # ── 9. TRADE-LINE RATIOS (pct_ fields) ──────────────────────────────
#         pct_of_active_TLs_ever     = round(pct_active, 3)
#         pct_opened_TLs_L6m_of_L12m = _re_float(
#             r'(?:opened|new)\s+accounts?\s*\(?6m\s*/\s*12m\)?[\s:\-]+([\d\.]+)', txt, 0.3, lo=0, hi=1)
#         pct_currentBal_all_TL      = _re_float(
#             r'current\s+balance\s+(?:ratio|pct|%)[\s:\-]+([\d\.]+)', txt, 0.3, lo=0, hi=10)
#         pct_PL_enq_L6m_of_L12m    = round(PL_enq_L6m / max(PL_enq_L12m, 1), 2) if PL_enq_L6m >= 0 else 0
#         pct_CC_enq_L6m_of_L12m    = round(CC_enq_L6m / max(CC_enq_L12m, 1), 2) if CC_enq_L6m >= 0 else 0
#         pct_PL_enq_L6m_of_ever    = round(PL_enq_L6m / max(PL_enq if PL_enq >= 0 else 1, 1), 2)
#         pct_CC_enq_L6m_of_ever    = round(CC_enq_L6m / max(CC_enq if CC_enq >= 0 else 1, 1), 2)

#         # ── 10. PRODUCT FLAGS ────────────────────────────────────────────────
#         CC_Flag = 1 if re.search(r'credit\s+card', txt, re.IGNORECASE) else 0
#         PL_Flag = 1 if re.search(r'personal\s+loan', txt, re.IGNORECASE) else 0
#         HL_Flag = 1 if re.search(r'home\s+loan|housing\s+loan', txt, re.IGNORECASE) else 0
#         GL_Flag = 1 if re.search(r'gold\s+loan', txt, re.IGNORECASE) else 0

#         prod_map = {r'personal\s+loan':'PL', r'credit\s+card':'CC',
#                     r'home\s+loan|housing':'HL', r'auto\s+loan|car\s+loan':'AL',
#                     r'gold\s+loan':'GL', r'business\s+loan':'BL'}
#         last_prod = first_prod = 'others'
#         for pat, label in prod_map.items():
#             if re.search(pat, txt, re.IGNORECASE):
#                 last_prod = first_prod = label; break

#         # ── 11. SANITY CHECK: high score vs bad history ──────────────────────
#         if credit_score >= 750 and (written_off_count > 0 or dpd_90_count > 0):
#             credit_score = min(credit_score, 550)

#         # ── 12. NET CASH SURPLUS PROXY ───────────────────────────────────────
#         # Try to extract if stated, else infer
#         net_cash_surplus = _re_int(
#             r'(?:net\s+(?:cash\s+)?surplus|disposable\s+income)[^\d₹]{0,20}[₹Rs\.]*\s*([0-9,]+)', txt, 0)
#         if net_cash_surplus == 0:
#             net_cash_surplus = int(_infer_surplus_from_cibil(credit_score, dpd_60_count, dpd_30_count, float(monthly_income)))

#         # ── 13. INWARD BOUNCE & SALARY STABILITY (60k-specific fields) ───────
#         # These are bank-statement fields; CIBIL PDF won't have them directly.
#         # We infer them from available signals.
#         inward_bounce_count_3m  = dpd_90_count + dpd_60_count      # proxy: each severe DPD → bounce
#         salary_missing_months   = 0                                  # can't determine from CIBIL
#         total_credit_6m         = _re_int(r'total\s+credits?\s*\(?6m\)?[\s:\-₹]*([0-9,]+)', txt, 0)
#         total_debit_6m          = _re_int(r'total\s+debits?\s*\(?6m\)?[\s:\-₹]*([0-9,]+)', txt, 0)

#         # ── 14. STAGE-1 60K DATASET FIELD MAPPING ────────────────────────────
#         # All columns from train_60k_rule_accepted.csv mapped from OCR data
#         s1 = {
#             # Income / salary
#             'AMT_INCOME_TOTAL':          monthly_income * 12,
#             'AMT_ANNUITY':               existing_emi if existing_emi > 0 else int(monthly_income * 0.25),
#             'avg_salary_6m':             float(monthly_income),
#             'salary_txn_count_6m':       6.0,       # assume regular salary
#             'salary_amount_cv':          0.05 if employment_type == 'Salaried' else 0.25,
#             'salary_date_std':           2.0,
#             'salary_creditor_consistent': 1.0 if employment_type == 'Salaried' else 0.7,
#             'salary_missing_months':     float(salary_missing_months),
#             # Delinquency
#             'dpd_15_count_6m':           0.0,
#             'dpd_30_count_6m':           float(dpd_30_count),
#             'dpd_90_count_6m':           float(dpd_90_count),
#             'max_dpd_6m':                float(max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)),
#             'dpd_30_count_3m':           float(dpd_30_count),
#             'total_payments_6m':         0.0,
#             'total_late_15_6m':          0.0,
#             'total_late_30_6m':          float(dpd_30_count),
#             'total_late_60_6m':          float(dpd_60_count),
#             'total_late_90_6m':          float(dpd_90_count),
#             'max_days_late_6m':          float(max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)),
#             'avg_days_late_6m':          float(dpd_30_count * 10 + dpd_60_count * 20 + dpd_90_count * 40) / max(total_accounts, 1),
#             'total_late_30_3m':          float(dpd_30_count),
#             'total_late_90_3m':          float(dpd_90_count),
#             # Credit card
#             'avg_balance_cc':            0.0,
#             'total_drawings_cc':         0.0,
#             'avg_credit_limit':          0.0,
#             'max_utilization':           (cc_util_pct / 100) if cc_util_pct > 0 else 0.0,
#             'total_payments_cc':         0.0,
#             'dpd_count_cc':              0.0,
#             # POS / installment
#             'avg_balance_pos':           0.0,
#             'dpd_count_pos':             0.0,
#             # Aggregate
#             'total_credit_activity':     float(total_accounts),
#             'total_dpd_count':           float(dpd_30_count + dpd_60_count + dpd_90_count),
#             'avg_monthly_balance_6m':    float(net_cash_surplus),
#             'total_emi_monthly':         float(existing_emi if existing_emi > 0 else int(monthly_income * 0.25)),
#             'net_cash_surplus_6m':       float(net_cash_surplus),
#             'total_credit_6m':           float(total_credit_6m),
#             'total_debit_6m':            float(total_debit_6m),
#             # Cashflow
#             'inward_bounce_count_3m':    float(inward_bounce_count_3m),
#             'recent_payment_stress':     float(dpd_30_count + dpd_60_count),
#             # Active loans
#             'active_loans_count':        float(active_count),
#             # Bureau
#             'bureau_score':              float(credit_score),
#             'hard_reject_flag':          1 if (dpd_90_count > 5 or written_off_count > 0 or credit_score < 550) else 0  # DPD90 1-5 = REVIEW not hard reject,
#         }

#         # ── 15. INFERRED CATEGORICAL FLAGS (60k) ─────────────────────────────
#         _inferred = infer_categorical_flags({
#             'Credit_Score': credit_score, 'num_times_30p_dpd': dpd_30_count,
#             'num_times_60p_dpd': dpd_60_count, 'num_lss': num_lss,
#             'num_dbt': num_dbt, 'CC_utilization': cc_util_pct / 100 if cc_util_pct > 0 else 0,
#             'NETMONTHLYINCOME': monthly_income, 'Time_With_Curr_Empr': employment_tenure_months,
#             'dpd_90_count_6m': dpd_90_count, 'inward_bounce_count_3m': inward_bounce_count_3m,
#             'salary_missing_months': salary_missing_months,
#             'net_cash_surplus_6m': net_cash_surplus,
#         })

#         # ── 16. STAGE-2 EXTERNAL CIBIL DATASET FIELD MAPPING ─────────────────
#         # All 62 columns from External_Cibil_Dataset.xlsx
#         s2 = {
#             'Credit_Score':               credit_score,
#             'AGE':                        age_extracted,
#             'GENDER':                     gender,
#             'MARITALSTATUS':              marital_status,
#             'EDUCATION':                  education,
#             'NETMONTHLYINCOME':           monthly_income,
#             'Time_With_Curr_Empr':        employment_tenure_months,
#             # Delinquency counts
#             'num_times_delinquent':       num_times_delinquent,
#             'max_delinquency_level':      max_delinquency_level,
#             'max_recent_level_of_deliq':  max(dpd_60_count*60, dpd_30_count*30),
#             'num_deliq_6mts':             num_deliq_6mts,
#             'num_deliq_12mts':            num_deliq_12mts,
#             'num_deliq_6_12mts':          num_deliq_6_12mts,
#             'max_deliq_6mts':             max_deliq_6mts,
#             'max_deliq_12mts':            max_deliq_12mts,
#             'num_times_30p_dpd':          num_times_30p_dpd,
#             'num_times_60p_dpd':          num_times_60p_dpd,
#             'recent_level_of_deliq':      recent_level_of_deliq,
#             # Standard / substandard / doubtful / loss
#             'num_std':                    num_std,
#             'num_std_6mts':               num_std_6mts,
#             'num_std_12mts':              num_std_12mts,
#             'num_sub':                    num_sub,
#             'num_sub_6mts':               num_sub_6mts,
#             'num_sub_12mts':              num_sub_12mts,
#             'num_dbt':                    num_dbt,
#             'num_dbt_6mts':               num_dbt_6mts,
#             'num_dbt_12mts':              num_dbt_12mts,
#             'num_lss':                    num_lss,
#             'num_lss_6mts':               num_lss_6mts,
#             'num_lss_12mts':              num_lss_12mts,
#             # Timings
#             'time_since_recent_payment':  time_since_recent_payment,
#             'time_since_first_deliquency': time_since_first_deliq,
#             'time_since_recent_deliquency': time_since_recent_deliq,
#             # Enquiries
#             'tot_enq':                    tot_enq,
#             'enq_L3m':                    enq_L3m,
#             'enq_L6m':                    enq_L6m,
#             'enq_L12m':                   enq_L12m,
#             'time_since_recent_enq':      time_since_recent_enq,
#             'CC_enq':                     CC_enq,
#             'CC_enq_L6m':                 CC_enq_L6m,
#             'CC_enq_L12m':                CC_enq_L12m,
#             'PL_enq':                     PL_enq,
#             'PL_enq_L6m':                 PL_enq_L6m,
#             'PL_enq_L12m':                PL_enq_L12m,
#             # Ratios / pct fields
#             'pct_of_active_TLs_ever':     pct_of_active_TLs_ever,
#             'pct_opened_TLs_L6m_of_L12m': pct_opened_TLs_L6m_of_L12m,
#             'pct_currentBal_all_TL':      pct_currentBal_all_TL,
#             'pct_PL_enq_L6m_of_L12m':     pct_PL_enq_L6m_of_L12m,
#             'pct_CC_enq_L6m_of_L12m':     pct_CC_enq_L6m_of_L12m,
#             'pct_PL_enq_L6m_of_ever':     pct_PL_enq_L6m_of_ever,
#             'pct_CC_enq_L6m_of_ever':     pct_CC_enq_L6m_of_ever,
#             # Utilisation
#             'CC_utilization':             cc_util_pct / 100 if cc_util_pct > 0 else -99999,
#             'PL_utilization':             pl_util,
#             'CC_Flag':                    CC_Flag,
#             'PL_Flag':                    PL_Flag,
#             'HL_Flag':                    HL_Flag,
#             'GL_Flag':                    GL_Flag,
#             'max_unsec_exposure_inPct':   cc_util_pct if cc_util_pct > 0 else 0,
#             'last_prod_enq2':             last_prod,
#             'first_prod_enq2':            first_prod,
#         }

#         # ── 17. MERGE AND RETURN ─────────────────────────────────────────────
#         return {
#             **s1, **s2,
#             # Stage-1 form-specific fields
#             'existing_emi':              existing_emi if existing_emi > 0 else s1['total_emi_monthly'],
#             'employment_type':           employment_type,
#             'business_vintage_years':    business_vintage,
#             'credit_utilization_pct':    cc_util_pct if cc_util_pct > 0 else 0,
#             # Inferred categoricals for Stage 1 form dropdowns
#             'salary_stability_flag':     _inferred['salary_stability_flag'],
#             'payment_discipline_flag':   _inferred['payment_discipline_flag'],
#             'cashflow_health':           _inferred['cashflow_health'],
#             'liquidity_flag':            _inferred['liquidity_flag'],
#             'bureau_risk_flag':          _inferred['bureau_risk_flag'],
#             # Computed extra signals
#             'written_off_count':         written_off_count,
#             'settled_count':             settled_count,
#             'high_util_flag':            1 if cc_util_pct > 75 else 0,
#             'recent_deliq_flag':         1 if (dpd_90_count > 0 or dpd_60_count > 0) else 0,
#             'account_quality_score':     max(0, 100 - written_off_count*20 - settled_count*10 - dpd_90_count*15 - dpd_30_count*5),
#             '_surplus_proxy':            int(net_cash_surplus),
#             # Passthrough for UI display / audit
#             'raw_text':                  full_text,
#             'success':                   True,
#             'extraction_method':         'OCR+FullDatasetMapping_v2',
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

#     NOTE A-3 — risk_score scale:
#       source='stage1' or 'batch' → risk_score is on 0-100 (Stage 1 engine output).
#       source='stage2'            → risk_score is the combined_risk_score on 0-1000
#                                    (Stage 1 normalised + Stage 2 tier, see stage2_engine.py).
#     The fairness dashboard currently uses risk_score only for the 'Avg Risk Score' summary
#     column. If cross-source comparisons are needed, normalise to a common scale first.
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
#     # D3 FIX: cap at 1000 entries to prevent unbounded memory growth per session
#     if len(st.session_state.fairness_log) > 1000:
#         st.session_state.fairness_log = st.session_state.fairness_log[-1000:]

# # =============================================================================
# # STAGE 2 BINARY RESOLVER  (defined early — called from page routing below)
# # =============================================================================
# def resolve_stage2_to_binary(stage2_result: dict) -> dict:
#     """
#     Normalise Stage 2 result to a binary APPROVE / REJECT decision.
#     REVIEW outcomes are resolved via tier mapping; score is used as tie-breaker.
#     Defined here (before page routing) so it is always in scope regardless of
#     which section of the file Streamlit is executing.
#     """
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

#     # AGE POLICY GATE — split by employment type per spec
#     # UI allows 18–70 for input flexibility, but policy enforces:
#     #   - All types:       age must be > 24  (≤ 24 → too young)
#     #   - Salaried:        age must be ≤ 65  (retirement risk)
#     #   - Self-Employed / Business: age must be ≤ 70
#     _is_salaried = employment_type == 'Salaried'
#     _max_age     = 65 if _is_salaried else 70
#     _age_label   = "24–65 for Salaried" if _is_salaried else "24–70 for Self-Employed/Business"
#     if age <= 24:
#         policy_checks['age'] = f"❌ Age {age} — Too young (Min: 25)"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Applicant too young (minimum age 25)", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 70.0, 'affordability_data': {}}
#     if age > _max_age:
#         policy_checks['age'] = f"❌ Age {age} — Exceeds max ({_age_label})"
#         return {'decision': "REJECT", 'reason': f"Policy Gate: Age exceeds maximum for {employment_type} ({_max_age})", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 70.0, 'affordability_data': {}}
#     policy_checks['age'] = f"✅ Age {age} (Valid — {_age_label})"

#     if not kyc_verified:
#         policy_checks['kyc'] = "❌ KYC Not Verified"
#         return {'decision': "REJECT", 'reason': "Policy Gate: KYC verification required", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 70.0, 'affordability_data': {}}
#     policy_checks['kyc'] = "✅ KYC Verified"

#     if not customer_dict.get('rbi_consent', False):
#         policy_checks['rbi_consent'] = "❌ RBI Consent not obtained"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Customer consent not obtained", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 70.0, 'affordability_data': {}}
#     policy_checks['rbi_consent'] = "✅ Consent Obtained"

#     if bankruptcy_flag:
#         policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Active bankruptcy", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 95.0, 'affordability_data': {}}
#     policy_checks['bankruptcy'] = "✅ No Bankruptcy"

#     if fraud_flag:
#         policy_checks['fraud'] = "❌ Fraud Flag"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Fraud detected", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 95.0, 'affordability_data': {}}
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
#                 'pd_percentage': 72.0, 'affordability_data': {}}
#     policy_checks['income'] = f"✅ Income ₹{monthly_income:,.0f}"

#     if employment_type == 'Salaried' and employment_tenure < 6:
#         policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient tenure", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 72.0, 'affordability_data': {}}
#     elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
#         policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient business vintage", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 72.0, 'affordability_data': {}}
#     policy_checks['tenure'] = (f"✅ Tenure {employment_tenure} months" if employment_type == 'Salaried'
#                                 else f"✅ Business Vintage {business_vintage} years")

#     bureau_score = customer_dict.get('bureau_score', 0)
#     # FIX 1: round DPD counts — synthetic data generator applied Gaussian jitter
#     # (×normal(1, 0.005)) to all numeric columns, producing float values like 0.9904,
#     # 1.982, 6.94 instead of integers 0, 1, 2. Without rounding, dpd_90 > 0 fires
#     # on 0.9904 (should be 0) and the tiered gate logic is wrong for every noisy row.
#     dpd_90 = int(round(float(customer_dict.get('dpd_90_count_6m', 0) or 0)))
#     credit_utilization = customer_dict.get('credit_utilization_pct', 0)
#     recent_inquiries = customer_dict.get('recent_inquiries_3m', 0)

#     if bureau_score < 550:
#         policy_checks['bureau'] = f"❌ Bureau Score {bureau_score} (Min: 550)"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Bureau score too low", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 82.0, 'affordability_data': {}}
#     policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"

#     # DPD90 TIERED GATE:
#     #   0     -> PASS (clean)
#     #   1-5   -> REVIEW flag (elevated risk, underwriter required)
#     #   > 5   -> REJECT (severe delinquency, hard stop)
#     dpd_90_review_flag = False
#     # DESIGN NOTE (M2): DPD90 gate is tiered — >5 = hard REJECT, 1-5 = REVIEW flag.
#     # Legacy calculate_risk_score() (fallback-only) uses softer penalty for DPD90=1;
#     # that path is NEVER reached in production. This gate is the intended behavior.
#     if dpd_90 > 5:
#         policy_checks['dpd'] = f"❌ {dpd_90} instance(s) of 90+ DPD — Hard Reject (Max: 5)"
#         return {'decision': "REJECT", 'reason': "Policy Gate: Severe delinquency (90+ DPD > 5)", 'confidence': 0,
#                 'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
#                 'pd_percentage': 88.0, 'affordability_data': {}}
#     elif dpd_90 >= 1:
#         dpd_90_review_flag = True
#         policy_checks['dpd'] = f"⚠️ {dpd_90} instance(s) of 90+ DPD — Underwriter Review Required"
#     else:
#         policy_checks['dpd'] = "✅ No 90+ DPD (Clean)"
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
#     if dpd_90_review_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"  # DPD90 1-5 forces review

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
#             customer_dict['ml_confidence'] = decision_data.get('confidence', 0)
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
#     _pd_color = '#48bb78' if pd_score < 5 else ('#ed8936' if pd_score < 10 else '#f56565')
#     _pd_label = 'Low Risk' if pd_score < 5 else ('Moderate Risk' if pd_score < 10 else 'High Risk')
#     with col2: st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{_pd_color}">{pd_score}%</div><div class="stat-label">PD Score</div><div style="font-size:11px;color:{_pd_color};font-weight:600">{_pd_label}</div></div>', unsafe_allow_html=True)
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

#     # Update session state — Stage 2 is the binding final decision
#     st.session_state['stage2_final_decision'] = final_decision

#     if final_decision == "APPROVE":
#         st.markdown(
#             '<div class="decision-card decision-card-approved" style="padding:2.5rem;">'
#             '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">✔  APPROVED</div>'
#             '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">✅ STAGE 2 FINAL DECISION — Proceed to Disbursement</div>'
#             '</div>', unsafe_allow_html=True)
#     elif final_decision == "REJECT":
#         st.markdown(
#             '<div class="decision-card decision-card-rejected" style="padding:2.5rem;">'
#             '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">✘  REJECTED</div>'
#             '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">❌ STAGE 2 FINAL DECISION — Application Declined</div>'
#             '</div>', unsafe_allow_html=True)
#     else:
#         st.markdown(
#             '<div class="decision-card decision-card-review" style="padding:2.5rem;">'
#             '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">⚑  REVIEW</div>'
#             '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">⚠️ STAGE 2 FINAL DECISION — Requires Manual Credit Officer Review</div>'
#             '</div>', unsafe_allow_html=True)

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
#                     # Top-level PD — used by audit header (must match pd_calculation_factors.final_pd)
#                     'pd_percentage':              _final_pd,
#                     'risk_score':                 _safe(combined_risk, 0),
#                     'confidence':                 _safe(stage2_confidence, 0),
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
#             st.warning("⚠️ PDF generation is not available. Ensure utils/pdf_generator.py is present and `reportlab` is installed (add to requirements.txt).")

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
#             <div class="data-row"><span class="data-label">Version</span><span class="data-value">8.7</span></div>
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
#     with col4: st.metric("🔄 Version", "8.7", "Latest")
#     st.markdown("<br>", unsafe_allow_html=True)
#     st.markdown("""
#         <div class="warning-box" style="background:#f0fff4;border:1px solid #9ae6b4;padding:1rem;border-radius:0.5rem;">
#             <strong>🆕 New in Version 8.7:</strong><br>
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
#                         # ── Stage 1: 60k dataset field autofill ──────────────
#                         st.session_state.pdf_age               = int(extraction_result.get('AGE', 35))
#                         st.session_state.pdf_bureau_score      = int(extraction_result.get('Credit_Score', 720))
#                         st.session_state.pdf_dpd_90            = int(extraction_result.get('dpd_90_count_6m', 0))
#                         st.session_state.pdf_dpd_30            = int(extraction_result.get('dpd_30_count_6m', 0))
#                         _cc_util_raw = extraction_result.get('CC_utilization', 0) or 0
#                         st.session_state.pdf_credit_util       = int(max(0, float(_cc_util_raw)) * 100) if _cc_util_raw > 0 else 0
#                         st.session_state.pdf_inquiries         = int(extraction_result.get('enq_L3m', 2))
#                         st.session_state.pdf_active_loans      = int(extraction_result.get('num_std', 1))
#                         _emi = int(extraction_result.get('existing_emi') or extraction_result.get('total_emi_monthly') or 0)
#                         st.session_state.pdf_existing_emi      = _emi
#                         _income = int(extraction_result.get('NETMONTHLYINCOME') or extraction_result.get('avg_salary_6m') or 50000)
#                         st.session_state.pdf_monthly_income    = _income
#                         st.session_state.pdf_annual_income     = int(extraction_result.get('AMT_INCOME_TOTAL') or _income * 12)
#                         _surplus = int(extraction_result.get('net_cash_surplus_6m') or extraction_result.get('_surplus_proxy') or 0)
#                         st.session_state.pdf_net_surplus       = _surplus
#                         st.session_state.pdf_employment_tenure = int(extraction_result.get('Time_With_Curr_Empr', 24))
#                         # Employment type (new — was never filled before)
#                         _emp = extraction_result.get('employment_type', 'Salaried')
#                         if _emp in ['Salaried', 'Self-Employed', 'Business']:
#                             st.session_state.pdf_employment_type = _emp
#                         # Business vintage (new)
#                         st.session_state.pdf_business_vintage  = int(extraction_result.get('business_vintage_years', 0))
#                         # Gender (new — was extracted but never applied to form)
#                         _g = extraction_result.get('GENDER', 'M')
#                         st.session_state.pdf_gender = 'Male' if _g == 'M' else 'Female'
#                         # Dependents: CIBIL PDFs rarely state this; leave at form default
#                         # Inward bounce & missing salary (inferred from delinquency)
#                         st.session_state.pdf_inward_bounce     = int(extraction_result.get('inward_bounce_count_3m', 0))
#                         st.session_state.pdf_salary_missing    = int(extraction_result.get('salary_missing_months', 0))
#                         # Categorical flags (now come directly from extraction, no second infer needed)
#                         st.session_state.pdf_salary_stability   = extraction_result.get('salary_stability_flag', 'MODERATE')
#                         st.session_state.pdf_payment_discipline = extraction_result.get('payment_discipline_flag', 'GOOD')
#                         st.session_state.pdf_cashflow_health    = extraction_result.get('cashflow_health', 'MODERATE')
#                         st.session_state.pdf_liquidity_flag     = extraction_result.get('liquidity_flag', 'MODERATE')
#                         st.session_state.pdf_bureau_risk_flag   = extraction_result.get('bureau_risk_flag', 'MODERATE')
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
#             age = st.number_input("Age", 25, 70, value=int(st.session_state.get('pdf_age', 35)), help="Min 25 per RBI lending policy")
#             employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business'],
#                 index=['Salaried','Self-Employed','Business'].index(st.session_state.get('pdf_employment_type','Salaried')))
#         with col2:
#             _gender_opts = ['Male', 'Female', 'Non-binary / Other', 'Prefer not to say']
#             _gender_default = st.session_state.get('pdf_gender', 'Male')
#             _gender_idx = _gender_opts.index(_gender_default) if _gender_default in _gender_opts else 0
#             gender = st.selectbox("Gender", _gender_opts, index=_gender_idx)
#             dependents = st.number_input("Number of Dependents", 0, 20, value=int(st.session_state.get('pdf_dependents', 2)))
#         with col3:
#             # City Tier — field for fairness monitoring.
#             # FIX A-6: Use format_func so the selectbox displays the full label to the user
#             # but city_tier is derived immediately from CITY_TIERS at render time —
#             # no deferred lookup needed. A caption confirms the stored code.
#             _city_keys = list(CITY_TIERS.keys())
#             city_tier_label = st.selectbox(
#                 "City Tier", _city_keys, index=0,
#                 format_func=lambda k: k  # full descriptive label shown to user
#             )
#             city_tier = CITY_TIERS[city_tier_label]   # short code: 'Tier 1' / 'Tier 2' / etc.
#             st.caption(f"Stored as: **{city_tier}**")
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
#             inward_bounce_count   = st.number_input("Inward Bounce Count (3M)", 0, 10, value=int(st.session_state.get('pdf_inward_bounce', 0)))
#             salary_missing_months = st.number_input("Missing Salary Months (6M)", 0, 6, value=int(st.session_state.get('pdf_salary_missing', 0)))

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

#         # Inject ML confidence so reason_codes.py can distinguish ML-driven REVIEW
#         customer_data['ml_confidence'] = decision_data.get('confidence', 0)
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
#             foir      = affordability.get('foir_percentage', 0)
#             total_emi = int(round(affordability.get('total_emi', 0)))
#             net_disp  = int(round(affordability.get('net_disposable', 0)))

#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 render_info_card("Identity & Eligibility", "👤",
#                     {f"Age: {age}": "", f"Employment: {employment_type}": "",
#                      f"City Tier: {city_tier}": "", f"Dependents: {dependents}": "",
#                      f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""},
#                     {f"Age: {age}": "pass" if (age > 24 and age <= (65 if employment_type == 'Salaried' else 70)) else "fail",
#                      f"Employment: {employment_type}": "pass",
#                      f"City Tier: {city_tier}": "pass",
#                      f"Dependents: {dependents}": "pass" if dependents <= 5 else "warning",
#                      f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_verified else "fail"})
#             with col2:
#                 render_info_card("Credit Bureau", "🏦",
#                     {f"Bureau Score: {bureau_score}": "", f"DPD 90+: {dpd_90_6m}": "",
#                      f"Utilization: {credit_utilization}%": ""},
#                     {f"Bureau Score: {bureau_score}": "pass" if bureau_score >= 550 else "fail",
#                      f"DPD 90+: {dpd_90_6m}": "pass" if dpd_90_6m == 0 else ("warning" if dpd_90_6m <= 5 else "fail"),
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
#                     st.warning("⚠️ PDF download unavailable — `reportlab` or `pdf_generator.py` not found. Check requirements.txt.")
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
#                 'model_version': '8.7',
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
#                     st.warning("⚠️ Audit PDF download unavailable — `reportlab` or `pdf_generator.py` not found. Check requirements.txt.")
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
#                 age_cibil = st.number_input("Age", 25, 70, int(stage1_customer.get('age', 35)), help="Min 25 per RBI lending policy")
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

#                         # ── Summary metrics ──────────────────────────────────
#                         c1, c2, c3, c4 = st.columns(4)
#                         c1.metric("Credit Score",    extraction_result.get('Credit_Score', 'N/A'))
#                         c2.metric("DPD 30+ Count",   extraction_result.get('num_times_30p_dpd', 0))
#                         c3.metric("DPD 60+ Count",   extraction_result.get('num_times_60p_dpd', 0))
#                         c4.metric("Active Accounts", extraction_result.get('num_std', 0))
#                         c1, c2, c3, c4 = st.columns(4)
#                         c1.metric("Monthly Income", f"₹{extraction_result.get('NETMONTHLYINCOME', 0):,}")
#                         c2.metric("Employment Tenure", f"{extraction_result.get('Time_With_Curr_Empr',0)} mo")
#                         c3.metric("Written Off",    extraction_result.get('num_lss', 0))
#                         c4.metric("Enquiries (3M)", extraction_result.get('enq_L3m', 0))
#                         c1, c2, c3, c4 = st.columns(4)
#                         c1.metric("Payment Discipline", extraction_result.get('payment_discipline_flag','—'))
#                         c2.metric("Cashflow Health",    extraction_result.get('cashflow_health','—'))
#                         c3.metric("Bureau Risk",        extraction_result.get('bureau_risk_flag','—'))
#                         c4.metric("Salary Stability",   extraction_result.get('salary_stability_flag','—'))

#                         if extraction_result.get('written_off_count', 0) > 0:
#                             st.warning(f"⚠️ {extraction_result['written_off_count']} written-off accounts detected — score may be overridden.")

#                         _surplus_proxy = extraction_result.get('_surplus_proxy', 0)
#                         if _surplus_proxy:
#                             st.info(f"💡 Bureau-only PDF — net surplus estimated from income: ₹{_surplus_proxy:,}")

#                         with st.expander("📋 View all extracted fields"):
#                             _display = {k: v for k, v in extraction_result.items() if k not in ('raw_text','success','extraction_method')}
#                             st.json(_display)

#                         # ── Build enhanced_customer_data ─────────────────────
#                         # Start from Stage 1 customer (has gender, city_tier, rbi_consent, loan details)
#                         enhanced_customer_data = stage1_customer.copy()

#                         # Apply ALL extracted fields directly — the new extractor maps every column
#                         _skip = {'raw_text', 'success', 'extraction_method',
#                                  'loan_amount', 'loan_tenure_months', 'interest_rate',
#                                  'rbi_consent', 'kyc_verified', 'bankruptcy_flag', 'fraud_flag'}
#                         for k, v in extraction_result.items():
#                             if k not in _skip and v is not None:
#                                 enhanced_customer_data[k] = v

#                         # Income safety: if CIBIL income << Stage 1 application income, keep Stage 1
#                         _s1_inc = stage1_customer.get('avg_salary_6m', 50000)
#                         _s2_inc = extraction_result.get('NETMONTHLYINCOME', 0) or 0
#                         if 0 < _s2_inc < _s1_inc * 0.4:
#                             enhanced_customer_data['avg_salary_6m'] = _s1_inc
#                             enhanced_customer_data['AMT_INCOME_TOTAL'] = _s1_inc * 12
#                             st.warning(f"⚠️ CIBIL income ₹{_s2_inc:,} << application income ₹{_s1_inc:,} — using application income for FOIR.")

#                         # Sentinel cleanup
#                         enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)

#                         with st.spinner("🔬 Running Stage 2 analysis..."):
#                             try:
#                                 stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
#                                 stage2_result = resolve_stage2_to_binary(stage2_result)
#                                 display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
#                             except Exception as e:
#                                 st.error(f"❌ Analysis failed: {str(e)}")
#                                 st.exception(e)
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
#                            file_name="credit_assessment_template_v8.7.csv",
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
#                 <p><strong>Version:</strong> 8.7 — Dead code removed, all audit fixes applied (M1–M4, D1–D4)</p>
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
Run with: streamlit run app.py (from inside the notebooks folder)
Author: Zen Meraki
Date: March 2026
VERSION: 8.7 - Renamed from test.py, dead code removed, all audit fixes applied (C1/H1/H2/M1/M2/M3/L1/L2/L3)
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
import base64
from typing import List, Any
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
    # FIX A-2: CURRENT_DIR is the notebooks/ folder where stage2_engine.py lives.
    # It was already present but listed alongside PROJECT_ROOT without emphasis.
    # Adding it first and also adding CURRENT_DIR / "utils" ensures both
    # stage2_engine.py and utils/pdf_generator.py are importable on Streamlit Cloud
    # regardless of the working directory at launch time.
    CURRENT_DIR,                          # notebooks/  ← stage2_engine.py lives here
    CURRENT_DIR / "utils",               # notebooks/utils/  (if utils is nested)
    PROJECT_ROOT,
    PROJECT_ROOT / "loan",
    PROJECT_ROOT / "utils",              # credit_risk_engine/utils/  ← pdf_generator etc.
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
        .stat-card { background: white; padding: 1rem; border-radius: 0.5rem;
                     box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }
        .stat-number { font-size: 1.8rem; font-weight: bold; color: #2d3748; }
        .stat-label { font-size: 0.875rem; color: #718096; }
        .info-card { background: white; border-radius: 0.5rem; padding: 1rem;
                     margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .info-card-title { font-weight: 600; margin-bottom: 0.5rem; color: #2d3748; }
        .info-card-content { font-size: 0.875rem; }
        .data-row { display: flex; justify-content: space-between;
                    padding: 0.25rem 0; border-bottom: 1px solid #e2e8f0; }
        .data-label { color: #4a5568; }
        .data-value { font-weight: 500; }
        .status-badge { padding: 0.25rem 0.5rem; border-radius: 0.25rem;
                        font-size: 0.75rem; margin-left: 0.5rem; }
        .badge-pass { background: #c6f6d5; color: #22543d; }
        .badge-fail { background: #fed7d7; color: #742a2a; }
        .badge-warning { background: #feebc8; color: #744210; }
        .reason-item { padding: 0.25rem 0; }
        .reason-icon { color: #587042; font-weight: bold; margin-right: 0.5rem; }
    </style>
    """
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# CITY TIER MAPPING
# =============================================================================
CITY_TIERS = {
    "Tier 1 – Metro (Mumbai, Delhi, Bengaluru, Chennai, Hyderabad, Kolkata, Pune, Ahmedabad)": "Tier 1",
    "Tier 2 – Large City (Jaipur, Lucknow, Kochi, Nagpur, Indore, Bhopal, Patna, Vadodara…)": "Tier 2",
    "Tier 3 – Small City / Town": "Tier 3",
    "Rural / Village": "Rural",
}

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    defaults = {
        'stage1_complete':       False,
        'stage1_decision':       None,
        'stage1_data':           None,
        'current_customer_data': None,
        'page_navigation':       "🏠 Home",
        'use_two_stage':         False,
        'stage2_selected_tab':   "Manual Entry",
        # Fairness log — persists across sessions in memory
        'fairness_log':          [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# =============================================================================
# IMPORT BUSINESS LOGIC MODULES
# =============================================================================
try:
    from affordability_engine import calculate_emi, calculate_affordability
    from reason_codes import generate_reason_codes
    from risk_engine import (
        calculate_final_risk_score, fill_missing_ml_fields,
        clean_sentinel_values
    )
    from affordability_engine import check_net_disposable
except ImportError as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.info("""
    Required files (place in notebooks/, loan/, utils/, or project root):
    - affordability_engine.py  |  reason_codes.py  |  risk_engine.py
    - utils/__init__.py  |  utils/pdf_generator.py
    """)
    st.stop()

# OCR module — imported SEPARATELY so a missing cv2/pytesseract does not
# crash the whole app. If the import fails, OCR_AVAILABLE stays False and
# the upload widgets show a clear error message instead.
try:
    from ocr_extractor import extract_cibil_from_pdf, infer_categorical_flags
except ImportError as _ocr_import_err:
    OCR_AVAILABLE = False
    OCR_ERROR_MSG = (f"OCR module import failed: {_ocr_import_err}. "
                     "Ensure ocr_extractor.py is in utils/ and cv2/pytesseract are installed.")
    # Provide no-op fallbacks so the rest of the app doesn't crash
    def extract_cibil_from_pdf(_f):
        return {'success': False, 'error': OCR_ERROR_MSG}
    def infer_categorical_flags(_d):
        return {'payment_discipline_flag': 'MODERATE', 'cashflow_health': 'MODERATE',
                'liquidity_flag': 'MODERATE', 'bureau_risk_flag': 'MEDIUM',
                'salary_stability_flag': 'MODERATE', '_inference_path': 'fallback'}

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
    def is_stage2_available(): return False
    def get_stage2_status(): return {"error": "Stage 2 engine module not found", "available": False}

# =============================================================================
# PDF GENERATION – SAFE FALLBACK
# FIX A-1: Use explicit try/except import blocks instead of a single-path import.
# Tries utils.pdf_generator first (standard install), then bare pdf_generator
# (notebooks/ deployment). Sets PDF_AVAILABLE=False and shows a visible warning
# in the UI if neither path works, so users know PDF download will be disabled.
# =============================================================================
PDF_AVAILABLE = False
generate_decision_pdf = None
generate_audit_pdf = None
try:
    from utils.pdf_generator import generate_decision_pdf, generate_audit_pdf
    PDF_AVAILABLE = True
except ImportError:
    try:
        from pdf_generator import generate_decision_pdf, generate_audit_pdf
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False  # UI will show warning — see A-4 note in pdf download buttons

# =============================================================================
# JSON SANITIZER
# =============================================================================
def sanitize_for_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)): return obj
    if isinstance(obj, set): return list(obj)
    if isinstance(obj, datetime): return obj.isoformat()
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [sanitize_for_json(item) for item in obj]
    try:
        json.dumps(obj); return obj
    except (TypeError, ValueError): return str(obj)

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
            try: assets = joblib.load(path); break
            except FileNotFoundError: continue
        if assets is None:
            raise FileNotFoundError("Could not find credit_risk_assets.pkl")
        return {
            'model': assets['model'], 'features': assets['features'],
            'le_map': assets['le_map'], 'target_le': assets['target_le'],
            'loaded': True, 'error': None
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

MODEL      = ASSETS['model']
TOP_FEATURES = ASSETS['features']
LE_MAP     = ASSETS['le_map']
TARGET_LE  = ASSETS['target_le']

# =============================================================================
# PD CALCULATION FUNCTIONS
# NOTE: calculate_emi, calculate_affordability, generate_reason_codes,
#       calculate_final_risk_score are imported from their respective modules.
#       The PD functions below are NOT in any module so are kept here.
# =============================================================================
def bureau_score_to_pd(bureau_score):
    if bureau_score >= 800: return 0.5 + (900 - bureau_score) / 200 * 0.5
    elif bureau_score >= 750: return 1.0 + (800 - bureau_score) / 50 * 1.0
    elif bureau_score >= 700: return 2.0 + (750 - bureau_score) / 50 * 1.5
    elif bureau_score >= 650: return 3.5 + (700 - bureau_score) / 50 * 2.5
    elif bureau_score >= 600: return 6.0 + (650 - bureau_score) / 50 * 4.0
    elif bureau_score >= 550: return 10.0 + (600 - bureau_score) / 50 * 5.0
    else: return min(25.0, 15.0 + (550 - bureau_score) / 50 * 10.0)

def foir_to_pd_adjustment(foir_percentage):
    if foir_percentage <= 30: return -0.75
    elif foir_percentage <= 40: return 0.00
    elif foir_percentage <= 45: return 0.75
    elif foir_percentage <= 50: return 1.50
    elif foir_percentage <= 55: return 2.25
    elif foir_percentage <= 60: return 3.50
    else: return 6.00

def delinquency_to_pd_multiplier(dpd_90_count, dpd_30_count=0):
    if dpd_90_count >= 3: return 5.0
    elif dpd_90_count == 2: return 3.0
    elif dpd_90_count == 1: return 2.0
    elif dpd_30_count >= 3: return 1.6
    elif dpd_30_count >= 1: return 1.3
    else: return 1.0

def employment_stability_to_pd_adjustment(employment_type, tenure_months, business_vintage_years=0):
    if employment_type == 'Salaried':
        if tenure_months >= 36: return -0.5
        elif tenure_months >= 12: return 0.0
        elif tenure_months >= 6: return 0.5
        else: return 2.0
    elif employment_type in ['Self-Employed', 'Business']:
        if business_vintage_years >= 5: return -0.5
        elif business_vintage_years >= 2: return 0.0
        else: return 1.5
    else: return 1.0

def inquiry_pattern_to_pd_adjustment(recent_inquiries_3m):
    if recent_inquiries_3m <= 1: return -0.3
    elif recent_inquiries_3m <= 3: return 0.0
    elif recent_inquiries_3m <= 5: return 0.8
    elif recent_inquiries_3m <= 8: return 1.5
    else: return 3.0

def ml_confidence_to_pd_adjustment(ml_confidence, ml_decision):
    if ml_decision == "APPROVE":
        if ml_confidence >= 90: return -0.5
        elif ml_confidence >= 70: return 0.0
        else: return 0.5
    elif ml_decision == "REVIEW": return 1.0
    else: return 5.0

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
    # FIX 2: raised ceiling from 25% to 50%.
    # Previous cap of 25% meant fraud+bankruptcy showed identical PD to a clean
    # 550-score borrower. Raw PDs for REJECT cases reach 124% before clamping;
    # 4.2% of rejects exceeded the old cap. 50% preserves discrimination in the
    # high-risk tail while staying within practical underwriting display ranges.
    return round(max(0.5, min(final_pd, 50.0)), 2)

# =============================================================================
# CATEGORICAL FLAG INFERENCE (v8.5 dual-dataset)
# =============================================================================
def _infer_surplus_from_cibil(score: int, dpd_60: int, dpd_30: int, income: float) -> float:
    if dpd_60 >= 3: return income * -0.5
    elif score < 650 or dpd_60 >= 1: return income * -0.2
    elif score < 700: return income * 0.1
    else: return income * 0.3

# extract_cibil_from_pdf and infer_categorical_flags removed from here.
# Imported from utils/ocr_extractor.py (v3.0) above — see import block.
# The v3.0 module adds: deskew pre-processing, high-DPI retry,
# full Stage-1/Stage-2 field mapping, gender bias fix ('U' default),
# recent_deliq_flag from actual DPD, and account_quality_score.

def log_decision_for_fairness(customer_data: dict, decision: str, risk_score: int, pd_pct: float,
                               application_id: str = None, source: str = 'stage1'):
    """
    Append a minimal record to the in-session fairness log.
    source = 'stage1' | 'stage2' | 'batch'
    When Stage 2 completes, it REPLACES the Stage 1 record for the same application_id,
    so the fairness dashboard always shows the FINAL binding decision.

    NOTE A-3 — risk_score scale:
      source='stage1' or 'batch' → risk_score is on 0-100 (Stage 1 engine output).
      source='stage2'            → risk_score is the combined_risk_score on 0-1000
                                   (Stage 1 normalised + Stage 2 tier, see stage2_engine.py).
    The fairness dashboard currently uses risk_score only for the 'Avg Risk Score' summary
    column. If cross-source comparisons are needed, normalise to a common scale first.
    """
    record = {
        'ts':              datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'application_id':  application_id or customer_data.get('application_id', ''),
        'source':          source,
        'decision':        decision,
        'risk_score':      risk_score,
        'pd_pct':          pd_pct,
        'gender':          customer_data.get('gender', 'Unknown'),
        'city_tier':       customer_data.get('city_tier', 'Unknown'),
        'employment_type': customer_data.get('employment_type', 'Unknown'),
        'bureau_score':    customer_data.get('bureau_score', 0),
        'age_band':        (
            '24-30' if customer_data.get('age', 0) < 31 else
            '31-40' if customer_data.get('age', 0) < 41 else
            '41-50' if customer_data.get('age', 0) < 51 else '51+'
        ),
    }
    st.session_state.fairness_log.append(record)
    # D3 FIX: cap at 1000 entries to prevent unbounded memory growth per session
    if len(st.session_state.fairness_log) > 1000:
        st.session_state.fairness_log = st.session_state.fairness_log[-1000:]

# =============================================================================
# STAGE 2 BINARY RESOLVER  (defined early — called from page routing below)
# =============================================================================
def resolve_stage2_to_binary(stage2_result: dict) -> dict:
    """
    Normalise Stage 2 result to a binary APPROVE / REJECT decision.
    REVIEW outcomes are resolved via tier mapping; score is used as tie-breaker.
    Defined here (before page routing) so it is always in scope regardless of
    which section of the file Streamlit is executing.
    """
    result = stage2_result.copy()
    tier  = result.get('stage2_tier', '')
    raw   = result.get('final_decision', '')
    score = result.get('combined_risk_score', 0) or 0
    TIER_MAP = {'P1': 'APPROVE', 'P2': 'APPROVE', 'P3': 'REJECT', 'P4': 'REJECT'}
    if raw == 'REJECT':
        result['final_decision'] = 'REJECT'
    elif raw == 'APPROVE':
        result['final_decision'] = TIER_MAP.get(tier, 'APPROVE')
    else:
        if tier in TIER_MAP:
            result['final_decision'] = TIER_MAP[tier]
            result['reason'] = result.get('reason', '') + f" [REVIEW resolved to {TIER_MAP[tier]} via tier {tier}]"
        else:
            resolved = 'APPROVE' if score >= 600 else 'REJECT'
            result['final_decision'] = resolved
            result['reason'] = result.get('reason', '') + f" [REVIEW resolved to {resolved} via score {score}]"
    if result['final_decision'] == 'APPROVE':
        result.setdefault('interest_rate_range', {'P1': '9.5%–11%', 'P2': '11%–13%'}.get(tier, '11%–14%'))
    else:
        result['interest_rate_range'] = 'N/A — Rejected'
    return result


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

    # AGE POLICY GATE — split by employment type per spec
    # UI allows 18–70 for input flexibility, but policy enforces:
    #   - All types:       age must be > 24  (≤ 24 → too young)
    #   - Salaried:        age must be ≤ 65  (retirement risk)
    #   - Self-Employed / Business: age must be ≤ 70
    _is_salaried = employment_type == 'Salaried'
    _max_age     = 65 if _is_salaried else 70
    _age_label   = "24–65 for Salaried" if _is_salaried else "24–70 for Self-Employed/Business"
    if age <= 24:
        policy_checks['age'] = f"❌ Age {age} — Too young (Min: 25)"
        return {'decision': "REJECT", 'reason': "Policy Gate: Applicant too young (minimum age 25)", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 70.0, 'affordability_data': {}}
    if age > _max_age:
        policy_checks['age'] = f"❌ Age {age} — Exceeds max ({_age_label})"
        return {'decision': "REJECT", 'reason': f"Policy Gate: Age exceeds maximum for {employment_type} ({_max_age})", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 70.0, 'affordability_data': {}}
    policy_checks['age'] = f"✅ Age {age} (Valid — {_age_label})"

    if not kyc_verified:
        policy_checks['kyc'] = "❌ KYC Not Verified"
        return {'decision': "REJECT", 'reason': "Policy Gate: KYC verification required", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 70.0, 'affordability_data': {}}
    policy_checks['kyc'] = "✅ KYC Verified"

    if not customer_dict.get('rbi_consent', False):
        policy_checks['rbi_consent'] = "❌ RBI Consent not obtained"
        return {'decision': "REJECT", 'reason': "Policy Gate: Customer consent not obtained", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 70.0, 'affordability_data': {}}
    policy_checks['rbi_consent'] = "✅ Consent Obtained"

    if bankruptcy_flag:
        policy_checks['bankruptcy'] = "❌ Active Bankruptcy"
        return {'decision': "REJECT", 'reason': "Policy Gate: Active bankruptcy", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 95.0, 'affordability_data': {}}
    policy_checks['bankruptcy'] = "✅ No Bankruptcy"

    if fraud_flag:
        policy_checks['fraud'] = "❌ Fraud Flag"
        return {'decision': "REJECT", 'reason': "Policy Gate: Fraud detected", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 95.0, 'affordability_data': {}}
    policy_checks['fraud'] = "✅ No Fraud History"

    dependents = customer_dict.get('dependents', 0)
    dependents_flag_review = dependents > 5
    policy_checks['dependents'] = (f"⚠️ Dependents {dependents} (>5: Review Required)"
                                   if dependents_flag_review else f"✅ Dependents {dependents} (Acceptable)")

    monthly_income = customer_dict.get('avg_salary_6m', 0)
    employment_tenure = customer_dict.get('employment_tenure_months', 0)
    business_vintage = customer_dict.get('business_vintage_years', 0)

    if monthly_income < 15000:
        policy_checks['income'] = f"❌ Income ₹{monthly_income:,.0f} (Min: ₹15,000)"
        return {'decision': "REJECT", 'reason': "Policy Gate: Income below minimum", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 72.0, 'affordability_data': {}}
    policy_checks['income'] = f"✅ Income ₹{monthly_income:,.0f}"

    if employment_type == 'Salaried' and employment_tenure < 6:
        policy_checks['tenure'] = f"❌ Tenure {employment_tenure} months (Min: 6)"
        return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient tenure", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 72.0, 'affordability_data': {}}
    elif employment_type in ['Self-Employed', 'Business'] and business_vintage < 2:
        policy_checks['tenure'] = f"❌ Business Vintage {business_vintage} years (Min: 2)"
        return {'decision': "REJECT", 'reason': "Policy Gate: Insufficient business vintage", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 72.0, 'affordability_data': {}}
    policy_checks['tenure'] = (f"✅ Tenure {employment_tenure} months" if employment_type == 'Salaried'
                                else f"✅ Business Vintage {business_vintage} years")

    bureau_score = customer_dict.get('bureau_score', 0)
    # FIX 1: round DPD counts — synthetic data generator applied Gaussian jitter
    # (×normal(1, 0.005)) to all numeric columns, producing float values like 0.9904,
    # 1.982, 6.94 instead of integers 0, 1, 2. Without rounding, dpd_90 > 0 fires
    # on 0.9904 (should be 0) and the tiered gate logic is wrong for every noisy row.
    dpd_90 = int(round(float(customer_dict.get('dpd_90_count_6m', 0) or 0)))
    credit_utilization = customer_dict.get('credit_utilization_pct', 0)
    recent_inquiries = customer_dict.get('recent_inquiries_3m', 0)

    if bureau_score < 550:
        policy_checks['bureau'] = f"❌ Bureau Score {bureau_score} (Min: 550)"
        return {'decision': "REJECT", 'reason': "Policy Gate: Bureau score too low", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 82.0, 'affordability_data': {}}
    policy_checks['bureau'] = f"✅ Bureau Score {bureau_score}"

    # DPD90 TIERED GATE:
    #   0     -> PASS (clean)
    #   1-5   -> REVIEW flag (elevated risk, underwriter required)
    #   > 5   -> REJECT (severe delinquency, hard stop)
    dpd_90_review_flag = False
    # DESIGN NOTE (M2): DPD90 gate is tiered — >5 = hard REJECT, 1-5 = REVIEW flag.
    # Legacy calculate_risk_score() (fallback-only) uses softer penalty for DPD90=1;
    # that path is NEVER reached in production. This gate is the intended behavior.
    if dpd_90 > 5:
        policy_checks['dpd'] = f"❌ {dpd_90} instance(s) of 90+ DPD — Hard Reject (Max: 5)"
        return {'decision': "REJECT", 'reason': "Policy Gate: Severe delinquency (90+ DPD > 5)", 'confidence': 0,
                'class_probs': {'REJECT': 100}, 'policy_checks': policy_checks, 'risk_score': 0,
                'pd_percentage': 88.0, 'affordability_data': {}}
    elif dpd_90 >= 1:
        dpd_90_review_flag = True
        policy_checks['dpd'] = f"⚠️ {dpd_90} instance(s) of 90+ DPD — Underwriter Review Required"
    else:
        policy_checks['dpd'] = "✅ No 90+ DPD (Clean)"
    policy_checks['utilization'] = (f"⚠️ High utilization {credit_utilization}%" if credit_utilization > 80
                                    else f"✅ Utilization {credit_utilization}%")
    policy_checks['inquiries'] = (f"⚠️ {recent_inquiries} recent inquiries" if recent_inquiries > 5
                                  else f"✅ {recent_inquiries} inquiries")

    active_loans = customer_dict.get('active_loans_count', 0)
    active_loans_flag = active_loans >= 5
    policy_checks['active_loans'] = (f"⚠️ High active loans ({int(active_loans)}) — Review"
                                     if active_loans_flag else f"✅ Active loans: {int(active_loans)}")

    salary_stability = customer_dict.get('salary_stability_flag', 'STABLE')
    salary_flag = salary_stability == 'UNSTABLE'
    policy_checks['salary'] = (
        "⚠️ Unstable salary — Review required" if salary_stability == 'UNSTABLE' else
        "⚠️ Moderate salary stability" if salary_stability == 'MODERATE' else "✅ Stable salary"
    )

    input_df = pd.DataFrame([customer_dict])
    for col in TOP_FEATURES:
        if col not in input_df.columns:
            input_df[col] = "Unknown" if col in LE_MAP else 0
    for col, le in LE_MAP.items():
        if col in input_df.columns:
            val = str(input_df[col].values[0])
            try: input_df[col] = le.transform([val])[0]
            except ValueError: input_df[col] = 0
    final_input = input_df[TOP_FEATURES]
    pred_idx = MODEL.predict(final_input)[0]
    ml_decision = TARGET_LE.inverse_transform([pred_idx])[0]
    try:
        pred_proba = MODEL.predict_proba(final_input)[0]
        confidence = max(pred_proba) * 100
        class_probs = {cls: prob * 100 for cls, prob in zip(TARGET_LE.classes_, pred_proba)}
    except Exception:
        confidence = 75.0
        class_probs = {ml_decision: 100.0}

    loan_amount   = customer_dict.get('loan_amount', 0)
    loan_tenure   = customer_dict.get('loan_tenure_months', 12)
    interest_rate = customer_dict.get('interest_rate', 10.5)
    existing_emi  = customer_dict.get('existing_emi', 0)
    affordability_data = calculate_affordability(monthly_income, loan_amount, interest_rate, loan_tenure, existing_emi)
    foir = affordability_data['foir_percentage']

    if foir > 50:
        ml_decision = "REJECT"
        policy_checks['foir'] = f"❌ FOIR {foir:.1f}% exceeds maximum allowed (50%)"

    if dependents_flag_review and ml_decision == "APPROVE": ml_decision = "REVIEW"
    if active_loans_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"
    if salary_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"
    if dpd_90_review_flag and ml_decision == "APPROVE": ml_decision = "REVIEW"  # DPD90 1-5 forces review

    risk_score = calculate_final_risk_score(
        bureau_score=bureau_score, ml_confidence=confidence, foir=foir,
        dpd_90=dpd_90, dpd_30=customer_dict.get('dpd_30_count_6m', 0),
        net_surplus=customer_dict.get('net_cash_surplus_6m', 0),
        bounces=customer_dict.get('inward_bounce_count_3m', 0),
        missing_months=customer_dict.get('salary_missing_months', 0),
        active_loans=active_loans
    )
    pd_percentage = calculate_final_pd(
        bureau_score=bureau_score, foir=foir, confidence=confidence,
        dpd_90_count=dpd_90, dpd_30_count=customer_dict.get('dpd_30_count_6m', 0),
        employment_type=employment_type, employment_tenure=employment_tenure,
        business_vintage=business_vintage, recent_inquiries=recent_inquiries,
        ml_decision=ml_decision
    )
    return {
        'decision': ml_decision, 'ml_raw_decision': ml_decision,
        'reason': "Decision based on comprehensive assessment",
        'confidence': confidence, 'class_probs': class_probs,
        'policy_checks': policy_checks, 'risk_score': risk_score,
        'pd_percentage': round(pd_percentage, 2), 'affordability_data': affordability_data
    }

# =============================================================================
# BATCH PREDICTION ENGINE
# =============================================================================
def process_batch_predictions(df):
    results = []
    required_fields = {
        'age': 35, 'employment_type': 'Salaried', 'kyc_verified': True,
        'bankruptcy_flag': False, 'fraud_flag': False, 'rbi_consent': True,
        'employment_tenure_months': 24, 'business_vintage_years': 0,
        'bureau_score': 700, 'dpd_90_count_6m': 0, 'dpd_30_count_6m': 0,
        'credit_utilization_pct': 30, 'recent_inquiries_3m': 0,
        'active_loans_count': 0, 'existing_emi': 0, 'avg_salary_6m': 50000,
        'AMT_INCOME_TOTAL': 600000, 'net_cash_surplus_6m': 20000,
        'salary_stability_flag': 'STABLE', 'loan_amount': 180000,
        'loan_tenure_months': 24, 'interest_rate': 10.5, 'AMT_ANNUITY': 8500,
        'dependents': 0, 'payment_discipline_flag': 'GOOD',
        'liquidity_flag': 'LOW', 'cashflow_health': 'MODERATE',
        'bureau_risk_flag': 'LOW', 'inward_bounce_count_3m': 0,
        'salary_missing_months': 0, 'gender': 'Unknown', 'city_tier': 'Unknown',
    }
    for idx, row in df.iterrows():
        customer_dict = row.to_dict()
        for k, v in customer_dict.items():
            if isinstance(v, str):
                if v.lower() in ['yes', 'true', '1']: customer_dict[k] = True
                elif v.lower() in ['no', 'false', '0']: customer_dict[k] = False
        for field, default in required_fields.items():
            if field not in customer_dict or pd.isna(customer_dict.get(field, None)):
                customer_dict[field] = default
        try:
            decision_data = make_hybrid_decision_enhanced(customer_dict)
            customer_dict['ml_confidence'] = decision_data.get('confidence', 0)
            reasons = generate_reason_codes(
                decision=decision_data.get('decision', 'ERROR'),
                customer_data=customer_dict,
                affordability_data=decision_data.get('affordability_data', {}),
                policy_checks=decision_data.get('policy_checks', {})
            )
            affordability = decision_data.get('affordability_data', {})
            result = {
                'application_id': f"BATCH_{idx+1:04d}",
                'decision': decision_data.get('decision', 'ERROR'),
                'risk_score': decision_data.get('risk_score', 0),
                'pd_percentage': decision_data.get('pd_percentage', 0),
                'confidence': round(decision_data.get('confidence', 0), 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'reason_1': reasons[0] if len(reasons) > 0 else '',
                'reason_2': reasons[1] if len(reasons) > 1 else '',
                'reason_3': reasons[2] if len(reasons) > 2 else '',
                'age': customer_dict.get('age', ''),
                'gender': customer_dict.get('gender', ''),
                'city_tier': customer_dict.get('city_tier', ''),
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
                'prob_approve': round(decision_data.get('class_probs', {}).get('APPROVE', 0), 2),
                'prob_review': round(decision_data.get('class_probs', {}).get('REVIEW', 0), 2),
                'prob_reject': round(decision_data.get('class_probs', {}).get('REJECT', 0), 2),
            }
        except Exception as e:
            result = {
                'application_id': f"BATCH_{idx+1:04d}", 'decision': 'ERROR',
                'risk_score': 0, 'pd_percentage': 0, 'confidence': 0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'reason_1': '', 'reason_2': '', 'reason_3': '',
                'age': customer_dict.get('age', ''), 'gender': customer_dict.get('gender', ''),
                'city_tier': customer_dict.get('city_tier', ''),
                'employment_type': customer_dict.get('employment_type', ''),
                'bureau_score': customer_dict.get('bureau_score', ''),
                'monthly_income': customer_dict.get('avg_salary_6m', ''),
                'loan_amount': customer_dict.get('loan_amount', ''),
                'error_message': str(e)
            }
        else:
            # Log to fairness monitor (success path only)
            log_decision_for_fairness(
                customer_dict,
                result['decision'],
                result['risk_score'],
                result['pd_percentage']
            )
        results.append(result)
    return pd.DataFrame(results)

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
        card_class = "decision-card decision-card-approved"; icon = "✓"; subtitle = "Application Approved Successfully"
    elif decision == "REJECT":
        card_class = "decision-card decision-card-rejected"; icon = "✗"; subtitle = "Application Not Approved"
    else:
        card_class = "decision-card decision-card-review"; icon = "⚠"; subtitle = "Requires Manual Review"
    st.markdown(f'<div class="{card_class}"><div class="decision-title">{icon} {decision}</div><div class="decision-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.markdown(f'<div class="stat-card"><div class="stat-number">{risk_score}</div><div class="stat-label">Risk Score (0–100)</div></div>', unsafe_allow_html=True)
    _pd_color = '#48bb78' if pd_score < 5 else ('#ed8936' if pd_score < 10 else '#f56565')
    _pd_label = 'Low Risk' if pd_score < 5 else ('Moderate Risk' if pd_score < 10 else 'High Risk')
    with col2: st.markdown(f'<div class="stat-card"><div class="stat-number" style="color:{_pd_color}">{pd_score}%</div><div class="stat-label">PD Score</div><div style="font-size:11px;color:{_pd_color};font-weight:600">{_pd_label}</div></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="stat-card"><div class="stat-number">₹{approved_amount:,.0f}</div><div class="stat-label">Loan Amount</div></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="stat-card"><div class="stat-number">{tenure}</div><div class="stat-label">Tenure (Months)</div></div>', unsafe_allow_html=True)
    with col5: st.markdown(f'<div class="stat-card"><div class="stat-number">{decision_data.get("confidence", 0):.0f}%</div><div class="stat-label">Confidence</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: st.markdown(f'<div class="info-box"><strong>📋 Application ID:</strong> {app_id}</div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="info-box"><strong>🕐 Decision Timestamp:</strong> {timestamp}</div>', unsafe_allow_html=True)

def render_info_card(title, icon, data_dict, status_dict=None):
    st.markdown(f'<div class="info-card"><div class="info-card-title">{icon} {title}</div><div class="info-card-content">', unsafe_allow_html=True)
    for label, value in data_dict.items():
        status = ""
        if status_dict and label in status_dict:
            if status_dict[label] == "pass": status = '<span class="status-badge badge-pass">✓</span>'
            elif status_dict[label] == "fail": status = '<span class="status-badge badge-fail">✗</span>'
            elif status_dict[label] == "warning": status = '<span class="status-badge badge-warning">⚠</span>'
        st.markdown(f'<div class="data-row"><span class="data-label">{label}</span><span class="data-value">{value} {status}</span></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

def render_reason_codes(reasons):
    st.markdown('<div class="info-card"><div class="info-card-title">📝 Decision Reasons</div><div class="info-card-content">', unsafe_allow_html=True)
    for i, reason in enumerate(reasons, 1):
        st.markdown(f'<div class="reason-item"><span class="reason-icon">{i}.</span>{reason}</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

def create_modern_gauge(value, title, max_value=100):
    color = "#f56565" if value <= 50 else "#ed8936" if value <= 75 else "#48bb78"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title, 'font': {'size': 18, 'color': '#2d3748'}},
        number={'suffix': "%", 'font': {'size': 36, 'color': '#2d3748'}},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': 'white', 'borderwidth': 0,
            'steps': [{'range': [0, 50], 'color': '#fed7d7'},
                      {'range': [50, 75], 'color': '#feebc8'},
                      {'range': [75, 100], 'color': '#c6f6d5'}]
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white')
    return fig

def create_modern_bar_chart(class_probs):
    df = pd.DataFrame({'Decision': list(class_probs.keys()), 'Probability': list(class_probs.values())})
    colors = {'REVIEW': '#ed8936', 'APPROVE': '#48bb78', 'REJECT': '#f56565'}
    fig = px.bar(df, x='Decision', y='Probability', title='Decision Probabilities',
                 color='Decision', color_discrete_map=colors, text='Probability')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
    fig.update_layout(showlegend=False, yaxis_title='Probability (%)', xaxis_title='', height=300,
                      margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='white', plot_bgcolor='white',
                      yaxis={'gridcolor': '#e2e8f0', 'range': [0, max(class_probs.values()) * 1.2]})
    return fig

# =============================================================================
# STAGE 2 BINARY RESOLVER

# =============================================================================
# STAGE 2 RESULTS DISPLAY
# =============================================================================
def display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data):
    st.markdown("---")
    st.markdown('<p class="main-header">🎯 Stage 2 Final Results</p>', unsafe_allow_html=True)
    final_decision    = stage2_result.get('final_decision', 'ERROR')
    interest_range    = stage2_result.get('interest_rate_range', 'N/A')
    stage2_tier       = stage2_result.get('stage2_tier', 'N/A')
    stage2_confidence = stage2_result.get('stage2_confidence', 0)
    combined_risk     = stage2_result.get('combined_risk_score', 0)

    # ── Fairness log: use Stage 2 FINAL decision, remove the earlier Stage 1 entry ──
    # Stage 1 logged a preliminary decision for this customer. Since Stage 2
    # is the BINDING final decision, we replace that entry so the fairness
    # dashboard always reflects the true outcome.
    app_id = stage1_customer.get('application_id', None)
    if app_id and 'fairness_log' in st.session_state:
        st.session_state.fairness_log = [
            r for r in st.session_state.fairness_log
            if r.get('application_id') != app_id
        ]
    log_decision_for_fairness(
        enhanced_customer_data,
        final_decision,
        combined_risk,
        stage2_result.get('pd_percentage', stage1_data.get('pd_percentage', 0)),
        application_id=app_id,
        source='stage2'
    )

    # Update session state — Stage 2 is the binding final decision
    st.session_state['stage2_final_decision'] = final_decision

    if final_decision == "APPROVE":
        st.markdown(
            '<div class="decision-card decision-card-approved" style="padding:2.5rem;">'
            '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">✔  APPROVED</div>'
            '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">✅ STAGE 2 FINAL DECISION — Proceed to Disbursement</div>'
            '</div>', unsafe_allow_html=True)
    elif final_decision == "REJECT":
        st.markdown(
            '<div class="decision-card decision-card-rejected" style="padding:2.5rem;">'
            '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">✘  REJECTED</div>'
            '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">❌ STAGE 2 FINAL DECISION — Application Declined</div>'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="decision-card decision-card-review" style="padding:2.5rem;">'
            '<div class="decision-title" style="font-size:3.5rem;font-weight:900;letter-spacing:2px;">⚑  REVIEW</div>'
            '<div class="decision-subtitle" style="font-size:1.2rem;margin-top:0.5rem;">⚠️ STAGE 2 FINAL DECISION — Requires Manual Credit Officer Review</div>'
            '</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Risk Tier", stage2_tier)
    with col2: st.metric("Interest Rate", interest_range)
    with col3: st.metric("Combined Risk Score (0–1000)", combined_risk)
    with col4: st.metric("Stage 2 Confidence", f"{stage2_confidence:.1f}%" if stage2_confidence else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔬 Analysis", "📋 Data", "📥 Download"])

    with tab1:
        s1_dec = st.session_state.get('stage1_decision', 'N/A')
        s2_label = "✅ APPROVE" if final_decision == "APPROVE" else "❌ REJECT"
        # FIX SCALE: Stage 1 is 0-100, Stage 2 combined is 0-1000.
        # Show them with explicit scale labels so they are never compared directly.
        s1_score_raw = stage1_data.get('risk_score', 'N/A')
        s1_score_display = f"{s1_score_raw}/100" if isinstance(s1_score_raw, (int, float)) else s1_score_raw
        s2_score_display = f"{combined_risk}/1000"
        comparison_df = pd.DataFrame([
            {'Stage': 'Stage 1 (Screening)', 'Decision': s1_dec,
             'Risk Score': s1_score_display, 'Scale': '0–100 (higher = riskier)',
             'Tier': 'N/A', 'Note': 'APPROVE/REVIEW → proceed to Stage 2'},
            {'Stage': 'Stage 2 — FINAL', 'Decision': s2_label,
             'Risk Score': s2_score_display, 'Scale': '0–1000 (higher = riskier)',
             'Tier': f"{stage2_tier} | {interest_range}", 'Note': 'Binding final decision'}
        ])
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        tier_info = {
            'P1': {'name': 'Premium → APPROVED', 'color': '#10B981', 'desc': 'Excellent credit profile — lowest interest rate band'},
            'P2': {'name': 'Standard → APPROVED', 'color': '#3B82F6', 'desc': 'Good credit profile — standard interest rate band'},
            'P3': {'name': 'Subprime → REJECTED', 'color': '#F59E0B', 'desc': 'Fair credit with elevated risk — application declined'},
            'P4': {'name': 'High Risk → REJECTED', 'color': '#EF4444', 'desc': 'High risk profile — application declined'},
        }
        if stage2_tier in tier_info:
            td = tier_info[stage2_tier]
            st.markdown(f'<div style="background:{td["color"]};color:white;padding:1rem;border-radius:0.5rem;"><h3 style="margin:0;color:white;">{stage2_tier}: {td["name"]}</h3><p style="margin:0.5rem 0 0 0;">{td["desc"]}</p></div>', unsafe_allow_html=True)
        st.info(stage2_result.get('reason', 'N/A'))

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Tier Probabilities**")
            if 'tier_probabilities' in stage2_result:
                for tier, prob in stage2_result['tier_probabilities'].items():
                    st.metric(tier, f"{prob:.1f}%")
        with col2:
            st.markdown("**Stage Scores**")
            s1_raw = stage1_data.get('risk_score', 'N/A')
            s1_display = f"{s1_raw} / 100" if isinstance(s1_raw, (int, float)) else s1_raw
            s2_raw = stage2_result.get('stage2_risk_score', 'N/A')
            s2_display = f"{s2_raw} / 1000" if isinstance(s2_raw, (int, float)) else s2_raw
            combined_display = f"{combined_risk} / 1000"
            st.metric("Stage 1 Risk Score (0–100)", s1_display)
            st.metric("Stage 2 Risk Score (0–1000)", s2_display)
            st.metric("Combined Score (0–1000)", combined_display)
            st.caption("⚠️ Stage 1 and Stage 2 scores use different scales — do not compare directly.")
        with st.expander("Complete Stage 2 Result (JSON)"):
            st.json(stage2_result)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Stage 1 Customer Data"): st.json(stage1_customer)
        with col2:
            with st.expander("Enhanced CIBIL Data"): st.json(enhanced_customer_data)

    with tab4:
        if PDF_AVAILABLE and generate_audit_pdf is not None:
            try:
                _safe = lambda v, d='N/A': v if v is not None else d
                # Build full pd_calculation_factors from enhanced customer data
                _bs  = enhanced_customer_data.get('bureau_score', stage1_customer.get('bureau_score', 0))
                _foir = stage1_data.get('affordability_data', {}).get('foir_percentage', 0)
                _conf = stage1_data.get('confidence', 0)
                _dpd90 = enhanced_customer_data.get('dpd_90_count_6m', stage1_customer.get('dpd_90_count_6m', 0))
                _dpd30 = enhanced_customer_data.get('dpd_30_count_6m', stage1_customer.get('dpd_30_count_6m', 0))
                _emp_type = enhanced_customer_data.get('employment_type', stage1_customer.get('employment_type', 'Salaried'))
                _emp_ten  = enhanced_customer_data.get('employment_tenure_months', stage1_customer.get('employment_tenure_months', 24))
                _biz_vin  = enhanced_customer_data.get('business_vintage_years', stage1_customer.get('business_vintage_years', 0))
                _inq      = enhanced_customer_data.get('recent_inquiries_3m', stage1_customer.get('recent_inquiries_3m', 0))
                _base_pd   = bureau_score_to_pd(_bs)
                _foir_adj  = foir_to_pd_adjustment(_foir)
                _deliq_mul = delinquency_to_pd_multiplier(_dpd90, _dpd30)
                _emp_adj   = employment_stability_to_pd_adjustment(_emp_type, _emp_ten, _biz_vin)
                _inq_adj   = inquiry_pattern_to_pd_adjustment(_inq)
                _ml_adj    = ml_confidence_to_pd_adjustment(_conf, stage1_data.get('decision','REVIEW'))
                _final_pd  = stage1_data.get('pd_percentage', round(max(0.5, min(
                    _base_pd * _deliq_mul + _foir_adj + _emp_adj + _inq_adj + _ml_adj, 25.0)), 2))

                report_data = {
                    'application_id':  _safe(stage1_customer.get('application_id')),
                    'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_version':   '8.7',
                    'decision':        _safe(stage1_data.get('decision')),
                    'stage2_final_decision':      _safe(final_decision),
                    'stage2_tier':                _safe(stage2_tier),
                    'stage2_interest_range':      _safe(interest_range),
                    'stage2_combined_risk_score': _safe(combined_risk, 0),
                    'stage2_confidence':          _safe(stage2_confidence, 0),
                    'stage2_reason':              _safe(stage2_result.get('reason')),
                    'stage2_tier_probabilities':  stage2_result.get('tier_probabilities') or {},
                    'stage2_complete_analysis':   stage2_result,
                    # Top-level PD — used by audit header (must match pd_calculation_factors.final_pd)
                    'pd_percentage':              _final_pd,
                    'risk_score':                 _safe(combined_risk, 0),
                    'confidence':                 _safe(stage2_confidence, 0),
                    # Policy gate results
                    'policy_checks': stage1_data.get('policy_checks', {}),
                    # Full PD calculation breakdown
                    'pd_calculation_factors': {
                        'bureau_score':           _bs,
                        'base_pd':                round(_base_pd, 2),
                        'dpd_90':                 _dpd90,
                        'dpd_30':                 _dpd30,
                        'delinquency_multiplier': round(_deliq_mul, 2),
                        'foir':                   round(_foir, 2),
                        'foir_adjustment':        round(_foir_adj, 2),
                        'employment_adjustment':  round(_emp_adj, 2),
                        'inquiry_adjustment':     round(_inq_adj, 2),
                        'ml_adjustment':          round(_ml_adj, 2),
                        'final_pd':               _final_pd,
                    },
                    # Reason codes from Stage 1
                    'reason_codes': stage1_customer.get('reason_codes', []),
                    # Raw data refs
                    'customer_data':          stage1_customer,
                    'enhanced_customer_data': enhanced_customer_data,
                }
                pdf_buffer = generate_audit_pdf(report_data)
                st.download_button("📥 Download PDF Report", data=pdf_buffer,
                                   file_name=f"stage2_report_{stage1_customer.get('application_id','X')}.pdf",
                                   mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"PDF generation failed: {str(e)}")
        else:
            st.warning("⚠️ PDF generation is not available. Ensure utils/pdf_generator.py is present and `reportlab` is installed (add to requirements.txt).")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 New Assessment", key="new_assessment_stage2", use_container_width=True):
            for k in ['stage1_complete','stage1_decision','stage1_data','current_customer_data']:
                st.session_state[k] = (False if k == 'stage1_complete' else None)
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
# FAIRNESS MONITORING DASHBOARD
# =============================================================================
def render_fairness_dashboard():
    st.markdown('<p class="main-header">⚖️ Fairness Monitoring</p>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-box">
            <strong>RBI Fair Lending Compliance Dashboard</strong><br>
            Tracks approval rates across demographic groups to detect potential disparate impact.
            <strong>Fairness is measured on the FINAL binding decision</strong> — Stage 2 outcome
            is used when available; Stage 1 (screening) entries are automatically replaced once
            Stage 2 completes for the same application.
            Data is session-based — decisions accumulate as applications are processed.
        </div>
    """, unsafe_allow_html=True)

    log = st.session_state.get('fairness_log', [])

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Log", use_container_width=True):
            st.session_state.fairness_log = []
            st.rerun()

    if not log:
        st.info("ℹ️ No decisions logged yet. Process some applications from the Assessment page to see fairness metrics here.")
        st.markdown("### 📊 What will appear here:")
        st.markdown("""
        - **Approval rate by Gender** — tracks if male/female/other applicants are treated equitably
        - **Approval rate by City Tier** — checks for geographic bias (Tier 1 vs Tier 3 vs Rural)
        - **Approval rate by Age Band** — identifies potential age discrimination
        - **Approval rate by Employment Type** — salaried vs self-employed equity check
        - **Average Risk Score & PD by group** — confirms scoring is not systematically biased
        """)
        return

    df = pd.DataFrame(log)
    df['approved'] = (df['decision'] == 'APPROVE').astype(int)
    n = len(df)

    # Source breakdown
    if 'source' in df.columns:
        n_s2    = int((df['source'] == 'stage2').sum())
        n_s1    = int((df['source'] == 'stage1').sum())
        n_batch = int((df['source'] == 'batch').sum())
        src_note = f"📌 {n_s2} Stage 2 (final) · {n_s1} Stage 1 (screening) · {n_batch} Batch"
        st.caption(src_note)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Decisions", n)
    with c2: st.metric("Approvals", int(df['approved'].sum()), f"{df['approved'].mean()*100:.1f}%")
    with c3: st.metric("Reviews", int((df['decision']=='REVIEW').sum()))
    with c4: st.metric("Rejections", int((df['decision']=='REJECT').sum()))

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Gender", "🏙️ City Tier", "📅 Age Band", "💼 Employment"])

    COLOR_MAP = {'APPROVE': '#48bb78', 'REVIEW': '#ed8936', 'REJECT': '#f56565'}

    def _approval_bar(group_col, title):
        grp = df.groupby(group_col).agg(
            Total=('decision', 'count'),
            Approved=('approved', 'sum'),
            Avg_Risk=('risk_score', 'mean'),
            Avg_PD=('pd_pct', 'mean'),
        ).reset_index()
        grp['Approval Rate %'] = (grp['Approved'] / grp['Total'] * 100).round(1)
        grp['Avg Risk Score'] = grp['Avg_Risk'].round(1)
        grp['Avg PD %'] = grp['Avg_PD'].round(2)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(grp, x=group_col, y='Approval Rate %',
                         title=title, text='Approval Rate %',
                         color='Approval Rate %',
                         color_continuous_scale=['#f56565', '#ed8936', '#48bb78'],
                         range_color=[0, 100])
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10),
                              coloraxis_showscale=False, paper_bgcolor='white', plot_bgcolor='white',
                              yaxis={'range': [0, 110], 'gridcolor': '#e2e8f0'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Summary Table**")
            display_df = grp[[group_col, 'Total', 'Approval Rate %', 'Avg Risk Score', 'Avg PD %']].copy()
            # Flag groups with approval rate deviation > 15pp from overall
            overall_rate = df['approved'].mean() * 100
            def _flag(rate):
                diff = rate - overall_rate
                if abs(diff) > 15: return f"{'🔴' if diff < 0 else '🟢'} {rate:.1f}%"
                return f"✅ {rate:.1f}%"
            display_df['Approval Rate %'] = display_df['Approval Rate %'].apply(_flag)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            overall_str = f"{overall_rate:.1f}%"
            st.caption(f"Overall approval rate: **{overall_str}**. 🔴 = >15pp below average (potential bias). 🟢 = >15pp above average.")

    with tab1:
        if df['gender'].nunique() > 1:
            _approval_bar('gender', 'Approval Rate by Gender')
            # Decision mix donut per gender
            fig2 = px.pie(df, names='decision', color='decision', color_discrete_map=COLOR_MAP,
                          title='Decision Mix (all)', hole=0.5)
            fig2.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Need 2+ gender values in decisions to show chart. Ensure Gender field is filled on the form.")

    with tab2:
        if df['city_tier'].nunique() > 1:
            _approval_bar('city_tier', 'Approval Rate by City Tier')
        else:
            st.info("Need 2+ city tier values. Ensure City Tier field is filled on the form.")

    with tab3:
        if df['age_band'].nunique() > 1:
            _approval_bar('age_band', 'Approval Rate by Age Band')
        else:
            st.info("Need decisions across multiple age bands (24-30, 31-40, 41-50, 51+).")

    with tab4:
        if df['employment_type'].nunique() > 1:
            _approval_bar('employment_type', 'Approval Rate by Employment Type')
        else:
            st.info("Need 2+ employment types in decisions.")

    st.markdown("---")
    st.markdown("### 📥 Export Fairness Report")
    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button("📥 Download Decision Log (CSV)", data=csv_data,
                           file_name=f"fairness_log_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv", use_container_width=True)
    with col2:
        st.caption("⚠️ **Note:** This log is session-based and resets when the app restarts. "
                   "For persistent fairness monitoring, connect to a database or export regularly.")


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("# 🏦 Credit Risk Engine")
    st.markdown("---")

    navigation_options = ["🏠 Home", "👤 Assessment", "📊 Batch Process", "⚖️ Fairness", "📈 Model Info", "ℹ️ About"]

    if (st.session_state.stage1_complete and st.session_state.stage1_decision in ['APPROVE', 'REVIEW']):
        navigation_options.insert(2, "🔬 Stage 2 Analysis")
        st.success(f"✅ Stage 1: {st.session_state.stage1_decision}")
        st.info("🔬 Stage 2 Analysis unlocked!")
    elif st.session_state.stage1_complete:
        st.warning(f"⚠️ Stage 1: {st.session_state.stage1_decision}")
        st.caption("Stage 2 only for APPROVE/REVIEW")

    page = st.radio("**Navigation**", navigation_options,
                    label_visibility="collapsed", key="page_navigation")

    st.markdown("---")
    stage2_indicator = '✅ Active' if STAGE2_AVAILABLE and is_stage2_available() else '❌ Inactive'
    ocr_indicator = '✅ Ready' if OCR_AVAILABLE else '❌ Not Installed'
    pdf_indicator = '✅ Ready' if PDF_AVAILABLE else '❌ Not Installed'
    fairness_count = len(st.session_state.fairness_log)

    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">System Status</div>
        <div class="info-card-content">
            <div class="data-row"><span class="data-label">Model</span><span class="data-value">✅ Loaded</span></div>
            <div class="data-row"><span class="data-label">Version</span><span class="data-value">8.7</span></div>
            <div class="data-row"><span class="data-label">Stage 2</span><span class="data-value">{stage2_indicator}</span></div>
            <div class="data-row"><span class="data-label">OCR</span><span class="data-value">{ocr_indicator}</span></div>
            <div class="data-row"><span class="data-label">PDF Gen</span><span class="data-value">{pdf_indicator}</span></div>
            <div class="data-row"><span class="data-label">Fairness Log</span><span class="data-value">{fairness_count} decisions</span></div>
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
            for k in ['stage1_complete','stage1_decision','stage1_data','current_customer_data','extracted_cibil_data']:
                st.session_state[k] = False if k == 'stage1_complete' else None
            st.rerun()

# =============================================================================
# PAGE ROUTING
# =============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">Credit Risk Engine</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><h3 style="margin-top:0;">🎯 AI-Powered Lending Decisions</h3><p style="margin-bottom:0;">Comprehensive credit risk evaluation combining hard policy rules, machine learning models, and affordability analysis.</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="info-card"><div class="info-card-title">🛡️ Policy Gates</div><div class="info-card-content"><ul><li>Age & KYC verification</li><li>RBI consent check</li><li>Employment stability</li><li>Minimum income checks</li><li>Credit bureau thresholds</li><li>Bankruptcy & fraud detection</li></ul></div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card"><div class="info-card-title">🤖 ML Assessment</div><div class="info-card-content"><ul><li>Random Forest classifier</li><li>60K+ training samples</li><li>Confidence scoring</li><li>Multi-class prediction</li><li>Feature importance</li></ul></div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-card"><div class="info-card-title">⚖️ Fairness Monitoring</div><div class="info-card-content"><ul><li>Approval rate by gender</li><li>Approval rate by city tier</li><li>Age band equity check</li><li>Employment type parity</li><li>RBI compliance ready</li></ul></div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🎯 Accuracy", "85%", "+2%")
    with col2: st.metric("⚡ Avg Response", "1.2s", "-0.3s")
    with col3: st.metric("📊 Features", len(TOP_FEATURES))
    with col4: st.metric("🔄 Version", "8.7", "Latest")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="warning-box" style="background:#f0fff4;border:1px solid #9ae6b4;padding:1rem;border-radius:0.5rem;">
            <strong>🆕 New in Version 8.7:</strong><br>
            • <strong>Cleaned codebase</strong> — removed ~210 lines of duplicate function definitions<br>
            • <strong>City Tier field</strong> — Tier 1/2/3/Rural captured on every application<br>
            • <strong>Gender field</strong> — explicit gender capture for fairness logging<br>
            • <strong>RBI Consent checkbox</strong> — required policy gate before assessment<br>
            • <strong>Fairness Monitoring dashboard</strong> — approval rates by gender, city tier, age band, employment type<br>
            • <strong>v8.5 features retained</strong> — dual-dataset OCR inference, categorical flag auto-fill
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
            st.info("⬇️ Scroll down — all form fields have been pre-filled from the PDF. Review and click **Assess Credit Risk**.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Credit Score", ex.get('Credit_Score', '—'))
            c2.metric("Monthly Income", f"₹{ex.get('NETMONTHLYINCOME') or ex.get('avg_salary_6m', 0):,}")
            c3.metric("DPD 60+ Count", ex.get('num_times_60p_dpd', 0))
            c4.metric("CC Utilization", f"{max(0, float(ex.get('CC_utilization', 0) or 0))*100:.0f}%")
            _inf = st.session_state.get('_last_inferred_flags', {})
            if _inf:
                st.markdown("**📊 Inferred Categorical Flags:**")
                fc1, fc2, fc3, fc4, fc5 = st.columns(5)
                fc1.metric("Payment Discipline", _inf.get('payment_discipline_flag', '—'))
                fc2.metric("Cashflow Health", _inf.get('cashflow_health', '—'))
                fc3.metric("Liquidity", _inf.get('liquidity_flag', '—'))
                fc4.metric("Bureau Risk", _inf.get('bureau_risk_flag', '—'))
                fc5.metric("Salary Stability", _inf.get('salary_stability_flag', '—'))
            if st.button("🔄 Upload a different PDF", key="reset_pdf"):
                st.session_state.pdf_just_extracted = False
                st.session_state.pop('_last_extraction', None)
                st.session_state.pop('_last_inferred_flags', None)
                st.rerun()
        else:
            st.markdown('<div class="info-box">💡 Complete the form below or upload a CIBIL PDF to auto‑fill bureau data.</div>', unsafe_allow_html=True)
            if not OCR_AVAILABLE:
                st.warning(f"⚠️ PDF auto-fill unavailable — OCR not installed. {OCR_ERROR_MSG or ''} Complete the form manually below.")
            else:
                uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="assessment_pdf")
                if uploaded_pdf is not None:
                    st.info(f"📄 File ready: **{uploaded_pdf.name}** ({uploaded_pdf.size/1024:.1f} KB)")
                    if st.button("🔍 Extract & Auto-fill Form", key="extract_assessment", type="primary", use_container_width=True):
                        with st.spinner("🔄 Running OCR on CIBIL PDF — this takes 10-30 seconds..."):
                            if uploaded_pdf is None:
                                st.error("❌ File was lost — please re-upload the PDF and try again.")
                                st.stop()
                            uploaded_pdf.seek(0)
                            extraction_result = extract_cibil_from_pdf(uploaded_pdf)
                        if extraction_result.get('success', False):
                            st.session_state.pdf_age               = int(extraction_result.get('AGE', 35))
                            st.session_state.pdf_bureau_score      = int(extraction_result.get('Credit_Score', 720))
                            st.session_state.pdf_dpd_90            = int(extraction_result.get('dpd_90_count_6m', 0))
                            st.session_state.pdf_dpd_30            = int(extraction_result.get('dpd_30_count_6m', 0))
                            _cc_util_raw = extraction_result.get('CC_utilization', 0) or 0
                            st.session_state.pdf_credit_util       = int(max(0, float(_cc_util_raw)) * 100) if _cc_util_raw > 0 else 0
                            st.session_state.pdf_inquiries         = int(extraction_result.get('enq_L3m', 2))
                            st.session_state.pdf_active_loans      = int(extraction_result.get('num_std', 1))
                            _emi = int(extraction_result.get('existing_emi') or extraction_result.get('total_emi_monthly') or 0)
                            st.session_state.pdf_existing_emi      = _emi
                            _income = int(extraction_result.get('NETMONTHLYINCOME') or extraction_result.get('avg_salary_6m') or 50000)
                            st.session_state.pdf_monthly_income    = _income
                            st.session_state.pdf_annual_income     = int(extraction_result.get('AMT_INCOME_TOTAL') or _income * 12)
                            _surplus = int(extraction_result.get('net_cash_surplus_6m') or extraction_result.get('_surplus_proxy') or 0)
                            st.session_state.pdf_net_surplus       = _surplus
                            st.session_state.pdf_employment_tenure = int(extraction_result.get('Time_With_Curr_Empr', 24))
                            _emp = extraction_result.get('employment_type', 'Salaried')
                            if _emp in ['Salaried', 'Self-Employed', 'Business']:
                                st.session_state.pdf_employment_type = _emp
                            st.session_state.pdf_business_vintage  = int(extraction_result.get('business_vintage_years', 0))
                            _g = extraction_result.get('GENDER', 'U')
                            if _g == 'F':
                                st.session_state.pdf_gender = 'Female'
                            elif _g == 'M':
                                st.session_state.pdf_gender = 'Male'
                            else:
                                st.session_state.pdf_gender = 'Prefer not to say'
                            st.session_state.pdf_inward_bounce     = int(extraction_result.get('inward_bounce_count_3m', 0))
                            st.session_state.pdf_salary_missing    = int(extraction_result.get('salary_missing_months', 0))
                            st.session_state.pdf_salary_stability   = extraction_result.get('salary_stability_flag', 'MODERATE')
                            st.session_state.pdf_payment_discipline = extraction_result.get('payment_discipline_flag', 'GOOD')
                            st.session_state.pdf_cashflow_health    = extraction_result.get('cashflow_health', 'MODERATE')
                            st.session_state.pdf_liquidity_flag     = extraction_result.get('liquidity_flag', 'MODERATE')
                            st.session_state.pdf_bureau_risk_flag   = extraction_result.get('bureau_risk_flag', 'LOW')
                            st.session_state.pdf_just_extracted     = True
                            st.session_state._last_extraction       = extraction_result
                            st.session_state._last_inferred_flags   = {
                                'payment_discipline_flag': extraction_result.get('payment_discipline_flag', '—'),
                                'cashflow_health':         extraction_result.get('cashflow_health', '—'),
                                'liquidity_flag':          extraction_result.get('liquidity_flag', '—'),
                                'bureau_risk_flag':        extraction_result.get('bureau_risk_flag', '—'),
                                'salary_stability_flag':   extraction_result.get('salary_stability_flag', '—'),
                            }
                            st.rerun()
                        else:
                            st.error(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")
                            if extraction_result.get('traceback'):
                                with st.expander("🔍 Error details"):
                                    st.code(extraction_result['traceback'])

    # ── Seed session state defaults before form renders ──────────────────────
    # This ensures every widget key has a value in st.session_state before
    # the form renders. OCR sets these same keys, so widgets pick up OCR
    # values automatically on rerun without needing value=/index= params.
    _form_defaults = {
        'pdf_age': 35, 'pdf_employment_type': 'Salaried', 'pdf_gender': 'Male',
        'pdf_dependents': 2, 'pdf_kyc': True, 'pdf_bankruptcy': False,
        'pdf_fraud': False, 'pdf_employment_tenure': 24, 'pdf_business_vintage': 3,
        'pdf_bureau_score': 720, 'pdf_dpd_90': 0, 'pdf_dpd_30': 0,
        'pdf_credit_util': 30, 'pdf_inquiries': 2, 'pdf_active_loans': 1,
        'pdf_existing_emi': 15000, 'pdf_monthly_income': 50000,
        'pdf_annual_income': 600000, 'pdf_net_surplus': 20000,
        'pdf_salary_stability': 'STABLE', 'pdf_loan_amount': 180000,
        'pdf_loan_tenure': 24, 'pdf_interest_rate': 10.5, 'pdf_amt_annuity': 8500,
        'pdf_payment_discipline': 'GOOD', 'pdf_liquidity_flag': 'LOW',
        'pdf_cashflow_health': 'MODERATE', 'pdf_bureau_risk_flag': 'LOW',
        'pdf_inward_bounce': 0, 'pdf_salary_missing': 0,
    }
    for _k, _v in _form_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    with st.form("assessment_form"):
        # ── Identity & Eligibility ─────────────────────────────────────────
        st.markdown('<p class="section-header">👤 Identity & Eligibility</p>', unsafe_allow_html=True)
        col_name1, col_name2 = st.columns([2, 2])
        with col_name1:
            customer_name = st.text_input("Customer Name (Optional)", value="", placeholder="e.g. Ramesh Kumar")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age", 25, 70, key='pdf_age', help="Min 25 per RBI lending policy")
            employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business'],
                key='pdf_employment_type')
        with col2:
            _gender_opts = ['Male', 'Female', 'Non-binary / Other', 'Prefer not to say']
            # selectbox doesn't support key= with arbitrary string values, so use index trick
            # but re-seed index from session state each render
            _gender_val = st.session_state.get('pdf_gender', 'Male')
            _gender_idx = _gender_opts.index(_gender_val) if _gender_val in _gender_opts else 0
            gender = st.selectbox("Gender", _gender_opts, index=_gender_idx)
            dependents = st.number_input("Number of Dependents", 0, 20, key='pdf_dependents')
        with col3:
            _city_keys = list(CITY_TIERS.keys())
            city_tier_label = st.selectbox(
                "City Tier", _city_keys, index=0,
                format_func=lambda k: k
            )
            city_tier = CITY_TIERS[city_tier_label]
            st.caption(f"Stored as: **{city_tier}**")
            kyc_verified = st.selectbox("KYC Verified", ['Yes', 'No'],
                index=0 if st.session_state.get('pdf_kyc', True) else 1) == 'Yes'
        with col4:
            bankruptcy_flag = st.selectbox("Bankruptcy Flag", ['No', 'Yes'],
                index=0 if not st.session_state.get('pdf_bankruptcy', False) else 1) == 'Yes'
            fraud_flag = st.selectbox("Fraud Flag", ['No', 'Yes'],
                index=0 if not st.session_state.get('pdf_fraud', False) else 1) == 'Yes'

        # RBI Consent — REQUIRED
        st.markdown('<p class="section-header">📜 RBI Compliance</p>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            rbi_consent = st.checkbox(
                "✅ I confirm the customer has been informed of and consented to: (a) credit bureau enquiry, "
                "(b) data usage for credit assessment, (c) Key Fact Statement (KFS) terms, and "
                "(d) grievance redressal process. **(Required — RBI Digital Lending Guidelines)**",
                value=False
            )
        with col2:
            st.markdown("""
                <div style="background:#fff3cd;border:1px solid #ffc107;padding:0.5rem;border-radius:0.4rem;font-size:0.82rem;">
                    ⚠️ Without consent, the application cannot proceed per RBI DLG 2022.
                </div>
            """, unsafe_allow_html=True)

        # Employment tenure
        st.markdown('<p class="section-header">💼 Employment</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if employment_type == 'Salaried':
                employment_tenure = st.number_input("Employment Tenure (months)", 0, 600,
                    key='pdf_employment_tenure')
                business_vintage = 0
            else:
                business_vintage = st.number_input("Business Vintage (years)", 0, 50,
                    key='pdf_business_vintage')
                employment_tenure = 0
        with col2:
            st.markdown("""
                <div class="info-box" style="margin-top:1rem;">
                    <strong>Policy thresholds:</strong><br>
                    Salaried: min 6 months<br>
                    Self-Employed/Business: min 2 years
                </div>
            """, unsafe_allow_html=True)

        # Credit Bureau
        st.markdown('<p class="section-header">🏦 Credit Bureau</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            bureau_score = st.number_input("Bureau Score", 300, 900, key='pdf_bureau_score', step=10)
            dpd_90_6m    = st.number_input("DPD 90+ (Last 6M)", 0, 20, key='pdf_dpd_90')
            dpd_30_6m    = st.number_input("DPD 30+ (Last 6M)", 0, 20, key='pdf_dpd_30')
        with col2:
            credit_utilization = st.number_input("Credit Utilization (%)", 0, 100, key='pdf_credit_util')
            recent_inquiries   = st.number_input("Recent Inquiries (3M)", 0, 20, key='pdf_inquiries')
        with col3:
            active_loans = st.number_input("Active Loans", 0, 10, key='pdf_active_loans')
            existing_emi = st.number_input("Existing Total EMI (₹)", 0, 200000, key='pdf_existing_emi', step=1000)

        # Income & Financial
        st.markdown('<p class="section-header">💰 Income & Financial</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_salary = st.number_input("Monthly Income (₹)", 0, 1000000, key='pdf_monthly_income', step=5000)
            amt_income = st.number_input("Annual Income (₹)", 0, 10000000, key='pdf_annual_income', step=10000)
        with col2:
            net_surplus = st.number_input("Net Cash Surplus (₹)", -100000, 500000, key='pdf_net_surplus', step=5000)
            _ss_opts = ['STABLE', 'MODERATE', 'UNSTABLE']
            _ss_val  = st.session_state.get('pdf_salary_stability', 'STABLE')
            salary_stability = st.selectbox("Salary Stability", _ss_opts,
                index=(_ss_opts.index(_ss_val) if _ss_val in _ss_opts else 0))
        with col3:
            loan_amount  = st.number_input("Loan Amount (₹)", 0, 5000000, key='pdf_loan_amount', step=10000)
            loan_tenure  = st.number_input("Tenure (months)", 3, 360, key='pdf_loan_tenure')
        with col4:
            interest_rate = st.number_input("Interest Rate (%)", 8.0, 20.0, key='pdf_interest_rate', step=0.5)
            amt_annuity   = st.number_input("Requested EMI (₹)", 0, 200000, key='pdf_amt_annuity', step=500)

        # Additional Credit Behaviour
        st.markdown('<p class="section-header">📋 Additional Credit Behaviour</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            _pd_opts = ['GOOD', 'MODERATE', 'POOR']
            _pd_val  = st.session_state.get('pdf_payment_discipline', 'GOOD')
            payment_discipline = st.selectbox("Payment Discipline", _pd_opts,
                index=(_pd_opts.index(_pd_val) if _pd_val in _pd_opts else 0))
            _lq_opts = ['LOW', 'ADEQUATE', 'MODERATE']
            _lq_val  = st.session_state.get('pdf_liquidity_flag', 'LOW')
            liquidity_flag = st.selectbox("Liquidity", _lq_opts,
                index=(_lq_opts.index(_lq_val) if _lq_val in _lq_opts else 0))
        with col2:
            _cf_opts = ['MODERATE', 'HEALTHY', 'STRESSED', 'STABLE']
            _cf_val  = st.session_state.get('pdf_cashflow_health', 'MODERATE')
            cashflow_health = st.selectbox("Cashflow Health", _cf_opts,
                index=(_cf_opts.index(_cf_val) if _cf_val in _cf_opts else 0))
            _br_opts = ['LOW', 'MEDIUM', 'HIGH']
            _br_val  = st.session_state.get('pdf_bureau_risk_flag', 'LOW')
            bureau_risk_flag = st.selectbox("Bureau Risk", _br_opts,
                index=(_br_opts.index(_br_val) if _br_val in _br_opts else 0))
        with col3:
            inward_bounce_count   = st.number_input("Inward Bounce Count (3M)", 0, 10, key='pdf_inward_bounce')
            salary_missing_months = st.number_input("Missing Salary Months (6M)", 0, 6, key='pdf_salary_missing')

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Assess Credit Risk", use_container_width=True)

    if submitted:
        timestamp = datetime.now()
        app_id = "PL" + timestamp.strftime("%Y%m%d%H%M%S")
        customer_data = {
            'name': customer_name.strip() if customer_name.strip() else 'N/A',
            'age': age, 'employment_type': employment_type,
            'gender': gender, 'city_tier': city_tier,
            'dependents': dependents, 'kyc_verified': kyc_verified,
            'rbi_consent': rbi_consent,
            'bankruptcy_flag': bankruptcy_flag, 'fraud_flag': fraud_flag,
            'employment_tenure_months': employment_tenure,
            'business_vintage_years': business_vintage,
            'bureau_score': bureau_score,
            'dpd_90_count_6m': dpd_90_6m, 'dpd_30_count_6m': dpd_30_6m,
            'credit_utilization_pct': credit_utilization, 'max_utilization': credit_utilization,
            'recent_inquiries_3m': recent_inquiries, 'active_loans_count': active_loans,
            'avg_salary_6m': avg_salary, 'AMT_INCOME_TOTAL': amt_income,
            'net_cash_surplus_6m': net_surplus, 'salary_stability_flag': salary_stability,
            'loan_amount': loan_amount, 'loan_tenure_months': loan_tenure,
            'interest_rate': interest_rate, 'existing_emi': existing_emi,
            'AMT_ANNUITY': amt_annuity, 'application_id': app_id,
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'payment_discipline_flag': payment_discipline,
            'liquidity_flag': liquidity_flag, 'cashflow_health': cashflow_health,
            'bureau_risk_flag': bureau_risk_flag,
            'inward_bounce_count_3m': inward_bounce_count,
            'salary_missing_months': salary_missing_months,
        }

        with st.spinner("🔄 Processing Stage 1 assessment..."):
            decision_data = make_hybrid_decision_enhanced(customer_data)

        # Inject ML confidence so reason_codes.py can distinguish ML-driven REVIEW
        customer_data['ml_confidence'] = decision_data.get('confidence', 0)
        reasons = generate_reason_codes(
            decision=decision_data.get('decision', 'ERROR'),
            customer_data=customer_data,
            affordability_data=decision_data.get('affordability_data', {}),
            policy_checks=decision_data.get('policy_checks', {})
        )
        customer_data['reason_codes'] = reasons

        # Log to fairness monitor (Stage 1 — may be replaced by Stage 2 final decision)
        log_decision_for_fairness(customer_data, decision_data.get('decision','ERROR'),
                                  decision_data.get('risk_score', 0), decision_data.get('pd_percentage', 0),
                                  application_id=customer_data.get('application_id'),
                                  source='stage1')

        st.session_state.stage1_complete = True
        st.session_state.stage1_decision = decision_data.get('decision', 'ERROR')
        st.session_state.stage1_data = decision_data
        st.session_state.current_customer_data = customer_data

        # Only clear the "just extracted" flag — keep pdf_* values intact so
        # the user can re-submit or navigate back without losing OCR data.
        # Values are cleared when a new PDF is uploaded (reset_pdf button).
        st.session_state.pdf_just_extracted = False
        st.session_state.pop('_last_extraction', None)
        st.session_state.pop('_last_inferred_flags', None)

        tab1, tab2, tab3, tab4 = st.tabs(["📋 Application", "📊 Decision", "🔍 Analysis", "📝 Audit"])

        with tab1:
            st.markdown('<p class="section-header">Application Summary</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                render_info_card("👤 Identity", "👤",
                                 {"Age": age, "Gender": gender, "City Tier": city_tier,
                                  "Employment": employment_type, "Dependents": dependents,
                                  "KYC Status": "Verified" if kyc_verified else "Not Verified",
                                  "RBI Consent": "✅ Obtained" if rbi_consent else "❌ Not obtained"})
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
                st.markdown('<div class="info-box" style="background:linear-gradient(135deg,#10B981,#059669);color:white;text-align:center;"><h3 style="margin:0;color:white;">✅ Eligible for Stage 2 Deep Dive</h3></div>', unsafe_allow_html=True)
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
                st.markdown('<div style="background:linear-gradient(135deg,#EF4444,#DC2626);color:white;padding:1rem;border-radius:0.5rem;text-align:center;"><h3 style="margin:0;color:white;">❌ Stage 2 Not Available</h3><p style="margin:0.5rem 0 0 0;">Application rejected. Stage 2 only for APPROVE/REVIEW.</p></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            affordability = decision_data.get('affordability_data', {})
            foir      = affordability.get('foir_percentage', 0)
            total_emi = int(round(affordability.get('total_emi', 0)))
            net_disp  = int(round(affordability.get('net_disposable', 0)))

            col1, col2, col3 = st.columns(3)
            with col1:
                render_info_card("Identity & Eligibility", "👤",
                    {f"Age: {age}": "", f"Employment: {employment_type}": "",
                     f"City Tier: {city_tier}": "", f"Dependents: {dependents}": "",
                     f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": ""},
                    {f"Age: {age}": "pass" if (age > 24 and age <= (65 if employment_type == 'Salaried' else 70)) else "fail",
                     f"Employment: {employment_type}": "pass",
                     f"City Tier: {city_tier}": "pass",
                     f"Dependents: {dependents}": "pass" if dependents <= 5 else "warning",
                     f"KYC: {'Verified' if kyc_verified else 'Not Verified'}": "pass" if kyc_verified else "fail"})
            with col2:
                render_info_card("Credit Bureau", "🏦",
                    {f"Bureau Score: {bureau_score}": "", f"DPD 90+: {dpd_90_6m}": "",
                     f"Utilization: {credit_utilization}%": ""},
                    {f"Bureau Score: {bureau_score}": "pass" if bureau_score >= 550 else "fail",
                     f"DPD 90+: {dpd_90_6m}": "pass" if dpd_90_6m == 0 else ("warning" if dpd_90_6m <= 5 else "fail"),
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
                    st.warning("⚠️ PDF download unavailable — `reportlab` or `pdf_generator.py` not found. Check requirements.txt.")
            with col2:
                if st.button("🔄 Re-Evaluate", key="reevaluate_btn", use_container_width=True):
                    st.rerun()

        with tab3:
            st.markdown('<p class="section-header">Model Analysis</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_modern_gauge(decision_data.get('confidence', 0), "Model Confidence"), use_container_width=True)
            with col2:
                st.plotly_chart(create_modern_bar_chart(decision_data.get('class_probs', {"APPROVE": 0, "REVIEW": 0, "REJECT": 0})), use_container_width=True)
            st.markdown('<p class="section-header">Policy Checks</p>', unsafe_allow_html=True)
            policy_df = pd.DataFrame([{'Check': k, 'Result': v} for k, v in decision_data.get('policy_checks', {}).items()])
            st.dataframe(policy_df, use_container_width=True, hide_index=True)
            st.markdown('<p class="section-header">PD Calculation Breakdown</p>', unsafe_allow_html=True)
            for factor, value in {
                'Bureau Score': f"{bureau_score} → Base PD: {bureau_score_to_pd(bureau_score):.1f}%",
                'Delinquency': f"DPD 90+: {dpd_90_6m}, DPD 30+: {dpd_30_6m} → Multiplier: {delinquency_to_pd_multiplier(dpd_90_6m, dpd_30_6m):.1f}x",
                'FOIR Impact': f"{foir:.1f}% → Adjustment: {foir_to_pd_adjustment(foir):.1f}%",
                'Final PD': f"{decision_data.get('pd_percentage', 0)}%"
            }.items():
                st.markdown(f"**{factor}:** {value}")

        with tab4:
            st.markdown('<p class="section-header">Audit Trail</p>', unsafe_allow_html=True)
            audit_log = sanitize_for_json({
                'application_id': app_id,
                'timestamp': timestamp.isoformat(),
                'decision': decision_data.get('decision', 'ERROR'),
                'risk_score': decision_data.get('risk_score', 0),
                'pd_percentage': decision_data.get('pd_percentage', 0),
                'confidence': round(decision_data.get('confidence', 0), 2),
                'model_version': '8.7',
                'gender': gender, 'city_tier': city_tier,
                'rbi_consent': rbi_consent,
                'reason_codes': reasons,
                'policy_checks': decision_data.get('policy_checks', {}),
                'affordability': decision_data.get('affordability_data', {}),
                'customer_data': {k: v for k, v in customer_data.items() if k not in ['application_id','timestamp','reason_codes']},
            })
            with st.expander("📋 View Audit Log (JSON)"):
                st.json(audit_log)
            col1, col2 = st.columns(2)
            with col1:
                if PDF_AVAILABLE and generate_audit_pdf is not None:
                    try:
                        audit_pdf_buffer = generate_audit_pdf(audit_log)
                        st.download_button("📥 Download Audit Trail (PDF)", data=audit_pdf_buffer,
                                           file_name=f"audit_trail_{app_id}.pdf", mime="application/pdf",
                                           use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating audit PDF: {str(e)}")
                else:
                    st.warning("⚠️ Audit PDF download unavailable — `reportlab` or `pdf_generator.py` not found. Check requirements.txt.")
            with col2:
                st.download_button("📥 Download Audit Log (JSON)",
                                   data=json.dumps(audit_log, indent=2),
                                   file_name=f"audit_{app_id}.json", mime="application/json",
                                   use_container_width=True)

elif page == "🔬 Stage 2 Analysis":
    st.markdown('<p class="main-header">Stage 2: CIBIL Deep Dive</p>', unsafe_allow_html=True)

    if not st.session_state.get('stage1_complete', False):
        st.error("❌ You must complete Stage 1 Assessment first!")
        if st.button("← Go to Assessment", use_container_width=True):
            st.session_state.page_navigation = "👤 Assessment"
            st.rerun()
        st.stop()

    if st.session_state.get('stage1_decision') not in ['APPROVE', 'REVIEW']:
        st.error("❌ Stage 2 is only available for APPROVED or REVIEW applications!")
        if st.button("← Go Back", use_container_width=True):
            st.session_state.page_navigation = "👤 Assessment"
            st.rerun()
        st.stop()

    if not (STAGE2_AVAILABLE and is_stage2_available()):
        st.error("❌ Stage 2 model not available! Please ensure `stage2_cibil_model.pkl` is in the project directory.")
        if st.button("← Go Back", use_container_width=True):
            st.session_state.page_navigation = "👤 Assessment"
            st.rerun()
        st.stop()

    stage1_data = st.session_state.get('stage1_data', {})
    stage1_customer = st.session_state.get('current_customer_data', {})

    st.markdown(f'<div class="info-box" style="background:linear-gradient(135deg,#3B82F6,#2563EB);color:white;"><h3 style="margin:0;color:white;">📊 Stage 1 Results</h3><p style="margin:0.5rem 0 0 0;"><strong>Decision:</strong> {st.session_state.get("stage1_decision","N/A")} | <strong>Risk Score:</strong> {stage1_data.get("risk_score","N/A")}<span style="font-size:0.8em;opacity:0.8">/100</span> | <strong>App ID:</strong> {stage1_customer.get("application_id","N/A")}</p><p style="margin:0.25rem 0 0 0;font-size:0.8em;opacity:0.8;">⚠ Stage 1 score is out of 100. Stage 2 combined score is out of 1000 — different scales, do not compare directly.</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_options = ["Manual Entry", "PDF Upload", "Batch Analysis"]
    default_tab = st.session_state.get('stage2_selected_tab', 'Manual Entry')
    selected_tab = st.radio("Select input method", tab_options,
                            index=tab_options.index(default_tab) if default_tab in tab_options else 0,
                            horizontal=True, label_visibility="collapsed")

    if selected_tab == "Manual Entry":
        st.markdown('<p class="section-header">Manual CIBIL Data Entry</p>', unsafe_allow_html=True)
        with st.form("stage2_manual_form"):
            st.markdown("### 👤 Demographics & Product Enquiries")
            col1, col2, col3 = st.columns(3)
            with col1:
                gender_s2 = st.selectbox("Gender", ["Male", "Female", "Others"])
                marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Others"])
                education = st.selectbox("Education", ["Graduate", "Post Graduate", "Under Graduate", "Professional", "Others"])
            with col2:
                st.markdown("**Credit Score & History**")
                cibil_score = st.number_input("Credit Score", 300, 900, 720, 10)
                max_delinquency = st.number_input("Max Delinquency Level", 0, 100, 0)
                num_times_30dpd = st.number_input("Times 30+ DPD", 0, 50, 0)
                num_times_60dpd = st.number_input("Times 60+ DPD", 0, 50, 0)
                num_times_delinquent = st.number_input("Total Delinquent", 0, 50, 0)
            with col3:
                st.markdown("**Recent Behavior**")
                num_deliq_6m = st.number_input("Delinquencies (6M)", 0, 20, 0)
                num_deliq_12m = st.number_input("Delinquencies (12M)", 0, 20, 0)
                max_deliq_6m = st.number_input("Max Delinq (6M)", 0, 100, 0)
                max_deliq_12m = st.number_input("Max Delinq (12M)", 0, 100, 0)
                enq_L3m = st.number_input("Inquiries (3M)", 0, 20, 2)
                enq_L6m = st.number_input("Inquiries (6M)", 0, 30, 4)
                enq_L12m = st.number_input("Inquiries (12M)", 0, 50, 6)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Account Quality**")
                num_std = st.number_input("Standard Accounts", 0, 50, 3)
                num_std_6m = st.number_input("Standard (6M)", 0, 50, 3)
                num_std_12m = st.number_input("Standard (12M)", 0, 50, 3)
                num_sub = st.number_input("Sub-standard", 0, 20, 0)
                num_sub_6m = st.number_input("Sub-standard (6M)", 0, 20, 0)
                num_dbt = st.number_input("Doubtful", 0, 10, 0)
                num_lss = st.number_input("Loss", 0, 10, 0)
            with col2:
                st.markdown("**Utilization**")
                pct_active_tls = st.number_input("% Active TLs", 0.0, 1.0, 0.60, 0.01)
                pct_current_bal = st.number_input("Current Balance %", 0.0, 1.0, 0.30, 0.01)
                cc_utilization = st.number_input("CC Utilization", 0.0, 1.0, 0.35, 0.01)
                pl_utilization = st.number_input("PL Utilization", 0.0, 1.0, 0.25, 0.01)
                max_unsec_exposure = st.number_input("Max Unsec Exposure %", 0, 100, 30)
            with col3:
                st.markdown("**Demographics & Products**")
                age_cibil = st.number_input("Age", 25, 70, int(stage1_customer.get('age', 35)), help="Min 25 per RBI lending policy")
                net_monthly_income = st.number_input("Net Monthly Income", 0, 1000000, int(stage1_customer.get('avg_salary_6m', 50000)), 5000)
                time_curr_employer = st.number_input("Employment Tenure (months)", 0, 600, int(stage1_customer.get('employment_tenure_months', 24)))
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
                    st.warning(f'⚠️ CIBIL income ₹{_s2_inc:,} much lower than application income ₹{_s1_inc:,}. Using application income.')
                enhanced_customer_data.update({
                    'bureau_score': cibil_score, 'age': age_cibil,
                    'avg_salary_6m': _final_income, 'employment_tenure_months': time_curr_employer,
                    'dpd_30_count_6m': num_times_30dpd, 'dpd_90_count_6m': num_times_60dpd,
                    'max_delinquency_level': max_delinquency, 'num_times_delinquent': num_times_delinquent,
                    'num_deliq_6mts': num_deliq_6m, 'num_deliq_12mts': num_deliq_12m,
                    'max_deliq_6mts': max_deliq_6m, 'max_deliq_12mts': max_deliq_12m,
                    'recent_inquiries_3m': enq_L3m, 'enq_L6m': enq_L6m, 'enq_L12m': enq_L12m,
                    'active_loans_count': num_std, 'num_std_6mts': num_std_6m, 'num_std_12mts': num_std_12m,
                    'num_sub': num_sub, 'num_sub_6mts': num_sub_6m,
                    'num_dbt': num_dbt, 'num_lss': num_lss,
                    'credit_utilization_pct': cc_utilization * 100,
                    'pct_of_active_TLs_ever': pct_active_tls, 'pct_currentBal_all_TL': pct_current_bal,
                    'CC_utilization': cc_utilization, 'PL_utilization': pl_utilization,
                    'max_unsec_exposure_inPct': max_unsec_exposure,
                    'CC_Flag': 1 if cc_flag else 0, 'PL_Flag': 1 if pl_flag else 0,
                    'HL_Flag': 1 if hl_flag else 0, 'GL_Flag': 1 if gl_flag else 0,
                    'GENDER': gender_s2, 'MARITALSTATUS': marital_status, 'EDUCATION': education,
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
            st.warning("Please use the **Manual Entry** tab.")
        else:
            uploaded_pdf = st.file_uploader("Upload CIBIL Report (PDF)", type=['pdf'], key="stage2_pdf")
            if uploaded_pdf is not None:
                st.success(f"✅ File uploaded: {uploaded_pdf.name} ({uploaded_pdf.size/1024:.1f} KB)")
                if st.button("🔬 Extract & Analyze", key="extract_analyze_stage2", type="primary", use_container_width=True):
                    with st.spinner("🔄 Extracting data from PDF..."):
                        extraction_result = extract_cibil_from_pdf(uploaded_pdf)
                    if extraction_result.get('success', False):
                        st.success("✅ PDF extraction successful!")

                        # ── Summary metrics ──────────────────────────────────
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Credit Score",    extraction_result.get('Credit_Score', 'N/A'))
                        c2.metric("DPD 30+ Count",   extraction_result.get('num_times_30p_dpd', 0))
                        c3.metric("DPD 60+ Count",   extraction_result.get('num_times_60p_dpd', 0))
                        c4.metric("Active Accounts", extraction_result.get('num_std', 0))
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Monthly Income", f"₹{extraction_result.get('NETMONTHLYINCOME', 0):,}")
                        c2.metric("Employment Tenure", f"{extraction_result.get('Time_With_Curr_Empr',0)} mo")
                        c3.metric("Written Off",    extraction_result.get('num_lss', 0))
                        c4.metric("Enquiries (3M)", extraction_result.get('enq_L3m', 0))
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Payment Discipline", extraction_result.get('payment_discipline_flag','—'))
                        c2.metric("Cashflow Health",    extraction_result.get('cashflow_health','—'))
                        c3.metric("Bureau Risk",        extraction_result.get('bureau_risk_flag','—'))
                        c4.metric("Salary Stability",   extraction_result.get('salary_stability_flag','—'))

                        if extraction_result.get('written_off_count', 0) > 0:
                            st.warning(f"⚠️ {extraction_result['written_off_count']} written-off accounts detected — score may be overridden.")

                        _surplus_proxy = extraction_result.get('_surplus_proxy', 0)
                        if _surplus_proxy:
                            st.info(f"💡 Bureau-only PDF — net surplus estimated from income: ₹{_surplus_proxy:,}")

                        with st.expander("📋 View all extracted fields"):
                            _display = {k: v for k, v in extraction_result.items() if k not in ('raw_text','success','extraction_method')}
                            st.json(_display)

                        # ── Build enhanced_customer_data ─────────────────────
                        # Start from Stage 1 customer (has gender, city_tier, rbi_consent, loan details)
                        enhanced_customer_data = stage1_customer.copy()

                        # Apply ALL extracted fields directly — the new extractor maps every column
                        _skip = {'raw_text', 'success', 'extraction_method',
                                 'loan_amount', 'loan_tenure_months', 'interest_rate',
                                 'rbi_consent', 'kyc_verified', 'bankruptcy_flag', 'fraud_flag'}
                        for k, v in extraction_result.items():
                            if k not in _skip and v is not None:
                                enhanced_customer_data[k] = v

                        # Income safety: if CIBIL income << Stage 1 application income, keep Stage 1
                        _s1_inc = stage1_customer.get('avg_salary_6m', 50000)
                        _s2_inc = extraction_result.get('NETMONTHLYINCOME', 0) or 0
                        if 0 < _s2_inc < _s1_inc * 0.4:
                            enhanced_customer_data['avg_salary_6m'] = _s1_inc
                            enhanced_customer_data['AMT_INCOME_TOTAL'] = _s1_inc * 12
                            st.warning(f"⚠️ CIBIL income ₹{_s2_inc:,} << application income ₹{_s1_inc:,} — using application income for FOIR.")

                        # Sentinel cleanup
                        enhanced_customer_data = clean_sentinel_values(enhanced_customer_data)

                        with st.spinner("🔬 Running Stage 2 analysis..."):
                            try:
                                stage2_result = make_two_stage_decision(enhanced_customer_data, stage1_function=make_hybrid_decision_enhanced)
                                stage2_result = resolve_stage2_to_binary(stage2_result)
                                display_stage2_results(stage2_result, stage1_data, stage1_customer, enhanced_customer_data)
                            except Exception as e:
                                st.error(f"❌ Analysis failed: {str(e)}")
                                st.exception(e)
                    else:
                        st.error("❌ PDF extraction failed: " + extraction_result.get('error', 'Unknown'))

    elif selected_tab == "Batch Analysis":
        st.info("📊 Stage 2 Batch analysis coming soon.")

elif page == "⚖️ Fairness":
    render_fairness_dashboard()

elif page == "📊 Batch Process":
    st.markdown('<p class="main-header">Batch Processing</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">📤 Upload a CSV file with customer data for bulk credit assessment.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} records")
            with st.expander("📄 Preview Uploaded Data"):
                st.dataframe(df.head(), use_container_width=True)
            required_cols = ['age', 'employment_type', 'avg_salary_6m', 'bureau_score', 'loan_amount']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("🚀 Process Batch Predictions", key="process_batch_btn", type="primary", use_container_width=True):
                    with st.spinner(f"🔍 Processing {len(df)} records..."):
                        results_df = process_batch_predictions(df)
                    st.success(f"✅ Completed {len(results_df)} records!")
                    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Analytics", "📥 Download"])
                    with tab1:
                        st.dataframe(results_df, use_container_width=True)
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: st.metric("✅ Approved", len(results_df[results_df['decision'] == 'APPROVE']))
                        with c2: st.metric("❌ Rejected", len(results_df[results_df['decision'] == 'REJECT']))
                        with c3: st.metric("⚠️ Review", len(results_df[results_df['decision'] == 'REVIEW']))
                        with c4: st.metric("📊 Avg Risk Score", f"{results_df['risk_score'].mean():.0f}")
                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            dc = results_df['decision'].value_counts()
                            fig1 = px.pie(values=dc.values, names=dc.index, title="Decision Distribution",
                                          color=dc.index, color_discrete_map={'APPROVE':'#48bb78','REVIEW':'#ed8936','REJECT':'#f56565'})
                            st.plotly_chart(fig1, use_container_width=True)
                        with col2:
                            fig2 = px.histogram(results_df, x='risk_score', title="Risk Score Distribution",
                                                nbins=20, color_discrete_sequence=['#587042'])
                            st.plotly_chart(fig2, use_container_width=True)
                        # Fairness charts from batch
                        if 'gender' in results_df.columns and results_df['gender'].nunique() > 1:
                            results_df['approved_num'] = (results_df['decision'] == 'APPROVE').astype(int)
                            grp = results_df.groupby('gender')['approved_num'].mean().reset_index()
                            grp['Approval Rate %'] = (grp['approved_num'] * 100).round(1)
                            fig3 = px.bar(grp, x='gender', y='Approval Rate %', title='Approval Rate by Gender (Batch)',
                                          color='Approval Rate %', color_continuous_scale=['#f56565','#48bb78'], range_color=[0,100])
                            st.plotly_chart(fig3, use_container_width=True)
                        if 'city_tier' in results_df.columns and results_df['city_tier'].nunique() > 1:
                            results_df['approved_num'] = (results_df['decision'] == 'APPROVE').astype(int)
                            grp2 = results_df.groupby('city_tier')['approved_num'].mean().reset_index()
                            grp2['Approval Rate %'] = (grp2['approved_num'] * 100).round(1)
                            fig4 = px.bar(grp2, x='city_tier', y='Approval Rate %', title='Approval Rate by City Tier (Batch)',
                                          color='Approval Rate %', color_continuous_scale=['#f56565','#48bb78'], range_color=[0,100])
                            st.plotly_chart(fig4, use_container_width=True)
                    with tab3:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button("📥 Download as CSV", data=results_df.to_csv(index=False),
                                               file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                               mime="text/csv", use_container_width=True)
                        with col2:
                            st.download_button("📥 Download as JSON", data=results_df.to_json(orient='records', indent=2),
                                               file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                               mime="application/json", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    else:
        st.markdown("---")
        st.markdown("### 📋 CSV Template")
        template_data = {
            'age': [35, 42, 28], 'gender': ['Male', 'Female', 'Male'],
            'city_tier': ['Tier 1', 'Tier 2', 'Tier 3'],
            'employment_type': ['Salaried', 'Self-Employed', 'Salaried'],
            'dependents': [2, 3, 6], 'kyc_verified': ['Yes', 'Yes', 'No'],
            'bankruptcy_flag': ['No', 'No', 'No'], 'fraud_flag': ['No', 'No', 'No'],
            'rbi_consent': ['Yes', 'Yes', 'Yes'],
            'employment_tenure_months': [24, 0, 18], 'business_vintage_years': [0, 5, 0],
            'bureau_score': [720, 680, 580], 'dpd_90_count_6m': [0, 1, 2],
            'dpd_30_count_6m': [0, 2, 1], 'credit_utilization_pct': [30, 45, 75],
            'recent_inquiries_3m': [2, 1, 5], 'active_loans_count': [1, 2, 3],
            'avg_salary_6m': [50000, 75000, 35000], 'AMT_INCOME_TOTAL': [600000, 900000, 420000],
            'net_cash_surplus_6m': [20000, 35000, 10000],
            'salary_stability_flag': ['STABLE', 'MODERATE', 'UNSTABLE'],
            'loan_amount': [180000, 250000, 100000], 'loan_tenure_months': [24, 36, 12],
            'interest_rate': [10.5, 11.0, 12.0], 'existing_emi': [15000, 20000, 8000],
            'AMT_ANNUITY': [8500, 9500, 4500],
            'payment_discipline_flag': ['GOOD', 'MODERATE', 'POOR'],
            'liquidity_flag': ['LOW', 'ADEQUATE', 'LOW'],
            'cashflow_health': ['HEALTHY', 'MODERATE', 'STRESSED'],
            'bureau_risk_flag': ['LOW', 'MEDIUM', 'HIGH'],
            'inward_bounce_count_3m': [0, 1, 3], 'salary_missing_months': [0, 0, 2],
        }
        template_df = pd.DataFrame(template_data)
        st.dataframe(template_df, use_container_width=True)
        st.caption("📝 New columns: `gender`, `city_tier`, `rbi_consent` — required for fairness monitoring and compliance.")
        st.download_button("📥 Download CSV Template", data=template_df.to_csv(index=False),
                           file_name="credit_assessment_template_v8.7.csv",
                           mime="text/csv", use_container_width=True)

elif page == "📈 Model Info":
    st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f'<div class="stat-card"><div class="stat-number">RF</div><div class="stat-label">Model Type</div></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TOP_FEATURES)}</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="stat-card"><div class="stat-number">{len(TARGET_LE.classes_)}</div><div class="stat-label">Classes</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Top Features</p>', unsafe_allow_html=True)
    feature_df = pd.DataFrame({'Rank': range(1, min(21, len(TOP_FEATURES)+1)), 'Feature': TOP_FEATURES[:20]})
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

elif page == "ℹ️ About":
    st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-card">
            <div class="info-card-title">🏦 Credit Risk Assessment Platform</div>
            <div class="info-card-content">
                <p><strong>Version:</strong> 8.7 — Dead code removed, all audit fixes applied (M1–M4, D1–D4)</p>
                <p><strong>Developer:</strong> Zen Meraki</p>
                <p><strong>Date:</strong> January 2026</p>
                <br>
                <p>A comprehensive credit risk evaluation system combining hard policy rules,
                machine learning, and affordability analysis for accurate and RBI-compliant lending decisions.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="info-card">
                <div class="info-card-title">🎯 Key Features</div>
                <div class="info-card-content"><ul style="margin:0;padding-left:1.25rem;">
                    <li>Three-layer decision engine</li>
                    <li>Real-time risk assessment</li>
                    <li>Industry-standard PD calculation</li>
                    <li>FOIR calculation & validation</li>
                    <li>Automated reason generation</li>
                    <li>Complete audit trail (PDF)</li>
                    <li>OCR auto-fill with categorical inference</li>
                    <li>⚖️ Fairness monitoring dashboard</li>
                    <li>🏙️ City Tier field for geographic equity</li>
                    <li>📜 RBI consent gate (DLG 2022)</li>
                </ul></div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="info-card">
                <div class="info-card-title">🛠️ Technology Stack</div>
                <div class="info-card-content"><ul style="margin:0;padding-left:1.25rem;">
                    <li>Streamlit (UI Framework)</li>
                    <li>Scikit-learn (ML)</li>
                    <li>Plotly (Visualizations)</li>
                    <li>Pandas (Data Processing)</li>
                    <li>ReportLab (PDF Generation)</li>
                    <li>Tesseract OCR + pdf2image</li>
                    <li>Python 3.8+</li>
                </ul></div>
            </div>
        """, unsafe_allow_html=True)
