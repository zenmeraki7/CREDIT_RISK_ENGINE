# """
# STAGE 2 CIBIL DEEP DIVE ENGINE
# Separate module for 2-stage credit risk system

# Author: Zen Meraki
# Date: February 2026
# Version: 1.0

# Usage in test.py:
#     from stage2_engine import make_two_stage_decision, is_stage2_available, get_stage2_status
# """

# import joblib
# import os
# import numpy as np
# import streamlit as st

# # =============================================================================
# # LOAD STAGE 2 MODEL (CIBIL DEEP DIVE)
# # =============================================================================

# @st.cache_resource
# def load_stage2_model():
#     """Load Stage 2 CIBIL model if available"""
#     try:
#         # Try multiple possible locations
#         model_paths = [
#             'stage2_cibil_model.pkl',
#             'models/stage2_cibil_model.pkl',
#             '/mnt/user-data/outputs/stage2_cibil_model.pkl',
#             './stage2_cibil_model.pkl'
#         ]
        
#         for path in model_paths:
#             if os.path.exists(path):
#                 assets = joblib.load(path)
#                 return {
#                     'loaded': True,
#                     'model': assets['model'],
#                     'features': assets['features'],
#                     'label_encoder': assets['label_encoder'],
#                     'feature_importance': assets.get('feature_importance', None),
#                     'test_accuracy': assets.get('test_accuracy', 0),
#                     'path': path,
#                     'error': None
#                 }
        
#         return {
#             'loaded': False,
#             'error': 'Stage 2 model not found. Place stage2_cibil_model.pkl in project root.'
#         }
    
#     except Exception as e:
#         return {
#             'loaded': False,
#             'error': f'Error loading Stage 2 model: {str(e)}'
#         }


# # Load Stage 2 model at module import
# STAGE2_ASSETS = load_stage2_model()


# # =============================================================================
# # MAP STAGE 1 DATA TO STAGE 2 CIBIL FORMAT
# # =============================================================================

# def prepare_stage2_input(customer_data, stage2_features):
#     """
#     Map customer data from Stage 1 format to Stage 2 CIBIL format
    
#     Args:
#         customer_data: Dictionary from Stage 1 (test.py format)
#         stage2_features: List of feature names expected by Stage 2 model
    
#     Returns:
#         Array of values in correct order for Stage 2 prediction
#     """
    
#     # Mapping from Stage 1 fields to Stage 2 CIBIL fields
#     mapping = {
#         # Credit Score & Basic Info
#         'Credit_Score': customer_data.get('bureau_score', 700),
#         'AGE': customer_data.get('age', 35),
#         'NETMONTHLYINCOME': customer_data.get('avg_salary_6m', 30000),
#         'Time_With_Curr_Empr': customer_data.get('employment_tenure_months', 24),
        
#         # Delinquency
#         'num_times_30p_dpd': customer_data.get('dpd_30_count_6m', 0),
#         'num_times_60p_dpd': customer_data.get('dpd_90_count_6m', 0),
#         'num_times_delinquent': customer_data.get('dpd_90_count_6m', 0) + customer_data.get('dpd_30_count_6m', 0),
#         'max_delinquency_level': 0 if customer_data.get('dpd_90_count_6m', 0) == 0 else 50,
        
#         # Delinquency (6 months)
#         'num_deliq_6mts': customer_data.get('dpd_90_count_6m', 0),
#         'max_deliq_6mts': 0 if customer_data.get('dpd_90_count_6m', 0) == 0 else 50,
        
#         # Delinquency (12 months) - estimate from 6 months
#         'num_deliq_12mts': customer_data.get('dpd_90_count_6m', 0),
#         'max_deliq_12mts': 0 if customer_data.get('dpd_90_count_6m', 0) == 0 else 50,
        
#         # Credit Inquiries
#         'enq_L3m': customer_data.get('recent_inquiries_3m', 0),
#         'enq_L6m': customer_data.get('recent_inquiries_3m', 0) * 2,
#         'enq_L12m': customer_data.get('recent_inquiries_3m', 0) * 4,
        
#         # Account Quality
#         'num_std': customer_data.get('active_loans_count', 1),
#         'num_std_6mts': customer_data.get('active_loans_count', 1),
#         'num_std_12mts': customer_data.get('active_loans_count', 1),
#         'num_sub': 0,  # Assume no sub-standard
#         'num_sub_6mts': 0,
#         'num_dbt': 0,  # Assume no doubtful
#         'num_lss': 0,  # Assume no loss
        
#         # Utilization
#         'pct_currentBal_all_TL': customer_data.get('credit_utilization_pct', 30) / 100,
#         'CC_utilization': customer_data.get('credit_utilization_pct', 30) / 100,
#         'PL_utilization': customer_data.get('credit_utilization_pct', 30) / 100,
#         'max_unsec_exposure_inPct': customer_data.get('credit_utilization_pct', 30),
#         'pct_of_active_TLs_ever': 0.5,
        
#         # Product Flags
#         'CC_Flag': 1 if customer_data.get('credit_utilization_pct', 0) > 0 else 0,
#         'PL_Flag': 1,
#         'HL_Flag': 0,
#         'GL_Flag': 0,
        
#         # Engineered Features
#         'delinq_severity_score': 0 if customer_data.get('dpd_90_count_6m', 0) == 0 else 16.67,
#         'high_dpd_risk': 1 if customer_data.get('dpd_90_count_6m', 0) > 0 else 0,
#         'recent_deliq_flag': 1 if customer_data.get('dpd_90_count_6m', 0) > 0 else 0,
#         'credit_hungry': 1 if customer_data.get('recent_inquiries_3m', 0) > 3 else 0,
#         'account_quality_score': customer_data.get('active_loans_count', 1) * 10,
#         'high_util_flag': 1 if customer_data.get('credit_utilization_pct', 0) > 75 else 0,
#         'employment_stable': 1 if customer_data.get('employment_tenure_months', 0) >= 24 else 0
#     }
    
#     # Build input array in the exact order expected by the model
#     input_array = []
#     for feature in stage2_features:
#         value = mapping.get(feature, 0)
#         input_array.append(value)
    
#     return input_array


# # =============================================================================
# # DECISION MATRIX FOR TWO-STAGE SYSTEM
# # =============================================================================

# def apply_two_stage_decision_matrix(stage1_decision, stage2_tier, 
#                                     stage1_risk_score, stage2_confidence):
#     """
#     Apply decision matrix to combine Stage 1 + Stage 2
    
#     Decision Rules:
#     ---------------
#     Stage 1 APPROVE:
#         + P1/P2 → APPROVE (Premium: 8.5-10%)
#         + P3    → APPROVE (Standard: 12-14%)
#         + P4    → MANUAL REVIEW
    
#     Stage 1 REVIEW:
#         + P1/P2 → APPROVE (Standard: 10-11%)
#         + P3    → MANUAL REVIEW
#         + P4    → REJECT
    
#     Stage 1 REJECT:
#         + Any   → REJECT (no Stage 2 needed)
#     """
    
#     if stage1_decision == "APPROVE":
#         if stage2_tier in ['P1', 'P2']:
#             return {
#                 'final_decision': 'APPROVE',
#                 'tier': f'{stage2_tier} - Premium',
#                 'interest_rate_range': '8.5% - 10.0%',
#                 'reason': f'Excellent credit profile. Stage 1: APPROVE, Stage 2: {stage2_tier}'
#             }
#         elif stage2_tier == 'P3':
#             return {
#                 'final_decision': 'APPROVE',
#                 'tier': 'P3 - Standard',
#                 'interest_rate_range': '12.0% - 14.0%',
#                 'reason': f'Acceptable credit with some concerns. Stage 1: APPROVE, Stage 2: P3'
#             }
#         else:  # P4
#             return {
#                 'final_decision': 'MANUAL_REVIEW',
#                 'tier': 'P4 - High Risk',
#                 'interest_rate_range': 'To be determined',
#                 'reason': f'High risk detected in CIBIL analysis. Needs underwriter review.'
#             }
    
#     elif stage1_decision == "REVIEW":
#         if stage2_tier in ['P1', 'P2']:
#             return {
#                 'final_decision': 'APPROVE',
#                 'tier': f'{stage2_tier} - Standard',
#                 'interest_rate_range': '10.0% - 11.0%',
#                 'reason': f'Stage 1 concerns overridden by strong CIBIL profile ({stage2_tier})'
#             }
#         elif stage2_tier == 'P3':
#             return {
#                 'final_decision': 'MANUAL_REVIEW',
#                 'tier': 'P3 - Borderline',
#                 'interest_rate_range': 'To be determined',
#                 'reason': 'Mixed signals from both stages. Requires underwriter judgment.'
#             }
#         else:  # P4
#             return {
#                 'final_decision': 'REJECT',
#                 'tier': None,
#                 'interest_rate_range': 'N/A',
#                 'reason': 'Failed both Stage 1 and Stage 2 validation'
#             }
    
#     else:  # REJECT
#         return {
#             'final_decision': 'REJECT',
#             'tier': None,
#             'interest_rate_range': 'N/A',
#             'reason': 'Failed Stage 1 policy gates'
#         }


# # =============================================================================
# # MAIN TWO-STAGE DECISION FUNCTION
# # =============================================================================

# def make_two_stage_decision(customer_data, stage1_function):
#     """
#     Complete two-stage decision engine
    
#     Flow:
#     1. Run Stage 1 (passed as parameter)
#     2. If REJECT → stop, return REJECT
#     3. If APPROVE/REVIEW → run Stage 2 CIBIL analysis
#     4. Apply decision matrix
#     5. Return final decision with tier and interest rate
    
#     Args:
#         customer_data: Customer information dictionary
#         stage1_function: The make_hybrid_decision_enhanced function from test.py
    
#     Returns:
#         Dictionary with final decision, tier, rates, and reasoning
#     """
    
#     # ========================================
#     # STAGE 1: QUICK SCREENING
#     # ========================================
    
#     stage1_result = stage1_function(customer_data)
    
#     stage1_decision = stage1_result['decision']
#     stage1_risk_score = stage1_result['risk_score']
#     stage1_pd = stage1_result['pd_percentage']
#     stage1_confidence = stage1_result['confidence']
    
#     # If rejected at Stage 1, no need for Stage 2
#     if stage1_decision == "REJECT":
#         return {
#             'final_decision': 'REJECT',
#             'tier': None,
#             'interest_rate_range': 'N/A',
#             'stage1_decision': stage1_decision,
#             'stage2_tier': None,
#             'stage2_confidence': None,
#             'combined_risk_score': stage1_risk_score,
#             'pd_percentage': stage1_pd,
#             'reason': stage1_result['reason'],
#             'stage1_details': stage1_result,
#             'stage2_used': False
#         }
    
#     # ========================================
#     # STAGE 2: CIBIL DEEP DIVE
#     # ========================================
    
#     if not STAGE2_ASSETS['loaded']:
#         # Stage 2 model not available, use Stage 1 only
#         return {
#             'final_decision': stage1_decision,
#             'tier': 'N/A (Stage 2 not available)',
#             'interest_rate_range': '10.0% - 12.0%',
#             'stage1_decision': stage1_decision,
#             'stage2_tier': None,
#             'stage2_confidence': None,
#             'combined_risk_score': stage1_risk_score,
#             'pd_percentage': stage1_pd,
#             'reason': stage1_result['reason'] + ' (Stage 2 model not loaded)',
#             'stage1_details': stage1_result,
#             'stage2_used': False
#         }
    
#     # Prepare input for Stage 2
#     stage2_features = STAGE2_ASSETS['features']
#     stage2_input = prepare_stage2_input(customer_data, stage2_features)
    
#     # Predict tier using Stage 2 model
#     stage2_model = STAGE2_ASSETS['model']
#     stage2_le = STAGE2_ASSETS['label_encoder']
    
#     try:
#         # Reshape input for prediction
#         stage2_input_array = np.array(stage2_input).reshape(1, -1)
        
#         # Predict
#         tier_idx = stage2_model.predict(stage2_input_array)[0]
#         tier_proba = stage2_model.predict_proba(stage2_input_array)[0]
        
#         # Decode
#         stage2_tier = stage2_le.inverse_transform([tier_idx])[0]
#         stage2_confidence = max(tier_proba) * 100
        
#         # Get tier probabilities
#         tier_probs = {
#             tier: prob * 100 
#             for tier, prob in zip(stage2_le.classes_, tier_proba)
#         }
        
#     except Exception as e:
#         # If Stage 2 prediction fails, fallback to Stage 1 only
#         return {
#             'final_decision': stage1_decision,
#             'tier': 'N/A (Stage 2 error)',
#             'interest_rate_range': '10.0% - 12.0%',
#             'stage1_decision': stage1_decision,
#             'stage2_tier': None,
#             'stage2_confidence': None,
#             'combined_risk_score': stage1_risk_score,
#             'pd_percentage': stage1_pd,
#             'reason': f'{stage1_result["reason"]} (Stage 2 failed: {str(e)})',
#             'stage1_details': stage1_result,
#             'stage2_used': False,
#             'stage2_error': str(e)
#         }
    
#     # ========================================
#     # APPLY DECISION MATRIX
#     # ========================================
    
#     matrix_result = apply_two_stage_decision_matrix(
#         stage1_decision=stage1_decision,
#         stage2_tier=stage2_tier,
#         stage1_risk_score=stage1_risk_score,
#         stage2_confidence=stage2_confidence
#     )
    
#     # Calculate combined risk score (weighted average)
#     # Stage 1: 40%, Stage 2: 60%
#     tier_score_map = {'P1': 900, 'P2': 750, 'P3': 600, 'P4': 450}
#     stage2_score = tier_score_map.get(stage2_tier, 500)
#     combined_risk_score = int(0.40 * stage1_risk_score + 0.60 * stage2_score)
    
#     # Return complete result
#     return {
#         'final_decision': matrix_result['final_decision'],
#         'tier': matrix_result['tier'],
#         'interest_rate_range': matrix_result['interest_rate_range'],
#         'stage1_decision': stage1_decision,
#         'stage2_tier': stage2_tier,
#         'stage2_confidence': round(stage2_confidence, 2),
#         'tier_probabilities': tier_probs,
#         'combined_risk_score': combined_risk_score,
#         'stage1_risk_score': stage1_risk_score,
#         'stage2_risk_score': stage2_score,
#         'pd_percentage': stage1_pd,
#         'reason': matrix_result['reason'],
#         'stage1_details': stage1_result,
#         'stage2_used': True,
#         'stage2_model_accuracy': STAGE2_ASSETS.get('test_accuracy', 0)
#     }


# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def is_stage2_available():
#     """Check if Stage 2 model is loaded and available"""
#     return STAGE2_ASSETS['loaded']


# def get_stage2_status():
#     """Get Stage 2 model status for display"""
#     if STAGE2_ASSETS['loaded']:
#         return {
#             'status': '✅ Loaded',
#             'accuracy': f"{STAGE2_ASSETS.get('test_accuracy', 0) * 100:.2f}%",
#             'features': len(STAGE2_ASSETS['features']),
#             'classes': ', '.join(STAGE2_ASSETS['label_encoder'].classes_),
#             'path': STAGE2_ASSETS.get('path', 'Unknown')
#         }
#     else:
#         return {
#             'status': '❌ Not Loaded',
#             'error': STAGE2_ASSETS.get('error', 'Unknown error'),
#             'accuracy': 'N/A',
#             'features': 0,
#             'classes': 'N/A',
#             'path': 'N/A'
#         }


# # =============================================================================
# # MODULE INFO
# # =============================================================================

# def get_module_info():
#     """Get module information"""
#     return {
#         'name': 'Stage 2 CIBIL Deep Dive Engine',
#         'version': '1.0',
#         'author': 'Zen Meraki',
#         'stage2_loaded': STAGE2_ASSETS['loaded'],
#         'stage2_status': get_stage2_status()
#     }


# # Print status on import
# if __name__ != "__main__":
#     status = get_stage2_status()
#     print(f"🔬 Stage 2 Engine: {status['status']}")
#     if STAGE2_ASSETS['loaded']:
#         print(f"   Accuracy: {status['accuracy']}")
#         print(f"   Features: {status['features']}")
#         print(f"   Path: {status['path']}")




"""
STAGE 2 CIBIL DEEP DIVE ENGINE
Separate module for 2-stage credit risk system

Author: Zen Meraki
Date: February 2026
Version: 1.0

Usage in test.py:
    from stage2_engine import make_two_stage_decision, is_stage2_available, get_stage2_status
"""

import joblib
import os
import numpy as np
import streamlit as st

import joblib

def load_model():
    model = joblib.load("notebooks/stage2_cibil_model.pkl")
    return model


# =============================================================================
# LOAD STAGE 2 MODEL (CIBIL DEEP DIVE)
# =============================================================================

@st.cache_resource
def load_stage2_model():
    """Load Stage 2 CIBIL model if available"""
    try:
        # Try multiple possible locations
        model_paths = [
            'stage2_cibil_model.pkl',
            'models/stage2_cibil_model.pkl',
              'notebooks/stage2_cibil_model.pkl',           # ADD THIS LINE
              './stage2_cibil_model.pkl',
              '../stage2_cibil_model.pkl'                   # ADD THIS LINE TOO
              ]
        
        for path in model_paths:
            if os.path.exists(path):
                
                assets = joblib.load(path)
                return {
                    'loaded': True,
                    'model': assets['model'],
                    'features': assets['features'],
                    'label_encoder': assets['label_encoder'],
                    'feature_importance': assets.get('feature_importance', None),
                    'test_accuracy': assets.get('test_accuracy', 0),
                    'path': path,
                    'error': None
                }
        
        return {
            'loaded': False,
            'error': 'Stage 2 model not found. Place stage2_cibil_model.pkl in project root.'
        }
    
    except Exception as e:
        return {
            'loaded': False,
            'error': f'Error loading Stage 2 model: {str(e)}'
        }


# Load Stage 2 model at module import
STAGE2_ASSETS = load_stage2_model()


# =============================================================================
# MAP STAGE 1 DATA TO STAGE 2 CIBIL FORMAT
# =============================================================================

def prepare_stage2_input(customer_data, stage2_features):
    """
    Map customer data from Stage 1 format to Stage 2 CIBIL format
    
    Args:
        customer_data: Dictionary from Stage 1 (test.py format)
        stage2_features: List of feature names expected by Stage 2 model
    
    Returns:
        Array of values in correct order for Stage 2 prediction
    """
    
    # Mapping from Stage 1 fields to Stage 2 CIBIL fields
    mapping = {
        # Credit Score & Basic Info
        'Credit_Score': customer_data.get('bureau_score', 700),
        'AGE': customer_data.get('age', 35),
        'NETMONTHLYINCOME': customer_data.get('avg_salary_6m', 30000),
        'Time_With_Curr_Empr': customer_data.get('employment_tenure_months', 24),
        
        # Delinquency
        'num_times_30p_dpd': customer_data.get('dpd_30_count_6m', 0),
        'num_times_60p_dpd': customer_data.get('dpd_90_count_6m', 0),
        'num_times_delinquent': customer_data.get('dpd_90_count_6m', 0) + customer_data.get('dpd_30_count_6m', 0),
        'max_delinquency_level': 0 if customer_data.get('dpd_90_count_6m', 0) == 0 else 50,
        
        # Delinquency (6 months)
        'num_deliq_6mts': customer_data.get('dpd_90_count_6m', 0),
        'max_deliq_6mts': 0 if customer_data.get('dpd_90_count_6m', 0) == 0 else 50,
        
        # Delinquency (12 months) - estimate from 6 months
        'num_deliq_12mts': customer_data.get('dpd_90_count_6m', 0),
        'max_deliq_12mts': 0 if customer_data.get('dpd_90_count_6m', 0) == 0 else 50,
        
        # Credit Inquiries
        'enq_L3m': customer_data.get('recent_inquiries_3m', 0),
        'enq_L6m': customer_data.get('recent_inquiries_3m', 0) * 2,
        'enq_L12m': customer_data.get('recent_inquiries_3m', 0) * 4,
        
        # Account Quality
        'num_std': customer_data.get('active_loans_count', 1),
        'num_std_6mts': customer_data.get('active_loans_count', 1),
        'num_std_12mts': customer_data.get('active_loans_count', 1),
        'num_sub': 0,  # Assume no sub-standard
        'num_sub_6mts': 0,
        'num_dbt': 0,  # Assume no doubtful
        'num_lss': 0,  # Assume no loss
        
        # Utilization
        'pct_currentBal_all_TL': customer_data.get('credit_utilization_pct', 30) / 100,
        'CC_utilization': customer_data.get('credit_utilization_pct', 30) / 100,
        'PL_utilization': customer_data.get('credit_utilization_pct', 30) / 100,
        'max_unsec_exposure_inPct': customer_data.get('credit_utilization_pct', 30),
        'pct_of_active_TLs_ever': 0.5,
        
        # Product Flags
        'CC_Flag': 1 if customer_data.get('credit_utilization_pct', 0) > 0 else 0,
        'PL_Flag': 1,
        'HL_Flag': 0,
        'GL_Flag': 0,
        
        # Engineered Features
        'delinq_severity_score': 0 if customer_data.get('dpd_90_count_6m', 0) == 0 else 16.67,
        'high_dpd_risk': 1 if customer_data.get('dpd_90_count_6m', 0) > 0 else 0,
        'recent_deliq_flag': 1 if customer_data.get('dpd_90_count_6m', 0) > 0 else 0,
        'credit_hungry': 1 if customer_data.get('recent_inquiries_3m', 0) > 3 else 0,
        'account_quality_score': customer_data.get('active_loans_count', 1) * 10,
        'high_util_flag': 1 if customer_data.get('credit_utilization_pct', 0) > 75 else 0,
        'employment_stable': 1 if customer_data.get('employment_tenure_months', 0) >= 24 else 0
    }
    
    # Build input array in the exact order expected by the model
    input_array = []
    for feature in stage2_features:
        value = mapping.get(feature, 0)
        input_array.append(value)
    
    return input_array


# =============================================================================
# DECISION MATRIX FOR TWO-STAGE SYSTEM
# =============================================================================

def apply_two_stage_decision_matrix(stage1_decision, stage2_tier, 
                                    stage1_risk_score, stage2_confidence):
    """
    Apply decision matrix to combine Stage 1 + Stage 2
    
    Decision Rules:
    ---------------
    Stage 1 APPROVE:
        + P1/P2 → APPROVE (Premium: 8.5-10%)
        + P3    → APPROVE (Standard: 12-14%)
        + P4    → MANUAL REVIEW
    
    Stage 1 REVIEW:
        + P1/P2 → APPROVE (Standard: 10-11%)
        + P3    → MANUAL REVIEW
        + P4    → REJECT
    
    Stage 1 REJECT:
        + Any   → REJECT (no Stage 2 needed)
    """
    
    if stage1_decision == "APPROVE":
        if stage2_tier in ['P1', 'P2']:
            return {
                'final_decision': 'APPROVE',
                'tier': f'{stage2_tier} - Premium',
                'interest_rate_range': '8.5% - 10.0%',
                'reason': f'Excellent credit profile. Stage 1: APPROVE, Stage 2: {stage2_tier}'
            }
        elif stage2_tier == 'P3':
            return {
                'final_decision': 'APPROVE',
                'tier': 'P3 - Standard',
                'interest_rate_range': '12.0% - 14.0%',
                'reason': f'Acceptable credit with some concerns. Stage 1: APPROVE, Stage 2: P3'
            }
        else:  # P4
            return {
                'final_decision': 'MANUAL_REVIEW',
                'tier': 'P4 - High Risk',
                'interest_rate_range': 'To be determined',
                'reason': f'High risk detected in CIBIL analysis. Needs underwriter review.'
            }
    
    elif stage1_decision == "REVIEW":
        if stage2_tier in ['P1', 'P2']:
            return {
                'final_decision': 'APPROVE',
                'tier': f'{stage2_tier} - Standard',
                'interest_rate_range': '10.0% - 11.0%',
                'reason': f'Stage 1 concerns overridden by strong CIBIL profile ({stage2_tier})'
            }
        elif stage2_tier == 'P3':
            return {
                'final_decision': 'MANUAL_REVIEW',
                'tier': 'P3 - Borderline',
                'interest_rate_range': 'To be determined',
                'reason': 'Mixed signals from both stages. Requires underwriter judgment.'
            }
        else:  # P4
            return {
                'final_decision': 'REJECT',
                'tier': None,
                'interest_rate_range': 'N/A',
                'reason': 'Failed both Stage 1 and Stage 2 validation'
            }
    
    else:  # REJECT
        return {
            'final_decision': 'REJECT',
            'tier': None,
            'interest_rate_range': 'N/A',
            'reason': 'Failed Stage 1 policy gates'
        }


# =============================================================================
# MAIN TWO-STAGE DECISION FUNCTION
# =============================================================================

def make_two_stage_decision(customer_data, stage1_function):
    """
    Complete two-stage decision engine
    
    Flow:
    1. Run Stage 1 (passed as parameter)
    2. If REJECT → stop, return REJECT
    3. If APPROVE/REVIEW → run Stage 2 CIBIL analysis
    4. Apply decision matrix
    5. Return final decision with tier and interest rate
    
    Args:
        customer_data: Customer information dictionary
        stage1_function: The make_hybrid_decision_enhanced function from test.py
    
    Returns:
        Dictionary with final decision, tier, rates, and reasoning
    """
    
    # ========================================
    # STAGE 1: QUICK SCREENING
    # ========================================
    
    stage1_result = stage1_function(customer_data)
    
    stage1_decision = stage1_result['decision']
    stage1_risk_score = stage1_result['risk_score']
    stage1_pd = stage1_result['pd_percentage']
    stage1_confidence = stage1_result['confidence']
    
    # If rejected at Stage 1, no need for Stage 2
    if stage1_decision == "REJECT":
        return {
            'final_decision': 'REJECT',
            'tier': None,
            'interest_rate_range': 'N/A',
            'stage1_decision': stage1_decision,
            'stage2_tier': None,
            'stage2_confidence': None,
            'combined_risk_score': stage1_risk_score,
            'pd_percentage': stage1_pd,
            'reason': stage1_result['reason'],
            'stage1_details': stage1_result,
            'stage2_used': False
        }
    
    # ========================================
    # STAGE 2: CIBIL DEEP DIVE
    # ========================================
    
    if not STAGE2_ASSETS['loaded']:
        # Stage 2 model not available, use Stage 1 only
        return {
            'final_decision': stage1_decision,
            'tier': 'N/A (Stage 2 not available)',
            'interest_rate_range': '10.0% - 12.0%',
            'stage1_decision': stage1_decision,
            'stage2_tier': None,
            'stage2_confidence': None,
            'combined_risk_score': stage1_risk_score,
            'pd_percentage': stage1_pd,
            'reason': stage1_result['reason'] + ' (Stage 2 model not loaded)',
            'stage1_details': stage1_result,
            'stage2_used': False
        }
    
    # Prepare input for Stage 2
    stage2_features = STAGE2_ASSETS['features']
    stage2_input = prepare_stage2_input(customer_data, stage2_features)
    
    # Predict tier using Stage 2 model
    stage2_model = STAGE2_ASSETS['model']
    stage2_le = STAGE2_ASSETS['label_encoder']
    
    try:
        # Reshape input for prediction
        stage2_input_array = np.array(stage2_input).reshape(1, -1)
        
        # Predict
        tier_idx = stage2_model.predict(stage2_input_array)[0]
        tier_proba = stage2_model.predict_proba(stage2_input_array)[0]
        
        # Decode
        stage2_tier = stage2_le.inverse_transform([tier_idx])[0]
        stage2_confidence = max(tier_proba) * 100
        
        # Get tier probabilities
        tier_probs = {
            tier: prob * 100 
            for tier, prob in zip(stage2_le.classes_, tier_proba)
        }
        
    except Exception as e:
        # If Stage 2 prediction fails, fallback to Stage 1 only
        return {
            'final_decision': stage1_decision,
            'tier': 'N/A (Stage 2 error)',
            'interest_rate_range': '10.0% - 12.0%',
            'stage1_decision': stage1_decision,
            'stage2_tier': None,
            'stage2_confidence': None,
            'combined_risk_score': stage1_risk_score,
            'pd_percentage': stage1_pd,
            'reason': f'{stage1_result["reason"]} (Stage 2 failed: {str(e)})',
            'stage1_details': stage1_result,
            'stage2_used': False,
            'stage2_error': str(e)
        }
    
    # ========================================
    # APPLY DECISION MATRIX
    # ========================================
    
    matrix_result = apply_two_stage_decision_matrix(
        stage1_decision=stage1_decision,
        stage2_tier=stage2_tier,
        stage1_risk_score=stage1_risk_score,
        stage2_confidence=stage2_confidence
    )
    
    # Calculate combined risk score (weighted average)
    # Stage 1: 40%, Stage 2: 60%
    tier_score_map = {'P1': 900, 'P2': 750, 'P3': 600, 'P4': 450}
    stage2_score = tier_score_map.get(stage2_tier, 500)
    combined_risk_score = int(0.40 * stage1_risk_score + 0.60 * stage2_score)
    
    # Return complete result WITH COMPATIBILITY KEYS
    return {
        # Two-Stage specific keys
        'final_decision': matrix_result['final_decision'],
        'tier': matrix_result['tier'],
        'interest_rate_range': matrix_result['interest_rate_range'],
        'stage1_decision': stage1_decision,
        'stage2_tier': stage2_tier,
        'stage2_confidence': round(stage2_confidence, 2),
        'tier_probabilities': tier_probs,
        'combined_risk_score': combined_risk_score,
        'stage1_risk_score': stage1_risk_score,
        'stage2_risk_score': stage2_score,
        'pd_percentage': stage1_pd,
        'reason': matrix_result['reason'],
        'stage1_details': stage1_result,
        'stage2_used': True,
        'stage2_model_accuracy': STAGE2_ASSETS.get('test_accuracy', 0),
        
        # Compatibility keys for existing test.py code
        'decision': matrix_result['final_decision'],  # Same as final_decision
        'risk_score': combined_risk_score,             # For compatibility
        'confidence': stage1_confidence,               # From Stage 1
        'class_probs': stage1_result.get('class_probs', {}),  # From Stage 1
        'policy_checks': stage1_result.get('policy_checks', {}),  # From Stage 1
        'affordability_data': stage1_result.get('affordability_data', {})  # From Stage 1
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_stage2_available():
    """Check if Stage 2 model is loaded and available"""
    return STAGE2_ASSETS['loaded']


def get_stage2_status():
    """Get Stage 2 model status for display"""
    if STAGE2_ASSETS['loaded']:
        return {
            'status': '✅ Loaded',
            'accuracy': f"{STAGE2_ASSETS.get('test_accuracy', 0) * 100:.2f}%",
            'features': len(STAGE2_ASSETS['features']),
            'classes': ', '.join(STAGE2_ASSETS['label_encoder'].classes_),
            'path': STAGE2_ASSETS.get('path', 'Unknown')
        }
    else:
        return {
            'status': '❌ Not Loaded',
            'error': STAGE2_ASSETS.get('error', 'Unknown error'),
            'accuracy': 'N/A',
            'features': 0,
            'classes': 'N/A',
            'path': 'N/A'
        }


# =============================================================================
# MODULE INFO
# =============================================================================

def get_module_info():
    """Get module information"""
    return {
        'name': 'Stage 2 CIBIL Deep Dive Engine',
        'version': '1.0',
        'author': 'Zen Meraki',
        'stage2_loaded': STAGE2_ASSETS['loaded'],
        'stage2_status': get_stage2_status()
    }


# Print status on import
if __name__ != "__main__":
    status = get_stage2_status()
    print(f"🔬 Stage 2 Engine: {status['status']}")
    if STAGE2_ASSETS['loaded']:
        print(f"   Accuracy: {status['accuracy']}")
        print(f"   Features: {status['features']}")
        print(f"   Path: {status['path']}")
