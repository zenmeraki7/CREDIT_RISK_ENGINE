"""
STAGE 2 CIBIL DEEP DIVE ENGINE
Separate module for 2-stage credit risk system

Author: Zen Meraki
Date: February 2026
Version: 1.1 - FIXED
"""

import joblib  # Bug fix 1: removed duplicate import
import os
import numpy as np
import streamlit as st


# =============================================================================
# LOAD STAGE 2 MODEL (CIBIL DEEP DIVE)
# =============================================================================

@st.cache_resource
def load_stage2_model():
    """Load Stage 2 CIBIL model if available"""
    try:
        model_paths = [
            'stage2_cibil_model.pkl',
            'models/stage2_cibil_model.pkl',
            'notebooks/stage2_cibil_model.pkl',
            './stage2_cibil_model.pkl',
            '../stage2_cibil_model.pkl'
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


STAGE2_ASSETS = load_stage2_model()


# =============================================================================
# MAP STAGE 1 DATA TO STAGE 2 CIBIL FORMAT
# =============================================================================

def _safe_util(value, default=0.0):
    """
    Bug fix 4 (dataset): Clean sentinel -99999 values before passing to Stage 2 model.
    92.8% of CIBIL records have CC_utilization = -99999 (no credit card).
    """
    if value is None or value < 0:
        return default
    return value


def prepare_stage2_input(customer_data, stage2_features):
    """
    Map customer data from Stage 1 format to Stage 2 CIBIL format.
    Includes sentinel value cleaning for CC/PL utilization fields.
    """
    # Clean utilization values first — sentinel -99999 = no product = 0 utilization
    cc_util_raw = customer_data.get('CC_utilization',
                    customer_data.get('credit_utilization_pct', 30) / 100)
    pl_util_raw = customer_data.get('PL_utilization', 0.25)
    cc_util = _safe_util(cc_util_raw, 0.0)
    pl_util = _safe_util(pl_util_raw, 0.0)

    mapping = {
        # Credit Score & Basic Info
        'Credit_Score': customer_data.get('bureau_score', 700),
        'AGE': customer_data.get('age', 35),
        'NETMONTHLYINCOME': customer_data.get('avg_salary_6m', 30000),
        'Time_With_Curr_Empr': customer_data.get('employment_tenure_months', 24),

        # Delinquency
        'num_times_30p_dpd': customer_data.get('dpd_30_count_6m', 0),
        'num_times_60p_dpd': customer_data.get('dpd_90_count_6m', 0),
        'num_times_delinquent': (customer_data.get('dpd_90_count_6m', 0) +
                                  customer_data.get('dpd_30_count_6m', 0)),
        'max_delinquency_level': 0 if customer_data.get('dpd_90_count_6m', 0) == 0 else 50,

        # Delinquency (6 months)
        'num_deliq_6mts': customer_data.get('dpd_90_count_6m', 0),
        'max_deliq_6mts': 0 if customer_data.get('dpd_90_count_6m', 0) == 0 else 50,

        # Delinquency (12 months)
        'num_deliq_12mts': customer_data.get('dpd_90_count_6m', 0),
        'max_deliq_12mts': 0 if customer_data.get('dpd_90_count_6m', 0) == 0 else 50,

        # Credit Inquiries
        'enq_L3m': customer_data.get('recent_inquiries_3m', 0),
        'enq_L6m': customer_data.get('enq_L6m',
                    customer_data.get('recent_inquiries_3m', 0) * 2),
        'enq_L12m': customer_data.get('enq_L12m',
                     customer_data.get('recent_inquiries_3m', 0) * 4),

        # Account Quality
        'num_std': customer_data.get('active_loans_count', 1),
        'num_std_6mts': customer_data.get('num_std_6mts',
                         customer_data.get('active_loans_count', 1)),
        'num_std_12mts': customer_data.get('num_std_12mts',
                          customer_data.get('active_loans_count', 1)),
        'num_sub': customer_data.get('num_sub', 0),
        'num_sub_6mts': customer_data.get('num_sub_6mts', 0),
        'num_dbt': customer_data.get('num_dbt', 0),
        'num_lss': customer_data.get('num_lss', 0),

        # Utilization — sentinel cleaned
        'pct_currentBal_all_TL': _safe_util(
            customer_data.get('pct_currentBal_all_TL'), cc_util),
        'CC_utilization': cc_util,
        'PL_utilization': pl_util,
        'max_unsec_exposure_inPct': _safe_util(
            customer_data.get('max_unsec_exposure_inPct'), cc_util * 100),
        'pct_of_active_TLs_ever': _safe_util(
            customer_data.get('pct_of_active_TLs_ever'), 0.5),

        # Product Flags
        'CC_Flag': customer_data.get('CC_Flag',
                    1 if customer_data.get('credit_utilization_pct', 0) > 0 else 0),
        'PL_Flag': customer_data.get('PL_Flag', 1),
        'HL_Flag': customer_data.get('HL_Flag', 0),
        'GL_Flag': customer_data.get('GL_Flag', 0),

        # Engineered Features
        'delinq_severity_score': (0 if customer_data.get('dpd_90_count_6m', 0) == 0
                                   else 16.67),
        'high_dpd_risk': 1 if customer_data.get('dpd_90_count_6m', 0) > 0 else 0,
        'recent_deliq_flag': 1 if customer_data.get('dpd_90_count_6m', 0) > 0 else 0,
        'credit_hungry': 1 if customer_data.get('recent_inquiries_3m', 0) > 3 else 0,
        'account_quality_score': customer_data.get('active_loans_count', 1) * 10,
        'high_util_flag': 1 if cc_util > 0.75 else 0,
        'employment_stable': (1 if customer_data.get('employment_tenure_months', 0) >= 24
                               else 0),
        # Demographic fields (Stage 2 only — not in Stage 1 form, use neutral defaults)
        'MARITALSTATUS': customer_data.get('MARITALSTATUS', 'Single'),
        'EDUCATION': customer_data.get('EDUCATION', 'GRADUATE'),
        'GENDER': customer_data.get('GENDER', 'M'),
        # Time-since fields — cleaned sentinels → 0 means no event ever
        'time_since_recent_payment': 0 if customer_data.get('time_since_recent_payment', -99999) < 0 else customer_data.get('time_since_recent_payment', 0),
        'time_since_first_deliquency': 0 if customer_data.get('time_since_first_deliquency', -99999) < 0 else customer_data.get('time_since_first_deliquency', 0),
        'time_since_recent_deliquency': 0 if customer_data.get('time_since_recent_deliquency', -99999) < 0 else customer_data.get('time_since_recent_deliquency', 0),
        'time_since_recent_enq': 0 if customer_data.get('time_since_recent_enq', -99999) < 0 else customer_data.get('time_since_recent_enq', 0),
        # Enquiry breakdowns
        'tot_enq': 0 if customer_data.get('tot_enq', -99999) < 0 else customer_data.get('tot_enq', customer_data.get('recent_inquiries_3m', 0) * 4),
        'CC_enq': 0 if customer_data.get('CC_enq', -99999) < 0 else customer_data.get('CC_enq', 0),
        'CC_enq_L6m': 0 if customer_data.get('CC_enq_L6m', -99999) < 0 else customer_data.get('CC_enq_L6m', 0),
        'CC_enq_L12m': 0 if customer_data.get('CC_enq_L12m', -99999) < 0 else customer_data.get('CC_enq_L12m', 0),
        'PL_enq': 0 if customer_data.get('PL_enq', -99999) < 0 else customer_data.get('PL_enq', 0),
        'PL_enq_L6m': 0 if customer_data.get('PL_enq_L6m', -99999) < 0 else customer_data.get('PL_enq_L6m', 0),
        'PL_enq_L12m': 0 if customer_data.get('PL_enq_L12m', -99999) < 0 else customer_data.get('PL_enq_L12m', 0),
        # Additional delinquency breakdowns
        'num_deliq_6_12mts': customer_data.get('num_deliq_6_12mts', 0),
        'num_dbt_6mts': customer_data.get('num_dbt_6mts', 0),
        'num_dbt_12mts': customer_data.get('num_dbt_12mts', 0),
        'num_lss_6mts': customer_data.get('num_lss_6mts', 0),
        'num_lss_12mts': customer_data.get('num_lss_12mts', 0),
        'num_sub_12mts': customer_data.get('num_sub_12mts', 0),
        # Product enquiry patterns
        'last_prod_enq2': customer_data.get('last_prod_enq2', 'others'),
        'first_prod_enq2': customer_data.get('first_prod_enq2', 'others'),
        'pct_opened_TLs_L6m_of_L12m': customer_data.get('pct_opened_TLs_L6m_of_L12m', 0),
        'pct_PL_enq_L6m_of_L12m': customer_data.get('pct_PL_enq_L6m_of_L12m', 0),
        'pct_PL_enq_L6m_of_ever': customer_data.get('pct_PL_enq_L6m_of_ever', 0),
        'pct_CC_enq_L6m_of_L12m': customer_data.get('pct_CC_enq_L6m_of_L12m', 0),
        'pct_CC_enq_L6m_of_ever': customer_data.get('pct_CC_enq_L6m_of_ever', 0),
        'recent_level_of_deliq': customer_data.get('recent_level_of_deliq', 0),
        'max_recent_level_of_deliq': customer_data.get('max_recent_level_of_deliq', 0),
    }

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
    Apply decision matrix to combine Stage 1 + Stage 2 results.
    Bug fix 2: all final_decision values use 'REVIEW' not 'MANUAL_REVIEW'
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
                'reason': 'Acceptable credit with some concerns. Stage 1: APPROVE, Stage 2: P3'
            }
        else:  # P4
            return {
                'final_decision': 'REVIEW',  # Bug fix 2: was 'MANUAL_REVIEW'
                'tier': 'P4 - High Risk',
                'interest_rate_range': 'To be determined',
                'reason': 'High risk detected in CIBIL analysis. Needs underwriter review.'
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
                'final_decision': 'REVIEW',  # Bug fix 2: was 'MANUAL_REVIEW'
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
    Complete two-stage decision engine.
    """

    # STAGE 1
    stage1_result = stage1_function(customer_data)

    stage1_decision = stage1_result['decision']
    stage1_risk_score = stage1_result['risk_score']
    stage1_pd = stage1_result['pd_percentage']
    stage1_confidence = stage1_result['confidence']

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
            'stage2_used': False,
            # Compatibility keys
            'decision': 'REJECT',
            'risk_score': stage1_risk_score,
            'confidence': stage1_confidence,
            'class_probs': stage1_result.get('class_probs', {}),
            'policy_checks': stage1_result.get('policy_checks', {}),
            'affordability_data': stage1_result.get('affordability_data', {})
        }

    if not STAGE2_ASSETS['loaded']:
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
            'stage2_used': False,
            'decision': stage1_decision,
            'risk_score': stage1_risk_score,
            'confidence': stage1_confidence,
            'class_probs': stage1_result.get('class_probs', {}),
            'policy_checks': stage1_result.get('policy_checks', {}),
            'affordability_data': stage1_result.get('affordability_data', {})
        }

    # STAGE 2
    stage2_features = STAGE2_ASSETS['features']
    stage2_input = prepare_stage2_input(customer_data, stage2_features)

    stage2_model = STAGE2_ASSETS['model']
    stage2_le = STAGE2_ASSETS['label_encoder']

    try:
        stage2_input_array = np.array(stage2_input).reshape(1, -1)

        tier_idx = stage2_model.predict(stage2_input_array)[0]
        tier_proba = stage2_model.predict_proba(stage2_input_array)[0]

        stage2_tier = stage2_le.inverse_transform([tier_idx])[0]
        stage2_confidence = max(tier_proba) * 100

        tier_probs = {
            tier: prob * 100
            for tier, prob in zip(stage2_le.classes_, tier_proba)
        }

    except Exception as e:
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
            'stage2_error': str(e),
            'decision': stage1_decision,
            'risk_score': stage1_risk_score,
            'confidence': stage1_confidence,
            'class_probs': stage1_result.get('class_probs', {}),
            'policy_checks': stage1_result.get('policy_checks', {}),
            'affordability_data': stage1_result.get('affordability_data', {})
        }

    # DECISION MATRIX
    matrix_result = apply_two_stage_decision_matrix(
        stage1_decision=stage1_decision,
        stage2_tier=stage2_tier,
        stage1_risk_score=stage1_risk_score,
        stage2_confidence=stage2_confidence
    )

    # Bug fix 3: Combined risk score — normalize both to same 0-100 scale
    # Stage 1 is already 0-100; convert Stage 2 tier to 0-100 equivalent
    tier_score_map_normalized = {'P1': 15, 'P2': 35, 'P3': 60, 'P4': 80}
    stage2_score_normalized = tier_score_map_normalized.get(stage2_tier, 50)
    # Weighted: Stage 1 = 40%, Stage 2 = 60% (CIBIL data is more current)
    combined_risk_score = int(0.40 * stage1_risk_score + 0.60 * stage2_score_normalized)

    # Keep original tier scores for display purposes (score out of 1000)
    tier_score_display = {'P1': 900, 'P2': 750, 'P3': 600, 'P4': 450}
    stage2_score_display = tier_score_display.get(stage2_tier, 500)

    return {
        'final_decision': matrix_result['final_decision'],
        'tier': matrix_result['tier'],
        'interest_rate_range': matrix_result['interest_rate_range'],
        'stage1_decision': stage1_decision,
        'stage2_tier': stage2_tier,
        'stage2_confidence': round(stage2_confidence, 2),
        'tier_probabilities': tier_probs,
        'combined_risk_score': combined_risk_score,
        'stage1_risk_score': stage1_risk_score,
        'stage2_risk_score': stage2_score_display,
        'pd_percentage': stage1_pd,
        'reason': matrix_result['reason'],
        'stage1_details': stage1_result,
        'stage2_used': True,
        'stage2_model_accuracy': STAGE2_ASSETS.get('test_accuracy', 0),
        # Compatibility keys
        'decision': matrix_result['final_decision'],
        'risk_score': combined_risk_score,
        'confidence': stage1_confidence,
        'class_probs': stage1_result.get('class_probs', {}),
        'policy_checks': stage1_result.get('policy_checks', {}),
        'affordability_data': stage1_result.get('affordability_data', {})
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_stage2_available():
    return STAGE2_ASSETS['loaded']


def get_stage2_status():
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


def get_module_info():
    return {
        'name': 'Stage 2 CIBIL Deep Dive Engine',
        'version': '1.1',
        'author': 'Zen Meraki',
        'stage2_loaded': STAGE2_ASSETS['loaded'],
        'stage2_status': get_stage2_status()
    }


if __name__ != "__main__":
    status = get_stage2_status()
    print(f"🔬 Stage 2 Engine: {status['status']}")
    if STAGE2_ASSETS['loaded']:
        print(f"   Accuracy: {status['accuracy']}")
        print(f"   Features: {status['features']}")
        print(f"   Path: {status['path']}")
