



# css_styles.py
"""
Ultra-Enhanced Sage Green & Yellow Theme CSS for Credit Risk Dashboard
Features: Glassmorphism, Neumorphism, Smooth Animations, Gradient Magic, 3D Effects
"""

CSS = """
<style>
/* ==================== FONT IMPORTS ==================== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@300;400;500;600;700;800&display=swap');

/* ==================== CSS VARIABLES ==================== */
:root {
    --fern-green: #262842;        /* Space Cadet */
    --sage: #293961;              /* Delfi Blue */
    --cosmic-latte: #E3E4FA;      /* Lavender (web) */
    --jasmine: #8897BD;           /* Cool Grey */
    --saffron: #2C497F;           /* Vikki Blue */

    --dark-fern: #1a1f30;         /* darker than Space Cadet */
    --light-sage: #8897BD;        /* reuse Cool Grey */
    --fern-dark: #1a1f30;
    --fern-light: #2C497F;         /* Vikki Blue as lighter primary */
    --sage-light: #8897BD;
    --sage-dark: #1f2a47;          /* approximate darker Delfi Blue */
    --gold: #8897BD;               /* reuse Cool Grey */
    --gold-light: #b0c2e0;         /* lighter version of Cool Grey */

    --white: #FFFFFF;
    --off-white: #FAFAFA;
    --gray-50: #F9FAFB;
    --gray-100: #F3F4F6;
    --gray-200: #E5E7EB;
    --gray-300: #D1D5DB;
    --gray-400: #9CA3AF;
    --gray-500: #6B7280;
    --gray-600: #4B5563;
    --gray-700: #374151;
    --gray-800: #1F2937;
    --gray-900: #111827;

    /* Semantic status colors (unchanged) */
    --success: #10B981;
    --success-light: #D1FAE5;
    --warning: #F59E0B;
    --warning-light: #FEF3C7;
    --danger: #EF4444;
    --danger-light: #FEE2E2;
    --info: #3B82F6;
    --info-light: #DBEAFE;

    /* Shadows – updated to use the new primary color (Space Cadet) */
    --shadow-sm: 0 1px 2px 0 rgba(38, 40, 66, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(38, 40, 66, 0.1), 0 2px 4px -1px rgba(38, 40, 66, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(38, 40, 66, 0.15), 0 4px 6px -2px rgba(38, 40, 66, 0.05);
    --shadow-xl: 0 20px 25px -5px rgba(38, 40, 66, 0.2), 0 10px 10px -5px rgba(38, 40, 66, 0.04);
    --shadow-2xl: 0 25px 50px -12px rgba(38, 40, 66, 0.25);

    --glass-bg: rgba(255, 255, 255, 0.85);
    --glass-border: rgba(38, 40, 66, 0.18);

    --transition-fast: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-slow: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-bounce: all 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

/* ==================== GLOBAL STYLES ==================== */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

body {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    overflow-x: hidden;
}

.main {
    background: linear-gradient(135deg, rgba(250, 247, 230, 0.4) 0%, rgba(255, 255, 255, 0.6) 100%);
    position: relative;
    z-index: 1;
}

.block-container {
    padding: 3rem 2rem;
    max-width: 1400px;
    background: transparent;
}

/* ==================== TYPOGRAPHY ==================== */
.main-header {
    font-family: 'Poppins', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: var(--fern-green);
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 2rem;
    position: relative;
    letter-spacing: -0.02em;
}

.main-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 4px;
    background: linear-gradient(90deg, transparent, var(--saffron), transparent);
    border-radius: 2px;
}

.section-header {
    font-family: 'Poppins', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--fern-green);
    margin-top: 3rem;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    padding-left: 1rem;
    border-left: 5px solid var(--saffron);
    border-bottom: 2px solid var(--sage-light);
    background: linear-gradient(90deg, rgba(246, 197, 49, 0.05) 0%, transparent 100%);
    border-radius: 0 8px 8px 0;
}

/* ==================== DECISION CARDS ==================== */
.decision-card {
    padding: 3rem 2.5rem;
    border-radius: 24px;
    margin: 2.5rem 0;
    color: white;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    box-shadow: var(--shadow-2xl);
    transition: var(--transition-base);
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.decision-card-approved {
    background: linear-gradient(135deg, 
        rgba(16, 185, 129, 0.95) 0%, 
        rgba(88, 112, 66, 0.95) 50%, 
        rgba(122, 158, 77, 0.95) 100%);
}

.decision-card-rejected {
    background: linear-gradient(135deg, 
        rgba(239, 68, 68, 0.95) 0%, 
        rgba(220, 38, 38, 0.95) 50%, 
        rgba(185, 28, 28, 0.95) 100%);
}

.decision-card-review {
    background: linear-gradient(135deg, 
        rgba(246, 197, 49, 0.95) 0%, 
        rgba(248, 222, 140, 0.95) 50%, 
        rgba(255, 215, 0, 0.95) 100%);
    color: var(--gray-800);
}

.decision-title {
    font-family: 'Poppins', sans-serif;
    font-size: 3.5rem;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    text-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    letter-spacing: -0.02em;
}

.decision-subtitle {
    font-size: 1.25rem;
    margin-top: 1rem;
    opacity: 0.95;
    font-weight: 500;
    text-align: center;
}

/* ==================== GLASSMORPHIC INFO CARDS ==================== */
.info-card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px) saturate(180%);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: var(--shadow-lg), inset 0 1px 0 rgba(255, 255, 255, 0.5);
    border: 1px solid var(--glass-border);
    margin-bottom: 1.5rem;
    transition: var(--transition-base);
    position: relative;
    overflow: hidden;
}

.info-card:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: var(--shadow-xl), inset 0 1px 0 rgba(255, 255, 255, 0.6);
    border-color: var(--fern-green);
}

.info-card-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--fern-green);
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid var(--sage-light);
}

.info-card-content {
    color: var(--gray-700);
    line-height: 1.7;
}

/* ==================== NEUMORPHIC STAT CARDS ==================== */
.stat-card {
    background: linear-gradient(145deg, #ffffff, #f0f0f0);
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    box-shadow: 
        8px 8px 16px rgba(163, 177, 198, 0.3),
        -8px -8px 16px rgba(255, 255, 255, 0.8),
        inset 0 1px 0 rgba(255, 255, 255, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.5);
    transition: var(--transition-bounce);
    position: relative;
    overflow: hidden;
}

.stat-card:hover {
    transform: translateY(-8px) scale(1.05);
    box-shadow: 
        12px 12px 24px rgba(163, 177, 198, 0.4),
        -12px -12px 24px rgba(255, 255, 255, 0.9),
        inset 0 2px 0 rgba(255, 255, 255, 0.6);
}

.stat-number {
    font-family: 'Poppins', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: var(--fern-green);
    margin-bottom: 0.75rem;
    display: block;
}

.stat-label {
    font-size: 0.875rem;
    font-weight: 700;
    color: var(--gray-600);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ==================== STATUS BADGES ==================== */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.625rem 1.25rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.875rem;
    gap: 0.5rem;
    transition: var(--transition-base);
    box-shadow: var(--shadow-md);
    border: 2px solid;
    position: relative;
    overflow: hidden;
}

.badge-pass {
    background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
    color: var(--success);
    border-color: var(--success);
}

.badge-fail {
    background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
    color: var(--danger);
    border-color: var(--danger);
}

.badge-warning {
    background: linear-gradient(135deg, var(--cosmic-latte) 0%, var(--jasmine) 100%);
    color: #92400E;
    border-color: var(--warning);
}

/* ==================== DATA ROWS ==================== */
.data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0.5rem;
    border-bottom: 1px solid rgba(169, 180, 148, 0.2);
    transition: var(--transition-fast);
    border-radius: 8px;
}

.data-row:hover {
    background: rgba(88, 112, 66, 0.03);
    padding-left: 1rem;
}

.data-row:last-child {
    border-bottom: none;
}

.data-label {
    font-weight: 600;
    color: var(--gray-700);
    position: relative;
    padding-left: 1rem;
}

.data-label::before {
    content: '▸';
    position: absolute;
    left: 0;
    color: var(--fern-green);
    opacity: 0;
    transition: var(--transition-fast);
}

.data-row:hover .data-label::before {
    opacity: 1;
}

.data-value {
    font-weight: 700;
    color: var(--fern-green);
}

/* ==================== REASON ITEMS ==================== */
.reason-item {
    background: linear-gradient(135deg, rgba(250, 247, 230, 0.8) 0%, rgba(248, 222, 140, 0.3) 100%);
    backdrop-filter: blur(10px);
    padding: 1.25rem 1.5rem;
    border-radius: 16px;
    border-left: 5px solid var(--saffron);
    margin-bottom: 1rem;
    color: #92400E;
    font-weight: 600;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    transition: var(--transition-base);
    box-shadow: var(--shadow-sm);
}

.reason-item:hover {
    transform: translateX(8px);
    box-shadow: var(--shadow-md);
    border-left-width: 8px;
}

.reason-icon {
    font-size: 1.5rem;
    color: var(--fern-green);
    flex-shrink: 0;
}

/* ==================== PRIMARY BUTTONS ==================== */
.stButton > button {
    background: linear-gradient(135deg, var(--fern-green) 0%, var(--fern-light) 100%);
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 1rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    transition: var(--transition-base) !important;
    box-shadow: 0 6px 20px rgba(88, 112, 66, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 28px rgba(88, 112, 66, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    background: linear-gradient(135deg, var(--fern-light) 0%, var(--fern-green) 100%) !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(88, 112, 66, 0.3) !important;
}

/* ==================== PDF DOWNLOAD BUTTONS (FIXED) ==================== */
.stDownloadButton button {
    background: linear-gradient(135deg, var(--saffron) 0%, var(--gold) 100%) !important;
    color: var(--gray-900) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: var(--transition-base) !important;
    box-shadow: 0 4px 12px rgba(246, 197, 49, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    cursor: pointer !important;
    width: 100% !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin: 0.25rem 0 !important;
}

.stDownloadButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(246, 197, 49, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    background: linear-gradient(135deg, var(--gold) 0%, var(--saffron) 100%) !important;
}

.stDownloadButton button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(246, 197, 49, 0.3) !important;
}

/* ==================== TABS ==================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    padding: 1.25rem;
    border-radius: 16px;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--glass-border);
}

.stTabs [data-baseweb="tab"] {
    height: 3.5rem;
    padding: 0 2rem;
    background: transparent;
    border-radius: 12px;
    color: var(--gray-600);
    font-weight: 700;
    font-size: 0.95rem;
    transition: var(--transition-base);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--fern-green) 0%, var(--fern-light) 100%);
    color: white !important;
    box-shadow: 0 6px 20px rgba(88, 112, 66, 0.3);
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--sage-light);
    color: var(--fern-dark);
}

/* ==================== SIDEBAR ==================== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, 
        rgba(250, 247, 230, 0.95) 0%, 
        rgba(245, 242, 224, 0.95) 100%);
    backdrop-filter: blur(20px) saturate(180%);
    border-right: 1px solid rgba(169, 180, 148, 0.3);
    box-shadow: 4px 0 20px rgba(88, 112, 66, 0.1);
}

/* ==================== INFO/WARNING/ERROR BOXES ==================== */
.info-box {
    background: linear-gradient(135deg, 
        rgba(59, 130, 246, 0.08) 0%, 
        rgba(147, 197, 253, 0.08) 100%);
    backdrop-filter: blur(10px);
    border-left: 5px solid var(--info);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    color: var(--gray-800);
    border: 1px solid rgba(59, 130, 246, 0.2);
    box-shadow: var(--shadow-md);
}

.warning-box {
    background: linear-gradient(135deg, 
        rgba(245, 158, 11, 0.08) 0%, 
        rgba(251, 191, 36, 0.08) 100%);
    backdrop-filter: blur(10px);
    border-left: 5px solid var(--warning);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    color: #92400E;
    border: 1px solid rgba(245, 158, 11, 0.2);
    box-shadow: var(--shadow-md);
}

.error-box {
    background: linear-gradient(135deg, 
        rgba(239, 68, 68, 0.08) 0%, 
        rgba(252, 129, 129, 0.08) 100%);
    backdrop-filter: blur(10px);
    border-left: 5px solid var(--danger);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    color: #991B1B;
    border: 1px solid rgba(239, 68, 68, 0.2);
    box-shadow: var(--shadow-md);
}

/* ==================== INPUT FIELDS ==================== */
.stNumberInput > div > div > input,
.stSelectbox > div > div > select,
.stTextInput > div > div > input {
    color: #FFFFFF !important;
    background: linear-gradient(135deg, #2D3748 0%, #1A202C 100%) !important;
    border: 2px solid var(--sage) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    transition: var(--transition-base) !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}

.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > select:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--fern-green) !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2), 0 0 0 4px rgba(88, 112, 66, 0.2) !important;
    background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%) !important;
}

.stNumberInput label,
.stSelectbox label,
.stTextInput label {
    color: var(--fern-green) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
    display: block !important;
}

/* ==================== METRICS ==================== */
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    font-family: 'Poppins', sans-serif !important;
    color: var(--fern-green) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.875rem !important;
    font-weight: 700 !important;
    color: var(--gray-600) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ==================== EXPANDER ==================== */
.streamlit-expanderHeader {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 12px !important;
    padding: 1rem 1.5rem !important;
    font-weight: 700 !important;
    color: var(--fern-green) !important;
    border: 2px solid var(--sage) !important;
    transition: var(--transition-base) !important;
}

.streamlit-expanderHeader:hover {
    border-color: var(--fern-green) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ==================== PROGRESS BAR ==================== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--fern-green) 0%, var(--fern-light) 50%, var(--saffron) 100%) !important;
    box-shadow: 0 2px 8px rgba(88, 112, 66, 0.4) !important;
    border-radius: 10px !important;
}

/* ==================== RADIO & CHECKBOX ==================== */
.stRadio > div > label,
.stCheckbox > label {
    padding: 0.5rem 1rem;
    border-radius: 8px;
    transition: var(--transition-fast);
    color: var(--gray-700);
}

.stRadio > div > label:hover,
.stCheckbox > label:hover {
    background: rgba(88, 112, 66, 0.05);
}

/* ==================== SCROLLBAR ==================== */
::-webkit-scrollbar {
    width: 12px;
    height: 12px;
}

::-webkit-scrollbar-track {
    background: linear-gradient(180deg, var(--cosmic-latte) 0%, #F5F2E0 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--sage) 0%, var(--fern-green) 100%);
    border-radius: 10px;
    border: 2px solid var(--cosmic-latte);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, var(--fern-green) 0%, var(--fern-dark) 100%);
}

/* ==================== RESPONSIVE ==================== */
@media (max-width: 768px) {
    .main-header {
        font-size: 2rem;
    }
    .decision-title {
        font-size: 2.5rem;
    }
    .stat-number {
        font-size: 2rem;
    }
    .stButton > button,
    .stDownloadButton button {
        padding: 0.75rem 1rem !important;
    }
}

/* ==================== MISC ==================== */
hr {
    margin: 3rem 0;
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--sage), var(--saffron), var(--sage), transparent);
    opacity: 0.5;
}

.dataframe {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    border: 2px solid var(--sage-light);
}

::placeholder {
    color: var(--gray-500) !important;
    opacity: 0.7 !important;
    font-weight: 500 !important;
}

::selection {
    background: var(--sage-light);
    color: var(--fern-dark);
}

</style>
"""
