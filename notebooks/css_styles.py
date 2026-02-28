# css_styles.py

CSS = """
<style>
/* ========== FONTS ========== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ========== CSS VARIABLES ========== */
:root {
    /* Modern neutral palette */
    --bg-light: #F8FAFC;
    --card-bg: #FFFFFF;
    --text-primary: #1E293B;
    --text-secondary: #475569;
    --border-subtle: #E2E8F0;
    --accent: #2563EB;
    --accent-soft: #3B82F6;

    /* Status colors */
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --info: #3B82F6;

    /* Shadows */
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
    --shadow-xl: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);

    /* Spacing */
    --space-xs: 0.5rem;
    --space-sm: 1rem;
    --space-md: 1.5rem;
    --space-lg: 2rem;
    --space-xl: 3rem;
}

/* ========== GLOBAL RESET ========== */
* {
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background-color: var(--bg-light);
    color: var(--text-primary);
}

.main {
    background: var(--bg-light);
    padding: var(--space-lg) 0;
}

.block-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 var(--space-lg);
}

/* ========== HEADERS ========== */
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--text-primary);
    text-align: center;
    margin: var(--space-xl) 0 var(--space-lg);
    letter-spacing: -0.02em;
    position: relative;
}
.main-header::after {
    content: '';
    display: block;
    width: 80px;
    height: 4px;
    background: var(--accent);
    margin: var(--space-sm) auto 0;
    border-radius: 2px;
}

.section-header {
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: var(--space-lg) 0 var(--space-md);
    padding-bottom: var(--space-xs);
    border-bottom: 2px solid var(--border-subtle);
}

/* ========== CARD BASE ========== */
.card, .info-card, .stat-card {
    background: var(--card-bg);
    border-radius: 20px;
    padding: var(--space-lg);
    border: 1px solid var(--border-subtle);
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}
.card:hover, .info-card:hover, .stat-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.info-card-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: var(--space-sm);
    border-bottom: 2px solid var(--border-subtle);
    padding-bottom: var(--space-xs);
    display: flex;
    align-items: center;
    gap: var(--space-xs);
}
.info-card-title i {
    color: var(--accent);
}

.info-card-content {
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ========== STAT CARDS (for metric displays) ========== */
.stat-card {
    text-align: center;
    padding: var(--space-lg);
}
.stat-number {
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
    display: block;
    margin-bottom: var(--space-xs);
}
.stat-label {
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ========== DECISION CARDS ========== */
.decision-card {
    border-radius: 32px;
    padding: var(--space-xl);
    color: white;
    text-align: center;
    background: linear-gradient(135deg, #1E293B, #0F172A);
    box-shadow: var(--shadow-xl);
    margin: var(--space-lg) 0;
}
.decision-card-approved {
    background: linear-gradient(135deg, var(--success), #059669);
}
.decision-card-rejected {
    background: linear-gradient(135deg, var(--danger), #DC2626);
}
.decision-card-review {
    background: linear-gradient(135deg, var(--warning), #D97706);
}

.decision-title {
    font-size: 3rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-sm);
}
.decision-subtitle {
    font-size: 1.2rem;
    margin-top: var(--space-sm);
    opacity: 0.9;
}

/* ========== STATUS BADGES ========== */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1.2rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.85rem;
    gap: 0.5rem;
    box-shadow: var(--shadow-sm);
    border: 2px solid transparent;
}
.badge-pass {
    background: #D1FAE5;
    color: var(--success);
    border-color: var(--success);
}
.badge-fail {
    background: #FEE2E2;
    color: var(--danger);
    border-color: var(--danger);
}
.badge-warning {
    background: #FEF3C7;
    color: #92400E;
    border-color: var(--warning);
}

/* ========== DATA ROWS ========== */
.data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--space-sm) 0;
    border-bottom: 1px solid var(--border-subtle);
    transition: background 0.2s;
    border-radius: 8px;
}
.data-row:hover {
    background: var(--bg-light);
    padding-left: var(--space-sm);
    padding-right: var(--space-sm);
}
.data-row:last-child {
    border-bottom: none;
}
.data-label {
    font-weight: 600;
    color: var(--text-primary);
}
.data-value {
    font-weight: 700;
    color: var(--accent);
}

/* ========== REASON ITEMS ========== */
.reason-item {
    background: rgba(37, 99, 235, 0.05);
    border-left: 5px solid var(--warning);
    padding: var(--space-md);
    border-radius: 12px;
    margin-bottom: var(--space-sm);
    display: flex;
    align-items: flex-start;
    gap: var(--space-sm);
    transition: all 0.2s;
}
.reason-item:hover {
    transform: translateX(4px);
    box-shadow: var(--shadow-sm);
}
.reason-icon {
    font-size: 1.5rem;
    color: var(--warning);
    flex-shrink: 0;
}

/* ========== INFO/WARNING/ERROR BOXES ========== */
.info-box, .warning-box, .error-box {
    border-radius: 12px;
    padding: var(--space-md);
    margin: var(--space-md) 0;
    border-left: 5px solid;
    background: rgba(0,0,0,0.02);
    box-shadow: var(--shadow-sm);
}
.info-box {
    border-left-color: var(--info);
    background: rgba(59,130,246,0.05);
}
.warning-box {
    border-left-color: var(--warning);
    background: rgba(245,158,11,0.05);
}
.error-box {
    border-left-color: var(--danger);
    background: rgba(239,68,68,0.05);
}

/* ========== BUTTONS ========== */
.stButton > button, .stDownloadButton button {
    border: none !important;
    border-radius: 40px !important;
    padding: 0.75rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow-sm) !important;
    cursor: pointer !important;
    width: 100% !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: white !important;
}
.stButton > button:hover {
    background: var(--accent-soft) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg) !important;
}

.stDownloadButton button {
    background: white !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
}
.stDownloadButton button:hover {
    background: var(--bg-light) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card-bg);
    border-radius: 60px;
    padding: 0.5rem;
    gap: 0.5rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 40px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600;
    color: var(--text-secondary);
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
    box-shadow: var(--shadow-md) !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: var(--bg-light);
    color: var(--text-primary);
}

/* ========== METRICS (Streamlit's built-in metric) ========== */
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricDelta"] {
    font-size: 0.85rem !important;
}

/* ========== INPUT FIELDS ========== */
.stNumberInput input, .stSelectbox select, .stTextInput input, .stDateInput input {
    border: 2px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 0.75rem 1rem !important;
    background: white !important;
    color: var(--text-primary) !important;
    font-size: 1rem !important;
    transition: border 0.2s, box-shadow 0.2s !important;
}
.stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus, .stDateInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
    outline: none !important;
}

.stNumberInput label, .stSelectbox label, .stTextInput label, .stDateInput label {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    margin-bottom: 0.25rem !important;
}

/* ========== CHECKBOX & RADIO ========== */
.stCheckbox > label, .stRadio > div {
    color: var(--text-primary);
}
.stCheckbox > label:hover, .stRadio > div:hover {
    color: var(--accent);
}

/* ========== DATA FRAMES ========== */
.dataframe {
    border-collapse: collapse;
    width: 100%;
    background: white;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-subtle);
}
.dataframe th {
    background: var(--bg-light);
    font-weight: 600;
    color: var(--text-primary);
    padding: var(--space-sm);
    text-align: left;
    border-bottom: 2px solid var(--border-subtle);
}
.dataframe td {
    padding: var(--space-sm);
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-secondary);
}
.dataframe tr:last-child td {
    border-bottom: none;
}

/* ========== PROGRESS BAR ========== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-soft)) !important;
    border-radius: 20px !important;
    box-shadow: 0 2px 4px rgba(37,99,235,0.2) !important;
}

/* ========== EXPANDER ========== */
.streamlit-expanderHeader {
    background: white !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 0.75rem 1rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    transition: all 0.2s !important;
}
.streamlit-expanderHeader:hover {
    border-color: var(--accent) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid var(--border-subtle);
    box-shadow: 2px 0 10px rgba(0,0,0,0.02);
}
[data-testid="stSidebar"] .block-container {
    padding: var(--space-lg) var(--space-md);
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: var(--bg-light);
}
::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 20px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94A3B8;
}

/* ========== RESPONSIVE ========== */
@media (max-width: 768px) {
    .main-header { font-size: 2.2rem; }
    .decision-title { font-size: 2.2rem; }
    .stat-number { font-size: 2rem; }
    .block-container { padding: 0 var(--space-md); }
}

/* ========== MISC ========== */
hr {
    margin: var(--space-lg) 0;
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
}
::placeholder {
    color: var(--text-secondary) !important;
    opacity: 0.6 !important;
}
::selection {
    background: var(--accent);
    color: white;
}
</style>
"""
