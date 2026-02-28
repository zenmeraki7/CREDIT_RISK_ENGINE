# css_styles.py

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    /* New palette with added warm accent */
    --navy-dark: #262842;        /* primary dark */
    --navy-mid: #293961;          /* secondary dark */
    --blue-deep: #2C497F;         /* deep blue */
    --blue-soft: #8897BD;         /* soft blue/grey */
    --lavender-light: #E3E4FA;    /* light background */
    --accent-warm: #F6C531;        /* warm gold (keep from original) */

    /* Derived colours */
    --primary-bg: #FFFFFF;
    --card-bg: rgba(227, 228, 250, 0.6);  /* lavender with transparency */
    --text-primary: #262842;
    --text-secondary: #4B5563;
    --border-light: rgba(38, 40, 66, 0.1);
    --shadow-color: rgba(38, 40, 66, 0.1);

    /* Semantic status (unchanged) */
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --info: #3B82F6;

    /* Shadows - subtle */
    --shadow-sm: 0 2px 8px var(--shadow-color);
    --shadow-md: 0 4px 12px var(--shadow-color);
    --shadow-lg: 0 8px 24px var(--shadow-color);
    --shadow-xl: 0 16px 32px rgba(38, 40, 66, 0.15);

    /* Transitions */
    --transition: all 0.2s ease;
}

/* ==================== GLOBAL ==================== */
* { font-family: 'Inter', sans-serif; }

.main {
    background: linear-gradient(145deg, #ffffff 0%, var(--lavender-light) 100%);
}

.block-container { padding: 2rem; }

/* ==================== HEADERS ==================== */
.main-header {
    font-size: 3rem;
    font-weight: 800;
    color: var(--navy-dark);
    text-align: center;
    margin: 2rem 0;
    letter-spacing: -0.02em;
    position: relative;
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 120px;
    height: 4px;
    background: linear-gradient(90deg, transparent, var(--accent-warm), transparent);
    border-radius: 2px;
}

.section-header {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--navy-dark);
    margin: 2rem 0 1rem;
    padding-left: 1rem;
    border-left: 5px solid var(--accent-warm);
}

/* ==================== CARDS ==================== */
.info-card {
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 1.8rem;
    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: var(--shadow-md);
    transition: var(--transition);
}
.info-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--accent-warm);
}

.info-card-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--navy-dark);
    margin-bottom: 1rem;
    border-bottom: 2px solid var(--blue-soft);
    padding-bottom: 0.5rem;
}

/* ==================== STAT CARDS ==================== */
.stat-card {
    background: white;
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-light);
    transition: var(--transition);
}
.stat-card:hover {
    transform: scale(1.02);
    box-shadow: var(--shadow-lg);
}
.stat-number {
    font-size: 3rem;
    font-weight: 800;
    color: var(--navy-dark);
    display: block;
    margin-bottom: 0.5rem;
}
.stat-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ==================== DECISION CARD ==================== */
.decision-card {
    padding: 2.5rem;
    border-radius: 30px;
    margin: 2rem 0;
    color: white;
    background: linear-gradient(135deg, var(--navy-dark), var(--blue-deep));
    box-shadow: var(--shadow-xl);
    border: 1px solid rgba(255,255,255,0.1);
}
.decision-card.approved { background: linear-gradient(135deg, #10B981, var(--blue-deep)); }
.decision-card.rejected { background: linear-gradient(135deg, #EF4444, var(--navy-dark)); }
.decision-card.review { background: linear-gradient(135deg, var(--accent-warm), var(--blue-deep)); color: var(--navy-dark); }

.decision-title {
    font-size: 3rem;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
}
.decision-subtitle {
    font-size: 1.2rem;
    text-align: center;
    margin-top: 1rem;
    opacity: 0.9;
}

/* ==================== BUTTONS ==================== */
.stButton > button, .stDownloadButton button {
    border: none;
    border-radius: 12px !important;
    padding: 0.8rem 1.8rem !important;
    font-weight: 600 !important;
    transition: var(--transition) !important;
    box-shadow: var(--shadow-sm) !important;
    cursor: pointer !important;
    width: 100% !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton > button {
    background: var(--navy-dark) !important;
    color: white !important;
}
.stButton > button:hover {
    background: var(--blue-deep) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg) !important;
}

.stDownloadButton button {
    background: var(--accent-warm) !important;
    color: var(--navy-dark) !important;
}
.stDownloadButton button:hover {
    background: #e0b01a !important;
    transform: translateY(-2px) !important;
}

/* ==================== TABS ==================== */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    padding: 0.8rem;
    border-radius: 50px;
    box-shadow: var(--shadow-sm);
}
.stTabs [data-baseweb="tab"] {
    padding: 0.8rem 2rem;
    border-radius: 30px;
    font-weight: 600;
    color: var(--text-secondary);
}
.stTabs [aria-selected="true"] {
    background: var(--navy-dark);
    color: white !important;
    box-shadow: var(--shadow-md);
}

/* ==================== METRICS ==================== */
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    color: var(--navy-dark) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
}

/* ==================== INPUTS ==================== */
.stNumberInput input, .stSelectbox select, .stTextInput input {
    border: 2px solid var(--border-light) !important;
    border-radius: 12px !important;
    padding: 0.8rem !important;
    background: white !important;
    color: var(--navy-dark) !important;
    transition: var(--transition) !important;
}
.stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus {
    border-color: var(--accent-warm) !important;
    box-shadow: 0 0 0 3px rgba(246,197,49,0.2) !important;
}

/* ==================== BADGES ==================== */
.status-badge {
    display: inline-block;
    padding: 0.5rem 1.2rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.85rem;
    background: var(--blue-soft);
    color: white;
}
.badge-pass { background: var(--success); }
.badge-fail { background: var(--danger); }
.badge-warning { background: var(--accent-warm); color: var(--navy-dark); }

/* ==================== DATA ROWS ==================== */
.data-row {
    display: flex;
    justify-content: space-between;
    padding: 1rem;
    border-bottom: 1px solid var(--border-light);
}
.data-label { font-weight: 600; color: var(--navy-dark); }
.data-value { font-weight: 700; color: var(--blue-deep); }

/* ==================== REASON ITEMS ==================== */
.reason-item {
    background: rgba(136,151,189,0.1);
    border-left: 5px solid var(--accent-warm);
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.reason-icon { font-size: 1.5rem; color: var(--accent-warm); }

/* ==================== SCROLLBAR ==================== */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--lavender-light); }
::-webkit-scrollbar-thumb { background: var(--blue-soft); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--blue-deep); }

/* ==================== RESPONSIVE ==================== */
@media (max-width: 768px) {
    .main-header { font-size: 2.2rem; }
    .decision-title { font-size: 2.2rem; }
    .stat-number { font-size: 2rem; }
}
</style>
"""
