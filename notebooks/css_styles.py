# css_styles.py

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    /* Primary palette – pastel version */
    --fern-green: #6C7A89;        /* muted slate blue (primary dark) */
    --sage: #C1D5C0;               /* soft sage green */
    --cosmic-latte: #FFF9E6;       /* light cream */
    --jasmine: #FFDAB9;             /* soft peach */
    --saffron: #FFF5BA;             /* pastel yellow (accent) */

    /* Derived shades */
    --dark-fern: #4A5A6A;           /* darker slate */
    --light-sage: #E0ECDE;          /* very light sage */
    --fern-dark: #4A5A6A;           /* alias for dark-fern */
    --fern-light: #8A9BA8;          /* lighter slate */
    --sage-light: #E0ECDE;
    --sage-dark: #A2B5A0;           /* muted green-grey */
    --gold: #FFE5B4;                 /* soft peach/gold */
    --gold-light: #FFF0DB;           /* very light peach */

    /* Greys – kept neutral but can be softened */
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

    /* Semantic status – pastel versions */
    --success: #A8E6CF;             /* soft mint green */
    --success-light: #D1FAE5;
    --warning: #FFD3B6;              /* soft apricot */
    --warning-light: #FEF3C7;
    --danger: #FFB7B2;               /* soft coral */
    --danger-light: #FEE2E2;
    --info: #B5EAD7;                  /* soft seafoam */
    --info-light: #DBEAFE;

    /* Shadows – use the primary colour with low opacity */
    --shadow-sm: 0 1px 2px 0 rgba(108, 122, 137, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(108, 122, 137, 0.1), 0 2px 4px -1px rgba(108, 122, 137, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(108, 122, 137, 0.15), 0 4px 6px -2px rgba(108, 122, 137, 0.05);
    --shadow-xl: 0 20px 25px -5px rgba(108, 122, 137, 0.2), 0 10px 10px -5px rgba(108, 122, 137, 0.04);
    --shadow-2xl: 0 25px 50px -12px rgba(108, 122, 137, 0.25);

    /* Glass effects */
    --glass-bg: rgba(255, 255, 255, 0.7);
    --glass-border: rgba(108, 122, 137, 0.15);

    /* Additional variables for components */
    --text-secondary: var(--gray-600);
    --border-light: rgba(108, 122, 137, 0.2);
    --transition: var(--transition-base);

    /* Transitions */
    --transition-fast: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-slow: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-bounce: all 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

/* ==================== GLOBAL ==================== */
* { font-family: 'Inter', sans-serif; }

.main {
    background: linear-gradient(145deg, #ffffff 0%, var(--cosmic-latte) 100%);
}

.block-container { padding: 2rem; }

/* ==================== HEADERS ==================== */
.main-header {
    font-size: 3rem;
    font-weight: 800;
    color: var(--fern-green);
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
    background: linear-gradient(90deg, transparent, var(--saffron), transparent);
    border-radius: 2px;
}

.section-header {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--fern-green);
    margin: 2rem 0 1rem;
    padding-left: 1rem;
    border-left: 5px solid var(--saffron);
}

/* ==================== CARDS ==================== */
.info-card {
    background: var(--glass-bg);
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
    border-color: var(--saffron);
}

.info-card-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--fern-green);
    margin-bottom: 1rem;
    border-bottom: 2px solid var(--sage);
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
    color: var(--fern-green);
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
    background: linear-gradient(135deg, var(--fern-green), var(--dark-fern));
    box-shadow: var(--shadow-xl);
    border: 1px solid rgba(255,255,255,0.1);
}
.decision-card.approved { background: linear-gradient(135deg, var(--success), var(--dark-fern)); }
.decision-card.rejected { background: linear-gradient(135deg, var(--danger), var(--fern-green)); }
.decision-card.review { background: linear-gradient(135deg, var(--saffron), var(--dark-fern)); color: var(--fern-green); }

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
    background: var(--fern-green) !important;
    color: white !important;
}
.stButton > button:hover {
    background: var(--dark-fern) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg) !important;
}

.stDownloadButton button {
    background: var(--saffron) !important;
    color: var(--fern-green) !important;
}
.stDownloadButton button:hover {
    background: #e6dba8 !important;  /* slightly darker pastel yellow */
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
    background: var(--fern-green);
    color: white !important;
    box-shadow: var(--shadow-md);
}

/* ==================== METRICS ==================== */
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    color: var(--fern-green) !important;
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
    color: var(--fern-green) !important;
    transition: var(--transition) !important;
}
.stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus {
    border-color: var(--saffron) !important;
    box-shadow: 0 0 0 3px rgba(255, 245, 186, 0.5) !important;
}

/* ==================== BADGES ==================== */
.status-badge {
    display: inline-block;
    padding: 0.5rem 1.2rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.85rem;
    background: var(--sage);
    color: white;
}
.badge-pass { background: var(--success); }
.badge-fail { background: var(--danger); }
.badge-warning { background: var(--saffron); color: var(--fern-green); }

/* ==================== DATA ROWS ==================== */
.data-row {
    display: flex;
    justify-content: space-between;
    padding: 1rem;
    border-bottom: 1px solid var(--border-light);
}
.data-label { font-weight: 600; color: var(--fern-green); }
.data-value { font-weight: 700; color: var(--dark-fern); }

/* ==================== REASON ITEMS ==================== */
.reason-item {
    background: rgba(108, 122, 137, 0.1);  /* using --fern-green with opacity */
    border-left: 5px solid var(--saffron);
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.reason-icon { font-size: 1.5rem; color: var(--saffron); }

/* ==================== SCROLLBAR ==================== */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--cosmic-latte); }
::-webkit-scrollbar-thumb { background: var(--sage); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--dark-fern); }

/* ==================== RESPONSIVE ==================== */
@media (max-width: 768px) {
    .main-header { font-size: 2.2rem; }
    .decision-title { font-size: 2.2rem; }
    .stat-number { font-size: 2rem; }
}
</style>
"""
