# # css_styles.py

# CSS = """
# <style>
# /* ========== FONTS ========== */
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

# /* ========== CSS VARIABLES ========== */
# :root {
#     /* New palette */
#     --navy-deep: #00002A;           /* deep navy – primary text, headers */
#     --navy-mid: #1A3F75;             /* medium blue – secondary, hover */
#     --teal-light: #44E6A9;           /* soft teal – accent, highlights */
#     --mint-fresh: #44EA4C;            /* fresh mint – success states */

#     /* Neutrals */
#     --bg-main: #FFFFFF;                /* pure white background */
#     --card-bg: #FFFFFF;                 /* white cards */
#     --text-primary: #1E293B;            /* dark slate (kept for softer text) */
#     --text-secondary: #64748B;           /* softer grey for labels */
#     --border-soft: #E2E8F0;              /* very light border */
#     --light-bg: #F8FAFC;                  /* very light blue-grey for subtle backgrounds */

#     /* Shadows – very subtle */
#     --shadow-sm: 0 2px 4px rgba(0,0,0,0.02);
#     --shadow-md: 0 4px 8px rgba(0,0,0,0.03);
#     --shadow-lg: 0 8px 16px rgba(0,0,0,0.04);
#     --shadow-xl: 0 12px 24px rgba(0,0,0,0.05);

#     /* Status colors */
#     --success: var(--mint-fresh);
#     --warning: #F59E0B;
#     --danger: #EF4444;
#     --info: var(--teal-light);

#     /* Spacing */
#     --space-xs: 0.5rem;
#     --space-sm: 1rem;
#     --space-md: 1.5rem;
#     --space-lg: 2rem;
#     --space-xl: 3rem;
# }

# /* ========== GLOBAL RESET ========== */
# * {
#     font-family: 'Inter', sans-serif;
#     margin: 0;
#     padding: 0;
#     box-sizing: border-box;
# }

# body {
#     background-color: var(--bg-main);
#     color: var(--text-primary);
#     line-height: 1.5;
# }

# .main {
#     background: var(--bg-main);
#     padding: var(--space-lg) 0;
# }

# .block-container {
#     max-width: 1400px;
#     margin: 0 auto;
#     padding: 0 var(--space-lg);
# }

# /* ========== TYPOGRAPHY ========== */
# .main-header {
#     font-size: 2.8rem;
#     font-weight: 700;
#     color: var(--navy-deep);
#     text-align: center;
#     margin: var(--space-xl) 0 var(--space-lg);
#     letter-spacing: -0.02em;
#     position: relative;
# }
# .main-header::after {
#     content: '';
#     display: block;
#     width: 80px;
#     height: 4px;
#     background: var(--teal-light);
#     margin: var(--space-sm) auto 0;
#     border-radius: 2px;
# }

# .section-header {
#     font-size: 1.8rem;
#     font-weight: 600;
#     color: var(--navy-deep);
#     margin: var(--space-lg) 0 var(--space-md);
#     padding-bottom: var(--space-xs);
#     border-bottom: 2px solid var(--teal-light);
# }

# /* ========== CARDS ========== */
# .card, .info-card {
#     background: var(--card-bg);
#     border-radius: 24px;
#     padding: var(--space-lg);
#     box-shadow: var(--shadow-md);
#     transition: all 0.2s ease;
#     border: 1px solid var(--border-soft);
# }
# .card:hover, .info-card:hover {
#     box-shadow: var(--shadow-lg);
#     transform: translateY(-2px);
#     border-color: var(--teal-light);
# }

# .info-card-title {
#     font-size: 1.2rem;
#     font-weight: 600;
#     color: var(--navy-deep);
#     margin-bottom: var(--space-sm);
#     border-bottom: 1px solid var(--border-soft);
#     padding-bottom: var(--space-xs);
#     display: flex;
#     align-items: center;
#     gap: var(--space-xs);
# }
# .info-card-title i {
#     color: var(--teal-light);
# }

# .info-card-content {
#     color: var(--text-secondary);
#     line-height: 1.6;
# }

# /* ========== STAT CARDS (for metric displays like RF, FEATURES, CLASSES) ========== */
# .stat-card {
#     background: var(--card-bg);
#     border-radius: 24px;
#     padding: var(--space-lg);
#     text-align: center;
#     box-shadow: var(--shadow-sm);
#     border: 1px solid var(--border-soft);
#     transition: all 0.2s ease;
# }
# .stat-card:hover {
#     box-shadow: var(--shadow-md);
#     border-color: var(--teal-light);
#     transform: scale(1.02);
# }
# .stat-number {
#     font-size: 2.8rem;
#     font-weight: 700;
#     color: var(--navy-deep);
#     line-height: 1.2;
#     display: block;
#     margin-bottom: var(--space-xs);
# }
# .stat-label {
#     font-size: 0.9rem;
#     font-weight: 500;
#     color: var(--text-secondary);
#     text-transform: uppercase;
#     letter-spacing: 0.05em;
# }

# /* ========== DECISION CARDS ========== */
# .decision-card {
#     border-radius: 32px;
#     padding: var(--space-xl);
#     color: white;
#     text-align: center;
#     background: linear-gradient(135deg, var(--navy-deep), var(--navy-mid));
#     box-shadow: var(--shadow-xl);
#     margin: var(--space-lg) 0;
# }
# .decision-card-approved {
#     background: linear-gradient(135deg, var(--mint-fresh), var(--teal-light));
#     color: var(--navy-deep);
# }
# .decision-card-rejected {
#     background: linear-gradient(135deg, var(--danger), #B91C1C);
# }
# .decision-card-review {
#     background: linear-gradient(135deg, var(--warning), #B45309);
#     color: white;
# }

# .decision-title {
#     font-size: 3rem;
#     font-weight: 800;
#     display: flex;
#     align-items: center;
#     justify-content: center;
#     gap: var(--space-sm);
# }
# .decision-subtitle {
#     font-size: 1.2rem;
#     margin-top: var(--space-sm);
#     opacity: 0.9;
# }

# /* ========== STATUS BADGES ========== */
# .status-badge {
#     display: inline-flex;
#     align-items: center;
#     padding: 0.5rem 1.2rem;
#     border-radius: 40px;
#     font-weight: 600;
#     font-size: 0.85rem;
#     gap: 0.5rem;
#     box-shadow: var(--shadow-sm);
#     border: 1px solid transparent;
# }
# .badge-pass {
#     background: rgba(68, 234, 76, 0.1);
#     color: #2B6E2B;
#     border-color: var(--mint-fresh);
# }
# .badge-fail {
#     background: #FEE2E2;
#     color: var(--danger);
#     border-color: var(--danger);
# }
# .badge-warning {
#     background: #FEF3C7;
#     color: #92400E;
#     border-color: var(--warning);
# }

# /* ========== DATA ROWS ========== */
# .data-row {
#     display: flex;
#     justify-content: space-between;
#     align-items: center;
#     padding: var(--space-sm) 0;
#     border-bottom: 1px solid var(--border-soft);
#     transition: background 0.2s;
#     border-radius: 8px;
# }
# .data-row:hover {
#     background: var(--light-bg);
#     padding-left: var(--space-sm);
#     padding-right: var(--space-sm);
# }
# .data-row:last-child {
#     border-bottom: none;
# }
# .data-label {
#     font-weight: 600;
#     color: var(--text-primary);
# }
# .data-value {
#     font-weight: 700;
#     color: var(--navy-mid);
# }

# /* ========== REASON ITEMS ========== */
# .reason-item {
#     background: var(--light-bg);
#     border-left: 5px solid var(--teal-light);
#     padding: var(--space-md);
#     border-radius: 16px;
#     margin-bottom: var(--space-sm);
#     display: flex;
#     align-items: flex-start;
#     gap: var(--space-sm);
#     transition: all 0.2s;
#     box-shadow: var(--shadow-sm);
# }
# .reason-item:hover {
#     transform: translateX(4px);
#     box-shadow: var(--shadow-md);
#     border-left-width: 8px;
# }
# .reason-icon {
#     font-size: 1.5rem;
#     color: var(--teal-light);
#     flex-shrink: 0;
# }

# /* ========== INFO/WARNING/ERROR BOXES ========== */
# .info-box, .warning-box, .error-box {
#     border-radius: 16px;
#     padding: var(--space-md);
#     margin: var(--space-md) 0;
#     border-left: 5px solid;
#     background: var(--light-bg);
#     box-shadow: var(--shadow-sm);
# }
# .info-box {
#     border-left-color: var(--teal-light);
# }
# .warning-box {
#     border-left-color: var(--warning);
# }
# .error-box {
#     border-left-color: var(--danger);
# }

# /* ========== BUTTONS ========== */
# .stButton > button, .stDownloadButton button {
#     border: none !important;
#     border-radius: 40px !important;
#     padding: 0.75rem 1.8rem !important;
#     font-weight: 600 !important;
#     font-size: 1rem !important;
#     transition: all 0.2s ease !important;
#     box-shadow: var(--shadow-sm) !important;
#     cursor: pointer !important;
#     width: 100% !important;
# }

# .stButton > button {
#     background: var(--navy-deep) !important;
#     color: white !important;
# }
# .stButton > button:hover {
#     background: var(--navy-mid) !important;
#     transform: translateY(-2px) !important;
#     box-shadow: var(--shadow-lg) !important;
# }

# .stDownloadButton button {
#     background: white !important;
#     color: var(--navy-deep) !important;
#     border: 1px solid var(--border-soft) !important;
# }
# .stDownloadButton button:hover {
#     background: var(--light-bg) !important;
#     transform: translateY(-2px) !important;
#     box-shadow: var(--shadow-md) !important;
#     border-color: var(--teal-light) !important;
# }

# /* ========== TABS ========== */
# .stTabs [data-baseweb="tab-list"] {
#     background: var(--card-bg);
#     border-radius: 60px;
#     padding: 0.5rem;
#     gap: 0.5rem;
#     box-shadow: var(--shadow-sm);
#     border: 1px solid var(--border-soft);
# }
# .stTabs [data-baseweb="tab"] {
#     border-radius: 40px !important;
#     padding: 0.6rem 2rem !important;
#     font-weight: 600;
#     color: var(--text-secondary);
#     transition: all 0.2s;
# }
# .stTabs [aria-selected="true"] {
#     background: var(--navy-deep) !important;
#     color: white !important;
#     box-shadow: var(--shadow-md) !important;
# }
# .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
#     background: var(--light-bg);
#     color: var(--navy-deep);
# }

# /* ========== METRICS (Streamlit's built-in metric) ========== */
# [data-testid="stMetricValue"] {
#     font-size: 2.5rem !important;
#     font-weight: 700 !important;
#     color: var(--navy-deep) !important;
# }
# [data-testid="stMetricLabel"] {
#     font-size: 0.9rem !important;
#     font-weight: 500 !important;
#     color: var(--text-secondary) !important;
#     text-transform: uppercase;
#     letter-spacing: 0.05em;
# }
# [data-testid="stMetricDelta"] {
#     font-size: 0.85rem !important;
# }

# /* ========== INPUT FIELDS ========== */
# .stNumberInput input, .stSelectbox select, .stTextInput input, .stDateInput input {
#     border: 2px solid var(--border-soft) !important;
#     border-radius: 16px !important;
#     padding: 0.75rem 1rem !important;
#     background: white !important;
#     color: var(--text-primary) !important;
#     font-size: 1rem !important;
#     transition: border 0.2s, box-shadow 0.2s !important;
# }
# .stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus, .stDateInput input:focus {
#     border-color: var(--teal-light) !important;
#     box-shadow: 0 0 0 4px rgba(68, 230, 169, 0.1) !important;
#     outline: none !important;
# }

# .stNumberInput label, .stSelectbox label, .stTextInput label, .stDateInput label {
#     font-size: 0.9rem !important;
#     font-weight: 600 !important;
#     color: var(--text-secondary) !important;
#     margin-bottom: 0.25rem !important;
# }

# /* ========== CHECKBOX & RADIO ========== */
# .stCheckbox > label, .stRadio > div {
#     color: var(--text-primary);
# }
# .stCheckbox > label:hover, .stRadio > div:hover {
#     color: var(--navy-mid);
# }

# /* ========== DATA FRAMES ========== */
# .dataframe {
#     border-collapse: collapse;
#     width: 100%;
#     background: white;
#     border-radius: 20px;
#     overflow: hidden;
#     box-shadow: var(--shadow-sm);
#     border: 1px solid var(--border-soft);
# }
# .dataframe th {
#     background: var(--light-bg);
#     font-weight: 600;
#     color: var(--navy-deep);
#     padding: var(--space-sm);
#     text-align: left;
#     border-bottom: 2px solid var(--teal-light);
# }
# .dataframe td {
#     padding: var(--space-sm);
#     border-bottom: 1px solid var(--border-soft);
#     color: var(--text-secondary);
# }
# .dataframe tr:last-child td {
#     border-bottom: none;
# }
# .dataframe tr:hover {
#     background: var(--light-bg);
# }

# /* ========== PROGRESS BAR ========== */
# .stProgress > div > div > div {
#     background: linear-gradient(90deg, var(--teal-light), var(--mint-fresh)) !important;
#     border-radius: 20px !important;
#     box-shadow: 0 2px 4px rgba(68, 230, 169, 0.2) !important;
# }

# /* ========== EXPANDER ========== */
# .streamlit-expanderHeader {
#     background: white !important;
#     border: 1px solid var(--border-soft) !important;
#     border-radius: 16px !important;
#     padding: 0.75rem 1rem !important;
#     font-weight: 600 !important;
#     color: var(--navy-deep) !important;
#     transition: all 0.2s !important;
# }
# .streamlit-expanderHeader:hover {
#     border-color: var(--teal-light) !important;
#     box-shadow: var(--shadow-sm) !important;
# }

# /* ========== SIDEBAR ========== */
# [data-testid="stSidebar"] {
#     background: white;
#     border-right: 1px solid var(--border-soft);
#     box-shadow: 2px 0 10px rgba(0,0,0,0.02);
# }
# [data-testid="stSidebar"] .block-container {
#     padding: var(--space-lg) var(--space-md);
# }
# [data-testid="stSidebar"] .sidebar-content {
#     background: var(--light-bg);
#     border-radius: 16px;
#     padding: var(--space-sm);
# }

# /* ========== SCROLLBAR ========== */
# ::-webkit-scrollbar {
#     width: 6px;
#     height: 6px;
# }
# ::-webkit-scrollbar-track {
#     background: var(--light-bg);
# }
# ::-webkit-scrollbar-thumb {
#     background: var(--teal-light);
#     border-radius: 20px;
# }
# ::-webkit-scrollbar-thumb:hover {
#     background: var(--navy-mid);
# }

# /* ========== RESPONSIVE ========== */
# @media (max-width: 768px) {
#     .main-header { font-size: 2.2rem; }
#     .decision-title { font-size: 2.2rem; }
#     .stat-number { font-size: 2rem; }
#     .block-container { padding: 0 var(--space-md); }
# }

# /* ========== MISC ========== */
# hr {
#     margin: var(--space-lg) 0;
#     border: none;
#     height: 2px;
#     background: linear-gradient(90deg, transparent, var(--teal-light), transparent);
# }
# ::placeholder {
#     color: var(--text-secondary) !important;
#     opacity: 0.6 !important;
# }
# ::selection {
#     background: var(--teal-light);
#     color: var(--navy-deep);
# }
# </style>






CSS = """
<style>
/* ========== FONTS ========== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ========== CSS VARIABLES ========== */
:root {
    /* Enhanced palette */
    --navy-deep: #0B1E33;           /* softer deep navy */
    --navy-mid: #1F3A5F;             /* richer medium blue */
    --teal-light: #4FD1C5;           /* refined teal */
    --mint-fresh: #68D391;            /* soft mint green */
    --accent-gold: #F6AD55;            /* warm gold accent */

    /* Neutrals – refined */
    --bg-main: #F7FAFC;                /* off-white with cool tint */
    --card-bg: #FFFFFF;                 /* pure white */
    --text-primary: #1A2B3C;            /* slightly softer dark */
    --text-secondary: #4A5B6B;           /* medium grey */
    --text-muted: #718096;                /* light grey for captions */
    --border-soft: #E2E8F0;              /* very light border */
    --light-bg: #EDF2F7;                  /* subtle grey for hover */

    /* Shadows – layered and more elegant */
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.05), 0 2px 4px rgba(0,0,0,0.05);
    --shadow-lg: 0 10px 15px rgba(0,0,0,0.05), 0 4px 6px rgba(0,0,0,0.05);
    --shadow-xl: 0 20px 25px rgba(0,0,0,0.05), 0 10px 10px rgba(0,0,0,0.02);

    /* Status colors (enhanced) */
    --success: #48BB78;
    --warning: #ED8936;
    --danger: #F56565;
    --info: var(--teal-light);

    /* Spacing (kept consistent) */
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
    background-color: var(--bg-main);
    color: var(--text-primary);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.main {
    background: var(--bg-main);
    padding: var(--space-lg) 0;
}

.block-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 var(--space-lg);
}

/* ========== TYPOGRAPHY ========== */
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--navy-deep);
    text-align: center;
    margin: var(--space-xl) 0 var(--space-lg);
    letter-spacing: -0.02em;
    position: relative;
}
.main-header::after {
    content: '';
    display: block;
    width: 100px;
    height: 4px;
    background: linear-gradient(90deg, var(--teal-light), var(--accent-gold));
    margin: var(--space-sm) auto 0;
    border-radius: 4px;
}

.section-header {
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--navy-deep);
    margin: var(--space-lg) 0 var(--space-md);
    padding-bottom: var(--space-xs);
    border-bottom: 2px solid var(--teal-light);
}

/* ========== CARDS ========== */
.card, .info-card {
    background: var(--card-bg);
    border-radius: 24px;
    padding: var(--space-lg);
    box-shadow: var(--shadow-md);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    border: 1px solid var(--border-soft);
    backdrop-filter: blur(10px);
}
.card:hover, .info-card:hover {
    box-shadow: var(--shadow-xl);
    transform: translateY(-4px);
    border-color: var(--teal-light);
}

.info-card-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--navy-deep);
    margin-bottom: var(--space-sm);
    border-bottom: 1px solid var(--border-soft);
    padding-bottom: var(--space-xs);
    display: flex;
    align-items: center;
    gap: var(--space-xs);
}
.info-card-title i {
    color: var(--teal-light);
}

.info-card-content {
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ========== STAT CARDS ========== */
.stat-card {
    background: var(--card-bg);
    border-radius: 24px;
    padding: var(--space-lg);
    text-align: center;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-soft);
    transition: all 0.3s ease;
}
.stat-card:hover {
    box-shadow: var(--shadow-lg);
    border-color: var(--teal-light);
    transform: scale(1.02);
}
.stat-number {
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--navy-deep);
    line-height: 1.2;
    display: block;
    margin-bottom: var(--space-xs);
}
.stat-label {
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ========== DECISION CARDS ========== */
.decision-card {
    border-radius: 32px;
    padding: var(--space-xl);
    color: white;
    text-align: center;
    background: linear-gradient(135deg, var(--navy-deep), var(--navy-mid));
    box-shadow: var(--shadow-xl);
    margin: var(--space-lg) 0;
    position: relative;
    overflow: hidden;
}
.decision-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    opacity: 0.3;
}
.decision-card-approved {
    background: linear-gradient(135deg, var(--mint-fresh), var(--teal-light));
    color: var(--navy-deep);
}
.decision-card-rejected {
    background: linear-gradient(135deg, var(--danger), #C53030);
}
.decision-card-review {
    background: linear-gradient(135deg, var(--warning), #DD6B20);
}

.decision-title {
    font-size: 3rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-sm);
    position: relative;
    z-index: 1;
}
.decision-subtitle {
    font-size: 1.2rem;
    margin-top: var(--space-sm);
    opacity: 0.9;
    position: relative;
    z-index: 1;
}

/* ========== STATUS BADGES ========== */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1.2rem;
    border-radius: 40px;
    font-weight: 600;
    font-size: 0.85rem;
    gap: 0.5rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid transparent;
    transition: all 0.2s;
}
.badge-pass {
    background: rgba(72, 187, 120, 0.1);
    color: #2F6E4A;
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
    border-bottom: 1px solid var(--border-soft);
    transition: background 0.2s;
    border-radius: 8px;
}
.data-row:hover {
    background: var(--light-bg);
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
    color: var(--navy-mid);
}

/* ========== REASON ITEMS ========== */
.reason-item {
    background: var(--light-bg);
    border-left: 5px solid var(--teal-light);
    padding: var(--space-md);
    border-radius: 16px;
    margin-bottom: var(--space-sm);
    display: flex;
    align-items: flex-start;
    gap: var(--space-sm);
    transition: all 0.3s;
    box-shadow: var(--shadow-sm);
}
.reason-item:hover {
    transform: translateX(4px);
    box-shadow: var(--shadow-lg);
    border-left-width: 8px;
}
.reason-icon {
    font-size: 1.5rem;
    color: var(--teal-light);
    flex-shrink: 0;
}

/* ========== INFO/WARNING/ERROR BOXES ========== */
.info-box, .warning-box, .error-box {
    border-radius: 16px;
    padding: var(--space-md);
    margin: var(--space-md) 0;
    border-left: 5px solid;
    background: var(--light-bg);
    box-shadow: var(--shadow-sm);
    transition: all 0.2s;
}
.info-box:hover, .warning-box:hover, .error-box:hover {
    box-shadow: var(--shadow-md);
}
.info-box {
    border-left-color: var(--teal-light);
}
.warning-box {
    border-left-color: var(--warning);
}
.error-box {
    border-left-color: var(--danger);
}

/* ========== BUTTONS ========== */
.stButton > button, .stDownloadButton button {
    border: none !important;
    border-radius: 40px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    box-shadow: var(--shadow-sm) !important;
    cursor: pointer !important;
    width: 100% !important;
    letter-spacing: 0.02em;
}

.stButton > button {
    background: linear-gradient(135deg, var(--navy-deep), var(--navy-mid)) !important;
    color: white !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--navy-mid), var(--navy-deep)) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-xl) !important;
}

.stDownloadButton button {
    background: linear-gradient(135deg, white, var(--light-bg)) !important;
    color: var(--navy-deep) !important;
    border: 1px solid var(--border-soft) !important;
}
.stDownloadButton button:hover {
    background: linear-gradient(135deg, var(--light-bg), white) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg) !important;
    border-color: var(--teal-light) !important;
}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card-bg);
    border-radius: 60px;
    padding: 0.5rem;
    gap: 0.5rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-soft);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 40px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600;
    color: var(--text-secondary);
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--navy-deep), var(--navy-mid)) !important;
    color: white !important;
    box-shadow: var(--shadow-md) !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: var(--light-bg);
    color: var(--navy-deep);
}

/* ========== METRICS (Streamlit's built-in metric) ========== */
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: var(--navy-deep) !important;
    line-height: 1.2;
}
[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricDelta"] {
    font-size: 0.85rem !important;
}

/* ========== INPUT FIELDS ========== */
.stNumberInput input, .stSelectbox select, .stTextInput input, .stDateInput input {
    border: 2px solid var(--border-soft) !important;
    border-radius: 16px !important;
    padding: 0.75rem 1rem !important;
    background: white !important;
    color: var(--text-primary) !important;
    font-size: 1rem !important;
    transition: all 0.2s !important;
}
.stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus, .stDateInput input:focus {
    border-color: var(--teal-light) !important;
    box-shadow: 0 0 0 4px rgba(79, 209, 197, 0.15) !important;
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
    transition: color 0.2s;
}
.stCheckbox > label:hover, .stRadio > div:hover {
    color: var(--navy-mid);
}

/* ========== DATA FRAMES ========== */
.dataframe {
    border-collapse: collapse;
    width: 100%;
    background: white;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-soft);
}
.dataframe th {
    background: linear-gradient(135deg, var(--light-bg), var(--bg-main));
    font-weight: 600;
    color: var(--navy-deep);
    padding: var(--space-sm);
    text-align: left;
    border-bottom: 2px solid var(--teal-light);
}
.dataframe td {
    padding: var(--space-sm);
    border-bottom: 1px solid var(--border-soft);
    color: var(--text-secondary);
}
.dataframe tr:last-child td {
    border-bottom: none;
}
.dataframe tr:hover {
    background: var(--light-bg);
}

/* ========== PROGRESS BAR ========== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--teal-light), var(--mint-fresh)) !important;
    border-radius: 20px !important;
    box-shadow: 0 2px 6px rgba(79, 209, 197, 0.3) !important;
}

/* ========== EXPANDER ========== */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, white, var(--light-bg)) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 16px !important;
    padding: 0.75rem 1rem !important;
    font-weight: 600 !important;
    color: var(--navy-deep) !important;
    transition: all 0.2s !important;
}
.streamlit-expanderHeader:hover {
    border-color: var(--teal-light) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--card-bg), var(--light-bg));
    border-right: 1px solid var(--border-soft);
    box-shadow: 2px 0 10px rgba(0,0,0,0.02);
}
[data-testid="stSidebar"] .block-container {
    padding: var(--space-lg) var(--space-md);
}
[data-testid="stSidebar"] .sidebar-content {
    background: var(--card-bg);
    border-radius: 16px;
    padding: var(--space-sm);
    box-shadow: var(--shadow-sm);
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: var(--light-bg);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: var(--teal-light);
    border-radius: 10px;
    border: 2px solid var(--light-bg);
}
::-webkit-scrollbar-thumb:hover {
    background: var(--navy-mid);
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
    background: linear-gradient(90deg, transparent, var(--teal-light), var(--accent-gold), transparent);
}
::placeholder {
    color: var(--text-muted) !important;
    opacity: 0.6 !important;
}
::selection {
    background: var(--teal-light);
    color: var(--navy-deep);
}
</style>
"""
# """
