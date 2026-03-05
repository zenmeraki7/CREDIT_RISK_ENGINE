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
# """



CSS = """
<style>
/* ========== FONTS ========== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ========== CSS VARIABLES ========== */
:root {
    /* Pastel Green Palette */
    --beige-soft: #F1F5D8;           /* main background */
    --tea-green: #D6FFC1;             /* card background, light sections */
    --light-green-1: #B9FFAF;          /* success, hover states */
    --light-green-2: #97FA9A;          /* accent for buttons, borders */
    --aquamarine: #8AF0BF;             /* primary accent (teal-like) */

    /* Dark text for contrast */
    --text-dark: #1E3A3A;               /* deep green‑grey for primary text */
    --text-muted: #3A5A5A;               /* softer for secondary text */

    /* Neutrals */
    --border-soft: #C0D9C0;              /* muted green border */
    --shadow-light: rgba(0,40,20,0.05);   /* very soft green shadow */

    /* Status colors (keep for consistency) */
    --success: var(--light-green-2);
    --warning: #F59E0B;
    --danger: #EF4444;

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
    background-color: var(--beige-soft);
    color: var(--text-dark);
    line-height: 1.5;
}

.main {
    background: var(--beige-soft);
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
    color: var(--text-dark);
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
    background: var(--aquamarine);
    margin: var(--space-sm) auto 0;
    border-radius: 2px;
}

.section-header {
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--text-dark);
    margin: var(--space-lg) 0 var(--space-md);
    padding-bottom: var(--space-xs);
    border-bottom: 2px solid var(--aquamarine);
}

/* ========== CARDS ========== */
.card, .info-card {
    background: var(--tea-green);
    border-radius: 24px;
    padding: var(--space-lg);
    box-shadow: 0 4px 12px var(--shadow-light);
    transition: all 0.2s ease;
    border: 1px solid var(--border-soft);
}
.card:hover, .info-card:hover {
    box-shadow: 0 8px 20px var(--shadow-light);
    transform: translateY(-2px);
    border-color: var(--aquamarine);
}

.info-card-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: var(--space-sm);
    border-bottom: 1px solid var(--border-soft);
    padding-bottom: var(--space-xs);
    display: flex;
    align-items: center;
    gap: var(--space-xs);
}
.info-card-title i {
    color: var(--aquamarine);
}

.info-card-content {
    color: var(--text-muted);
    line-height: 1.6;
}

/* ========== STAT CARDS ========== */
.stat-card {
    background: var(--tea-green);
    border-radius: 24px;
    padding: var(--space-lg);
    text-align: center;
    box-shadow: 0 2px 8px var(--shadow-light);
    border: 1px solid var(--border-soft);
    transition: all 0.2s ease;
}
.stat-card:hover {
    box-shadow: 0 4px 12px var(--shadow-light);
    border-color: var(--aquamarine);
    transform: scale(1.02);
}
.stat-number {
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--text-dark);
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
    background: linear-gradient(135deg, var(--light-green-2), var(--aquamarine));
    box-shadow: 0 12px 24px var(--shadow-light);
    margin: var(--space-lg) 0;
}
.decision-card-approved {
    background: linear-gradient(135deg, var(--light-green-1), var(--light-green-2));
    color: var(--text-dark);
}
.decision-card-rejected {
    background: linear-gradient(135deg, var(--danger), #B91C1C);
}
.decision-card-review {
    background: linear-gradient(135deg, var(--warning), #B45309);
    color: white;
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
    border-radius: 40px;
    font-weight: 600;
    font-size: 0.85rem;
    gap: 0.5rem;
    box-shadow: 0 2px 6px var(--shadow-light);
    border: 1px solid transparent;
}
.badge-pass {
    background: rgba(151, 250, 154, 0.2);
    color: #1E4A1E;
    border-color: var(--light-green-2);
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
    background: var(--tea-green);
    padding-left: var(--space-sm);
    padding-right: var(--space-sm);
}
.data-row:last-child {
    border-bottom: none;
}
.data-label {
    font-weight: 600;
    color: var(--text-dark);
}
.data-value {
    font-weight: 700;
    color: var(--text-dark);
}

/* ========== REASON ITEMS ========== */
.reason-item {
    background: var(--tea-green);
    border-left: 5px solid var(--aquamarine);
    padding: var(--space-md);
    border-radius: 16px;
    margin-bottom: var(--space-sm);
    display: flex;
    align-items: flex-start;
    gap: var(--space-sm);
    transition: all 0.2s;
    box-shadow: 0 2px 8px var(--shadow-light);
}
.reason-item:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 12px var(--shadow-light);
    border-left-width: 8px;
}
.reason-icon {
    font-size: 1.5rem;
    color: var(--aquamarine);
    flex-shrink: 0;
}

/* ========== INFO/WARNING/ERROR BOXES ========== */
.info-box, .warning-box, .error-box {
    border-radius: 16px;
    padding: var(--space-md);
    margin: var(--space-md) 0;
    border-left: 5px solid;
    background: var(--tea-green);
    box-shadow: 0 2px 8px var(--shadow-light);
}
.info-box {
    border-left-color: var(--aquamarine);
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
    padding: 0.75rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px var(--shadow-light) !important;
    cursor: pointer !important;
    width: 100% !important;
}

.stButton > button {
    background: var(--light-green-2) !important;
    color: var(--text-dark) !important;
}
.stButton > button:hover {
    background: var(--aquamarine) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 14px var(--shadow-light) !important;
}

.stDownloadButton button {
    background: var(--tea-green) !important;
    color: var(--text-dark) !important;
    border: 1px solid var(--border-soft) !important;
}
.stDownloadButton button:hover {
    background: var(--light-green-1) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px var(--shadow-light) !important;
    border-color: var(--aquamarine) !important;
}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"] {
    background: var(--tea-green);
    border-radius: 60px;
    padding: 0.5rem;
    gap: 0.5rem;
    box-shadow: 0 2px 8px var(--shadow-light);
    border: 1px solid var(--border-soft);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 40px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600;
    color: var(--text-muted);
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: var(--light-green-2) !important;
    color: var(--text-dark) !important;
    box-shadow: 0 4px 10px var(--shadow-light) !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: var(--beige-soft);
    color: var(--text-dark);
}

/* ========== METRICS (Streamlit's built-in metric) ========== */
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: var(--text-dark) !important;
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
    color: var(--text-dark) !important;
    font-size: 1rem !important;
    transition: border 0.2s, box-shadow 0.2s !important;
}
.stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus, .stDateInput input:focus {
    border-color: var(--aquamarine) !important;
    box-shadow: 0 0 0 4px rgba(138, 240, 191, 0.2) !important;
    outline: none !important;
}

.stNumberInput label, .stSelectbox label, .stTextInput label, .stDateInput label {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    margin-bottom: 0.25rem !important;
}

/* ========== CHECKBOX & RADIO ========== */
.stCheckbox > label, .stRadio > div {
    color: var(--text-dark);
}
.stCheckbox > label:hover, .stRadio > div:hover {
    color: var(--aquamarine);
}

/* ========== DATA FRAMES ========== */
.dataframe {
    border-collapse: collapse;
    width: 100%;
    background: white;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 2px 8px var(--shadow-light);
    border: 1px solid var(--border-soft);
}
.dataframe th {
    background: var(--tea-green);
    font-weight: 600;
    color: var(--text-dark);
    padding: var(--space-sm);
    text-align: left;
    border-bottom: 2px solid var(--aquamarine);
}
.dataframe td {
    padding: var(--space-sm);
    border-bottom: 1px solid var(--border-soft);
    color: var(--text-muted);
}
.dataframe tr:last-child td {
    border-bottom: none;
}
.dataframe tr:hover {
    background: var(--beige-soft);
}

/* ========== PROGRESS BAR ========== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--light-green-2), var(--aquamarine)) !important;
    border-radius: 20px !important;
    box-shadow: 0 2px 6px var(--shadow-light) !important;
}

/* ========== EXPANDER ========== */
.streamlit-expanderHeader {
    background: var(--tea-green) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 16px !important;
    padding: 0.75rem 1rem !important;
    font-weight: 600 !important;
    color: var(--text-dark) !important;
    transition: all 0.2s !important;
}
.streamlit-expanderHeader:hover {
    border-color: var(--aquamarine) !important;
    box-shadow: 0 2px 8px var(--shadow-light) !important;
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] {
    background: var(--tea-green);
    border-right: 1px solid var(--border-soft);
    box-shadow: 2px 0 10px var(--shadow-light);
}
[data-testid="stSidebar"] .block-container {
    padding: var(--space-lg) var(--space-md);
}
[data-testid="stSidebar"] .sidebar-content {
    background: var(--beige-soft);
    border-radius: 16px;
    padding: var(--space-sm);
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--beige-soft);
}
::-webkit-scrollbar-thumb {
    background: var(--aquamarine);
    border-radius: 20px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--light-green-2);
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
    background: linear-gradient(90deg, transparent, var(--aquamarine), transparent);
}
::placeholder {
    color: var(--text-muted) !important;
    opacity: 0.6 !important;
}
::selection {
    background: var(--aquamarine);
    color: var(--text-dark);
}
</style>
"""
