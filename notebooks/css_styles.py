


# css_styles.py

CSS = """
<style>
/* ========== FONTS ========== */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

/* ========== DESIGN TOKENS ========== */
:root {
    /* ── Warm Ivory Base ── */
    --ivory-50:  #FDFAF5;   /* page background */
    --ivory-100: #F8F3E8;   /* card background */
    --ivory-200: #F0E9D6;   /* elevated card */
    --ivory-300: #E5D9BC;   /* borders */
    --ivory-400: #D4C49A;   /* strong borders */

    /* ── Deep Indigo Ink ── */
    --ink-900:  #1A1040;    /* primary headings */
    --ink-800:  #2D1F6E;    /* secondary headings */
    --ink-700:  #3D2D8C;    /* hover accent */
    --ink-600:  #5243A3;    /* interactive elements */
    --ink-500:  #7B6FBC;    /* muted accent */

    /* ── Saffron Gold ── */
    --saffron-500: #E8890C;   /* primary accent */
    --saffron-400: #F5A623;   /* lighter accent */
    --saffron-300: #F7BE5A;   /* soft gold */
    --saffron-100: #FEF3DC;   /* gold tint bg */
    --saffron-glow: rgba(232, 137, 12, 0.18);

    /* ── Semantic Status ── */
    --emerald-600: #047857;
    --emerald-500: #059669;
    --emerald-100: #D1FAE5;
    --emerald-glow: rgba(5, 150, 105, 0.15);

    --rose-600: #BE123C;
    --rose-500: #E11D48;
    --rose-100: #FFE4E6;
    --rose-glow: rgba(225, 29, 72, 0.15);

    --amber-600: #B45309;
    --amber-500: #D97706;
    --amber-100: #FEF3C7;
    --amber-glow: rgba(217, 119, 6, 0.15);

    /* ── Typography ── */
    --text-900: #0F0A2A;    /* headings */
    --text-700: #2D1F6E;    /* subheadings */
    --text-500: #4A4068;    /* body */
    --text-300: #8B7EA8;    /* secondary */
    --text-200: #B5A8CC;    /* placeholder */

    /* ── Refined Shadows ── */
    --shadow-xs: 0 1px 2px rgba(26, 16, 64, 0.04);
    --shadow-sm: 0 2px 8px rgba(26, 16, 64, 0.06), 0 1px 3px rgba(26, 16, 64, 0.08);
    --shadow-md: 0 4px 16px rgba(26, 16, 64, 0.08), 0 2px 6px rgba(26, 16, 64, 0.06);
    --shadow-lg: 0 8px 32px rgba(26, 16, 64, 0.10), 0 4px 12px rgba(26, 16, 64, 0.08);
    --shadow-xl: 0 16px 48px rgba(26, 16, 64, 0.12), 0 8px 24px rgba(26, 16, 64, 0.08);
    --shadow-gold: 0 4px 20px rgba(232, 137, 12, 0.25);
    --shadow-ink:  0 4px 20px rgba(61, 45, 140, 0.20);

    /* ── Spacing ── */
    --s-xs: 0.5rem;
    --s-sm: 1rem;
    --s-md: 1.5rem;
    --s-lg: 2rem;
    --s-xl: 3rem;

    /* ── Radius ── */
    --r-xs: 6px;
    --r-sm: 12px;
    --r-md: 18px;
    --r-lg: 24px;
    --r-xl: 32px;
    --r-full: 9999px;
}

/* ========== GLOBAL ========== */
*, *::before, *::after {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    box-sizing: border-box;
}

.stApp, body {
    background-color: var(--ivory-50) !important;
    background-image:
        radial-gradient(ellipse 70% 40% at 0% 0%, rgba(82, 67, 163, 0.04) 0%, transparent 55%),
        radial-gradient(ellipse 50% 35% at 100% 100%, rgba(232, 137, 12, 0.05) 0%, transparent 55%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%231A1040' fill-opacity='0.012'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") !important;
    color: var(--text-500) !important;
}

.main, .block-container {
    background: transparent !important;
    padding-top: var(--s-md) !important;
    max-width: 1440px !important;
}

/* ========== TYPOGRAPHY ========== */
.main-header {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 3rem;
    font-weight: 700;
    color: var(--ink-900);
    text-align: center;
    margin: var(--s-lg) 0 var(--s-md);
    letter-spacing: -0.02em;
    line-height: 1.2;
    position: relative;
    padding-bottom: var(--s-sm);
}
.main-header::after {
    content: '';
    display: block;
    width: 72px;
    height: 4px;
    background: linear-gradient(90deg, var(--saffron-500), var(--saffron-300));
    margin: var(--s-xs) auto 0;
    border-radius: var(--r-full);
    box-shadow: var(--shadow-gold);
}

.section-header {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 1.75rem;
    font-weight: 600;
    font-style: italic;
    color: var(--ink-900);
    margin: var(--s-lg) 0 var(--s-md);
    padding-bottom: var(--s-xs);
    border-bottom: 1px solid var(--ivory-300);
    position: relative;
}
.section-header::before {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, var(--saffron-500), transparent);
    border-radius: var(--r-full);
}

h1, h2, h3 {
    color: var(--ink-900) !important;
    font-family: 'Playfair Display', Georgia, serif !important;
}
h4, h5, h6 {
    color: var(--ink-800) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
}

p, li    { color: var(--text-500) !important; line-height: 1.7; }
label    { color: var(--text-300) !important; }
strong   { color: var(--text-700) !important; font-weight: 700 !important; }

/* ========== STREAMLIT OVERRIDES ========== */
[data-testid="stApp"]         { background: var(--ivory-50) !important; }
.stMarkdown p                 { color: var(--text-500) !important; }
section[data-testid="stSidebar"]       { background: var(--ivory-100) !important; border-right: 1px solid var(--ivory-300) !important; }
section[data-testid="stSidebar"] > div { background: transparent !important; }

/* ========== CARDS ========== */
.card, .info-card {
    background: #FFFFFF;
    border-radius: var(--r-lg);
    padding: var(--s-lg);
    box-shadow: var(--shadow-md);
    border: 1px solid var(--ivory-300);
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
    overflow: hidden;
}
.card::before, .info-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--ink-700), var(--saffron-500));
    opacity: 0;
    transition: opacity 0.3s;
}
.card:hover, .info-card:hover {
    box-shadow: var(--shadow-lg), var(--shadow-ink);
    transform: translateY(-4px);
    border-color: var(--ivory-400);
}
.card:hover::before, .info-card:hover::before { opacity: 1; }

.info-card-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-300);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: var(--s-sm);
    padding-bottom: var(--s-xs);
    border-bottom: 1px solid var(--ivory-200);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.info-card-content {
    color: var(--text-500);
    line-height: 1.7;
    font-size: 0.93rem;
}

/* ========== STAT CARDS ========== */
.stat-card {
    background: #FFFFFF;
    border-radius: var(--r-lg);
    padding: var(--s-lg);
    text-align: center;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--ivory-300);
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--ink-600), var(--saffron-500));
}
.stat-card:hover {
    box-shadow: var(--shadow-md), var(--shadow-ink);
    transform: translateY(-3px) scale(1.01);
    border-color: var(--ink-500);
}
.stat-number {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3rem;
    font-weight: 700;
    color: var(--ink-800);
    line-height: 1.15;
    display: block;
    margin-bottom: var(--s-xs);
}
.stat-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-300);
    text-transform: uppercase;
    letter-spacing: 0.13em;
}

/* ========== DECISION CARDS ========== */
.decision-card {
    border-radius: var(--r-xl);
    padding: var(--s-xl) var(--s-lg);
    text-align: center;
    background: #FFFFFF;
    border: 1px solid var(--ivory-300);
    box-shadow: var(--shadow-xl);
    margin: var(--s-lg) 0;
    position: relative;
    overflow: hidden;
}
.decision-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(150deg, rgba(255,255,255,0.8) 0%, transparent 60%);
    pointer-events: none;
}
.decision-card-approved {
    background: linear-gradient(145deg, #F0FDF4 0%, #ECFDF5 60%, #D1FAE5 100%) !important;
    border-color: rgba(5, 150, 105, 0.25) !important;
    box-shadow: var(--shadow-lg), 0 0 0 1px rgba(5,150,105,0.1), 0 8px 40px rgba(5,150,105,0.12) !important;
}
.decision-card-rejected {
    background: linear-gradient(145deg, #FFF1F2 0%, #FFF0F1 60%, #FFE4E6 100%) !important;
    border-color: rgba(225, 29, 72, 0.25) !important;
    box-shadow: var(--shadow-lg), 0 0 0 1px rgba(225,29,72,0.1), 0 8px 40px rgba(225,29,72,0.12) !important;
}
.decision-card-review {
    background: linear-gradient(145deg, #FFFBEB 0%, #FEF9EC 60%, #FEF3C7 100%) !important;
    border-color: rgba(217, 119, 6, 0.25) !important;
    box-shadow: var(--shadow-lg), 0 0 0 1px rgba(217,119,6,0.1), 0 8px 40px rgba(217,119,6,0.12) !important;
}

.decision-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--s-sm);
    position: relative;
    z-index: 1;
}
.decision-title.approved { color: var(--emerald-600); }
.decision-title.rejected { color: var(--rose-600); }
.decision-title.review   { color: var(--amber-600); }

.decision-subtitle {
    font-size: 1rem;
    margin-top: var(--s-sm);
    color: var(--text-300);
    letter-spacing: 0.02em;
    position: relative;
    z-index: 1;
}

/* ========== STATUS BADGES ========== */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.9rem;
    border-radius: var(--r-full);
    font-weight: 700;
    font-size: 0.72rem;
    gap: 0.4rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-pass {
    background: var(--emerald-100);
    color: var(--emerald-600);
    border: 1px solid rgba(5, 150, 105, 0.25);
    box-shadow: 0 1px 4px var(--emerald-glow);
}
.badge-fail {
    background: var(--rose-100);
    color: var(--rose-600);
    border: 1px solid rgba(225, 29, 72, 0.25);
    box-shadow: 0 1px 4px var(--rose-glow);
}
.badge-warning {
    background: var(--amber-100);
    color: var(--amber-600);
    border: 1px solid rgba(217, 119, 6, 0.25);
    box-shadow: 0 1px 4px var(--amber-glow);
}

/* ========== DATA ROWS ========== */
.data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.7rem var(--s-sm);
    border-bottom: 1px solid var(--ivory-200);
    border-radius: var(--r-xs);
    transition: all 0.15s ease;
}
.data-row:hover {
    background: var(--ivory-100);
    border-bottom-color: transparent;
    padding-left: var(--s-md);
}
.data-row:last-child { border-bottom: none; }
.data-label {
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--text-300);
}
.data-value {
    font-family: 'Fira Code', monospace;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--ink-800);
}

/* ========== REASON ITEMS ========== */
.reason-item {
    background: var(--ivory-100);
    border-left: 3px solid var(--saffron-500);
    padding: var(--s-md);
    border-radius: 0 var(--r-md) var(--r-md) 0;
    margin-bottom: var(--s-xs);
    display: flex;
    align-items: flex-start;
    gap: var(--s-sm);
    transition: all 0.2s ease;
    box-shadow: var(--shadow-xs);
}
.reason-item:hover {
    transform: translateX(5px);
    border-left-color: var(--ink-600);
    background: #FFFFFF;
    box-shadow: var(--shadow-sm);
}
.reason-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }

/* ========== INFO BOXES ========== */
.info-box, .warning-box, .error-box {
    border-radius: var(--r-md);
    padding: var(--s-md);
    margin: var(--s-md) 0;
    font-size: 0.93rem;
    line-height: 1.65;
    border: 1px solid;
}
.info-box {
    background: linear-gradient(135deg, rgba(82,67,163,0.04), rgba(232,137,12,0.04));
    border-color: rgba(82, 67, 163, 0.18);
    color: var(--ink-800);
}
.warning-box {
    background: var(--amber-100);
    border-color: rgba(217, 119, 6, 0.3);
    color: var(--amber-600);
}
.error-box {
    background: var(--rose-100);
    border-color: rgba(225, 29, 72, 0.3);
    color: var(--rose-600);
}

/* ========== BUTTONS ========== */
.stButton > button {
    background: linear-gradient(135deg, var(--ink-800) 0%, var(--ink-700) 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--r-full) !important;
    padding: 0.65rem 1.6rem !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    box-shadow: 0 4px 14px rgba(26, 16, 64, 0.25) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(26, 16, 64, 0.35) !important;
    background: linear-gradient(135deg, var(--ink-900) 0%, var(--ink-800) 100%) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.stDownloadButton button {
    background: #FFFFFF !important;
    color: var(--ink-800) !important;
    border: 1.5px solid var(--ivory-400) !important;
    border-radius: var(--r-full) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-xs) !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton button:hover {
    background: var(--saffron-100) !important;
    border-color: var(--saffron-500) !important;
    color: var(--amber-600) !important;
    box-shadow: var(--shadow-gold) !important;
    transform: translateY(-2px) !important;
}

.stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--saffron-500) 0%, var(--saffron-400) 100%) !important;
    color: var(--ink-900) !important;
    border: none !important;
    border-radius: var(--r-full) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 20px rgba(232, 137, 12, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}
.stFormSubmitButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 32px rgba(232, 137, 12, 0.45) !important;
}

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"] {
    background: var(--ivory-200) !important;
    border-radius: var(--r-full) !important;
    padding: 0.3rem !important;
    gap: 0.2rem !important;
    border: 1px solid var(--ivory-300) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--r-full) !important;
    padding: 0.45rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: var(--text-300) !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--ink-800) !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 10px rgba(26, 16, 64, 0.25) !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: var(--ivory-100) !important;
    color: var(--ink-800) !important;
}

/* ========== METRICS ========== */
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: var(--ink-900) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    color: var(--text-200) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid var(--ivory-300) !important;
    border-radius: var(--r-md) !important;
    padding: var(--s-md) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: var(--shadow-md), var(--shadow-ink) !important;
    border-color: var(--ink-500) !important;
    transform: translateY(-1px) !important;
}

/* ========== INPUTS ========== */
.stNumberInput input,
.stTextInput input,
.stDateInput input {
    background: #FFFFFF !important;
    border: 1.5px solid var(--ivory-300) !important;
    border-radius: var(--r-sm) !important;
    color: var(--ink-900) !important;
    font-size: 0.92rem !important;
    padding: 0.55rem 0.85rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    font-family: 'Fira Code', monospace !important;
    box-shadow: var(--shadow-xs) !important;
}
.stNumberInput input:focus,
.stTextInput input:focus,
.stDateInput input:focus {
    border-color: var(--ink-600) !important;
    box-shadow: 0 0 0 3px rgba(82, 67, 163, 0.1) !important;
    outline: none !important;
}
.stNumberInput label,
.stTextInput label,
.stDateInput label,
.stSelectbox label,
.stTextArea label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: var(--text-300) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #FFFFFF !important;
    border: 1.5px solid var(--ivory-300) !important;
    border-radius: var(--r-sm) !important;
    color: var(--ink-900) !important;
    box-shadow: var(--shadow-xs) !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--ink-600) !important;
    box-shadow: 0 0 0 3px rgba(82, 67, 163, 0.1) !important;
}

/* Textarea */
.stTextArea textarea {
    background: #FFFFFF !important;
    border: 1.5px solid var(--ivory-300) !important;
    border-radius: var(--r-md) !important;
    color: var(--ink-900) !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.87rem !important;
    box-shadow: var(--shadow-xs) !important;
}
.stTextArea textarea:focus {
    border-color: var(--ink-600) !important;
    box-shadow: 0 0 0 3px rgba(82, 67, 163, 0.1) !important;
}

/* Checkbox / Radio */
.stCheckbox > label {
    color: var(--text-500) !important;
    font-size: 0.9rem !important;
}
.stRadio > div > label {
    color: var(--text-500) !important;
    font-size: 0.9rem !important;
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--ink-900) !important; }
[data-testid="stSidebar"] p  { color: var(--text-500) !important; }
[data-testid="stSidebar"] .stRadio > div > label {
    color: var(--text-500) !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 0.8rem !important;
    border-radius: var(--r-sm) !important;
    transition: all 0.15s !important;
    display: block !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: var(--ivory-200) !important;
    color: var(--ink-700) !important;
}

/* ========== DATAFRAMES ========== */
[data-testid="stDataFrame"] {
    border-radius: var(--r-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--ivory-300) !important;
    box-shadow: var(--shadow-sm) !important;
}
.dataframe {
    background: #FFFFFF;
    border-collapse: collapse;
    width: 100%;
}
.dataframe th {
    background: var(--ivory-100);
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-300);
    padding: var(--s-sm);
    text-align: left;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 2px solid var(--saffron-500);
}
.dataframe td {
    padding: 0.7rem var(--s-sm);
    border-bottom: 1px solid var(--ivory-200);
    color: var(--text-500);
    font-size: 0.88rem;
    font-family: 'Fira Code', monospace;
}
.dataframe tr:last-child td { border-bottom: none; }
.dataframe tr:hover td {
    background: var(--ivory-100);
    color: var(--ink-800);
}

/* ========== EXPANDER ========== */
.streamlit-expanderHeader {
    background: #FFFFFF !important;
    border: 1px solid var(--ivory-300) !important;
    border-radius: var(--r-md) !important;
    color: var(--text-500) !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    box-shadow: var(--shadow-xs) !important;
}
.streamlit-expanderHeader:hover {
    border-color: var(--ink-500) !important;
    color: var(--ink-800) !important;
    box-shadow: var(--shadow-sm) !important;
}
.streamlit-expanderContent {
    background: #FFFFFF !important;
    border: 1px solid var(--ivory-300) !important;
    border-top: none !important;
    border-radius: 0 0 var(--r-md) var(--r-md) !important;
}

/* ========== PROGRESS ========== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--ink-700), var(--saffron-500)) !important;
    border-radius: var(--r-full) !important;
    box-shadow: 0 2px 8px var(--saffron-glow) !important;
}
.stProgress > div > div {
    background: var(--ivory-300) !important;
    border-radius: var(--r-full) !important;
}

/* ========== DIVIDER ========== */
hr {
    margin: var(--s-lg) 0;
    border: none;
    height: 1px;
    background: linear-gradient(90deg,
        transparent,
        var(--ivory-300) 15%,
        var(--saffron-400) 50%,
        var(--ivory-300) 85%,
        transparent
    );
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--ivory-100); }
::-webkit-scrollbar-thumb {
    background: var(--ivory-400);
    border-radius: var(--r-full);
}
::-webkit-scrollbar-thumb:hover { background: var(--ink-500); }

/* ========== MISC ========== */
::selection {
    background: rgba(232, 137, 12, 0.2);
    color: var(--ink-900);
}
::placeholder { color: var(--text-200) !important; opacity: 1 !important; }

/* ========== RESPONSIVE ========== */
@media (max-width: 768px) {
    .main-header    { font-size: 2.2rem; }
    .section-header { font-size: 1.45rem; }
    .decision-title { font-size: 2.5rem; }
    .stat-number    { font-size: 2.4rem; }
}

/* ========== UTILITIES ========== */
.text-saffron { color: var(--saffron-500) !important; }
.text-ink     { color: var(--ink-800) !important; }
.text-emerald { color: var(--emerald-600) !important; }
.text-rose    { color: var(--rose-600) !important; }
.text-amber   { color: var(--amber-600) !important; }
.text-muted   { color: var(--text-200) !important; }
.text-mono    { font-family: 'Fira Code', monospace !important; }
.text-serif   { font-family: 'Playfair Display', serif !important; }
.shadow-gold  { box-shadow: var(--shadow-gold) !important; }
.shadow-ink   { box-shadow: var(--shadow-ink) !important; }
</style>
"""
# """
