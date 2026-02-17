# css_style.py
"""
Ultra-Enhanced Sage Green & Yellow Theme CSS for Credit Risk Dashboard
Features: Glassmorphism, Neumorphism, Smooth Animations, Gradient Magic, 3D Effects
"""

CSS = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@300;400;500;600;700;800&display=swap');

/* ==================== CSS VARIABLES ==================== */
:root {
    /* Primary Palette */
    --fern-green: #587042;
    --sage: #A9B494;
    --cosmic-latte: #FAF7E6;
    --jasmine: #F8DE8C;
    --saffron: #F6C531;
    --dark-fern: #486032;
    --light-sage: #D4DBC4;
    
    /* Extended Palette */
    --fern-dark: #3d5230;
    --fern-light: #6d8a57;
    --sage-light: #c4d1b3;
    --sage-dark: #8e9d7f;
    --gold: #FFD700;
    --gold-light: #FFED4E;
    
    /* Neutrals */
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
    
    /* Semantic Colors */
    --success: #10B981;
    --success-light: #D1FAE5;
    --warning: #F59E0B;
    --warning-light: #FEF3C7;
    --danger: #EF4444;
    --danger-light: #FEE2E2;
    --info: #3B82F6;
    --info-light: #DBEAFE;
    
    /* Shadows */
    --shadow-sm: 0 1px 2px 0 rgba(88, 112, 66, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(88, 112, 66, 0.1), 0 2px 4px -1px rgba(88, 112, 66, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(88, 112, 66, 0.15), 0 4px 6px -2px rgba(88, 112, 66, 0.05);
    --shadow-xl: 0 20px 25px -5px rgba(88, 112, 66, 0.2), 0 10px 10px -5px rgba(88, 112, 66, 0.04);
    --shadow-2xl: 0 25px 50px -12px rgba(88, 112, 66, 0.25);
    
    /* Glass Effects */
    --glass-bg: rgba(255, 255, 255, 0.85);
    --glass-border: rgba(88, 112, 66, 0.18);
    
    /* Transitions */
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
    position: relative;
    overflow-x: hidden;
}

/* Animated Background Pattern */
.main::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(circle at 20% 50%, rgba(88, 112, 66, 0.03) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(246, 197, 49, 0.03) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
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
    background: linear-gradient(135deg, var(--fern-green) 0%, var(--fern-light) 50%, var(--sage) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 4px 8px rgba(88, 112, 66, 0.1);
    animation: fadeInDown 0.8s ease-out;
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
    animation: shimmer 2s infinite;
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
    position: relative;
    animation: slideInLeft 0.6s ease-out;
    background: linear-gradient(90deg, rgba(246, 197, 49, 0.05) 0%, transparent 100%);
    border-radius: 0 8px 8px 0;
}

/* ==================== DECISION CARDS - PREMIUM EDITION ==================== */
.decision-card {
    padding: 3rem 2.5rem;
    border-radius: 24px;
    margin: 2.5rem 0;
    color: white;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    box-shadow: var(--shadow-2xl);
    animation: scaleIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    transition: var(--transition-base);
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.decision-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transform: rotate(45deg);
    animation: shine 3s infinite;
}

.decision-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 30px 60px -12px rgba(88, 112, 66, 0.35);
}

.decision-card-approved {
    background: linear-gradient(135deg, 
        rgba(16, 185, 129, 0.95) 0%, 
        rgba(88, 112, 66, 0.95) 50%, 
        rgba(122, 158, 77, 0.95) 100%);
    border-color: rgba(255, 255, 255, 0.3);
}

.decision-card-rejected {
    background: linear-gradient(135deg, 
        rgba(239, 68, 68, 0.95) 0%, 
        rgba(220, 38, 38, 0.95) 50%, 
        rgba(185, 28, 28, 0.95) 100%);
    border-color: rgba(255, 255, 255, 0.3);
}

.decision-card-review {
    background: linear-gradient(135deg, 
        rgba(246, 197, 49, 0.95) 0%, 
        rgba(248, 222, 140, 0.95) 50%, 
        rgba(255, 215, 0, 0.95) 100%);
    border-color: rgba(255, 255, 255, 0.3);
    color: var(--gray-800);
}

.decision-title {
    font-family: 'Poppins', sans-serif;
    font-size: 3.5rem;
    font-weight: 900;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    text-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    animation: bounceIn 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    letter-spacing: -0.02em;
}

.decision-subtitle {
    font-size: 1.25rem;
    margin-top: 1rem;
    opacity: 0.95;
    font-weight: 500;
    text-align: center;
    animation: fadeIn 1s ease-out 0.3s both;
}

/* ==================== STAGE 2 PREMIUM CARDS ==================== */
.stage2-decision-card {
    padding: 3.5rem 3rem;
    border-radius: 28px;
    margin: 3rem 0;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(30px);
    box-shadow: 
        var(--shadow-2xl),
        inset 0 1px 0 rgba(255, 255, 255, 0.3),
        0 0 40px rgba(88, 112, 66, 0.15);
    animation: zoomIn 0.7s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    border: 3px solid;
}

.stage2-decision-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.2) 0%, transparent 60%),
        radial-gradient(circle at 70% 70%, rgba(255, 255, 255, 0.15) 0%, transparent 60%);
    pointer-events: none;
}

.stage2-approved {
    background: linear-gradient(135deg, 
        rgba(16, 185, 129, 0.92) 0%, 
        rgba(5, 150, 105, 0.92) 100%);
    border-color: rgba(167, 243, 208, 0.5);
}

.stage2-review {
    background: linear-gradient(135deg, 
        rgba(245, 158, 11, 0.92) 0%, 
        rgba(217, 119, 6, 0.92) 100%);
    border-color: rgba(253, 230, 138, 0.5);
}

.stage2-rejected {
    background: linear-gradient(135deg, 
        rgba(239, 68, 68, 0.92) 0%, 
        rgba(220, 38, 38, 0.92) 100%);
    border-color: rgba(254, 202, 202, 0.5);
}

.stage2-title {
    font-family: 'Poppins', sans-serif;
    font-size: 4rem;
    font-weight: 900;
    margin-bottom: 1rem;
    text-shadow: 
        0 2px 4px rgba(0, 0, 0, 0.1),
        0 4px 8px rgba(0, 0, 0, 0.1),
        0 8px 16px rgba(0, 0, 0, 0.1);
    animation: bounceIn 0.9s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    color: white;
    text-align: center;
    letter-spacing: -0.03em;
}

.stage2-subtitle {
    font-size: 1.5rem;
    font-weight: 600;
    opacity: 0.98;
    text-align: center;
    animation: fadeIn 1.2s ease-out 0.4s both;
    color: white;
}

/* ==================== TIER BADGES - LUXURY EDITION ==================== */
.tier-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 1rem 2.5rem;
    border-radius: 50px;
    font-size: 1.5rem;
    font-weight: 800;
    font-family: 'Poppins', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    box-shadow: 
        0 10px 25px rgba(0, 0, 0, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
    position: relative;
    overflow: hidden;
    animation: pulse 2s infinite;
    border: 2px solid rgba(255, 255, 255, 0.3);
}

.tier-badge::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transform: rotate(45deg);
    animation: shimmer 3s infinite;
}

.tier-p1 {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: white;
    box-shadow: 
        0 10px 30px rgba(16, 185, 129, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.tier-p2 {
    background: linear-gradient(135deg, #34D399 0%, #10B981 100%);
    color: white;
    box-shadow: 
        0 10px 30px rgba(52, 211, 153, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.tier-p3 {
    background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
    color: white;
    box-shadow: 
        0 10px 30px rgba(245, 158, 11, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.tier-p4 {
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    color: white;
    box-shadow: 
        0 10px 30px rgba(239, 68, 68, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

/* ==================== GLASSMORPHIC INFO CARDS ==================== */
.info-card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px) saturate(180%);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 
        var(--shadow-lg),
        inset 0 1px 0 rgba(255, 255, 255, 0.5);
    border: 1px solid var(--glass-border);
    margin-bottom: 1.5rem;
    transition: var(--transition-base);
    position: relative;
    overflow: hidden;
}

.info-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--fern-green), var(--saffron), var(--fern-green));
    opacity: 0;
    transition: var(--transition-base);
}

.info-card:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 
        var(--shadow-xl),
        inset 0 1px 0 rgba(255, 255, 255, 0.6);
    border-color: var(--fern-green);
}

.info-card:hover::before {
    opacity: 1;
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

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--fern-green), var(--saffron));
    transform: scaleX(0);
    transform-origin: left;
    transition: var(--transition-base);
}

.stat-card:hover {
    transform: translateY(-8px) scale(1.05);
    box-shadow: 
        12px 12px 24px rgba(163, 177, 198, 0.4),
        -12px -12px 24px rgba(255, 255, 255, 0.9),
        inset 0 2px 0 rgba(255, 255, 255, 0.6);
}

.stat-card:hover::before {
    transform: scaleX(1);
}

.stat-number {
    font-family: 'Poppins', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--fern-green) 0%, var(--fern-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
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

/* ==================== STATUS BADGES WITH GLOW ==================== */
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

.status-badge::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.status-badge:hover::before {
    width: 300px;
    height: 300px;
}

.badge-pass {
    background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
    color: var(--success);
    border-color: var(--success);
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
}

.badge-fail {
    background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
    color: var(--danger);
    border-color: var(--danger);
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
}

.badge-warning {
    background: linear-gradient(135deg, var(--cosmic-latte) 0%, var(--jasmine) 100%);
    color: #92400E;
    border-color: var(--warning);
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);
}

/* ==================== PREMIUM DATA ROWS ==================== */
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
    background: linear-gradient(135deg, var(--fern-green) 0%, var(--fern-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ==================== REASON ITEMS WITH ICONS ==================== */
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
    animation: bounce 2s infinite;
}

/* ==================== PREMIUM BUTTONS ==================== */
.stButton > button {
    background: linear-gradient(135deg, var(--fern-green) 0%, var(--fern-light) 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 1rem 2rem;
    font-weight: 700;
    font-size: 1rem;
    transition: var(--transition-base);
    box-shadow: 
        0 6px 20px rgba(88, 112, 66, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    transition: var(--transition-base);
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 
        0 12px 28px rgba(88, 112, 66, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
    background: linear-gradient(135deg, var(--fern-light) 0%, var(--fern-green) 100%);
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:active {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(88, 112, 66, 0.3);
}

/* Download Buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--saffron) 0%, var(--gold) 100%);
    color: var(--gray-900);
    border: none;
    border-radius: 12px;
    padding: 1rem 2rem;
    font-weight: 700;
    transition: var(--transition-base);
    box-shadow: 
        0 6px 20px rgba(246, 197, 49, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.stDownloadButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(246, 197, 49, 0.4);
    background: linear-gradient(135deg, var(--gold) 0%, var(--saffron) 100%);
}

/* ==================== ENHANCED TABS ==================== */
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
    position: relative;
    overflow: hidden;
}

.stTabs [data-baseweb="tab"]::before {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--fern-green), var(--saffron));
    transform: scaleX(0);
    transition: var(--transition-base);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--fern-green) 0%, var(--fern-light) 100%);
    color: white;
    box-shadow: 0 6px 20px rgba(88, 112, 66, 0.3);
}

.stTabs [aria-selected="true"]::before {
    transform: scaleX(1);
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--sage-light);
    color: var(--fern-dark);
}

/* ==================== GLASSMORPHIC SIDEBAR ==================== */
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
    animation: slideInRight 0.5s ease-out;
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
    animation: slideInRight 0.5s ease-out;
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
    animation: slideInRight 0.5s ease-out;
}

/* ==================== STAGE PROGRESS INDICATOR ==================== */
.stage-progress {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2rem;
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    margin: 2.5rem 0;
    border: 2px solid var(--glass-border);
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
}

.stage-progress::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 6px;
    background: linear-gradient(90deg, var(--fern-green), var(--saffron), var(--fern-green));
    animation: progressShimmer 3s infinite;
}

.stage-step {
    text-align: center;
    flex: 1;
    position: relative;
    z-index: 1;
}

.stage-step-complete {
    color: var(--success);
    font-weight: 700;
    animation: bounceIn 0.6s ease-out;
}

.stage-step-active {
    color: var(--fern-green);
    font-weight: 800;
    animation: pulse 2s infinite;
}

.stage-step-pending {
    color: var(--gray-400);
    font-weight: 500;
    opacity: 0.6;
}

/* ==================== CIBIL UPLOAD BOX ==================== */
.cibil-upload-box {
    background: linear-gradient(135deg, 
        rgba(88, 112, 66, 0.05) 0%, 
        rgba(169, 180, 148, 0.05) 100%);
    border: 4px dashed var(--sage);
    border-radius: 24px;
    padding: 4rem 3rem;
    text-align: center;
    margin: 2.5rem 0;
    transition: var(--transition-base);
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.cibil-upload-box::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(246, 197, 49, 0.1) 0%, transparent 70%);
    opacity: 0;
    transition: var(--transition-base);
}

.cibil-upload-box:hover {
    border-color: var(--fern-green);
    background: linear-gradient(135deg, 
        rgba(88, 112, 66, 0.08) 0%, 
        rgba(246, 197, 49, 0.08) 100%);
    transform: scale(1.02);
    box-shadow: var(--shadow-lg);
}

.cibil-upload-box:hover::before {
    opacity: 1;
    animation: rotate 4s linear infinite;
}

/* ==================== ANIMATIONS ==================== */
@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-50px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(50px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes scaleIn {
    from {
        opacity: 0;
        transform: scale(0.9);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

@keyframes zoomIn {
    from {
        opacity: 0;
        transform: scale(0.8);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

@keyframes bounceIn {
    0% {
        opacity: 0;
        transform: scale(0.3);
    }
    50% {
        opacity: 1;
        transform: scale(1.05);
    }
    70% {
        transform: scale(0.9);
    }
    100% {
        transform: scale(1);
    }
}

@keyframes bounce {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-10px);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.8;
        transform: scale(1.05);
    }
}

@keyframes shimmer {
    0% {
        transform: translateX(-100%);
    }
    100% {
        transform: translateX(100%);
    }
}

@keyframes shine {
    0% {
        transform: translateX(-100%) translateY(-100%) rotate(45deg);
    }
    100% {
        transform: translateX(100%) translateY(100%) rotate(45deg);
    }
}

@keyframes progressShimmer {
    0% {
        background-position: -200% 0;
    }
    100% {
        background-position: 200% 0;
    }
}

@keyframes rotate {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

/* ==================== ENHANCED SCROLLBAR ==================== */
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
    transition: var(--transition-base);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, var(--fern-green) 0%, var(--fern-dark) 100%);
}

/* ==================== INPUT FIELD ENHANCEMENTS ==================== */
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
    box-shadow: 
        inset 0 2px 4px rgba(0, 0, 0, 0.2),
        0 0 0 4px rgba(88, 112, 66, 0.2) !important;
    background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%) !important;
}

/* Form Labels */
.stNumberInput label,
.stSelectbox label,
.stTextInput label {
    color: var(--fern-green) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
    display: block !important;
}

/* ==================== METRIC ENHANCEMENTS ==================== */
[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    font-family: 'Poppins', sans-serif !important;
    background: linear-gradient(135deg, var(--fern-green) 0%, var(--fern-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

[data-testid="stMetricLabel"] {
    font-size: 0.875rem !important;
    font-weight: 700 !important;
    color: var(--gray-600) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ==================== EXPANDER STYLING ==================== */
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

/* ==================== FEATURE GRID ==================== */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.feature-card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    border: 2px solid var(--sage-light);
    box-shadow: var(--shadow-md);
    transition: var(--transition-base);
    position: relative;
    overflow: hidden;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--fern-green), var(--saffron));
    transform: scaleX(0);
    transition: var(--transition-base);
}

.feature-card:hover {
    transform: translateY(-8px);
    box-shadow: var(--shadow-xl);
    border-color: var(--fern-green);
}

.feature-card:hover::before {
    transform: scaleX(1);
}

.feature-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--gray-600);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.feature-value {
    font-size: 1.75rem;
    font-weight: 800;
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, var(--fern-green) 0%, var(--fern-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ==================== PROGRESS BAR ==================== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, 
        var(--fern-green) 0%, 
        var(--fern-light) 50%, 
        var(--saffron) 100%) !important;
    box-shadow: 0 2px 8px rgba(88, 112, 66, 0.4) !important;
    border-radius: 10px !important;
}

/* ==================== RADIO & CHECKBOX ==================== */
.stRadio > div,
.stCheckbox > div {
    transition: var(--transition-fast);
}

.stRadio > div > label,
.stCheckbox > label {
    padding: 0.75rem 1rem;
    border-radius: 8px;
    transition: var(--transition-fast);
}

.stRadio > div > label:hover,
.stCheckbox > label:hover {
    background: rgba(88, 112, 66, 0.05);
}

/* ==================== ICON STYLES ==================== */
.icon {
    font-size: 1.75rem;
    margin-right: 0.75rem;
    color: var(--fern-green);
    filter: drop-shadow(0 2px 4px rgba(88, 112, 66, 0.2));
}

/* ==================== MISCELLANEOUS ENHANCEMENTS ==================== */
hr {
    margin: 3rem 0;
    border: none;
    height: 2px;
    background: linear-gradient(90deg, 
        transparent, 
        var(--sage), 
        var(--saffron), 
        var(--sage), 
        transparent);
    opacity: 0.5;
}

/* Table Styling */
.dataframe {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    border: 2px solid var(--sage-light);
}

/* Placeholder Text */
::placeholder {
    color: var(--gray-500) !important;
    opacity: 0.7 !important;
    font-weight: 500 !important;
}

/* Selection Styling */
::selection {
    background: var(--sage-light);
    color: var(--fern-dark);
}

/* ==================== RESPONSIVE DESIGN ==================== */
@media (max-width: 768px) {
    .main-header {
        font-size: 2rem;
    }
    
    .decision-title {
        font-size: 2.5rem;
    }
    
    .stage2-title {
        font-size: 2.75rem;
    }
    
    .stat-number {
        font-size: 2rem;
    }
    
    .feature-grid {
        grid-template-columns: 1fr;
    }
}

/* ==================== ANIMATED ENTRY CLASS ==================== */
.animated-entry {
    animation: fadeInUp 0.6s ease-out;
}

</style>
"""
