

"""
CIBIL PDF OCR Extractor — Production Grade (v3.1 — Fully Fixed)
================================================================
Extracts ALL fields from TransUnion CIBIL reports including:
  - Personal & employment info
  - Credit score & risk flags
  - Account-level details (limits, balances, product types)
  - Utilization per product type (CC, PL, HL, GL)
  - Enquiry breakdown
  - Derived model features aligned with credit_risk_engine schema

Fixes in v3.1 vs v3.0:
  ✅ FIX-A: pdfplumber import made optional — won't crash if missing
  ✅ FIX-B: OCR fallback (pytesseract + pdf2image) when pdfplumber gets no text
            OR when pdfplumber itself is not installed
  ✅ FIX-C: Detailed traceback ALWAYS returned in 'traceback' key so app.py
            expander can display it
  ✅ FIX-D: 'error' key always contains the real exception message (not 'Unknown error')
  ✅ FIX-E: PDF bytes rewound before pdfplumber to avoid 0-byte reads on
            Streamlit UploadedFile objects (seek(0) call added)
  ✅ FIX-F: extraction_method key returned so UI can show how data was extracted
  Original FIX-1 through FIX-3 retained.
"""

import re
import io
import math
import traceback as _tb
from datetime import datetime
from typing import Optional, Union

# ── Optional imports — degrade gracefully ──────────────────────────────────
_PDFPLUMBER_OK = False
try:
    import pdfplumber
    _PDFPLUMBER_OK = True
except ImportError:
    pass

_OCR_OK = False
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    import shutil as _shutil
    _tess = _shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if _tess:
        pytesseract.pytesseract.tesseract_cmd = _tess
    pytesseract.get_tesseract_version()
    _OCR_OK = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_rs(text: str) -> float:
    """Parse 'Rs. 3,200,000' or 'Rs. 8000,000' into a float."""
    if not text:
        return 0.0
    cleaned = re.sub(r'[Rrs.\s,]', '', text.replace('Rs.', '').replace('Rs', ''))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _parse_pct(text: str) -> float:
    """Parse '12%' → 0.12."""
    m = re.search(r'([\d.]+)\s*%', text)
    if m:
        return round(float(m.group(1)) / 100, 4)
    return 0.0


def _parse_date(text: str) -> Optional[datetime]:
    """Parse dates like '08-Dec-1986' or 'Jul-2016'."""
    for fmt in ('%d-%b-%Y', '%b-%Y', '%d-%m-%Y'):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def _age_from_dob(dob_str: str) -> int:
    """Return age in full years from DOB string."""
    dob = _parse_date(dob_str)
    if dob is None:
        return -1
    today = datetime.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _safe_int(val) -> int:
    try:
        return int(val) if not (isinstance(val, float) and math.isnan(val)) else -99999
    except (TypeError, ValueError):
        return -99999


# ---------------------------------------------------------------------------
# Text extraction layer — tries pdfplumber first, falls back to OCR
# ---------------------------------------------------------------------------

def _extract_text_from_source(pdf_source) -> tuple[str, str]:
    """
    Returns (full_text, method) where method is 'pdfplumber' | 'ocr' | 'failed'.
    FIX-E: Always seek(0) before reading to handle Streamlit UploadedFile objects
           whose read pointer may have advanced from a previous read() call.
    FIX-B: Falls back to pytesseract OCR if pdfplumber yields no text.
    """
    # Normalise to bytes — read once, reuse
    if hasattr(pdf_source, 'read'):
        # FIX-E: seek to start before reading
        if hasattr(pdf_source, 'seek'):
            pdf_source.seek(0)
        pdf_bytes = pdf_source.read()
    else:
        with open(pdf_source, 'rb') as f:
            pdf_bytes = f.read()

    full_text = ""

    # ── Attempt 1: pdfplumber (text-based PDF) ────────────────────────────
    if _PDFPLUMBER_OK:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                full_text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except Exception:
            full_text = ""

    if full_text.strip():
        return full_text, "pdfplumber"

    # ── Attempt 2: pytesseract OCR (scanned / image-only PDF) ────────────
    if _OCR_OK:
        try:
            images = convert_from_bytes(pdf_bytes, dpi=300)
            pages_text = []
            for img in images:
                text = pytesseract.image_to_string(img, lang='eng')
                pages_text.append(text)
            full_text = "\n".join(pages_text)
        except Exception:
            full_text = ""

    if full_text.strip():
        return full_text, "ocr"

    # ── Nothing worked ────────────────────────────────────────────────────
    raise ValueError(
        "Could not extract text from PDF.\n"
        f"  pdfplumber available: {_PDFPLUMBER_OK}\n"
        f"  OCR (tesseract) available: {_OCR_OK}\n"
        "The file may be:\n"
        "  • Encrypted or password-protected\n"
        "  • A scanned image without OCR support installed\n"
        "Solutions:\n"
        "  • Add 'pdfplumber' to requirements.txt\n"
        "  • Add 'pytesseract' + 'pdf2image' to requirements.txt AND 'tesseract-ocr poppler-utils' to packages.txt"
    )


# ---------------------------------------------------------------------------
# Main extractor — public API
# ---------------------------------------------------------------------------

def extract_cibil_from_pdf(pdf_source: Union[str, object]) -> dict:
    """
    Parse a TransUnion CIBIL PDF and return a flat feature dict
    compatible with the credit_risk_engine schema.

    Accepts:
      - a file path string (e.g. "cibil.pdf")
      - a Streamlit UploadedFile object (has .read() method)
      - any file-like object with .read()

    Always returns a dict with 'success' key:
      {'success': True,  'extraction_method': '...', ...fields...}
      {'success': False, 'error': '...', 'traceback': '...'}
    """
    try:
        return _extract_impl(pdf_source)
    except Exception as e:
        # FIX-C: always return full traceback so app.py expander can show it
        # FIX-D: error is the real exception string, never 'Unknown error'
        return {
            'success':   False,
            'error':     f"{type(e).__name__}: {str(e)}",
            'traceback': _tb.format_exc(),
        }


def _extract_impl(pdf_source: Union[str, object]) -> dict:
    """Internal implementation. Raises on error."""

    full_text, extraction_method = _extract_text_from_source(pdf_source)

    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    text_block = " ".join(lines)

    # ------------------------------------------------------------------
    # 1. CREDIT SCORE
    # ------------------------------------------------------------------
    bureau_score = 0
    score_section = re.search(r'CIBIL SCORE(.*?)(?:EMPLOYMENT|ACCOUNT)', text_block)
    if score_section:
        for tok in score_section.group(1).split():
            if re.fullmatch(r'[3-9]\d{2}', tok):
                bureau_score = int(tok)
                break
    if bureau_score == 0:
        m = re.search(r'\b([89]\d{2}|[3-7]\d{2})\b', text_block)
        if m:
            bureau_score = int(m.group(1))

    # ------------------------------------------------------------------
    # 2. PERSONAL INFORMATION
    # ------------------------------------------------------------------
    dob_str = ""
    m = re.search(r'Date of Birth\s+[\w\s]+?(\d{2}-[A-Za-z]{3}-\d{4})', text_block)
    if m:
        dob_str = m.group(1).strip()

    age = _age_from_dob(dob_str) if dob_str else -1

    gender_raw = ""
    m = re.search(r'Gender\s+PAN.*?(\bMale\b|\bFemale\b)', text_block, re.IGNORECASE)
    if m:
        gender_raw = m.group(1).capitalize()
    gender = "M" if gender_raw.upper() == "MALE" else ("F" if gender_raw.upper() == "FEMALE" else "U")

    marital_status = ""
    m = re.search(r'Marital Status.*?(\bMarried\b|\bSingle\b|\bDivorced\b|\bWidowed\b)', text_block, re.IGNORECASE)
    if m:
        marital_status = m.group(1).capitalize()

    # ------------------------------------------------------------------
    # 3. EMPLOYMENT INFORMATION
    # ------------------------------------------------------------------
    employment_type = "Salaried"
    m = re.search(r'Employment Type\s+(Salaried|Self.Employed.*?|Business.*?)\s', text_block, re.IGNORECASE)
    if m:
        employment_type = m.group(1).strip()

    net_monthly_income = 0.0
    m = re.search(
        r'Net Monthly Income\s+With Current Employer\s+Rs\.\s*([\d,]+)\s+(\d+)\s+months',
        text_block, re.IGNORECASE
    )
    if m:
        net_monthly_income = _parse_rs(m.group(1))
    if net_monthly_income == 0:
        m = re.search(r'Rs\.\s*([\d,]+)\s+\d+\s+months', text_block)
        if m:
            net_monthly_income = _parse_rs(m.group(1))

    time_with_curr_empr = 0
    m = re.search(r'(\d+)\s+months', text_block)
    if m:
        time_with_curr_empr = int(m.group(1))

    # ------------------------------------------------------------------
    # 4. ACCOUNT SUMMARY
    # ------------------------------------------------------------------
    total_accounts = active_accounts = closed_accounts = settled_count = 0
    written_off_count = 0
    current_balance_total = 0.0
    overdue_amount = 0.0

    m = re.search(
        r'Total Accounts\s+Active\s+Closed\s+Settled\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
        text_block
    )
    if m:
        total_accounts  = int(m.group(1))
        active_accounts = int(m.group(2))
        closed_accounts = int(m.group(3))
        settled_count   = int(m.group(4))

    m = re.search(
        r'Written Off\s+Current Balance\s+Overdue Amount\s+(\d+)\s+(Rs\.\s*[\d,]+)\s+(Rs\.\s*[\d,]+)',
        text_block
    )
    if m:
        written_off_count     = int(m.group(1))
        current_balance_total = _parse_rs(m.group(2))
        overdue_amount        = _parse_rs(m.group(3))

    # ------------------------------------------------------------------
    # 5. CREDIT UTILISATION
    # ------------------------------------------------------------------
    cc_util_pct   = 0.0
    pl_util_pct   = 0.0
    max_unsec_pct = 0.0
    credit_hungry = 0

    m = re.search(
        r'CC Utilization\s*%\s*PL Utilization\s*%\s*Max Unsecured Exposure\s*%\s*Credit Hungry Flag'
        r'\s*([\d.]+)%\s*([\d.]+)%\s*([\d.]+)%\s*(Yes|No)',
        text_block, re.IGNORECASE
    )
    if m:
        cc_util_pct   = float(m.group(1))
        pl_util_pct   = float(m.group(2))
        max_unsec_pct = float(m.group(3))
        credit_hungry = 0 if m.group(4).lower() == 'no' else 1
    else:
        m2 = re.search(r'CC Utilization.*?([\d.]+)%', text_block)
        if m2: cc_util_pct = float(m2.group(1))
        m2 = re.search(r'PL Utilization.*?([\d.]+)%', text_block)
        if m2: pl_util_pct = float(m2.group(1))
        m2 = re.search(r'Max Unsecured Exposure.*?([\d.]+)%', text_block)
        if m2: max_unsec_pct = float(m2.group(1))
        m2 = re.search(r'Credit Hungry Flag\s+(Yes|No)', text_block, re.IGNORECASE)
        if m2: credit_hungry = 0 if m2.group(1).lower() == 'no' else 1

    # ------------------------------------------------------------------
    # 6. ACCOUNT DETAILS
    # ------------------------------------------------------------------
    PRODUCT_TYPES = {
        'Home Loan':     'HL',
        'Car Loan':      'GL',
        'Auto Loan':     'GL',
        'Gold Loan':     'GL',
        'Personal Loan': 'PL',
        'Credit Card':   'CC',
        'Overdraft':     'CC',
    }

    account_rows = []
    account_section = re.search(
        r'ACCOUNT DETAILS\s+Lender.*?DPD 6M(.*?)(?:ENQUIRY DETAILS|SCORE FACTORS)',
        text_block, re.DOTALL
    )
    if account_section:
        rows_text = account_section.group(1)
        row_pattern = re.compile(
            r'([\w\s&]+?)(Credit Card|Car Loan|Home Loan|Personal Loan|Gold Loan|Auto Loan|Overdraft|Business Loan)'
            r'\s+([A-Z][a-z]{2}-\d{4})'
            r'\s+Rs\.\s*([\d,]+)'
            r'\s+Rs\.\s*([\d,]+)'
            r'\s+(Active|Closed|Settled|Written[- ]Off)'
            r'(\d+)',
            re.IGNORECASE
        )
        for match in row_pattern.finditer(rows_text):
            product_raw = match.group(2).strip()
            account_rows.append({
                'lender':  match.group(1).strip(),
                'product': product_raw,
                'ptype':   PRODUCT_TYPES.get(product_raw, 'OTH'),
                'opened':  match.group(3),
                'limit':   _parse_rs(match.group(4)),
                'balance': _parse_rs(match.group(5)),
                'status':  match.group(6).strip(),
                'dpd':     int(match.group(7)),
            })

    cc_rows = [r for r in account_rows if r['ptype'] == 'CC']
    pl_rows = [r for r in account_rows if r['ptype'] == 'PL']
    hl_rows = [r for r in account_rows if r['ptype'] == 'HL']
    gl_rows = [r for r in account_rows if r['ptype'] == 'GL']

    avg_balance_cc   = sum(r['balance'] for r in cc_rows) / len(cc_rows) if cc_rows else 0.0
    avg_credit_limit = sum(r['limit']   for r in cc_rows) / len(cc_rows) if cc_rows else 0.0

    pl_total_limit   = sum(r['limit']   for r in pl_rows)
    pl_total_balance = sum(r['balance'] for r in pl_rows)
    pl_utilization   = round(pl_total_balance / pl_total_limit, 4) if pl_total_limit > 0 else 0.0

    CC_Flag = 1 if cc_rows else 0
    PL_Flag = 1 if pl_rows else 0
    HL_Flag = 1 if hl_rows else 0
    GL_Flag = 1 if gl_rows else 0

    cc_utilization       = round(cc_util_pct / 100, 4)
    pl_utilization_cibil = round(pl_util_pct / 100, 4)
    pl_utilization_final = pl_utilization_cibil if pl_util_pct > 0 else pl_utilization

    # ------------------------------------------------------------------
    # 7. ENQUIRY DETAILS
    # ------------------------------------------------------------------
    tot_enq = enq_L3m = enq_L6m = enq_L12m = 0
    m = re.search(r'Total Enquiries:\s*(\d+)', text_block)
    if m: tot_enq = int(m.group(1))
    m = re.search(r'Last 3 Months:\s*(\d+)', text_block)
    if m: enq_L3m = int(m.group(1))
    m = re.search(r'Last 6 Months:\s*(\d+)', text_block)
    if m: enq_L6m = int(m.group(1))
    m = re.search(r'Last 12 Months:\s*(\d+)', text_block)
    if m: enq_L12m = int(m.group(1))

    time_since_recent_enq = -99999
    enq_section_m = re.search(r'ENQUIRY DETAILS(.*?)(?:SCORE FACTORS|$)', text_block, re.DOTALL)
    if enq_section_m:
        enq_date_m = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4})', enq_section_m.group(1))
        if enq_date_m:
            enq_date = _parse_date(enq_date_m.group(1))
            if enq_date:
                time_since_recent_enq = (datetime.today() - enq_date).days

    last_prod_enq2  = "OTH"
    first_prod_enq2 = "OTH"
    if enq_section_m:
        prod_matches = re.findall(
            r'\d{2}-[A-Za-z]{3}-\d{4}\s+([\w\s]+?)(?=\s+Rs\.)',
            enq_section_m.group(1), re.IGNORECASE
        )
        if prod_matches:
            for prod_raw, dest in [(prod_matches[0], 'last'), (prod_matches[-1], 'first')]:
                p = prod_raw.strip().lower()
                code = ('PL' if 'personal' in p else
                        'CC' if 'credit card' in p else
                        'HL' if 'home' in p else
                        'GL' if ('car' in p or 'auto' in p) else 'OTH')
                if dest == 'last':  last_prod_enq2  = code
                else:               first_prod_enq2 = code

    # ------------------------------------------------------------------
    # 8. DERIVED INCOME & EMI FEATURES
    # ------------------------------------------------------------------
    amt_income_total = round(net_monthly_income * 12, 2)

    hl_balance = sum(r['balance'] for r in hl_rows)
    hl_emi_est = round(hl_balance / 120, 2) if hl_balance > 0 else 0.0
    gl_balance = sum(r['balance'] for r in gl_rows)
    gl_emi_est = round(gl_balance / 120, 2) if gl_balance > 0 else 0.0
    pl_balance = sum(r['balance'] for r in pl_rows)
    pl_emi_est = round(pl_balance / 24, 2) if pl_balance > 0 else 0.0

    existing_emi       = hl_emi_est + gl_emi_est + pl_emi_est
    amt_annuity        = existing_emi
    total_emi_monthly  = existing_emi
    net_cash_surplus_6m    = net_monthly_income - existing_emi
    avg_monthly_balance_6m = net_cash_surplus_6m

    # ------------------------------------------------------------------
    # 9. DELINQUENCY FEATURES
    # ------------------------------------------------------------------
    max_dpd_6m       = max((r['dpd'] for r in account_rows), default=0)
    dpd_15_count_6m  = sum(1 for r in account_rows if r['dpd'] >= 15)
    dpd_30_count_6m  = sum(1 for r in account_rows if r['dpd'] >= 30)
    dpd_60_count_6m  = sum(1 for r in account_rows if r['dpd'] >= 60)
    dpd_90_count_6m  = sum(1 for r in account_rows if r['dpd'] >= 90)
    dpd_30_count_3m  = 0
    total_dpd_count  = dpd_30_count_6m

    high_dpd_risk         = 1 if max_dpd_6m >= 60 else 0
    recent_deliq_flag     = 1 if dpd_30_count_6m > 0 else 0
    delinq_severity_score = min(dpd_90_count_6m * 30 + dpd_30_count_6m * 10, 100)
    account_quality_score = max(0, 100 - delinq_severity_score - (written_off_count * 20) - (settled_count * 5))

    # ------------------------------------------------------------------
    # 10. BANK STATEMENT PLACEHOLDERS
    # ------------------------------------------------------------------
    salary_txn_count_6m         = 6
    salary_amount_cv            = 0.05
    salary_date_std             = 2
    salary_creditor_consistent  = 1
    salary_missing_months       = 0
    total_payments_6m           = 3
    total_late_15_6m            = 0
    total_late_30_6m            = dpd_30_count_6m
    total_late_60_6m            = 0
    total_late_90_6m            = 0
    max_days_late_6m            = max_dpd_6m
    avg_days_late_6m            = max_dpd_6m / max(1, dpd_30_count_6m) if dpd_30_count_6m > 0 else 0.0
    total_late_30_3m            = 0
    total_late_90_3m            = 0
    total_payments_cc           = 0
    dpd_count_cc                = 0
    avg_balance_pos             = 0.0
    dpd_count_pos               = 0
    total_credit_6m             = 0
    total_debit_6m              = 0
    inward_bounce_count_3m      = 0
    recent_payment_stress       = 0

    # ------------------------------------------------------------------
    # 11. QUALITATIVE FLAGS
    # ------------------------------------------------------------------
    high_util_flag = 1 if cc_utilization > 0.30 else 0

    bureau_risk_flag = (
        "HIGH"   if (bureau_score < 600 or max_dpd_6m >= 60 or written_off_count > 0)
        else "MEDIUM" if (bureau_score < 700 or max_dpd_6m >= 30)
        else "LOW"
    )
    salary_stability_flag = (
        "UNSTABLE"  if salary_amount_cv > 0.20
        else "MODERATE" if salary_amount_cv > 0.12
        else "STABLE"
    )
    cashflow_health = (
        "STRESSED"  if net_cash_surplus_6m < 0
        else "MODERATE" if net_cash_surplus_6m < net_monthly_income * 0.2
        else "HEALTHY"
    )
    liquidity_flag = (
        "LOW"      if avg_monthly_balance_6m < existing_emi
        else "MODERATE" if avg_monthly_balance_6m < existing_emi * 1.5
        else "ADEQUATE"
    )
    payment_discipline_flag = (
        "BAD"  if (dpd_30_count_6m > 2 or written_off_count > 0)
        else "FAIR" if dpd_15_count_6m > 0
        else "GOOD"
    )

    pct_of_active_TLs_ever = round(active_accounts / total_accounts, 2) if total_accounts > 0 else 0

    # ------------------------------------------------------------------
    # 12. ASSEMBLE AND RETURN
    # ------------------------------------------------------------------
    total_limit_all       = sum(r['limit']   for r in account_rows)
    total_balance_all     = sum(r['balance'] for r in account_rows)
    pct_currentBal_all_TL = round(total_balance_all / total_limit_all, 4) if total_limit_all > 0 else 0.0

    return {
        'success':            True,
        'extraction_method':  extraction_method,  # FIX-F

        # ── Income & Annuity ──────────────────────────────────────────
        "AMT_INCOME_TOTAL":          amt_income_total,
        "AMT_ANNUITY":               amt_annuity,

        # ── Salary / Bank Statement Features ─────────────────────────
        "avg_salary_6m":             net_monthly_income,
        "salary_txn_count_6m":       salary_txn_count_6m,
        "salary_amount_cv":          salary_amount_cv,
        "salary_date_std":           salary_date_std,
        "salary_creditor_consistent":salary_creditor_consistent,
        "salary_missing_months":     salary_missing_months,

        # ── DPD / Payment Behaviour ───────────────────────────────────
        "dpd_15_count_6m":           dpd_15_count_6m,
        "dpd_30_count_6m":           dpd_30_count_6m,
        "dpd_60_count_6m":           dpd_60_count_6m,
        "dpd_90_count_6m":           dpd_90_count_6m,
        "max_dpd_6m":                max_dpd_6m,
        "dpd_30_count_3m":           dpd_30_count_3m,
        "total_payments_6m":         total_payments_6m,
        "total_late_15_6m":          total_late_15_6m,
        "total_late_30_6m":          total_late_30_6m,
        "total_late_60_6m":          total_late_60_6m,
        "total_late_90_6m":          total_late_90_6m,
        "max_days_late_6m":          max_days_late_6m,
        "avg_days_late_6m":          avg_days_late_6m,
        "total_late_30_3m":          total_late_30_3m,
        "total_late_90_3m":          total_late_90_3m,

        # ── Credit Card Metrics ───────────────────────────────────────
        "avg_balance_cc":            avg_balance_cc,
        "total_drawings_cc":         0.0,
        "avg_credit_limit":          avg_credit_limit,
        "max_utilization":           cc_utilization,
        "total_payments_cc":         total_payments_cc,
        "dpd_count_cc":              dpd_count_cc,

        # ── POS / Other Balances ──────────────────────────────────────
        "avg_balance_pos":           avg_balance_pos,
        "dpd_count_pos":             dpd_count_pos,

        # ── Portfolio Overview ────────────────────────────────────────
        "total_credit_activity":     total_accounts,
        "total_dpd_count":           total_dpd_count,
        "avg_monthly_balance_6m":    avg_monthly_balance_6m,
        "total_emi_monthly":         total_emi_monthly,
        "net_cash_surplus_6m":       net_cash_surplus_6m,
        "total_credit_6m":           total_credit_6m,
        "total_debit_6m":            total_debit_6m,
        "inward_bounce_count_3m":    inward_bounce_count_3m,
        "recent_payment_stress":     recent_payment_stress,
        "active_loans_count":        active_accounts,

        # ── Bureau Score ──────────────────────────────────────────────
        "bureau_score":              bureau_score,
        "Credit_Score":              bureau_score,

        # ── Demographics ─────────────────────────────────────────────
        "AGE":                       age,
        "GENDER":                    gender,
        "MARITALSTATUS":             marital_status,
        "EDUCATION":                 "GRADUATE",
        "NETMONTHLYINCOME":          net_monthly_income,
        "Time_With_Curr_Empr":       time_with_curr_empr,

        # ── CIBIL Delinquency Features ────────────────────────────────
        "num_times_delinquent":       dpd_30_count_6m,
        "max_delinquency_level":      max_dpd_6m,
        "max_recent_level_of_deliq":  max_dpd_6m,
        "num_deliq_6mts":             dpd_30_count_6m,
        "num_deliq_12mts":            dpd_30_count_6m,
        "num_deliq_6_12mts":          0,
        "max_deliq_6mts":             max_dpd_6m if dpd_30_count_6m > 0 else -99999,
        "max_deliq_12mts":            max_dpd_6m if dpd_30_count_6m > 0 else -99999,
        "num_times_30p_dpd":          dpd_30_count_6m,
        "num_times_60p_dpd":          dpd_60_count_6m,
        "recent_level_of_deliq":      max_dpd_6m,
        "num_std":                    active_accounts,
        "num_std_6mts":               active_accounts,
        "num_std_12mts":              active_accounts,
        "num_sub":                    dpd_30_count_6m,
        "num_sub_6mts":               dpd_30_count_6m,
        "num_sub_12mts":              dpd_30_count_6m,
        "num_dbt":                    0,
        "num_dbt_6mts":               0,
        "num_dbt_12mts":              0,
        "num_lss":                    written_off_count,
        "num_lss_6mts":               0,
        "num_lss_12mts":              0,
        "time_since_recent_payment":  -99999,
        "time_since_first_deliquency":  -99999 if dpd_30_count_6m == 0 else 180,
        "time_since_recent_deliquency": -99999 if dpd_30_count_6m == 0 else 30,

        # ── Enquiry Features ──────────────────────────────────────────
        "tot_enq":                   tot_enq,
        "enq_L3m":                   enq_L3m,
        "enq_L6m":                   enq_L6m,
        "enq_L12m":                  enq_L12m,
        "time_since_recent_enq":     time_since_recent_enq,
        "CC_enq":                    -99999,
        "CC_enq_L6m":                0,
        "CC_enq_L12m":               0,
        "PL_enq":                    -99999,
        "PL_enq_L6m":                0,
        "PL_enq_L12m":               0,

        # ── Portfolio Ratios ──────────────────────────────────────────
        "pct_of_active_TLs_ever":     pct_of_active_TLs_ever,
        "pct_opened_TLs_L6m_of_L12m": 0.3,
        "pct_currentBal_all_TL":      pct_currentBal_all_TL,
        "pct_PL_enq_L6m_of_L12m":    0.0,
        "pct_CC_enq_L6m_of_L12m":    0.0,
        "pct_PL_enq_L6m_of_ever":    0.0,
        "pct_CC_enq_L6m_of_ever":    0.0,

        # ── Utilization ───────────────────────────────────────────────
        "CC_utilization":            cc_utilization,
        "PL_utilization":            pl_utilization_final,
        "CC_Flag":                   CC_Flag,
        "PL_Flag":                   PL_Flag,
        "HL_Flag":                   HL_Flag,
        "GL_Flag":                   GL_Flag,

        # ── Exposure & EMI ────────────────────────────────────────────
        "max_unsec_exposure_inPct":  max_unsec_pct,
        "last_prod_enq2":            last_prod_enq2,
        "first_prod_enq2":           first_prod_enq2,
        "existing_emi":              existing_emi,
        "employment_type":           employment_type,
        "business_vintage_years":    0,
        "credit_utilization_pct":    round(cc_utilization * 100, 1),

        # ── Qualitative Flags ─────────────────────────────────────────
        "salary_stability_flag":     salary_stability_flag,
        "payment_discipline_flag":   payment_discipline_flag,
        "cashflow_health":           cashflow_health,
        "liquidity_flag":            liquidity_flag,
        "bureau_risk_flag":          bureau_risk_flag,
        "written_off_count":         written_off_count,
        "settled_count":             settled_count,
        "high_util_flag":            high_util_flag,
        "credit_hungry":             credit_hungry,
        "delinq_severity_score":     delinq_severity_score,
        "high_dpd_risk":             high_dpd_risk,
        "recent_deliq_flag":         recent_deliq_flag,
        "account_quality_score":     account_quality_score,
        "_surplus_proxy":            net_cash_surplus_6m,

        # ── Extra audit fields ────────────────────────────────────────
        "_current_balance_total":    current_balance_total,
        "_overdue_amount":           overdue_amount,
        "_total_accounts":           total_accounts,
        "_closed_accounts":          closed_accounts,
        "_pl_total_limit":           pl_total_limit,
        "_pl_total_balance":         pl_total_balance,
        "_account_rows":             account_rows,
    }


# ---------------------------------------------------------------------------
# Categorical flag inference (for cases where CIBIL PDF is unavailable)
# ---------------------------------------------------------------------------

def infer_categorical_flags(features: dict) -> dict:
    """
    Given a features dict (possibly with numeric-only values),
    infer / recompute the qualitative string flags.
    """
    bureau_score      = features.get("bureau_score", 0)
    max_dpd_6m        = features.get("max_dpd_6m", 0)
    written_off_count = features.get("written_off_count", 0)
    dpd_30_count_6m   = features.get("dpd_30_count_6m", 0)
    dpd_15_count_6m   = features.get("dpd_15_count_6m", 0)
    salary_cv         = features.get("salary_amount_cv", 0)
    surplus           = features.get("net_cash_surplus_6m", 0)
    income            = features.get("avg_salary_6m", 1)
    emi               = features.get("total_emi_monthly", 0)
    balance           = features.get("avg_monthly_balance_6m", 0)
    cc_util           = features.get("CC_utilization", 0)

    bureau_risk_flag = (
        "HIGH"    if (bureau_score < 600 or max_dpd_6m >= 60 or written_off_count > 0)
        else "MEDIUM" if (bureau_score < 700 or max_dpd_6m >= 30)
        else "LOW"
    )
    salary_stability_flag = (
        "UNSTABLE"  if salary_cv > 0.20
        else "MODERATE" if salary_cv > 0.12
        else "STABLE"
    )
    cashflow_health = (
        "STRESSED"  if surplus < 0
        else "MODERATE" if surplus < income * 0.2
        else "HEALTHY"
    )
    liquidity_flag = (
        "LOW"       if balance < emi
        else "MODERATE" if balance < emi * 1.5
        else "ADEQUATE"
    )
    payment_discipline_flag = (
        "BAD"  if (dpd_30_count_6m > 2 or written_off_count > 0)
        else "FAIR" if dpd_15_count_6m > 0
        else "GOOD"
    )
    high_util_flag = 1 if cc_util > 0.30 else 0

    features.update({
        "bureau_risk_flag":         bureau_risk_flag,
        "salary_stability_flag":    salary_stability_flag,
        "cashflow_health":          cashflow_health,
        "liquidity_flag":           liquidity_flag,
        "payment_discipline_flag":  payment_discipline_flag,
        "high_util_flag":           high_util_flag,
    })
    return features


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pprint
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_cibil.pdf"
    result = extract_cibil_from_pdf(path)
    result.pop("_account_rows", None)
    pprint.pprint(result)
