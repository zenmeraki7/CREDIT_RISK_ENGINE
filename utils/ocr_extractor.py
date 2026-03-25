
# r"""
# ocr_extractor.py  —  CIBIL PDF Extraction Engine  v4.0
# =======================================================
# Production-grade OCR extraction from CIBIL / Bureau PDF reports.
 
# Root causes fixed vs v3.0
# --------------------------
# BUG 1 — Score in coloured box missed
#   Tesseract PSM 6 (single block) treats the score box as a separate
#   image region and never reads the number inside it.
#   Fix: multi-pass OCR — PSM 6 + PSM 11 (sparse text). PSM 11 picks up
#   isolated numbers in coloured boxes. Merge both texts before parsing.
 
# BUG 2 — DPD matched from balance, not from DPD column
#   Line: "Credit Card ICICI 45,000 030"
#   Old regex r'\b(0\d0|\d{3})\b' matches "000" from "45,000" first.
#   re.search returns the FIRST match, so DPD was always read as 000.
#   Fix: use negative lookbehind/ahead (?<![,\d])(\d{3})(?![,\d]) to
#   match only standalone 3-digit codes, then take the LAST one per line
#   (DPD is always the rightmost column).
 
# BUG 3 — OCR noise in numbers: "0)" → should be "0"
#   Fix: strip all non-digit characters from DPD values before parsing.
 
# BUG 4 — Words merged without spaces: "PersonalLoan", "CREDITINFORMATIONREPORT"
#   Fix: camelCase split + known-header normalization before section detection.
 
# BUG 5 — Income in lakhs not handled: "Rs. 7.80 L p.m."
#   Fix: lakhs/annual income patterns with divisor conversion.
 
# BUG 6 — Co-applicant income bleeding into primary
#   Fix: isolate PRIMARY APPLICANT section before income extraction.
 
# BUG 7 — DPD text codes (NIL, SMA, SUB, DBT, LSS) not handled
#   Fix: _DPD_TEXT_MAP converts text codes to integer days.
 
# BUG 8 — Score with spaces between digits: "7 4 2"
#   Fix: collapse spaced digits before parsing.
 
# Retained from v3.0
# ------------------
# - Multi-pass deskew + adaptive threshold + CLAHE contrast
# - Low-confidence pages re-run at 450 DPI
# - Gender defaults to 'U' (Unknown) — bias-neutral
# - recent_deliq_flag from actual DPD counts (FIX S-2)
# - surplus_for_return always initialised (FIX L2)
# """
 
# import re
# from datetime import datetime
# from typing import Optional
 
# try:
#     import cv2
#     import numpy as np
#     from pdf2image import convert_from_bytes
#     import pytesseract
#     OCR_AVAILABLE = True
#     OCR_ERROR_MSG = None
# except ImportError as _e:
#     OCR_AVAILABLE = False
#     OCR_ERROR_MSG = str(_e)
 
 
# # ---------------------------------------------------------------------------
# # HELPERS
# # ---------------------------------------------------------------------------
 
# def _re_int(pattern, text, default, lo=None, hi=None):
#     m = re.search(pattern, text, re.IGNORECASE)
#     if m:
#         try:
#             v = int(str(m.group(1)).replace(',', '').replace(' ', ''))
#             if lo is not None and v < lo: return default
#             if hi is not None and v > hi: return default
#             return v
#         except Exception:
#             pass
#     return default
 
 
# def _re_float(pattern, text, default, lo=None, hi=None):
#     m = re.search(pattern, text, re.IGNORECASE)
#     if m:
#         try:
#             v = float(str(m.group(1)).replace(',', '').replace(' ', ''))
#             if lo is not None and v < lo: return default
#             if hi is not None and v > hi: return default
#             return v
#         except Exception:
#             pass
#     return default
 
 
# def _fix_spaced_digits(text):
#     """Collapse OCR-spaced score digits: '7 4 2' → '742'."""
#     return re.sub(r'\b(\d)\s(\d)\s(\d)\b', r'\1\2\3', text)
 
 
# def _split_merged_words(text):
#     """
#     Fix common OCR word-merge issues:
#       'PersonalLoan' → 'Personal Loan'
#       'CreditCard'   → 'Credit Card'
#     """
#     # camelCase split
#     text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
#     # Known header normalisation
#     replacements = {
#         'CREDITINFORMATIONREPORT': 'CREDIT INFORMATION REPORT',
#         'ACCOUNTDETAILS': 'ACCOUNT DETAILS',
#         'ENQUIRYDETAILS': 'ENQUIRY DETAILS',
#         'PERSONALINFORMATION': 'PERSONAL INFORMATION',
#         'ENQUIRIESINFORMATION': 'ENQUIRIES INFORMATION',
#     }
#     for merged, spaced in replacements.items():
#         text = re.sub(merged, spaced, text, flags=re.IGNORECASE)
#     return text
 
 
# def _clean_ocr_noise(text):
#     """Remove common OCR symbol injections near numbers."""
#     # Remove stray ) ] } near digits
#     text = re.sub(r'(\d)[)\]}\|]', r'\1', text)
#     text = re.sub(r'[)\]}\|](\d)', r'\1', text)
#     # § near letters
#     text = re.sub(r'§', '', text)
#     return text
 
 
# def _infer_surplus_from_cibil(score, dpd_60, dpd_30, income, dpd_90=0):
#     """
#     Estimate net cash surplus from bureau signals.
 
#     FIX: Added dpd_90 parameter. Previous code only checked dpd_60 (60-89 day
#     bucket). PDFs where all late accounts have DPD >=90 (e.g. 090, 120) produce
#     dpd_60=0 and dpd_30=0, causing the function to return base=+15k for a
#     score-300 borrower with 10 written-off accounts → HEALTHY (wrong).
 
#     Also fixed: score < 650 alone now gives negative surplus (it didn't before).
#     """
#     base = income * 0.30
#     if score >= 750 and dpd_90 == 0 and dpd_60 == 0 and dpd_30 == 0:
#         return base * 1.5
#     if score >= 700 and dpd_90 == 0:
#         return base * 1.1
#     if dpd_90 >= 1:           # severe delinquency (90+ days) → negative
#         return base * -0.5
#     if dpd_60 >= 1:           # moderate delinquency (60-89 days)
#         return base * 0.2
#     if score < 650:           # poor bureau score alone → negative
#         return base * -0.2
#     if dpd_30 >= 2:
#         return base * 0.5
#     return base
 
 
# # DPD text code → days
# _DPD_TEXT_MAP = {
#     'NIL': 0, 'NA': 0, 'N/A': 0, 'XXX': 0, '-': 0, 'NONE': 0,
#     'STD': 0, 'SMA': 30, 'SUB': 90, 'DBT': 180,
#     'LSS': 365, 'NPA': 90, 'WO': 365,
# }
 
 
# def _parse_dpd_value(raw):
#     """
#     Convert any DPD representation to integer days.
#     Handles: text codes (NIL/STD/SMA/SUB/DBT/LSS), numeric (000/030/090),
#     and OCR-noisy strings ('0)' → 0).
#     """
#     s = str(raw).strip().upper()
#     if s in _DPD_TEXT_MAP:
#         return _DPD_TEXT_MAP[s]
#     # Strip non-digit characters (OCR noise like "0)")
#     clean = re.sub(r'[^0-9]', '', s)
#     try:
#         return int(clean) if clean else 0
#     except Exception:
#         return 0
 
 
# # ---------------------------------------------------------------------------
# # OCR PRE-PROCESSING
# # ---------------------------------------------------------------------------
 
# def _preprocess_page(pil_image, dpi=300):
#     """PIL → cleaned binary array. CLAHE → deskew → adaptive threshold → close."""
#     img  = np.array(pil_image)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
 
#     # CLAHE contrast enhancement
#     try:
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         gray  = clahe.apply(gray)
#     except Exception:
#         pass
 
#     # Deskew ±20°
#     try:
#         coords = np.column_stack(np.where(gray < 200))
#         if len(coords) > 100:
#             angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
#             if angle < -45: angle = 90 + angle
#             if 0.3 < abs(angle) < 20:
#                 (h, w) = gray.shape
#                 M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
#                 gray = cv2.warpAffine(gray, M, (w, h),
#                                       flags=cv2.INTER_CUBIC,
#                                       borderMode=cv2.BORDER_REPLICATE)
#     except Exception:
#         pass
 
#     binary = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY, blockSize=31, C=10)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
#     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     return binary
 
 
# def _ocr_page_multipass(pil_image, dpi=300):
#     """
#     Multi-pass OCR: PSM 6 (uniform block) + PSM 11 (sparse text).
#     PSM 11 catches isolated numbers in coloured boxes that PSM 6 misses.
#     Returns merged text + average confidence.
#     """
#     binary = _preprocess_page(pil_image, dpi)
 
#     # PSM 6 — standard full-page layout
#     cfg6  = '--oem 3 --psm 6 -l eng'
#     data6 = pytesseract.image_to_data(binary, config=cfg6,
#                                        output_type=pytesseract.Output.DICT)
#     text6 = pytesseract.image_to_string(binary, config=cfg6)
#     confs = [c for c in data6['conf'] if c != -1]
#     avg_conf = (sum(confs) / len(confs)) if confs else 0.0
 
#     # PSM 11 — sparse text (catches isolated numbers in boxes/headers)
#     cfg11  = '--oem 3 --psm 11 -l eng'
#     text11 = pytesseract.image_to_string(binary, config=cfg11)
 
#     # Merge: use PSM 6 as base, append PSM 11 lines not already present
#     lines6  = set(l.strip() for l in text6.splitlines() if l.strip())
#     extra   = [l for l in text11.splitlines()
#                if l.strip() and l.strip() not in lines6]
#     merged  = text6 + '\n' + '\n'.join(extra)
 
#     return merged, avg_conf
 
 
# def _ocr_pdf(pdf_bytes, low_conf_threshold=60.0):
#     """PDF bytes → full merged text. Low-confidence pages re-run at 450 DPI."""
#     full_text = ""
#     images_300 = convert_from_bytes(pdf_bytes, dpi=300)
#     for idx, page_img in enumerate(images_300):
#         text, conf = _ocr_page_multipass(page_img, dpi=300)
#         if conf < low_conf_threshold:
#             try:
#                 hi = convert_from_bytes(pdf_bytes, dpi=450,
#                                         first_page=idx+1, last_page=idx+1)
#                 text_hq, conf_hq = _ocr_page_multipass(hi[0], dpi=450)
#                 if conf_hq > conf:
#                     text = text_hq
#             except Exception:
#                 pass
#         full_text += text + "\n"
 
#     # Post-process: clean noise + split merged words
#     full_text = _clean_ocr_noise(full_text)
#     full_text = _split_merged_words(full_text)
#     full_text = _fix_spaced_digits(full_text)
#     return full_text
 
 
# # ---------------------------------------------------------------------------
# # CREDIT SCORE EXTRACTION
# # ---------------------------------------------------------------------------
 
# def _extract_credit_score(txt):
#     """
#     Extract bureau credit score (300-900).
#     Strategy:
#       1. Isolate PRIMARY APPLICANT section
#       2. Try explicit label patterns (most reliable)
#       3. Fallback: scan standalone 3-digit numbers, pick closest to 'score' keyword
#       4. NH/NTC → 650
 
#     FIX: Pattern 5 (generic score/rating + 3-digit) was matching
#     "Score range 300-549" and returning 300. Fixed by:
#       a) Excluding "score range NNN" from matches (it's a label not a value)
#       b) In fallback, picking the candidate CLOSEST to a score keyword
#          rather than the first one found.
#     """
#     # Isolate primary section if multi-applicant
#     primary = txt
#     m = re.search(
#         r'primary\s+applicant(.*?)(?:co[-\s]?applicant|guarantor|$)',
#         txt, re.IGNORECASE | re.DOTALL)
#     if m:
#         primary = m.group(1)
 
#     # Pre-process: blank out "score range NNN-NNN" so it can't be matched
#     # as an actual score value (it's a description of the score range, not the score)
#     def _mask_range(t):
#         return re.sub(
#             r'score\s+range\s+\d{3}\s*[-–]\s*\d{3}[^.]*', '', t, flags=re.IGNORECASE)
 
#     patterns = [
#         # Explicit label with colon/dash/paren (most reliable)
#         r'(?:cibil|experian|equifax|highmark|crif|bureau)\s*'
#         r'(?:trans\s*union\s*)?score\s*[:\-\(]?\s*(\d{3})',
#         # "Credit Score: 490"
#         r'(?:credit\s+)?score\s*[:\-]\s*(\d{3})',
#         # Score followed immediately by rating band on same or next line
#         r'\b(8[0-9]{2}|7[0-9]{2}|6[0-9]{2}|[3-5][0-9]{2})\s*'
#         r'(?:EXCELLENT|VERY\s*GOOD|GOOD|FAIR|SUBPRIME|POOR)\b',
#         r'\b(\d{3})\s+out\s+of\s+900\b',
#         r'(?:your\s+score\s+is|score\s+is)\s+(\d{3})',
#         # FIX: negative lookahead to exclude "score range" matches
#         r'(?:score|rating)\s*(?!range)[^\n\r]{0,30}?(\d{3})',
#     ]
#     for section in (primary, txt):
#         masked = _mask_range(section)
#         for pat in patterns:
#             m2 = re.search(pat, masked, re.IGNORECASE)
#             if m2:
#                 v = int(m2.group(1))
#                 if 300 <= v <= 900:
#                     return v
 
#     # Fallback: find all standalone 3-digit numbers in valid score range,
#     # then pick the one NEAREST to a score-related keyword.
#     # This prevents "Score range 300-549" from returning 300.
#     masked_full = _mask_range(txt)
#     candidates_pos = [(int(x), m.start()) for m in re.finditer(
#         r'(?<![,\d])(\d{3})(?![,\d])', masked_full)
#         for x in [m.group(1)] if 300 <= int(x) <= 900]
 
#     if candidates_pos:
#         # Find positions of score-related keywords
#         kw_positions = [m.start() for m in re.finditer(
#             r'(?:cibil|bureau|credit)\s+score', masked_full, re.IGNORECASE)]
#         if kw_positions:
#             # Pick candidate closest to nearest keyword
#             best = min(candidates_pos,
#                        key=lambda cp: min(abs(cp[1] - kp) for kp in kw_positions))
#             return best[0]
#         # No keyword found — return first candidate (but after masking range text)
#         return candidates_pos[0][0]
 
#     # NH / NTC = No History or New To Credit
#     if re.search(r'\b(?:NH|NTC|no\s+history|new\s+to\s+credit)\b',
#                  txt, re.IGNORECASE):
#         return 650
 
#     return 720  # safe fallback
 
 
# # ---------------------------------------------------------------------------
# # INCOME EXTRACTION
# # ---------------------------------------------------------------------------
 
# def _extract_income(txt):
#     """
#     Extract net monthly income in rupees.
#     Handles: rupees, lakhs (L), annual (p.a.), spaces in numbers.
#     """
#     # Filter co-applicant sections
#     section = txt
#     m = re.search(
#         r'(?:primary\s+applicant|applicant\s+details?)(.*?)'
#         r'(?:co[-\s]?applicant|guarantor|$)',
#         txt, re.IGNORECASE | re.DOTALL)
#     if m:
#         section = m.group(1)
 
#     # Clean spaces in numbers: "65 000" → "65000"
#     def _clean(t):
#         return re.sub(r'(\d)\s(\d{3})\b', r'\1\2', t)
 
#     # (regex, transform_fn)
#     # transform_fn takes the matched string and returns monthly rupees
#     def _rupees(s):      return float(s.replace(',', ''))
#     def _annual(s):      return float(s.replace(',', '')) / 12
#     def _lakhs_pm(s):    return float(s.replace(',', '')) * 100_000        # "7.80 L p.m."
#     def _lakhs_pa(s):    return float(s.replace(',', '')) * 100_000 / 12   # "7.80 L p.a."
 
#     patterns = [
#         # Direct monthly rupee patterns
#         (r'net\s+monthly\s+income[\s:\-₹Rs\.]*([0-9,]+)',               _rupees),
#         (r'monthly\s+(?:take.?home|salary|income)[\s:\-₹Rs\.]*([0-9,]+)', _rupees),
#         (r'(?:salary|income)\s+per\s+month[\s:\-₹Rs\.]*([0-9,]+)',      _rupees),
#         (r'take\s+home[\s:\-₹Rs\.]*([0-9,]+)',                          _rupees),
#         (r'₹\s*([0-9,]+)\s*/?\s*(?:per\s+month|p\.?m\.?|monthly)',     _rupees),
#         (r'([0-9,]+)\s*(?:per\s+month|p\.?m\.?)',                       _rupees),
#         # FIX BUG 2: CIBIL PDFs split 'Net Monthly Income' and 'Rs. 28,000'
#         # across table columns/rows. Need multiline [\s\S] matching.
#         (r'net\s+monthly[\s\S]{0,50}?Rs\.?\s*([0-9,]+)',                _rupees),
#         (r'(?:income|salary|earning)[^\n]{0,100}\nRs\.?\s*([0-9,]+)',   _rupees),
#         (r'Rs\.?\s*([0-9,]+)\s*\n(?:income|salary|earning)',            _rupees),
#         # Lakhs per month: "7.80 L p.m." → ₹7,80,000/month
#         (r'([0-9]+\.?[0-9]*)\s*(?:L|lakh|lakhs?)\s*'
#          r'(?:p\.?m\.?|per\s+month|monthly)',                            _lakhs_pm),
#         # Annual rupees → monthly
#         (r'(?:annual|yearly)\s+income[\s:\-₹Rs\.]*([0-9,]+)',           _annual),
#         (r'([0-9,]+)\s*p\.?a\.?',                                       _annual),
#         # Lakhs per annum → monthly
#         (r'([0-9]+\.?[0-9]*)\s*(?:L|lakh|lakhs?)\s*'
#          r'(?:p\.?a\.?|per\s+annum|annually)',                           _lakhs_pa),
#         # INCOME DETAILS section
#         (r'(?:total|gross)\s+income[\s:\-₹Rs\.]*([0-9,]+)',             _rupees),
#     ]
 
#     for src in (_clean(section), _clean(txt)):
#         for pat, transform in patterns:
#             m2 = re.search(pat, src, re.IGNORECASE)
#             if m2:
#                 try:
#                     monthly = transform(m2.group(1))
#                     if 5_000 < monthly < 5_000_000:
#                         return int(monthly)
#                 except Exception:
#                     pass
#     return 50_000
 
 
# # ---------------------------------------------------------------------------
# # ENQUIRY PARSER
# # ---------------------------------------------------------------------------
 
# def _parse_enquiries(txt):
#     enq_section = ""
#     m = re.search(
#         r'enquir(?:y|ies)\s+details?(.*?)(?:account\s+summary|$)',
#         txt, re.IGNORECASE | re.DOTALL)
#     if m:
#         enq_section = m.group(1)
 
#     raw_dates = re.findall(r'\b(\d{2}-[A-Za-z]{3}-\d{4})\b', enq_section)
#     today = datetime.now().date()
#     parsed_dates = []
#     for ds in raw_dates:
#         for fmt in ('%d-%b-%Y', '%d-%B-%Y'):
#             try:
#                 parsed_dates.append(datetime.strptime(ds, fmt).date())
#                 break
#             except ValueError:
#                 continue
 
#     def _win(days):
#         return sum(1 for d in parsed_dates if (today - d).days <= days)
 
#     # Narrative pattern: "applied for 5 loans in last 3 months"
#     narr3 = _re_int(
#         r'(?:applied|enquired|enquiries?)\s+(?:for\s+)?(\d+)\s+'
#         r'(?:loans?|cards?|products?)\s+in\s+(?:last\s+)?3\s+months?',
#         txt, 0)
 
#     enq_L3m  = max(_win(90),  narr3, _re_int(r'enquir(?:y|ies)\s*\(?3\s*m[^\d]{0,5}\)?[\s:\-]+(\d+)', txt, 0))
#     enq_L6m  = max(_win(180),        _re_int(r'enquir(?:y|ies)\s*\(?6\s*m[^\d]{0,5}\)?[\s:\-]+(\d+)', txt, 0))
#     enq_L12m = max(_win(365),        _re_int(r'enquir(?:y|ies)\s*\(?12\s*m[^\d]{0,5}\)?[\s:\-]+(\d+)', txt, 0))
#     tot_enq  = max(len(parsed_dates), _re_int(r'total\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, 0))
 
#     time_since_recent_enq = -99999
#     if parsed_dates:
#         time_since_recent_enq = (today - max(parsed_dates)).days
 
#     CC_enq      = _re_int(r'credit\s+card\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, -99999)
#     PL_enq      = _re_int(r'personal\s+loan\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, -99999)
#     CC_enq_L6m  = 0; CC_enq_L12m = 0
#     PL_enq_L6m  = 0; PL_enq_L12m = 0
 
#     last_prod = first_prod = 'others'
#     for pat, label in {
#         r'personal\s+loan': 'PL', r'credit\s+card': 'CC',
#         r'home\s+loan|housing': 'HL', r'auto\s+loan|car\s+loan': 'AL',
#         r'gold\s+loan': 'GL', r'business\s+loan': 'BL',
#     }.items():
#         if re.search(pat, enq_section or txt, re.IGNORECASE):
#             last_prod = first_prod = label
#             break
 
#     return dict(
#         tot_enq=tot_enq, enq_L3m=enq_L3m, enq_L6m=enq_L6m, enq_L12m=enq_L12m,
#         time_since_recent_enq=time_since_recent_enq,
#         CC_enq=CC_enq, CC_enq_L6m=CC_enq_L6m, CC_enq_L12m=CC_enq_L12m,
#         PL_enq=PL_enq, PL_enq_L6m=PL_enq_L6m, PL_enq_L12m=PL_enq_L12m,
#         last_prod_enq2=last_prod, first_prod_enq2=first_prod,
#     )
 
 
# # ---------------------------------------------------------------------------
# # ACCOUNT / DPD TABLE PARSER
# # ---------------------------------------------------------------------------
 
# def _parse_accounts(txt):
#     """
#     Parse account table → DPD counts.
 
#     Handles TWO layouts:
#       A) Scanned/OCR PDF — all cells on ONE line per account:
#          "Muthoot Finance  Personal Loan  Feb-2023  Rs.100,000  Active  090"
#          → parse line-by-line, take last 3-digit code as DPD
 
#       B) Digital PDF (pdfminer) — each cell on its OWN line:
#          "Muthoot Finance\\nPersonal Loan\\nFeb-2023\\nRs.100,000\\nActive\\n090"
#          → collect lenders, statuses, DPD codes in order, then zip them
 
#     FIX BUG 3 (this session): Digital PDFs caused mismatched status/DPD because
#     "Active" and "090" were parsed as separate 1-cell rows with dpd=0 and
#     status='active' respectively — producing 3 rows per account instead of 1.
#     """
#     # Isolate account section
#     idx_s = -1
#     idx_e = len(txt)
#     for m in re.finditer(r'ACCOUNT\s+DETAILS?|LOAN\s+DETAILS?|CREDIT\s+FACILITIES', txt, re.IGNORECASE):
#         idx_s = m.end(); break
#     for m in re.finditer(r'ENQUIRY\s+DETAILS?|SCORE\s+FACTORS', txt, re.IGNORECASE):
#         if m.start() > idx_s:
#             idx_e = m.start(); break
#     if idx_s < 0:
#         section = txt
#     else:
#         section = txt[idx_s:idx_e]
 
#     lines = [l.strip() for l in section.split('\n') if l.strip()]
 
#     # ── STRATEGY DETECTION ──────────────────────────────────────────────────
#     # Detect digital PDF layout: are lender names and DPD codes on separate lines?
#     lender_lines_all = [l for l in lines if re.search(
#         r'\bBank\b|\bFinance\b|\bCapital\b|\bNBFC\b|\bLtd\b|\bSBI\b|\bBajaj\b|\bTata\b|\bMuthoot\b',
#         l, re.IGNORECASE)]
#     dpd_only_lines = [l for l in lines if re.fullmatch(r'\d{3}', l)]
#     status_only_lines = [l for l in lines if re.fullmatch(
#         r'Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss|Sub[-\s]?Standard|Standard',
#         l, re.IGNORECASE)]
 
#     # Use digital strategy if we have lenders + standalone DPD codes
#     use_digital = (len(lender_lines_all) >= 1 and
#                    len(dpd_only_lines) >= 1 and
#                    len(dpd_only_lines) >= len(lender_lines_all) - 1)
 
#     accounts = []
 
#     if use_digital:
#         # ── DIGITAL PDF: zip lenders, statuses, DPD codes in document order ──
#         # All three lists appear in the same row-order in the PDF column layout
#         n = min(len(lender_lines_all), len(dpd_only_lines))
#         for i in range(n):
#             dpd_val = _parse_dpd_value(dpd_only_lines[i])
#             if i < len(status_only_lines):
#                 raw_s = status_only_lines[i].lower().replace(' ', '_').replace('-', '_')
#             else:
#                 raw_s = 'active'
#             # Detect product from lender line
#             prod_label = None
#             for pat, label in {
#                 r'credit\s+card': 'CC', r'personal\s+loan': 'PL',
#                 r'home\s+loan|housing': 'HL', r'auto\s+loan|car\s+loan': 'AL',
#                 r'gold\s+loan': 'GL', r'business\s+loan': 'BL',
#             }.items():
#                 if re.search(pat, lender_lines_all[i], re.IGNORECASE):
#                     prod_label = label; break
#             accounts.append({'dpd': dpd_val, 'status': raw_s,
#                              'product': prod_label or 'others'})
 
#         # If we found more statuses than DPD codes (some accounts have text DPD)
#         # handle text-code statuses (NPA, Written-Off) as both status and DPD
#         for i in range(n, len(lender_lines_all)):
#             # No DPD code found — check if status implies DPD
#             raw_s = status_only_lines[i].lower().replace(' ', '_').replace('-', '_') if i < len(status_only_lines) else 'active'
#             dpd_val = 0
#             if 'npa' in raw_s or 'written' in raw_s or 'loss' in raw_s:
#                 dpd_val = 90  # conservative default for NPA/Written-Off
#             accounts.append({'dpd': dpd_val, 'status': raw_s, 'product': 'others'})
 
#         # Also handle text-DPD codes that appeared inline (e.g. SMA, SUB at end)
#         text_dpd_lines = [l for l in lines if re.fullmatch(
#             r'NIL|NA|N/A|XXX|STD|SMA|SUB|DBT|LSS|NPA|WO', l.strip(), re.IGNORECASE)]
#         # (already handled via status_only_lines for NPA; others are rare in digital)
 
#     else:
#         # ── SCANNED/OCR PDF: original line-by-line parser ────────────────────
#         seen_keys = set()
#         for line in section.split('\n'):
#             if not line.strip():
#                 continue
#             stat_m = re.search(
#                 r'\b(Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss|'
#                 r'Sub[-\s]?Standard|Standard|Special\s+Mention|SMA|SUB|DBT|LSS)\b',
#                 line, re.IGNORECASE)
#             standalone_codes = re.findall(r'(?<![,\d])(\d{3})(?![,\d])', line)
#             text_dpd_m = re.search(
#                 r'\b(NIL|NA|N/A|XXX|STD|SMA|SUB|DBT|LSS|NPA|WO)\s*$',
#                 line.strip(), re.IGNORECASE)
#             prod_label = None
#             for pat, label in {
#                 r'credit\s+card|CC': 'CC', r'personal\s+loan|PL': 'PL',
#                 r'home\s+loan|HL|housing': 'HL', r'auto\s+loan|car\s+loan|AL': 'AL',
#                 r'gold\s+loan|GL': 'GL', r'business\s+loan|BL': 'BL',
#             }.items():
#                 if re.search(pat, line, re.IGNORECASE):
#                     prod_label = label; break
#             is_account_row = (
#                 stat_m or standalone_codes or text_dpd_m or
#                 re.search(r'\bINR\b|\bBank\b|\bFinance\b|\bCapital\b|\bNBFC\b|\bLtd\b',
#                           line, re.IGNORECASE))
#             if not is_account_row:
#                 continue
#             if text_dpd_m:
#                 dpd_val = _parse_dpd_value(text_dpd_m.group(1))
#             elif standalone_codes:
#                 dpd_val = _parse_dpd_value(standalone_codes[-1])
#             else:
#                 dpd_val = 0
#             status = (stat_m.group(1).lower().replace(' ', '_').replace('-', '_')
#                       if stat_m else 'active')
#             dedup_key = (prod_label, dpd_val, status)
#             if dedup_key in seen_keys and prod_label == 'CC':
#                 continue
#             seen_keys.add(dedup_key)
#             accounts.append({'dpd': dpd_val, 'status': status,
#                              'product': prod_label or 'others'})
 
#     # ── AGGREGATE ────────────────────────────────────────────────────────────
#     dpd_90 = dpd_60 = dpd_30 = 0
#     written_off = settled = active = sub_std = 0
 
#     if accounts:
#         for acc in accounts:
#             d, s = acc['dpd'], acc['status']
#             if d >= 90:   dpd_90 += 1
#             elif d >= 60: dpd_60 += 1
#             elif d >= 30: dpd_30 += 1
#             if any(x in s for x in ('written', 'npa', 'loss', 'lss', 'wo')):
#                 written_off += 1
#             elif 'settled' in s:
#                 settled += 1
#             elif any(x in s for x in ('active', 'standard', 'std')):
#                 active += 1
#             if d >= 30 or any(x in s for x in ('sub', 'doubtful', 'dbt', 'npa', 'sma')):
#                 sub_std += 1
#     else:
#         # Fallback keyword scan
#         written_off = len(re.findall(r'\bwritten[-\s]?off\b|\bNPA\b', txt, re.IGNORECASE))
#         settled     = len(re.findall(r'\bsettled\b', txt, re.IGNORECASE))
#         dpd_90      = len(re.findall(r'\b090\b|\b120\b|\b150\b|\b180\b|90\+?\s*dpd', txt, re.IGNORECASE))
#         dpd_60      = len(re.findall(r'\b060\b|60\+?\s*dpd', txt, re.IGNORECASE))
#         dpd_30      = len(re.findall(r'\b030\b|30\+?\s*dpd', txt, re.IGNORECASE))
#         active      = min(len(re.findall(r'\bactive\b', txt, re.IGNORECASE)), 15)
 
#     total_accounts = max(len(accounts), active + settled + written_off, 1)
#     return dict(
#         accounts=accounts,
#         dpd_90_count=dpd_90, dpd_60_count=dpd_60, dpd_30_count=dpd_30,
#         written_off_count=written_off, settled_count=settled,
#         active_count=active, sub_std=sub_std,
#         total_accounts=total_accounts,
#         pct_active=active / total_accounts,
#     )
 
#     # Aggregate
#     dpd_90 = dpd_60 = dpd_30 = 0
#     written_off = settled = active = sub_std = 0
 
#     if accounts:
#         for acc in accounts:
#             d, s = acc['dpd'], acc['status']
#             if d >= 90:   dpd_90 += 1
#             elif d >= 60: dpd_60 += 1
#             elif d >= 30: dpd_30 += 1
#             if any(x in s for x in ('written', 'npa', 'loss', 'lss', 'wo')):
#                 written_off += 1
#             elif 'settled' in s:
#                 settled += 1
#             elif any(x in s for x in ('active', 'standard', 'std')):
#                 active += 1
#             if d >= 30 or any(x in s for x in ('sub', 'doubtful', 'dbt', 'npa', 'sma')):
#                 sub_std += 1
#     else:
#         # Fallback keyword scan
#         written_off = len(re.findall(r'\bwritten[-\s]?off\b|\bNPA\b', txt, re.IGNORECASE))
#         settled     = len(re.findall(r'\bsettled\b', txt, re.IGNORECASE))
#         dpd_90      = len(re.findall(r'\b090\b|\b120\b|\b150\b|\b180\b|90\+?\s*dpd', txt, re.IGNORECASE))
#         dpd_60      = len(re.findall(r'\b060\b|60\+?\s*dpd', txt, re.IGNORECASE))
#         dpd_30      = len(re.findall(r'\b030\b|30\+?\s*dpd', txt, re.IGNORECASE))
#         active      = min(len(re.findall(r'\bactive\b', txt, re.IGNORECASE)), 15)
 
#     total_accounts = max(len(accounts), active + settled + written_off, 1)
#     return dict(
#         accounts=accounts,
#         dpd_90_count=dpd_90, dpd_60_count=dpd_60, dpd_30_count=dpd_30,
#         written_off_count=written_off, settled_count=settled,
#         active_count=active, sub_std=sub_std,
#         total_accounts=total_accounts,
#         pct_active=active / total_accounts,
#     )
 
 
# # ---------------------------------------------------------------------------
# # CATEGORICAL FLAG INFERENCE
# # ---------------------------------------------------------------------------
 
# def infer_categorical_flags(extraction_result: dict) -> dict:
#     score       = int(extraction_result.get('Credit_Score', 700) or 700)
#     dpd_30      = int(extraction_result.get('num_times_30p_dpd', 0) or 0)
#     dpd_60      = int(extraction_result.get('num_times_60p_dpd', 0) or 0)
#     written_off = int(extraction_result.get('num_lss', 0) or
#                       extraction_result.get('written_off_count', 0) or 0)
#     doubtful    = int(extraction_result.get('num_dbt', 0) or 0)
#     cc_util_raw = extraction_result.get('CC_utilization', 0) or 0
#     cc_util     = float(cc_util_raw) if cc_util_raw > 0 else 0.0
#     income      = float(extraction_result.get('NETMONTHLYINCOME', 0) or
#                         extraction_result.get('avg_salary_6m', 50_000) or 50_000)
#     tenure      = int(extraction_result.get('Time_With_Curr_Empr', 24) or 24)
 
#     is_bureau_only = (
#         'NETMONTHLYINCOME' in extraction_result
#         and 'net_cash_surplus_6m' not in extraction_result
#         and 'net_surplus' not in extraction_result
#     )
 
#     surplus = 0.0  # FIX L2: always initialised
 
#     if is_bureau_only:
#         dpd_90_bureau = int(extraction_result.get('dpd_90_count_6m', 0) or 0)
#         surplus = _infer_surplus_from_cibil(score, dpd_60, dpd_30, income, dpd_90=dpd_90_bureau)
#         payment_discipline = ('POOR'     if (dpd_60 >= 1 or dpd_90_bureau >= 1 or dpd_30 >= 3)
#                               else 'MODERATE' if dpd_30 >= 1 else 'GOOD')
#         cashflow_health    = ('HEALTHY'  if surplus >= 14_000
#                               else 'STABLE'   if surplus >= 600
#                               else 'STRESSED' if surplus < -1_000
#                               else 'MODERATE')
#         liquidity_flag     = ('ADEQUATE' if surplus > 14_000
#                               else 'LOW' if surplus < -32_000 else 'MODERATE')
#         bureau_risk        = ('HIGH'   if (written_off >= 1 or doubtful >= 1
#                                            or dpd_60 >= 3 or score < 580)
#                               else 'MEDIUM' if (score < 650 or
#                                                (dpd_30 >= 2 and cc_util > 0.60))
#                               else 'LOW')
#         salary_stability   = ('UNSTABLE' if tenure < 6
#                               else 'STABLE' if (tenure >= 24 and score >= 700
#                                                and dpd_30 == 0) else 'MODERATE')
#     else:
#         dpd_90  = int(extraction_result.get('dpd_90_count_6m', 0) or 0)
#         bounces = int(extraction_result.get('inward_bounce_count_3m', 0) or 0)
#         missing = int(extraction_result.get('salary_missing_months', 0) or 0)
#         surplus = float(extraction_result.get('net_cash_surplus_6m') or
#                         extraction_result.get('net_surplus') or -50_000)
#         payment_discipline = ('POOR'     if (dpd_90 >= 1 or bounces >= 2)
#                               else 'MODERATE' if (bounces == 1 or dpd_30 >= 1)
#                               else 'GOOD')
#         cashflow_health    = ('HEALTHY'  if surplus >= 14_000
#                               else 'STABLE'   if 600 <= surplus < 14_000
#                               else 'STRESSED' if surplus < -1_000
#                               else 'MODERATE')
#         liquidity_flag     = ('ADEQUATE' if surplus > 14_000
#                               else 'LOW' if surplus < -32_000 else 'MODERATE')
#         bureau_risk        = ('HIGH'   if (dpd_90 >= 3 or written_off >= 1
#                                            or (dpd_90 >= 1 and dpd_30 >= 2))
#                               else 'MEDIUM' if (score < 580 or
#                                                (dpd_30 >= 2 and cc_util > 0.60))
#                               else 'LOW')
#         salary_stability   = ('UNSTABLE' if missing >= 1
#                               else 'STABLE' if (missing == 0 and score >= 700
#                                                and dpd_30 == 0 and bounces == 0)
#                               else 'MODERATE')
 
#     return {
#         'payment_discipline_flag': payment_discipline,
#         'cashflow_health':         cashflow_health,
#         'liquidity_flag':          liquidity_flag,
#         'bureau_risk_flag':        bureau_risk,
#         'salary_stability_flag':   salary_stability,
#         '_inference_path':         'bureau_only' if is_bureau_only else 'bank_statement',
#         '_surplus_estimate':       float(surplus),
#     }
 
 
# # ---------------------------------------------------------------------------
# # MAIN EXTRACTION FUNCTION
# # ---------------------------------------------------------------------------
 
# def extract_cibil_from_pdf(uploaded_file) -> dict:
#     """CIBIL PDF → structured dict. Returns success=True/False."""
#     if not OCR_AVAILABLE:
#         return {'success': False,
#                 'error': OCR_ERROR_MSG or 'OCR libraries not installed.'}
#     try:
#         pdf_bytes = uploaded_file.read()
 
#         # 1. OCR (multi-pass: PSM 6 + PSM 11, low-conf retry at 450 DPI)
#         txt = _ocr_pdf(pdf_bytes)
 
#         # 2. Credit score
#         credit_score = _extract_credit_score(txt)
 
#         # 3. Age / DOB
#         # Two strategies to handle digital PDFs where pdfminer splits table cells:
#         #   "22-Mar-1982" → "22-Mar-198\n\nGender\n\nMale\n\n2"
#         # Strategy 1: complete DOB present (scanned PDFs, clean digital PDFs)
#         # Strategy 2: partial DOB + lone completing digit nearby (split digital PDFs)
#         age_extracted = 35
#         # Strategy 1 — complete date on one or two lines
#         for pat in [
#             r'(?:date\s+of\s+birth|dob|d\.o\.b)[\s:\-\n]+(\d{2}[-/][A-Za-z]{3}[-/]\d{4})',
#             r'(?:date\s+of\s+birth|dob)[\s:\-]+(\d{2}[-/]\d{2}[-/]\d{4})',
#             r'born[\s:]+(\d{2}[-/][A-Za-z]{3}[-/]\d{4})',
#             r'\b(\d{2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4})\b',
#         ]:
#             m = re.search(pat, txt, re.IGNORECASE)
#             if m:
#                 for fmt in ('%d-%b-%Y', '%d/%b/%Y', '%d-%b-%y',
#                             '%d-%m-%Y', '%d/%m/%Y'):
#                     try:
#                         dob = datetime.strptime(m.group(1), fmt)
#                         candidate = int((datetime.now() - dob).days / 365.25)
#                         if 18 <= candidate <= 80:
#                             age_extracted = candidate
#                             break
#                     except Exception:
#                         continue
#                 if age_extracted != 35:
#                     break
 
#         # Strategy 2 — partial DOB (year split across columns in digital PDF)
#         if age_extracted == 35:
#             m_partial = re.search(
#                 r'(?:date\s+of\s+birth|dob)[\s\S]{0,20}?(\d{2}-[A-Za-z]{3}-\d{3})',
#                 txt, re.IGNORECASE)
#             if m_partial:
#                 partial   = m_partial.group(1)      # e.g. "22-Mar-198"
#                 after_pos = m_partial.end()
#                 # Find the lone digit within 200 chars that completes the year
#                 m_digit = re.search(r'(?<!\d)(\d)(?!\d)',
#                                     txt[after_pos:after_pos + 200])
#                 if m_digit:
#                     year_int = int(partial[-3:] + m_digit.group(1))
#                     if 1940 <= year_int <= 2010:   # adult applicant range
#                         try:
#                             dob = datetime.strptime(partial + m_digit.group(1), '%d-%b-%Y')
#                             candidate = int((datetime.now() - dob).days / 365.25)
#                             if 18 <= candidate <= 80:
#                                 age_extracted = candidate
#                         except Exception:
#                             pass
 
#         # Strategy 3 — explicit "Age: NN" text
#         if age_extracted == 35:
#             age_extracted = _re_int(r'(?:^|\s)age[\s:\-]+(\d{2})\b',
#                                     txt, 35, lo=18, hi=80)
 
#         # 4. Gender — default 'U' (bias-neutral)
#         if re.search(r'\bfemale\b|\bF\b|\bShe\b|\bHer\b', txt, re.IGNORECASE):
#             gender = 'F'
#         elif re.search(r'\bmale\b|\bM\b|\bHe\b|\bHis\b', txt, re.IGNORECASE):
#             gender = 'M'
#         else:
#             gender = 'U'
 
#         if re.search(r'\bsingle\b|\bunmarried\b', txt, re.IGNORECASE):
#             marital_status = 'Single'
#         elif re.search(r'\bmarried\b|\bspouse\b', txt, re.IGNORECASE):
#             marital_status = 'Married'
#         else:
#             marital_status = 'Unknown'
 
#         # 5. Education
#         education = 'GRADUATE'
#         for pat, val in [
#             (r'post.?grad(uate)?|m\.?tech|mba|mca', 'POST-GRADUATE'),
#             (r'professional|ca\b|cs\b|icai',          'PROFESSIONAL'),
#             (r'\b12th\b|\bhsc\b|\binter(mediate)?\b', '12TH'),
#             (r'\bssc\b|\b10th\b|\bmatric',             'SSC'),
#             (r'under.?grad(uate)?',                    'UNDER GRADUATE'),
#             (r'\bgrad(uate)?\b|\bb\.?tech\b|\bb\.?e\b|\bb\.?sc\b|\bb\.?com\b', 'GRADUATE'),
#         ]:
#             if re.search(pat, txt, re.IGNORECASE):
#                 education = val; break
 
#         # 6. Income
#         monthly_income = _extract_income(txt)
 
#         # 7. Employment
#         employment_type = 'Salaried'
#         if re.search(r'self.?employed|self\s+employ|proprietor|freelance',
#                      txt, re.IGNORECASE):
#             employment_type = 'Self-Employed'
#         elif re.search(r'\bbusiness\b|\bfirm\b|\bpartner(ship)?\b',
#                        txt, re.IGNORECASE):
#             employment_type = 'Business'
 
#         employment_tenure_months = 36
#         m = re.search(
#             r'(?:with\s+current\s+employer|employment\s+tenure|'
#             r'employed\s+(?:since|for))[^\d]{0,20}(\d+)\s*(?:year|yr)',
#             txt, re.IGNORECASE)
#         if m:
#             employment_tenure_months = int(m.group(1)) * 12
#         else:
#             m = re.search(
#                 r'(?:with\s+current\s+employer|tenure)[^\d]{0,20}(\d+)\s*(?:month|mth)',
#                 txt, re.IGNORECASE)
#             if m: employment_tenure_months = int(m.group(1))
 
#         existing_emi = 0
#         for emi_pat in [
#             r'(?:total\s+emi|existing\s+emi|current\s+emi|monthly\s+emi)'
#             r'[^\d₹]{0,20}[₹Rs\.]*\s*([0-9,]+)',
#             r'emi\s+(?:outflow|obligation)[^\d]{0,20}([0-9,]+)',
#             r'total\s+(?:monthly\s+)?obligation[\s:\-₹]*([0-9,]+)',
#         ]:
#             mm = re.search(emi_pat, txt, re.IGNORECASE)
#             if mm:
#                 v = int(mm.group(1).replace(',', ''))
#                 if 500 < v < 500_000:
#                     existing_emi = v; break
 
#         business_vintage = 0
#         mb = re.search(
#             r'(?:business\s+(?:since|established|vintage|age|started))'
#             r'[^\d]{0,20}(\d+)\s*(?:year|yr)', txt, re.IGNORECASE)
#         if mb: business_vintage = int(mb.group(1))
 
#         # 8. Credit utilisation
#         cc_util_pct = -99999
#         mc = re.search(
#             r'(?:credit\s+card\s+utiliz[ao]tion|cc\s+utiliz[ao]tion|'
#             r'utiliz[ao]tion\s+ratio)[^\d]{0,20}(\d{1,3})\s*%?',
#             txt, re.IGNORECASE)
#         if mc: cc_util_pct = int(mc.group(1))
#         pl_util = _re_float(
#             r'(?:personal\s+loan\s+utiliz[ao]tion|pl\s+utiliz[ao]tion)'
#             r'[^\d]{0,20}([\d\.]+)', txt, 0.25, lo=0, hi=5)
 
#         # 9. Enquiries
#         enq_data = _parse_enquiries(txt)
 
#         # 10. Accounts / DPD  (BUG 2 + 3 + 7 fixed in _parse_accounts)
#         acc = _parse_accounts(txt)
#         dpd_90_count      = acc['dpd_90_count']
#         dpd_60_count      = acc['dpd_60_count']
#         dpd_30_count      = acc['dpd_30_count']
#         written_off_count = acc['written_off_count']
#         settled_count     = acc['settled_count']
#         active_count      = acc['active_count']
#         sub_std           = acc['sub_std']
#         total_accounts    = acc['total_accounts']
#         pct_active        = acc['pct_active']
#         num_std = active_count
#         num_sub = sub_std
#         num_dbt = dpd_90_count
#         num_lss = written_off_count
 
#         # 11. Sanity check: high score + bad history
#         if credit_score >= 750 and (written_off_count > 0 or dpd_90_count > 0):
#             credit_score = min(credit_score, 550)
 
#         # 12. Delinquency timings
#         recent_level_of_deliq = max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)
#         num_deliq_6mts    = dpd_30_count + dpd_60_count + dpd_90_count
#         num_deliq_12mts   = num_deliq_6mts
#         num_deliq_6_12mts = 0
#         max_deliq_6mts    = -99999 if num_deliq_6mts == 0 else recent_level_of_deliq
#         max_deliq_12mts   = max_deliq_6mts
#         num_std_6mts  = min(num_std, _re_int(r'standard\s+accounts?\s*\(?6m\)?[\s:\-]+(\d+)', txt, num_std))
#         num_std_12mts = _re_int(r'standard\s+accounts?\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_std)
#         num_sub_6mts  = _re_int(r'sub.?standard\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
#         num_sub_12mts = _re_int(r'sub.?standard\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_sub)
#         num_dbt_6mts  = _re_int(r'doubtful\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
#         num_dbt_12mts = _re_int(r'doubtful\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_dbt)
#         num_lss_6mts  = _re_int(r'loss\s*\(?6m\)?[\s:\-]+(\d+)', txt, 0)
#         num_lss_12mts = _re_int(r'loss\s*\(?12m\)?[\s:\-]+(\d+)', txt, num_lss)
#         num_times_delinquent  = dpd_30_count + dpd_60_count + dpd_90_count
#         num_times_30p_dpd     = dpd_30_count + dpd_60_count + dpd_90_count
#         num_times_60p_dpd     = dpd_60_count + dpd_90_count
#         max_delinquency_level = max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)
 
#         time_since_recent_payment = _re_int(
#             r'(?:last|recent)\s+payment[\s:\-]+(\d+)\s*days?', txt, -99999)
#         if time_since_recent_payment == -99999:
#             mv = re.search(r'(?:last|recent)\s+payment[\s:\-]+(\d+)\s*(?:month|mth)',
#                            txt, re.IGNORECASE)
#             if mv: time_since_recent_payment = int(mv.group(1)) * 30
 
#         time_since_first_deliq = (
#             -99999 if num_times_delinquent == 0 else
#             _re_int(r'first\s+delinquency[\s:\-]+(\d+)\s*days?', txt, 365))
#         time_since_recent_deliq = (
#             -99999 if num_times_delinquent == 0 else
#             _re_int(r'(?:last|recent)\s+delinquency[\s:\-]+(\d+)\s*days?', txt, 90))
 
#         # 13. Trade-line ratios
#         pct_of_active_TLs_ever     = round(pct_active, 3)
#         pct_opened_TLs_L6m_of_L12m = _re_float(
#             r'(?:opened|new)\s+accounts?\s*\(?6m\s*/\s*12m\)?[\s:\-]+([\d\.]+)',
#             txt, 0.3, lo=0, hi=1)
#         pct_currentBal_all_TL = _re_float(
#             r'current\s+balance\s+(?:ratio|pct|%)[\s:\-]+([\d\.]+)',
#             txt, 0.3, lo=0, hi=10)
#         PL_enq_L6m  = enq_data['PL_enq_L6m'];  PL_enq_L12m = enq_data['PL_enq_L12m']
#         PL_enq      = enq_data['PL_enq']
#         CC_enq_L6m  = enq_data['CC_enq_L6m'];  CC_enq_L12m = enq_data['CC_enq_L12m']
#         CC_enq      = enq_data['CC_enq']
#         pct_PL_enq_L6m_of_L12m = round(PL_enq_L6m / max(PL_enq_L12m, 1), 2) if PL_enq_L6m >= 0 else 0
#         pct_CC_enq_L6m_of_L12m = round(CC_enq_L6m / max(CC_enq_L12m, 1), 2) if CC_enq_L6m >= 0 else 0
#         pct_PL_enq_L6m_of_ever = round(PL_enq_L6m / max(PL_enq if PL_enq >= 0 else 1, 1), 2)
#         pct_CC_enq_L6m_of_ever = round(CC_enq_L6m / max(CC_enq if CC_enq >= 0 else 1, 1), 2)
 
#         # 14. Product flags
#         CC_Flag = 1 if re.search(r'credit\s+card', txt, re.IGNORECASE) else 0
#         PL_Flag = 1 if re.search(r'personal\s+loan', txt, re.IGNORECASE) else 0
#         HL_Flag = 1 if re.search(r'home\s+loan|housing\s+loan', txt, re.IGNORECASE) else 0
#         GL_Flag = 1 if re.search(r'gold\s+loan', txt, re.IGNORECASE) else 0
 
#         # 15. Net cash surplus
#         net_cash_surplus = _re_int(
#             r'(?:net\s+(?:cash\s+)?surplus|disposable\s+income)'
#             r'[^\d₹]{0,20}[₹Rs\.]*\s*([0-9,]+)', txt, 0)
#         if net_cash_surplus == 0:
#             net_cash_surplus = int(_infer_surplus_from_cibil(
#                 credit_score, dpd_60_count, dpd_30_count, float(monthly_income),
#                 dpd_90=dpd_90_count))
 
#         # 16. Bank-statement proxies
#         inward_bounce_count_3m = dpd_90_count + dpd_60_count
#         salary_missing_months  = 0
#         total_credit_6m = _re_int(r'total\s+credits?\s*\(?6m\)?[\s:\-₹]*([0-9,]+)', txt, 0)
#         total_debit_6m  = _re_int(r'total\s+debits?\s*\(?6m\)?[\s:\-₹]*([0-9,]+)', txt, 0)
 
#         # 17. Stage-1 field mapping
#         s1 = {
#             'AMT_INCOME_TOTAL':           monthly_income * 12,
#             'AMT_ANNUITY':                existing_emi if existing_emi > 0 else int(monthly_income * 0.25),
#             'avg_salary_6m':              float(monthly_income),
#             'salary_txn_count_6m':        6.0,
#             'salary_amount_cv':           0.05 if employment_type == 'Salaried' else 0.25,
#             'salary_date_std':            2.0,
#             'salary_creditor_consistent': 1.0 if employment_type == 'Salaried' else 0.7,
#             'salary_missing_months':      float(salary_missing_months),
#             'dpd_15_count_6m':            0.0,
#             'dpd_30_count_6m':            float(dpd_30_count),
#             'dpd_90_count_6m':            float(dpd_90_count),
#             'max_dpd_6m':                 float(max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)),
#             'dpd_30_count_3m':            float(dpd_30_count),
#             'total_payments_6m':          3.0,
#             'total_late_15_6m':           0.0,
#             'total_late_30_6m':           float(dpd_30_count),
#             'total_late_60_6m':           float(dpd_60_count),
#             'total_late_90_6m':           float(dpd_90_count),
#             'max_days_late_6m':           float(max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)),
#             'avg_days_late_6m':           float(dpd_30_count*10 + dpd_60_count*20 + dpd_90_count*40) / max(total_accounts, 1),
#             'total_late_30_3m':           float(dpd_30_count),
#             'total_late_90_3m':           float(dpd_90_count),
#             'avg_balance_cc':             0.0, 'total_drawings_cc': 0.0,
#             'avg_credit_limit':           0.0, 'max_utilization': (cc_util_pct / 100) if cc_util_pct > 0 else 0.0,
#             'total_payments_cc':          0.0, 'dpd_count_cc': 0.0,
#             'avg_balance_pos':            0.0, 'dpd_count_pos': 0.0,
#             'total_credit_activity':      float(total_accounts),
#             'total_dpd_count':            float(dpd_30_count + dpd_60_count + dpd_90_count),
#             'avg_monthly_balance_6m':     float(net_cash_surplus),
#             'total_emi_monthly':          float(existing_emi if existing_emi > 0 else int(monthly_income * 0.25)),
#             'net_cash_surplus_6m':        float(net_cash_surplus),
#             'total_credit_6m':            float(total_credit_6m),
#             'total_debit_6m':             float(total_debit_6m),
#             'inward_bounce_count_3m':     float(inward_bounce_count_3m),
#             'recent_payment_stress':      float(dpd_30_count + dpd_60_count),
#             'active_loans_count':         float(active_count),
#             'bureau_score':               float(credit_score),
#         }
 
#         # 18. Stage-2 field mapping
#         s2 = {
#             'Credit_Score': credit_score, 'AGE': age_extracted,
#             'GENDER': gender, 'MARITALSTATUS': marital_status,
#             'EDUCATION': education, 'NETMONTHLYINCOME': monthly_income,
#             'Time_With_Curr_Empr': employment_tenure_months,
#             'num_times_delinquent': num_times_delinquent,
#             'max_delinquency_level': max_delinquency_level,
#             'max_recent_level_of_deliq': max(dpd_60_count*60, dpd_30_count*30),
#             'num_deliq_6mts': num_deliq_6mts, 'num_deliq_12mts': num_deliq_12mts,
#             'num_deliq_6_12mts': num_deliq_6_12mts,
#             'max_deliq_6mts': max_deliq_6mts, 'max_deliq_12mts': max_deliq_12mts,
#             'num_times_30p_dpd': num_times_30p_dpd, 'num_times_60p_dpd': num_times_60p_dpd,
#             'recent_level_of_deliq': recent_level_of_deliq,
#             'num_std': num_std, 'num_std_6mts': num_std_6mts, 'num_std_12mts': num_std_12mts,
#             'num_sub': num_sub, 'num_sub_6mts': num_sub_6mts, 'num_sub_12mts': num_sub_12mts,
#             'num_dbt': num_dbt, 'num_dbt_6mts': num_dbt_6mts, 'num_dbt_12mts': num_dbt_12mts,
#             'num_lss': num_lss, 'num_lss_6mts': num_lss_6mts, 'num_lss_12mts': num_lss_12mts,
#             'time_since_recent_payment': time_since_recent_payment,
#             'time_since_first_deliquency': time_since_first_deliq,
#             'time_since_recent_deliquency': time_since_recent_deliq,
#             'tot_enq': enq_data['tot_enq'], 'enq_L3m': enq_data['enq_L3m'],
#             'enq_L6m': enq_data['enq_L6m'], 'enq_L12m': enq_data['enq_L12m'],
#             'time_since_recent_enq': enq_data['time_since_recent_enq'],
#             'CC_enq': CC_enq, 'CC_enq_L6m': CC_enq_L6m, 'CC_enq_L12m': CC_enq_L12m,
#             'PL_enq': PL_enq, 'PL_enq_L6m': PL_enq_L6m, 'PL_enq_L12m': PL_enq_L12m,
#             'pct_of_active_TLs_ever': pct_of_active_TLs_ever,
#             'pct_opened_TLs_L6m_of_L12m': pct_opened_TLs_L6m_of_L12m,
#             'pct_currentBal_all_TL': pct_currentBal_all_TL,
#             'pct_PL_enq_L6m_of_L12m': pct_PL_enq_L6m_of_L12m,
#             'pct_CC_enq_L6m_of_L12m': pct_CC_enq_L6m_of_L12m,
#             'pct_PL_enq_L6m_of_ever': pct_PL_enq_L6m_of_ever,
#             'pct_CC_enq_L6m_of_ever': pct_CC_enq_L6m_of_ever,
#             'CC_utilization': cc_util_pct / 100 if cc_util_pct > 0 else -99999,
#             'PL_utilization': pl_util,
#             'CC_Flag': CC_Flag, 'PL_Flag': PL_Flag, 'HL_Flag': HL_Flag, 'GL_Flag': GL_Flag,
#             'max_unsec_exposure_inPct': cc_util_pct if cc_util_pct > 0 else 0,
#             'last_prod_enq2': enq_data['last_prod_enq2'],
#             'first_prod_enq2': enq_data['first_prod_enq2'],
#         }
 
#         # 19. Categorical flags
#         _inf = infer_categorical_flags({
#             'Credit_Score': credit_score, 'num_times_30p_dpd': dpd_30_count,
#             'num_times_60p_dpd': dpd_60_count, 'num_lss': num_lss,
#             'num_dbt': num_dbt,
#             'CC_utilization': cc_util_pct / 100 if cc_util_pct > 0 else 0,
#             'NETMONTHLYINCOME': monthly_income, 'Time_With_Curr_Empr': employment_tenure_months,
#             'dpd_90_count_6m': dpd_90_count, 'inward_bounce_count_3m': inward_bounce_count_3m,
#             'salary_missing_months': salary_missing_months,
#             'net_cash_surplus_6m': net_cash_surplus,
#         })
 
#         # 20. FIX S-2: recent_deliq_flag from actual DPD
#         recent_deliq_flag = 1 if (dpd_90_count > 0 or dpd_60_count > 0) else 0
 
#         return {
#             **s1, **s2,
#             'existing_emi':              existing_emi if existing_emi > 0 else s1['total_emi_monthly'],
#             'employment_type':           employment_type,
#             'business_vintage_years':    business_vintage,
#             'credit_utilization_pct':    cc_util_pct if cc_util_pct > 0 else 0,
#             'salary_stability_flag':     _inf['salary_stability_flag'],
#             'payment_discipline_flag':   _inf['payment_discipline_flag'],
#             'cashflow_health':           _inf['cashflow_health'],
#             'liquidity_flag':            _inf['liquidity_flag'],
#             'bureau_risk_flag':          _inf['bureau_risk_flag'],
#             'written_off_count':         written_off_count,
#             'settled_count':             settled_count,
#             'high_util_flag':            1 if cc_util_pct > 75 else 0,
#             'recent_deliq_flag':         recent_deliq_flag,
#             'account_quality_score':     max(0, 100 - written_off_count*20
#                                              - settled_count*10 - dpd_90_count*15
#                                              - dpd_30_count*5),
#             '_surplus_proxy':            int(net_cash_surplus),
#             'raw_text':                  txt,
#             'success':                   True,
#             'extraction_method':         'OCR+MultiPass_v4',
#         }
 
#     except Exception as e:
#         import traceback
#         return {
#             'error':     str(e),
#             'message':   f'Error extracting CIBIL data: {str(e)}',
#             'traceback': traceback.format_exc(),
#             'success':   False,
#         }









# ocr_extractor.py — CIBIL PDF Extraction Engine v5.5
# =====================================================
# Hybrid extractor: pdfplumber for digital PDFs, OCR fallback.
#
# v5.5 fixes vs v5.4
# -------------------
# BUG 1 — pdfplumber tables are single-column merged cells
#   e.g. Account row arrives as one string:
#   "Kotak Bank Credit Card May-2017 Rs. 150,000 Rs. 57,000 Active 000"
#   _parse_accounts_table() required 7+ columns → always got 0 accounts.
#   Fix: parse the merged-cell string with regex instead of column indexing.
#
# BUG 2 — Age hardcoded to 35 in pdfplumber path
#   DOB is present in full_text ("03-Sep-1985") but age extraction was
#   never called. Fix: added _extract_age_from_text() called before return.
#
# BUG 3 — infer_categorical_flags() never called in pdfplumber path
#   payment_discipline, cashflow_health, liquidity, bureau_risk,
#   salary_stability all defaulted to missing → expander showed '—'.
#   Fix: call infer_categorical_flags() and merge result into return dict.
#
# BUG 4 — CC utilisation, income, enquiries extracted correctly from
#   full_text but then overridden by sentinel -99999 / 0 defaults later.
#   Fix: use consistent variable names throughout pdfplumber path.

import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from io import BytesIO
import traceback

# ---------------------------------------------------------------------------
# OPTIONAL DEPENDENCIES
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
    OCR_ERROR_MSG = None
except ImportError as e:
    OCR_AVAILABLE = False
    OCR_ERROR_MSG = str(e)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _re_int(pattern, text, default, lo=None, hi=None):
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            v = int(str(m.group(1)).replace(',', '').replace(' ', ''))
            if lo is not None and v < lo: return default
            if hi is not None and v > hi: return default
            return v
        except Exception: pass
    return default

def _re_float(pattern, text, default, lo=None, hi=None):
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            v = float(str(m.group(1)).replace(',', '').replace(' ', ''))
            if lo is not None and v < lo: return default
            if hi is not None and v > hi: return default
            return v
        except Exception: pass
    return default

def _fix_spaced_digits(text):
    return re.sub(r'\b(\d)\s(\d)\s(\d)\b', r'\1\2\3', text)

def _split_merged_words(text):
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    for merged, spaced in {
        'CREDITINFORMATIONREPORT': 'CREDIT INFORMATION REPORT',
        'ACCOUNTDETAILS': 'ACCOUNT DETAILS',
        'ENQUIRYDETAILS': 'ENQUIRY DETAILS',
        'PERSONALINFORMATION': 'PERSONAL INFORMATION',
    }.items():
        text = re.sub(merged, spaced, text, flags=re.IGNORECASE)
    return text

def _clean_ocr_noise(text):
    text = re.sub(r'(\d)[)\]}\|]', r'\1', text)
    text = re.sub(r'[)\]}\|](\d)', r'\1', text)
    text = re.sub(r'§', '', text)
    return text

_DPD_TEXT_MAP = {
    'NIL': 0, 'NA': 0, 'N/A': 0, 'XXX': 0, '-': 0, 'NONE': 0,
    'STD': 0, 'SMA': 30, 'SUB': 90, 'DBT': 180, 'LSS': 365, 'NPA': 90, 'WO': 365,
}

def _parse_dpd_value(raw):
    s = str(raw).strip().upper()
    if s in _DPD_TEXT_MAP: return _DPD_TEXT_MAP[s]
    clean = re.sub(r'[^0-9]', '', s)
    try: return int(clean) if clean else 0
    except: return 0

def _infer_surplus_from_cibil(score, dpd_60, dpd_30, income, dpd_90=0):
    base = income * 0.30
    if score >= 750 and dpd_90 == 0 and dpd_60 == 0 and dpd_30 == 0: return base * 1.5
    if score >= 700 and dpd_90 == 0: return base * 1.1
    if dpd_90 >= 1: return base * -0.5
    if dpd_60 >= 1: return base * 0.2
    if score < 650: return base * -0.2
    if dpd_30 >= 2: return base * 0.5
    return base

# ---------------------------------------------------------------------------
# CATEGORICAL FLAG INFERENCE
# ---------------------------------------------------------------------------
def infer_categorical_flags(extraction_result: dict) -> dict:
    score       = int(extraction_result.get('Credit_Score', 700) or 700)
    dpd_30      = int(extraction_result.get('num_times_30p_dpd', 0) or 0)
    dpd_60      = int(extraction_result.get('num_times_60p_dpd', 0) or 0)
    written_off = int(extraction_result.get('num_lss', 0) or
                      extraction_result.get('written_off_count', 0) or 0)
    doubtful    = int(extraction_result.get('num_dbt', 0) or 0)
    cc_util_raw = extraction_result.get('CC_utilization', 0) or 0
    cc_util     = float(cc_util_raw) if cc_util_raw > 0 else 0.0
    income      = float(extraction_result.get('NETMONTHLYINCOME', 0) or
                        extraction_result.get('avg_salary_6m', 50_000) or 50_000)
    tenure      = int(extraction_result.get('Time_With_Curr_Empr', 24) or 24)

    is_bureau_only = (
        'NETMONTHLYINCOME' in extraction_result
        and 'net_cash_surplus_6m' not in extraction_result
        and 'net_surplus' not in extraction_result
    )
    surplus = 0.0

    if is_bureau_only:
        dpd_90_bureau = int(extraction_result.get('dpd_90_count_6m', 0) or 0)
        surplus = _infer_surplus_from_cibil(score, dpd_60, dpd_30, income, dpd_90=dpd_90_bureau)
        payment_discipline = ('POOR'     if (dpd_60 >= 1 or dpd_90_bureau >= 1 or dpd_30 >= 3)
                              else 'MODERATE' if dpd_30 >= 1 else 'GOOD')
        cashflow_health    = ('HEALTHY'  if surplus >= 14_000
                              else 'STABLE'   if surplus >= 600
                              else 'STRESSED' if surplus < -1_000
                              else 'MODERATE')
        liquidity_flag     = ('ADEQUATE' if surplus > 14_000
                              else 'LOW' if surplus < -32_000 else 'MODERATE')
        bureau_risk        = ('HIGH'   if (written_off >= 1 or doubtful >= 1
                                           or dpd_60 >= 3 or score < 580)
                              else 'MEDIUM' if (score < 650 or
                                               (dpd_30 >= 2 and cc_util > 0.60))
                              else 'LOW')
        salary_stability   = ('UNSTABLE' if tenure < 6
                              else 'STABLE' if (tenure >= 24 and score >= 700
                                               and dpd_30 == 0) else 'MODERATE')
    else:
        dpd_90  = int(extraction_result.get('dpd_90_count_6m', 0) or 0)
        bounces = int(extraction_result.get('inward_bounce_count_3m', 0) or 0)
        missing = int(extraction_result.get('salary_missing_months', 0) or 0)
        surplus = float(extraction_result.get('net_cash_surplus_6m') or
                        extraction_result.get('net_surplus') or -50_000)
        payment_discipline = ('POOR'     if (dpd_90 >= 1 or bounces >= 2)
                              else 'MODERATE' if (bounces == 1 or dpd_30 >= 1)
                              else 'GOOD')
        cashflow_health    = ('HEALTHY'  if surplus >= 14_000
                              else 'STABLE'   if 600 <= surplus < 14_000
                              else 'STRESSED' if surplus < -1_000
                              else 'MODERATE')
        liquidity_flag     = ('ADEQUATE' if surplus > 14_000
                              else 'LOW' if surplus < -32_000 else 'MODERATE')
        bureau_risk        = ('HIGH'   if (dpd_90 >= 3 or written_off >= 1
                                           or (dpd_90 >= 1 and dpd_30 >= 2))
                              else 'MEDIUM' if (score < 580 or
                                               (dpd_30 >= 2 and cc_util > 0.60))
                              else 'LOW')
        salary_stability   = ('UNSTABLE' if missing >= 1
                              else 'STABLE' if (missing == 0 and score >= 700
                                               and dpd_30 == 0 and bounces == 0)
                              else 'MODERATE')

    return {
        'payment_discipline_flag': payment_discipline,
        'cashflow_health':         cashflow_health,
        'liquidity_flag':          liquidity_flag,
        'bureau_risk_flag':        bureau_risk,
        'salary_stability_flag':   salary_stability,
        '_inference_path':         'bureau_only' if is_bureau_only else 'bank_statement',
        '_surplus_estimate':       float(surplus),
    }

# ---------------------------------------------------------------------------
# OCR PRE-PROCESSING
# ---------------------------------------------------------------------------
def _preprocess_page(pil_image, dpi=300):
    img  = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
    except: pass
    try:
        coords = np.column_stack(np.where(gray < 200))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
            if angle < -45: angle = 90 + angle
            if 0.3 < abs(angle) < 20:
                (h, w) = gray.shape
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
    except: pass
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

def _ocr_page_multipass(pil_image, dpi=300):
    binary = _preprocess_page(pil_image, dpi)
    cfg6   = '--oem 3 --psm 6 -l eng'
    data6  = pytesseract.image_to_data(binary, config=cfg6,
                                       output_type=pytesseract.Output.DICT)
    text6  = pytesseract.image_to_string(binary, config=cfg6)
    confs  = [c for c in data6['conf'] if c != -1]
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0
    cfg11  = '--oem 3 --psm 11 -l eng'
    text11 = pytesseract.image_to_string(binary, config=cfg11)
    lines6 = set(l.strip() for l in text6.splitlines() if l.strip())
    extra  = [l for l in text11.splitlines() if l.strip() and l.strip() not in lines6]
    return text6 + '\n' + '\n'.join(extra), avg_conf

def _ocr_pdf(pdf_bytes, low_conf_threshold=60.0):
    full_text = ""
    images_300 = convert_from_bytes(pdf_bytes, dpi=300)
    for idx, page_img in enumerate(images_300):
        text, conf = _ocr_page_multipass(page_img, dpi=300)
        if conf < low_conf_threshold:
            try:
                hi = convert_from_bytes(pdf_bytes, dpi=450,
                                        first_page=idx+1, last_page=idx+1)
                text_hq, conf_hq = _ocr_page_multipass(hi[0], dpi=450)
                if conf_hq > conf: text = text_hq
            except: pass
        full_text += text + "\n"
    full_text = _clean_ocr_noise(full_text)
    full_text = _split_merged_words(full_text)
    full_text = _fix_spaced_digits(full_text)
    return full_text

# ---------------------------------------------------------------------------
# TEXT PARSERS (shared by both pdfplumber and OCR paths)
# ---------------------------------------------------------------------------
def _extract_credit_score(txt: str) -> int:
    """Extract credit score 300-900 from text."""
    txt = re.sub(r'score\s+range\s+\d{3}\s*[-–]\s*\d{3}[^.]*', '', txt, flags=re.IGNORECASE)

    def _mask_amounts(t):
        t = re.sub(r'Rs?\\.?\\s*[0-9,]{4,}', '', t, flags=re.IGNORECASE)
        t = re.sub(r'₹\\s*[0-9,]{4,}', '', t)
        return t

    patterns = [
        # Header on line above number (TransUnion layout: "CIBIL SCORE\n736")
        r'(?:CIBIL\s+SCORE|YOUR\s+(?:CIBIL\s+)?SCORE|CREDIT\s+SCORE)\s*\n+\s*(\d{3})\b',
        # Same-line explicit label
        r'(?:cibil|experian|equifax|highmark|crif|bureau)\s*(?:trans\s*union\s*)?score\s*[:\-\(]?\s*(\d{3})',
        r'(?:credit\s+)?score\s*[:\-]\s*(\d{3})',
        # Number followed by rating band (same or next line)
        r'\b(8[0-9]{2}|7[0-9]{2}|6[0-9]{2}|[3-5][0-9]{2})\s*\n?\s*(?:EXCELLENT|VERY\s*GOOD|GOOD|FAIR|SUBPRIME|POOR)\b',
        r'\b(\d{3})\s+out\s+of\s+900\b',
        r'(?:your\s+score\s+is|score\s+is)\s+(\d{3})',
        r'(?:score|rating)\s*(?!range)[^\n\r]{0,30}?(\d{3})',
    ]
    for pat in patterns:
        m = re.search(pat, txt, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 300 <= v <= 900:
                return v

    # Fallback: standalone 3-digit near score keyword
    masked = _mask_amounts(txt)
    candidates = [(int(mx.group(1)), mx.start())
                  for mx in re.finditer(r'(?<![,\d])(\d{3})(?![,\d])', masked)
                  if 300 <= int(mx.group(1)) <= 900]
    if candidates:
        kw_pos = [mx.start() for mx in re.finditer(
            r'(?:cibil|bureau|credit)\s+score', masked, re.IGNORECASE)]
        if kw_pos:
            return min(candidates, key=lambda cp: min(abs(cp[1]-kp) for kp in kw_pos))[0]
        return candidates[0][0]

    if re.search(r'\b(?:NH|NTC|no\s+history|new\s+to\s+credit)\b', txt, re.IGNORECASE):
        return 650
    return 720

def _extract_age(txt: str) -> int:
    """Extract age from DOB or explicit age field."""
    for pat in [
        r'(?:date\s+of\s+birth|dob|d\.o\.b)[\s:\-\n]+(\d{2}[-/][A-Za-z]{3}[-/]\d{4})',
        r'\b(\d{2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4})\b',
        r'(?:date\s+of\s+birth|dob)[\s:\-]+(\d{2}[-/]\d{2}[-/]\d{4})',
    ]:
        m = re.search(pat, txt, re.IGNORECASE)
        if m:
            for fmt in ('%d-%b-%Y', '%d/%b/%Y', '%d-%b-%y', '%d-%m-%Y', '%d/%m/%Y'):
                try:
                    dob = datetime.strptime(m.group(1), fmt)
                    age = int((datetime.now() - dob).days / 365.25)
                    if 18 <= age <= 80:
                        return age
                except: continue
    # Explicit age text
    return _re_int(r'(?:^|\s)age[\s:\-]+(\d{2})\b', txt, 35, lo=18, hi=80)

def _extract_income(txt: str) -> int:
    """Extract net monthly income."""
    # Primary: "Net Monthly Income" anywhere in text
    m = re.search(r'net\s+monthly\s+income[^\d]{0,80}?([\d,]+)', txt, re.IGNORECASE | re.DOTALL)
    if m:
        try:
            v = int(m.group(1).replace(',', ''))
            if 5_000 < v < 5_000_000:
                return v
        except: pass

    # Secondary patterns
    patterns = [
        r'monthly\s+(?:take.?home|salary|income)[\s:\-₹Rs\.]*([0-9,]+)',
        r'(?:salary|income)\s+per\s+month[\s:\-₹Rs\.]*([0-9,]+)',
        r'₹\s*([0-9,]+)\s*/?\s*(?:per\s+month|p\.?m\.?|monthly)',
        r'([0-9,]+)\s*(?:per\s+month|p\.?m\.?)',
        r'net\s+monthly[\s\S]{0,50}?Rs\.?\s*([0-9,]+)',
        r'Rs\.?\s*([0-9,]+)\s*\n(?:income|salary|earning)',
        r'([0-9]+\.?[0-9]*)\s*(?:L|lakh|lakhs?)\s*(?:p\.?m\.?|per\s+month|monthly)',
        r'(?:annual|yearly)\s+income[\s:\-₹Rs\.]*([0-9,]+)',
        r'(?:total|gross)\s+income[\s:\-₹Rs\.]*([0-9,]+)',
    ]
    for pat in patterns:
        m2 = re.search(pat, txt, re.IGNORECASE | re.DOTALL)
        if m2:
            try:
                s = m2.group(1).replace(',', '').replace(' ', '')
                v = float(s) * 100_000 if '.' in s and float(s) < 100 else int(s)
                if 'annual' in pat or 'yearly' in pat: v = v / 12
                if 5_000 < v < 5_000_000:
                    return int(v)
            except: pass
    return 50_000

def _extract_cc_util(txt: str) -> float:
    """Extract CC utilisation as a 0.0-1.0 fraction."""
    # "CC Utilization % ... 38%"
    patterns = [
        r'CC\s+Utilization\s+%\s*\n?\s*(\d{1,3})\s*%',
        r'CC\s+Utilization\s+%?\s*(\d{1,3})\s*%',
        r'CC\s+Utiliz[ao]tion[^\d]{0,20}(\d{1,3})\s*%?',
        r'credit\s+card\s+utiliz[ao]tion[^\d]{0,20}(\d{1,3})\s*%?',
        r'utiliz[ao]tion[^\d]{0,30}?(\d{1,3})\s*%',
    ]
    for pat in patterns:
        m = re.search(pat, txt, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 100:
                return v / 100.0
    return 0.0

def _extract_pl_util(txt: str) -> float:
    """Extract PL utilisation as a 0.0-1.0 fraction."""
    m = re.search(r'PL\s+Utilization\s+%?\s*\n?\s*(\d{1,3})\s*%', txt, re.IGNORECASE)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v / 100.0
    return _re_float(r'personal\s+loan\s+utiliz[ao]tion[^\d]{0,20}([\d\.]+)',
                     txt, 0.42, lo=0, hi=1)

def _extract_employment(txt: str) -> tuple:
    """Returns (employment_type, tenure_months, business_vintage_years)."""
    emp_type = 'Salaried'
    if re.search(r'self.?employed|self\s+employ|proprietor|freelance', txt, re.IGNORECASE):
        emp_type = 'Self-Employed'
    elif re.search(r'\bbusiness\b|\bfirm\b|\bpartner(ship)?\b', txt, re.IGNORECASE):
        emp_type = 'Business'

    tenure = 24
    m = re.search(
        r'(?:with\s+current\s+employer|employment\s+tenure|employed\s+(?:since|for))'
        r'[^\d]{0,20}(\d+)\s*(?:year|yr)', txt, re.IGNORECASE)
    if m:
        tenure = int(m.group(1)) * 12
    else:
        # TransUnion layout: "With Current Employer\nSalaried Manager Rs. 68,000 36 months"
        # [^\d] stops at digits in the salary amount, so use [\s\S] (any char) instead.
        m = re.search(
            r'(?:with\s+current\s+employer)[\s\S]{0,120}?(\d+)\s*months?',
            txt, re.IGNORECASE)
        if not m:
            m = re.search(
                r'(?:tenure)[^\d]{0,20}(\d+)\s*(?:month|mth)',
                txt, re.IGNORECASE)
        if m: tenure = int(m.group(1))

    vintage = 0
    mb = re.search(
        r'(?:business\s+(?:since|established|vintage|age|started))[^\d]{0,20}(\d+)\s*(?:year|yr)',
        txt, re.IGNORECASE)
    if mb: vintage = int(mb.group(1))

    return emp_type, tenure, vintage

def _extract_gender(txt: str) -> str:
    if re.search(r'\bfemale\b', txt, re.IGNORECASE): return 'F'
    # Check for standalone 'Male' but not 'Female'
    if re.search(r'\bmale\b', txt, re.IGNORECASE): return 'M'
    return 'U'

def _extract_marital(txt: str) -> str:
    if re.search(r'\bsingle\b|\bunmarried\b', txt, re.IGNORECASE): return 'Single'
    if re.search(r'\bmarried\b|\bspouse\b', txt, re.IGNORECASE): return 'Married'
    return 'Unknown'

def _extract_education(txt: str) -> str:
    for pat, val in [
        (r'post.?grad(uate)?|m\.?tech|mba|mca', 'POST-GRADUATE'),
        (r'professional|ca\b|cs\b|icai',          'PROFESSIONAL'),
        (r'\b12th\b|\bhsc\b|\binter(mediate)?\b', '12TH'),
        (r'\bssc\b|\b10th\b|\bmatric',             'SSC'),
        (r'under.?grad(uate)?',                    'UNDER GRADUATE'),
        (r'\bgrad(uate)?\b|\bb\.?tech\b|\bb\.?e\b|\bb\.?sc\b|\bb\.?com\b', 'GRADUATE'),
    ]:
        if re.search(pat, txt, re.IGNORECASE): return val
    return 'GRADUATE'

def _extract_existing_emi(txt: str) -> int:
    for emi_pat in [
        r'(?:total\s+emi|existing\s+emi|current\s+emi|monthly\s+emi)[^\d₹]{0,20}[₹Rs\.]*\s*([0-9,]+)',
        r'emi\s+(?:outflow|obligation)[^\d]{0,20}([0-9,]+)',
        r'total\s+(?:monthly\s+)?obligation[\s:\-₹]*([0-9,]+)',
    ]:
        mm = re.search(emi_pat, txt, re.IGNORECASE)
        if mm:
            v = int(mm.group(1).replace(',', ''))
            if 500 < v < 500_000: return v
    return 0

def _parse_enquiries(txt: str) -> dict:
    enq_section = ""
    m = re.search(r'enquir(?:y|ies)\s+details?(.*?)(?:score\s+factors|$)',
                  txt, re.IGNORECASE | re.DOTALL)
    if m: enq_section = m.group(1)

    # Prefer explicit counts from "Total Enquiries: 4 | Last 3 Months: 1 ..."
    tot_explicit = _re_int(r'total\s+enquir(?:y|ies)[:\s]+(\d+)', txt, 0)
    l3_explicit  = _re_int(r'last\s+3\s+months?[:\s]+(\d+)', txt, 0)
    l6_explicit  = _re_int(r'last\s+6\s+months?[:\s]+(\d+)', txt, 0)
    l12_explicit = _re_int(r'last\s+12\s+months?[:\s]+(\d+)', txt, 0)

    # Date-based counts as fallback
    raw_dates = re.findall(r'\b(\d{2}-[A-Za-z]{3}-\d{4})\b', enq_section)
    today = datetime.now().date()
    parsed_dates = []
    for ds in raw_dates:
        for fmt in ('%d-%b-%Y', '%d-%B-%Y'):
            try: parsed_dates.append(datetime.strptime(ds, fmt).date()); break
            except ValueError: continue

    def _win(days): return sum(1 for d in parsed_dates if (today - d).days <= days)

    enq_L3m  = l3_explicit  or _win(90)
    enq_L6m  = l6_explicit  or _win(180)
    enq_L12m = l12_explicit or _win(365)
    tot_enq  = tot_explicit or max(len(parsed_dates), enq_L12m)

    time_since_recent_enq = (today - max(parsed_dates)).days if parsed_dates else -99999

    CC_enq = _re_int(r'credit\s+card\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, -99999)
    PL_enq = _re_int(r'personal\s+loan\s+enquir(?:y|ies)[\s:\-]+(\d+)', txt, -99999)

    last_prod = first_prod = 'others'
    for pat, label in {
        r'personal\s+loan': 'PL', r'credit\s+card': 'CC',
        r'home\s+loan|housing': 'HL', r'auto\s+loan|car\s+loan': 'AL',
        r'gold\s+loan': 'GL', r'business\s+loan': 'BL',
    }.items():
        if re.search(pat, enq_section or txt, re.IGNORECASE):
            last_prod = first_prod = label; break

    return dict(
        tot_enq=tot_enq, enq_L3m=enq_L3m, enq_L6m=enq_L6m, enq_L12m=enq_L12m,
        time_since_recent_enq=time_since_recent_enq,
        CC_enq=CC_enq, CC_enq_L6m=0, CC_enq_L12m=0,
        PL_enq=PL_enq, PL_enq_L6m=0, PL_enq_L12m=0,
        last_prod_enq2=last_prod, first_prod_enq2=first_prod,
    )

# ---------------------------------------------------------------------------
# ACCOUNT / DPD PARSER — handles BOTH pdfplumber merged-cell AND OCR layouts
# ---------------------------------------------------------------------------
def _parse_accounts(txt: str) -> dict:
    """
    Parse the ACCOUNT DETAILS section.

    pdfplumber returns each row as a single merged string:
      "Kotak Bank Credit Card May-2017 Rs. 150,000 Rs. 57,000 Active 000"
    OCR returns similar. Both are handled by the same regex approach.

    Strategy: isolate the account section, then for each line try to extract
    (status, dpd) from the rightmost tokens.
    """
    # Isolate section between ACCOUNT DETAILS and ENQUIRY DETAILS / SCORE FACTORS
    idx_s = -1
    idx_e = len(txt)
    for m in re.finditer(r'ACCOUNT\s+DETAILS?|LOAN\s+DETAILS?', txt, re.IGNORECASE):
        idx_s = m.end(); break
    for m in re.finditer(r'ENQUIRY\s+DETAILS?|SCORE\s+FACTORS', txt, re.IGNORECASE):
        if m.start() > max(idx_s, 0):
            idx_e = m.start(); break
    section = txt[idx_s:idx_e] if idx_s >= 0 else txt

    dpd_90 = dpd_60 = dpd_30 = 0
    written_off = settled = active = sub_std = 0
    accounts = []

    STATUS_RE = re.compile(
        r'\b(Active|Settled|Written[-\s]?Off|Closed|NPA|Doubtful|Loss|'
        r'Sub[-\s]?Standard|Standard|SMA|SUB|DBT|LSS)\b', re.IGNORECASE)
    DPD_CODE_RE = re.compile(r'\b(\d{3})\b')

    lender_keywords = re.compile(
        r'\bBank\b|\bFinance\b|\bFinserv\b|\bCapital\b|\bLtd\b|\bSBI\b|\bBajaj\b'
        r'|\bTata\b|\bKotak\b|\bHDFC\b|\bICICI\b|\bAxis\b|\bIndusInd\b|\bRBL\b',
        re.IGNORECASE)

    for line in section.split('\n'):
        line = line.strip()
        if not line: continue

        # Skip header row
        if re.search(r'Lender\s+Product|DPD\s+6M', line, re.IGNORECASE): continue
        # Must look like an account row (has a lender name OR status)
        has_lender = bool(lender_keywords.search(line))
        stat_m = STATUS_RE.search(line)
        if not has_lender and not stat_m: continue

        # Extract DPD: last 3-digit code on the line (DPD is rightmost column)
        dpd_codes = DPD_CODE_RE.findall(line)
        # Filter to plausible DPD values (0-180 days or known codes)
        valid_dpd = [int(d) for d in dpd_codes if int(d) <= 180]
        dpd_val = valid_dpd[-1] if valid_dpd else 0

        # Extract status
        status = stat_m.group(1).lower().replace(' ', '_').replace('-', '_') if stat_m else 'active'

        # Product type
        prod_label = None
        for pat, label in [
            (r'credit\s+card', 'CC'), (r'personal\s+loan', 'PL'),
            (r'home\s+loan|housing', 'HL'), (r'car\s+loan|auto\s+loan', 'AL'),
            (r'gold\s+loan', 'GL'), (r'business\s+loan', 'BL'),
        ]:
            if re.search(pat, line, re.IGNORECASE):
                prod_label = label; break

        accounts.append({'dpd': dpd_val, 'status': status, 'product': prod_label or 'others'})

    # Aggregate
    for acc in accounts:
        d, s = acc['dpd'], acc['status']
        if d >= 90:   dpd_90 += 1
        elif d >= 60: dpd_60 += 1
        elif d >= 30: dpd_30 += 1
        if any(x in s for x in ('written', 'npa', 'loss', 'lss', 'wo')): written_off += 1
        elif 'settled' in s: settled += 1
        elif any(x in s for x in ('active', 'standard', 'std')): active += 1
        if d >= 30 or any(x in s for x in ('sub', 'doubtful', 'dbt', 'npa', 'sma')): sub_std += 1

    # If nothing parsed, fall back to keyword scan
    if not accounts:
        written_off = len(re.findall(r'\bwritten[-\s]?off\b|\bNPA\b', txt, re.IGNORECASE))
        settled     = len(re.findall(r'\bsettled\b', txt, re.IGNORECASE))
        dpd_90      = len(re.findall(r'\b090\b|\b120\b|\b150\b|\b180\b', txt))
        dpd_60      = len(re.findall(r'\b060\b', txt))
        dpd_30      = len(re.findall(r'\b030\b', txt))
        active      = min(len(re.findall(r'\bactive\b', txt, re.IGNORECASE)), 15)

    total = max(len(accounts), active + settled + written_off, 1)
    return dict(
        accounts=accounts,
        dpd_90_count=dpd_90, dpd_60_count=dpd_60, dpd_30_count=dpd_30,
        written_off_count=written_off, settled_count=settled,
        active_count=active, sub_std=sub_std,
        total_accounts=total, pct_active=active / total,
    )

# ---------------------------------------------------------------------------
# BUILD FULL RESULT DICT (shared by both paths)
# ---------------------------------------------------------------------------
def _build_result(credit_score, age, gender, marital_status, education,
                  monthly_income, emp_type, tenure_months, business_vintage,
                  existing_emi, cc_util_frac, pl_util_frac,
                  acc, enq_data, txt, extraction_method):
    """
    Assemble the complete extraction result dict from parsed components.
    Called by both the pdfplumber and OCR paths.
    """
    dpd_90_count      = acc['dpd_90_count']
    dpd_60_count      = acc['dpd_60_count']
    dpd_30_count      = acc['dpd_30_count']
    written_off_count = acc['written_off_count']
    settled_count     = acc['settled_count']
    active_count      = acc['active_count']
    sub_std           = acc['sub_std']
    total_accounts    = acc['total_accounts']
    pct_active        = acc['pct_active']

    # Sanity: very high score + bad history → cap score
    if credit_score >= 750 and (written_off_count > 0 or dpd_90_count > 0):
        credit_score = min(credit_score, 550)

    # cc_util for categorical flag inference (0-1 fraction)
    cc_util_for_flags = cc_util_frac  # already 0-1

    # Net surplus
    net_cash_surplus = int(_infer_surplus_from_cibil(
        credit_score, dpd_60_count, dpd_30_count, float(monthly_income),
        dpd_90=dpd_90_count))

    inward_bounce = dpd_90_count + dpd_60_count
    salary_missing = 0

    # Categorical flags — always computed
    _inf = infer_categorical_flags({
        'Credit_Score': credit_score,
        'num_times_30p_dpd': dpd_30_count,
        'num_times_60p_dpd': dpd_60_count,
        'num_lss': written_off_count,
        'num_dbt': dpd_90_count,
        'CC_utilization': cc_util_for_flags,
        'NETMONTHLYINCOME': monthly_income,
        'Time_With_Curr_Empr': tenure_months,
        'dpd_90_count_6m': dpd_90_count,
        'inward_bounce_count_3m': inward_bounce,
        'salary_missing_months': salary_missing,
        'net_cash_surplus_6m': net_cash_surplus,
    })

    # Delinquency derived fields
    num_times_30p_dpd = dpd_30_count + dpd_60_count + dpd_90_count
    num_times_60p_dpd = dpd_60_count + dpd_90_count
    recent_level      = max(dpd_90_count*90, dpd_60_count*60, dpd_30_count*30)
    num_deliq         = dpd_30_count + dpd_60_count + dpd_90_count
    max_deliq         = -99999 if num_deliq == 0 else recent_level

    # Enquiry shortcuts
    enq = enq_data
    PL_enq = enq['PL_enq']; CC_enq = enq['CC_enq']

    # Product flags
    CC_Flag = 1 if re.search(r'credit\s+card', txt, re.IGNORECASE) else 0
    PL_Flag = 1 if re.search(r'personal\s+loan', txt, re.IGNORECASE) else 0
    HL_Flag = 1 if re.search(r'home\s+loan|housing', txt, re.IGNORECASE) else 0
    GL_Flag = 1 if re.search(r'gold\s+loan', txt, re.IGNORECASE) else 0

    # Stage-1 fields
    s1 = {
        'AMT_INCOME_TOTAL':           monthly_income * 12,
        'AMT_ANNUITY':                existing_emi if existing_emi > 0 else int(monthly_income * 0.25),
        'avg_salary_6m':              float(monthly_income),
        'salary_txn_count_6m':        6.0,
        'salary_amount_cv':           0.05 if emp_type == 'Salaried' else 0.25,
        'salary_date_std':            2.0,
        'salary_creditor_consistent': 1.0 if emp_type == 'Salaried' else 0.7,
        'salary_missing_months':      float(salary_missing),
        'dpd_15_count_6m':            0.0,
        'dpd_30_count_6m':            float(dpd_30_count),
        'dpd_90_count_6m':            float(dpd_90_count),
        'dpd_60_count_6m':            float(dpd_60_count),
        'max_dpd_6m':                 float(recent_level),
        'dpd_30_count_3m':            float(dpd_30_count),
        'total_payments_6m':          3.0,
        'total_late_15_6m':           0.0,
        'total_late_30_6m':           float(dpd_30_count),
        'total_late_60_6m':           float(dpd_60_count),
        'total_late_90_6m':           float(dpd_90_count),
        'max_days_late_6m':           float(recent_level),
        'avg_days_late_6m':           float(dpd_30_count*10 + dpd_60_count*20 + dpd_90_count*40) / max(total_accounts, 1),
        'total_late_30_3m':           float(dpd_30_count),
        'total_late_90_3m':           float(dpd_90_count),
        'avg_balance_cc':             0.0, 'total_drawings_cc': 0.0,
        'avg_credit_limit':           0.0,
        'max_utilization':            cc_util_frac,
        'total_payments_cc':          0.0, 'dpd_count_cc': 0.0,
        'avg_balance_pos':            0.0, 'dpd_count_pos': 0.0,
        'total_credit_activity':      float(total_accounts),
        'total_dpd_count':            float(num_times_30p_dpd),
        'avg_monthly_balance_6m':     float(net_cash_surplus),
        'total_emi_monthly':          float(existing_emi if existing_emi > 0 else int(monthly_income * 0.25)),
        'net_cash_surplus_6m':        float(net_cash_surplus),
        'total_credit_6m':            0.0,
        'total_debit_6m':             0.0,
        'inward_bounce_count_3m':     float(inward_bounce),
        'recent_payment_stress':      float(dpd_30_count + dpd_60_count),
        'active_loans_count':         float(active_count),
        'bureau_score':               float(credit_score),
    }

    # Stage-2 fields
    s2 = {
        'Credit_Score': credit_score, 'AGE': age,
        'GENDER': gender, 'MARITALSTATUS': marital_status,
        'EDUCATION': education, 'NETMONTHLYINCOME': monthly_income,
        'Time_With_Curr_Empr': tenure_months,
        'num_times_delinquent': num_deliq,
        'max_delinquency_level': recent_level,
        'max_recent_level_of_deliq': max(dpd_60_count*60, dpd_30_count*30),
        'num_deliq_6mts': num_deliq, 'num_deliq_12mts': num_deliq,
        'num_deliq_6_12mts': 0,
        'max_deliq_6mts': max_deliq, 'max_deliq_12mts': max_deliq,
        'num_times_30p_dpd': num_times_30p_dpd,
        'num_times_60p_dpd': num_times_60p_dpd,
        'recent_level_of_deliq': recent_level,
        'num_std': active_count, 'num_std_6mts': active_count, 'num_std_12mts': active_count,
        'num_sub': sub_std, 'num_sub_6mts': 0, 'num_sub_12mts': sub_std,
        'num_dbt': dpd_90_count, 'num_dbt_6mts': 0, 'num_dbt_12mts': dpd_90_count,
        'num_lss': written_off_count, 'num_lss_6mts': 0, 'num_lss_12mts': written_off_count,
        'time_since_recent_payment': -99999,
        'time_since_first_deliquency': -99999 if num_deliq == 0 else 365,
        'time_since_recent_deliquency': -99999 if num_deliq == 0 else 90,
        'tot_enq': enq['tot_enq'], 'enq_L3m': enq['enq_L3m'],
        'enq_L6m': enq['enq_L6m'], 'enq_L12m': enq['enq_L12m'],
        'time_since_recent_enq': enq['time_since_recent_enq'],
        'CC_enq': CC_enq, 'CC_enq_L6m': 0, 'CC_enq_L12m': 0,
        'PL_enq': PL_enq, 'PL_enq_L6m': 0, 'PL_enq_L12m': 0,
        'pct_of_active_TLs_ever': round(pct_active, 3),
        'pct_opened_TLs_L6m_of_L12m': 0.3,
        'pct_currentBal_all_TL': 0.3,
        'pct_PL_enq_L6m_of_L12m': 0, 'pct_CC_enq_L6m_of_L12m': 0,
        'pct_PL_enq_L6m_of_ever': 0, 'pct_CC_enq_L6m_of_ever': 0,
        'CC_utilization': cc_util_frac if cc_util_frac > 0 else -99999,
        'PL_utilization': pl_util_frac,
        'CC_Flag': CC_Flag, 'PL_Flag': PL_Flag, 'HL_Flag': HL_Flag, 'GL_Flag': GL_Flag,
        'max_unsec_exposure_inPct': int(cc_util_frac * 100) if cc_util_frac > 0 else 0,
        'last_prod_enq2': enq['last_prod_enq2'],
        'first_prod_enq2': enq['first_prod_enq2'],
    }

    recent_deliq_flag = 1 if (dpd_90_count > 0 or dpd_60_count > 0) else 0

    return {
        **s1, **s2,
        'existing_emi':              existing_emi if existing_emi > 0 else s1['total_emi_monthly'],
        'employment_type':           emp_type,
        'business_vintage_years':    business_vintage,
        'credit_utilization_pct':    int(cc_util_frac * 100),
        'salary_stability_flag':     _inf['salary_stability_flag'],
        'payment_discipline_flag':   _inf['payment_discipline_flag'],
        'cashflow_health':           _inf['cashflow_health'],
        'liquidity_flag':            _inf['liquidity_flag'],
        'bureau_risk_flag':          _inf['bureau_risk_flag'],
        'written_off_count':         written_off_count,
        'settled_count':             settled_count,
        'high_util_flag':            1 if cc_util_frac > 0.75 else 0,
        'credit_hungry':             1 if enq['enq_L3m'] >= 5 else 0,
        'delinq_severity_score':     dpd_90_count * 3 + dpd_30_count,
        'high_dpd_risk':             1 if dpd_90_count >= 2 else 0,
        'recent_deliq_flag':         recent_deliq_flag,
        'account_quality_score':     max(0, 100 - written_off_count*20
                                         - settled_count*10 - dpd_90_count*15
                                         - dpd_30_count*5),
        '_surplus_proxy':            int(net_cash_surplus),
        'raw_text':                  txt,
        'success':                   True,
        'extraction_method':         extraction_method,
    }

# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def extract_cibil_from_pdf(uploaded_file) -> dict:
    """
    Extract all CIBIL fields from an uploaded PDF.
    Tries pdfplumber first (digital PDFs), falls back to OCR.
    """
    if not OCR_AVAILABLE and not PDFPLUMBER_AVAILABLE:
        return {'success': False, 'error': 'Neither pdfplumber nor OCR libraries are available.'}

    pdf_bytes = uploaded_file.read()

    # ── pdfplumber path (digital PDF) ────────────────────────────────────────
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

            if full_text.strip():
                # All parsers work on full_text — no column splitting needed
                credit_score  = _extract_credit_score(full_text)
                age           = _extract_age(full_text)
                gender        = _extract_gender(full_text)
                marital       = _extract_marital(full_text)
                education     = _extract_education(full_text)
                monthly_income = _extract_income(full_text)
                emp_type, tenure, vintage = _extract_employment(full_text)
                existing_emi  = _extract_existing_emi(full_text)
                cc_util       = _extract_cc_util(full_text)
                pl_util       = _extract_pl_util(full_text)
                acc           = _parse_accounts(full_text)
                enq           = _parse_enquiries(full_text)

                return _build_result(
                    credit_score, age, gender, marital, education,
                    monthly_income, emp_type, tenure, vintage,
                    existing_emi, cc_util, pl_util,
                    acc, enq, full_text, 'pdfplumber+v5.5'
                )
        except Exception:
            pass  # fall through to OCR

    # ── OCR fallback (scanned PDFs) ──────────────────────────────────────────
    if not OCR_AVAILABLE:
        return {'success': False, 'error': OCR_ERROR_MSG or 'OCR not available'}

    try:
        txt = _ocr_pdf(pdf_bytes)

        credit_score   = _extract_credit_score(txt)
        age            = _extract_age(txt)
        gender         = _extract_gender(txt)
        marital        = _extract_marital(txt)
        education      = _extract_education(txt)
        monthly_income = _extract_income(txt)
        emp_type, tenure, vintage = _extract_employment(txt)
        existing_emi   = _extract_existing_emi(txt)
        cc_util        = _extract_cc_util(txt)
        pl_util        = _extract_pl_util(txt)
        acc            = _parse_accounts(txt)
        enq            = _parse_enquiries(txt)

        return _build_result(
            credit_score, age, gender, marital, education,
            monthly_income, emp_type, tenure, vintage,
            existing_emi, cc_util, pl_util,
            acc, enq, txt, 'OCR+MultiPass+v5.5'
        )

    except Exception as e:
        return {
            'error': str(e),
            'message': f'Error extracting CIBIL data: {str(e)}',
            'traceback': traceback.format_exc(),
            'success': False,
        }
