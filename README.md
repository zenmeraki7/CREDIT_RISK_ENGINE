# Credit Risk Assessment Engine
**Version 8.3 / Rule Audit v2.0** — Zen Meraki, January 2026

AI-powered loan decision system combining hard policy rules, Random Forest ML, and affordability analysis.

---

## Repo Structure

```
credit_risk_engine/
├── notebooks/
│   ├── test.py                  ← Main Streamlit app (run this)
│   ├── affordability_engine.py  ← EMI, FOIR, net disposable
│   ├── risk_engine.py           ← Risk score + utility functions
│   ├── reason_codes.py          ← Human-readable decision reasons
│   ├── decision_summary.py      ← UI card components
│   ├── stage2_engine.py         ← Stage 2 CIBIL deep-dive
│   └── utils/
│       ├── __init__.py
│       └── pdf_generator.py     ← PDF audit report generation
├── credit_risk_assets.pkl       ← Trained Stage 1 RF model
├── stage2_cibil_model.pkl       ← Trained Stage 2 tier model
├── CHANGELOG.md
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn plotly joblib \
            pytesseract pdf2image opencv-python pillow reportlab

# Run the app (from repo root or notebooks/)
streamlit run notebooks/test.py
```

---

## Decision Rules

### Stage 1 — Hard Reject Gates (any failure = instant REJECT)

| Rule | Threshold | Source |
|------|-----------|--------|
| Age | 24–70 years | `make_hybrid_decision_enhanced()` |
| KYC | Must be verified | `make_hybrid_decision_enhanced()` |
| Bankruptcy | Must be none | `make_hybrid_decision_enhanced()` |
| Fraud | Must be none | `make_hybrid_decision_enhanced()` |
| Monthly income | ≥ ₹15,000 | `make_hybrid_decision_enhanced()` |
| Salaried tenure | ≥ 6 months | `make_hybrid_decision_enhanced()` |
| Business vintage | ≥ 2 years | `make_hybrid_decision_enhanced()` |
| Bureau score | ≥ 550 | `make_hybrid_decision_enhanced()` |
| DPD 90+ | = 0 | `make_hybrid_decision_enhanced()` |

### Stage 1 — Soft Flags (APPROVE → REVIEW)

| Condition | Threshold | Action |
|-----------|-----------|--------|
| FOIR | > 50% | → REVIEW |
| Dependents | > 5 | → REVIEW |

### FOIR Bands (your original design)

| FOIR | Status | Color |
|------|--------|-------|
| ≤ 35% | Excellent | 🟢 Green |
| ≤ 40% | Acceptable | 🟡 Yellow |
| ≤ 50% | High – Review Required | 🟠 Orange |
| > 50% | Over-leveraged | 🔴 Red → REJECT |

### Stage 2 — Decision Matrix

| Stage 1 | Stage 2 Tier | Final Decision | Rate |
|---------|-------------|----------------|------|
| APPROVE | P1/P2 | APPROVE | 8.5–10% |
| APPROVE | P3 | APPROVE | 12–14% |
| APPROVE | P4 | REVIEW | TBD |
| REVIEW  | P1/P2 | APPROVE | 10–11% |
| REVIEW  | P3 | REVIEW | TBD |
| REVIEW  | P4 | REJECT | — |
| REJECT  | any | REJECT | — |

---

## PD Calculation

**Base PD from bureau score:**

| Score | Base PD |
|-------|---------|
| 800+ | 0.5–1% |
| 750–800 | 1–2% |
| 700–750 | 2–3.5% |
| 650–700 | 3.5–6% |
| 600–650 | 6–10% |
| 550–600 | 10–15% |
| < 550 | 15–25% |

**DPD multipliers:** 3×90DPD=×5, 2×90DPD=×3, 1×90DPD=×2, 3×30DPD=×1.6, 1×30DPD=×1.3

**FOIR adjustment:** ≤30%=−0.75%, ≤40%=0%, ≤45%=+0.75%, ≤50%=+1.5%, ≤55%=+2.25%, ≤60%=+3.5%, >60%=+6%

**PD capped between 0.5% and 25%.**

---

## Known Rule Gaps (Pending Policy Decision)

See `CHANGELOG.md` → "Rule Gaps Identified" section for 9 items awaiting Credit Policy team decision.

---

## Environment Variables / Tesseract

Set Tesseract path if not auto-detected:
```python
# In test.py before pytesseract calls:
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
```

---

## Model Assets

| File | Description |
|------|-------------|
| `credit_risk_assets.pkl` | Stage 1: RF model + feature list + label encoders |
| `stage2_cibil_model.pkl` | Stage 2: tier classifier + feature list + label encoder |

Place both `.pkl` files in the same directory as `test.py`, or at project root.
