# 💳 Credit Risk Assessment Platform

A **production-grade hybrid credit decision system** that combines **policy rules + machine learning + affordability analysis** to assess loan applications in a way that is **explainable, auditable, and regulator-ready**.

This project demonstrates how **banks, NBFCs, and fintech lenders** actually make credit decisions — not just a pure ML prediction app.

![Version](https://img.shields.io/badge/version-8.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

---

## 🎯 Key Features

* ✅ **Hybrid Decisioning**: Rule-based policy engine + ML risk model + affordability overlay
* 📊 **Explainable Decisions**: Human-readable reason codes for every decision
* 📝 **Audit Ready**: Complete decision logs with full traceability
* 💰 **Affordability Engine**: FOIR calculation & income-obligation analysis
* 👥 **Manual Review Support**: Refer/override flows with mandatory comments
* 🎨 **Professional UI**: Modern white-themed Streamlit dashboard
* 🔢 **Risk Scoring**: 0-1000 risk score + PD (Probability of Default) percentage
* 🏷️ **Reason Code System**: Automated generation of top 3 decision factors
* 📄 **Complete Audit Logs**: JSON-formatted decision trails for regulatory compliance

---

## 🧠 Decision Architecture

The system implements a **three-layer policy-first credit decision flow**:

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: HARD POLICY GATES (Auto-Reject if Failed)        │
├─────────────────────────────────────────────────────────────┤
│  • Age Validation (18-65 salaried, 18-70 self-employed)    │
│  • KYC Verification Status                                  │
│  • Bankruptcy & Fraud Checks                                │
│  • Minimum Income (₹15,000/month)                          │
│  • Employment Stability (6+ months salaried, 2+ years self) │
│  • Credit Bureau Score (≥550)                               │
│  • Severe Delinquency (0 instances of 90+ DPD)             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: ML RISK ASSESSMENT                                │
├─────────────────────────────────────────────────────────────┤
│  • Random Forest Classifier                                 │
│  • Trained on 60,000+ applications                         │
│  • Multi-feature risk prediction                           │
│  • Confidence scoring                                       │
│  • Class probabilities (Approve/Review/Reject)             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: AFFORDABILITY OVERLAY                             │
├─────────────────────────────────────────────────────────────┤
│  • EMI Calculation (reducing balance method)                │
│  • FOIR Calculation (max 50%, recommended 40%)             │
│  • Net Disposable Income Check                             │
│  • Debt Burden Assessment                                   │
│  • Can override ML approval if FOIR > 45%                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
              FINAL DECISION + REASON CODES
```

👉 **Rules always override ML** — exactly how real lenders operate.

---

## 📊 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend/UI** | Streamlit 1.31+ |
| **ML Model** | Random Forest (scikit-learn) |
| **Risk Scoring** | Multi-factor weighted scoring |
| **Explainability** | Rule-based reason code generation |
| **Data Processing** | Pandas, NumPy |
| **Visualizations** | Plotly |
| **Styling** | Custom CSS (Professional white theme) |

---

## 📁 Project Structure

```
CREDIT_RISK_ENGINE/
│
├── 📄 test.py                          # Main Streamlit application
├── 📄 requirements.txt                 # Python dependencies
├── 📄 README.md                        # This file
│
├── 📂 models/
│   └── credit_risk_assets.pkl          # Trained Random Forest model
│
├── 📂 data/
│   ├── raw/                            # Original datasets
│   └── processed/                      # Engineered features
│
├── 📂 logs/
│   └── audit_logs/                     # Detailed decision breakdowns (JSON)
│
└── 📂 notebooks/
    └── model_training.ipynb            # Model development & training
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/zenmeraki7/credit-risk-engine.git
cd credit-risk-engine

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run test.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 💻 Usage

### Single Application Assessment

1. **Navigate to Assessment Page**: Click "👤 Assessment" in the sidebar

2. **Fill Application Form** with customer details:
   - **Identity & Eligibility**: Age, employment type, KYC status, tenure
   - **Credit Bureau**: Bureau score, DPD, utilization, inquiries
   - **Income & Financial**: Monthly income, loan amount, tenure, interest rate
   - **Existing Obligations**: Current EMI obligations

3. **Submit & View Results**: Click "🔍 Assess Credit Risk"

4. **Review Decision** across 4 tabs:
   - **📋 Application**: Summary of submitted data
   - **📊 Decision**: Visual decision summary with assessment cards
   - **🔍 Analysis**: Model confidence & probability charts
   - **📝 Audit**: Complete JSON audit trail

5. **Download Reports**: Download audit log (JSON format)

### Decision Output

The decision summary displays:

- **Decision Header**: Status (Approved/Rejected/Review), risk score (0-1000), PD percentage
- **Three Assessment Cards**: 
  - Identity & Eligibility (with pass/fail indicators)
  - Credit Bureau (with risk level badges)
  - Affordability (with FOIR breakdown)
- **Reason Codes**: Top 3 factors influencing the decision
- **Complete Audit Trail**: All policy checks, ML model version, timestamps

---

## 🧠 ML Risk Engine

### Model Specifications

| Attribute | Value |
|-----------|-------|
| **Algorithm** | Random Forest Classifier |
| **Training Samples** | 60,000+ loan applications |
| **Features** | 50+ engineered features |
| **Output** | 3-class prediction (Approve/Review/Reject) |
| **Confidence Scoring** | Probability distribution across classes |

### Key Features

**Credit Behavior**
- Bureau score (normalized)
- Credit utilization ratio
- Number of active loans
- Recent credit inquiries
- DPD history (15/30/90 days)
- Payment history score

**Income & Affordability**
- Monthly income
- Debt-to-Income ratio
- EMI-to-Income ratio
- Net cash surplus
- Income stability flag

**Employment & Stability**
- Employment tenure (months)
- Employment type
- Salary stability indicator
- Industry sector

**Loan Characteristics**
- Loan amount requested
- Tenure requested
- Loan-to-income ratio

---

## 🎯 Decision Logic & Thresholds

### Hard Policy Rules (Auto-Reject)

| Rule | Threshold |
|------|-----------|
| Age (Salaried) | 18-65 years |
| Age (Self-Employed) | 18-70 years |
| Minimum Income | ₹15,000/month |
| Employment Tenure (Salaried) | ≥6 months |
| Business Vintage (Self-Employed) | ≥2 years |
| Minimum Bureau Score | 550 |
| Maximum DPD 90+ | 0 instances |

### Affordability Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| FOIR | ≤40% | Excellent - Auto Approve |
| FOIR | 40-50% | Acceptable - Approve with caution |
| FOIR | >50% | Over-leveraged - Reject |
| Net Disposable Income | ≥₹10,000 | Minimum cushion required |

### Risk Score Bands

| Score Range | Risk Band | Typical Action |
|-------------|-----------|----------------|
| 750-1000 | Very Low Risk | Auto Approve |
| 650-749 | Low-Medium Risk | Manual Review |
| 550-649 | Medium-High Risk | Manual Review |
| 0-549 | High Risk | Reject |

---

## 🧾 Explainability & Reason Codes

### Approval Reasons (Examples)

✅ Excellent credit score (780)  
✅ Stable employment history (36 months)  
✅ Affordable EMI burden (FOIR: 35%)  
✅ Clean payment history (No DPD)  
✅ Strong monthly income (₹75,000)  
✅ Low credit utilization (25%)

### Rejection Reasons (Examples)

❌ Credit score below minimum (540 < 550)  
❌ EMI burden too high (FOIR: 55% > 50%)  
❌ Severe payment delays (2 instances of 90+ DPD)  
❌ Income below minimum threshold (₹12,000 < ₹15,000)  
❌ Insufficient employment tenure (4 months < 6)  
❌ Active bankruptcy detected

### Review Reasons (Examples)

⚠️ Credit score in borderline range (680)  
⚠️ EMI burden moderate (FOIR: 45%)  
⚠️ Recent employment change requiring verification  
⚠️ Mixed credit indicators requiring human review

---

## 📝 Audit & Logging

### Every Application Records:

```json
{
  "application_id": "PL20250131123456",
  "timestamp": "2025-01-31T12:34:56",
  "customer_id": "CUST001",
  
  "policy_checks": {
    "age": "✅ Age 32 (Valid)",
    "kyc": "✅ KYC Verified",
    "income": "✅ Income ₹50,000",
    "bureau": "✅ Bureau Score 720",
    "dpd": "✅ No 90+ DPD"
  },
  
  "ml_assessment": {
    "model_name": "Random Forest",
    "model_version": "8.0",
    "confidence": 85.5,
    "class_probabilities": {
      "APPROVE": 85.5,
      "REVIEW": 12.3,
      "REJECT": 2.2
    }
  },
  
  "affordability": {
    "foir_percentage": 38.5,
    "total_emi": 19250,
    "net_disposable": 30750,
    "affordable": true
  },
  
  "final_decision": {
    "decision": "APPROVE",
    "risk_score": 742,
    "pd_percentage": 2.8,
    "reason_codes": [
      "Excellent credit score (720)",
      "Affordable EMI burden (FOIR: 38.5%)",
      "Clean payment history"
    ]
  }
}
```

### Audit Trail Features

✅ **Full Traceability**: Every decision can be reproduced  
✅ **Model Versioning**: Know which model made which decision  
✅ **Timestamp Tracking**: Exact decision time recorded  
✅ **Reason Code Storage**: Explanations stored with decision  
✅ **Override Logging**: Manual reviewer decisions tracked  
✅ **Regulatory Compliance**: RBI/Basel audit-ready format

---

## 🔒 Security & Compliance

### Data Protection

- ✅ PAN/Aadhaar masking in logs (show only last 4 digits)
- ✅ Role-based access control framework
- ✅ Audit trail immutability
- ✅ Secure data handling practices

### Regulatory Alignment

- ✅ **RBI Guidelines**: FOIR limits, minimum income thresholds
- ✅ **Fair Lending**: No discriminatory features (gender, caste, religion blocked)
- ✅ **Explainability**: Reason codes for all rejections
- ✅ **Right to Know**: Customers informed of decision basis
- ✅ **Grievance Mechanism**: Reference numbers in audit logs

### Feature Governance

**Blocked Features** (Never Used):
- Gender, sex, marital status
- Religion, caste, race, ethnicity
- Political affiliation
- Disability, health conditions
- Sexual orientation

**Allowed Features**:
- Credit score, income, employment
- Debt obligations, payment history
- Age (within legal bounds)
- Geographic tier (for risk segmentation only)

---

## 📊 Performance Metrics

### Model Performance

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.82 |
| **Gini Coefficient** | 0.64 |
| **KS Statistic** | 0.45 |
| **Precision (Approve)** | 0.78 |
| **Recall (Approve)** | 0.85 |

### System Performance

| Metric | Target | Current |
|--------|--------|---------|
| **Avg Decision Time** | <2s | 1.2s ✅ |
| **Fallback Rate** | <5% | 2.1% ✅ |
| **Manual Review Rate** | 10-20% | 15% ✅ |
| **Approval Rate** | 40-50% | 45% ✅ |

---

## 📖 Core Components

### System Modules

**Affordability Engine** (`calculate_affordability()`)
- EMI calculation using reducing balance method
- FOIR percentage calculation
- Net disposable income computation
- Affordability status determination

**Reason Code Generator** (`generate_reason_codes()`)
- Automated reason generation based on decision
- Top 3 factor extraction
- Human-readable templates for all scenarios

**Risk Scoring** (`calculate_final_risk_score()`)
- Multi-factor weighted scoring (0-1000 scale)
- Bureau score contribution (40%)
- ML confidence contribution (40%)
- FOIR contribution (20%)

**Decision Engine** (`make_hybrid_decision_enhanced()`)
- Three-layer policy-first architecture
- Complete policy gate validation
- ML model integration
- Affordability overlay
- Final decision synthesis

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new features

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 Zen Meraki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👨‍💻 Author

**Zen Meraki**

- 🌐 GitHub: [@zenmeraki7](https://github.com/zenmeraki7)
- 📧 Email: contact@zenmeraki.dev
- 💼 LinkedIn: [Zen Meraki](https://linkedin.com/in/zenmeraki)

---

## 🙏 Acknowledgments

- Inspired by real-world credit risk systems at leading NBFCs and fintech companies
- Built with insights from RBI guidelines and Basel III framework
- UI/UX inspired by modern fintech applications
- Community feedback from credit risk professionals

---

## ⭐ Project Philosophy

> **This project focuses on real-world credit decisioning, not just ML accuracy.**

### Why This Matters

Most ML projects predict outcomes. This project **makes business decisions** with:

1. **Regulatory Compliance**: Built-in policy rules aligned with RBI guidelines
2. **Explainability**: Every decision has clear, human-readable reasons
3. **Auditability**: Complete trail for regulatory scrutiny
4. **Risk Management**: Multi-layer defense (policy + ML + affordability)
5. **Production Readiness**: Handles edge cases, fallbacks, and overrides

### Key Differentiators

| Traditional ML App | This Credit Risk Engine |
|-------------------|-------------------------|
| ✅ Predicts risk | ✅ Makes credit decisions |
| ❌ Black box output | ✅ Explainable reason codes |
| ❌ No policy enforcement | ✅ Hard rule gates |
| ❌ ML overrides everything | ✅ Rules can override ML |
| ❌ No audit trail | ✅ Complete decision logs |
| ❌ Single-layer decision | ✅ Three-layer architecture |

---

## 🎓 Learning Outcomes

Understanding this system means understanding:

✅ How banks actually approve/reject loans  
✅ Why ML alone isn't enough for lending  
✅ How policy rules interact with AI models  
✅ What regulators expect from credit systems  
✅ How to build explainable AI for finance  
✅ Production-grade software architecture  
✅ Audit and compliance requirements  

---

## 📞 Support

### Issues & Bug Reports

If you encounter any issues:
1. Check existing [Issues](https://github.com/zenmeraki7/credit-risk-engine/issues)
2. Create a new issue with clear description, steps to reproduce, and expected vs actual behavior

### Questions

For questions about usage or implementation:
- Email: support@zenmeraki.dev
- GitHub Discussions: [Ask a question](https://github.com/zenmeraki7/credit-risk-engine/discussions)

---

## 📈 Project Stats

![GitHub stars](https://img.shields.io/github/stars/zenmeraki7/credit-risk-engine?style=social)
![GitHub forks](https://img.shields.io/github/forks/zenmeraki7/credit-risk-engine?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/zenmeraki7/credit-risk-engine?style=social)

---

<div align="center">

**Made with ❤️ for the fintech community**

If you find this project useful, please consider giving it a ⭐!

[⬆ Back to Top](#-credit-risk-assessment-platform)

</div>
