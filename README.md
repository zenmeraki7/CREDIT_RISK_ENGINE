```markdown
# Credit Risk Assessment Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://your-app-url.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive credit risk evaluation system combining **hard policy rules**, **machine learning models**, and **affordability analysis** for accurate lending decisions. The dashboard provides a two-stage assessment pipeline with an intuitive UI and detailed audit trails.

![Dashboard Preview](docs/screenshot.png) <!-- Add actual screenshot later -->

## ✨ Features

### Stage 1 – Initial Assessment
- **Policy Gates** – Age, KYC, bankruptcy, fraud, income, employment tenure, bureau score, and delinquency checks.
- **ML Prediction** – Random Forest classifier trained on 60K+ samples with confidence scoring.
- **Affordability Analysis** – EMI calculation, FOIR (Fixed Obligation to Income Ratio), net disposable income.
- **PD Calculation** – Industry‑standard probability of default with dynamic adjustments (bureau, delinquency, FOIR, employment stability, inquiries, ML confidence).
- **Reason Code Generation** – Automated reasons for approve/reject/review decisions.
- **Audit Trail** – JSON export and optional PDF report.

### Stage 2 – CIBIL Deep Dive
- **Enhanced Data Input** – Manual entry, PDF upload (OCR), or batch analysis.
- **Tiered Risk Classification** – Predicts risk tier (P1/P2/P3/P4) and interest rate range.
- **OCR PDF Extraction** – Automatically extracts bureau data from uploaded CIBIL reports (requires Tesseract).
- **Combined Decision** – Overrides Stage 1 concerns with strong CIBIL profile.

### Batch Processing
- Upload CSV with multiple applications.
- Bulk predictions with decision distribution analytics.
- Download results as CSV or JSON.

### Modern UI/UX
- Sage green & yellow theme.
- Responsive layout with metric cards, gauges, and interactive charts.
- Streamlit‑native navigation and session state management.

## 🛠️ Technology Stack

| Component         | Technology                         |
|-------------------|------------------------------------|
| Frontend          | Streamlit 1.29.0                   |
| Machine Learning  | scikit‑learn 1.3.2 (Random Forest) |
| Data Processing   | pandas, numpy                      |
| Visualizations    | Plotly, Plotly Express              |
| PDF Generation    | ReportLab                           |
| OCR               | pytesseract, pdf2image, OpenCV      |
| Serialization     | joblib                              |

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip / uv
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for PDF upload)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/credit-risk-engine.git
cd credit-risk-engine/notebooks
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Tesseract OCR (System Dependency)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**  
Download installer from [GitHub UB‑Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.

> **Note for Streamlit Cloud / Deployments:**  
> Create a `packages.txt` file in the root with `tesseract-ocr` to install via `apt`.

### Step 5: Download Model Assets
Ensure the trained models are present:
- `credit_risk_assets.pkl` (Stage 1 Random Forest)
- `stage2_cibil_model.pkl` (Stage 2 CIBIL model) – place in `notebooks/` or project root.

If missing, run the training notebooks provided separately.

## ▶️ Running the App

```bash
streamlit run test.py
```

The app will open in your default browser at `http://localhost:8501`.

## 📖 Usage Guide

### Navigation
- **🏠 Home** – Overview and version info.
- **👤 Assessment** – Single application form.
- **🔬 Stage 2 Analysis** – Unlocks after an APPROVE/REVIEW decision (requires Stage 2 model).
- **📊 Batch Process** – Upload CSV for bulk predictions.
- **📈 Model Info** – Feature list and model details.
- **ℹ️ About** – Project information.

### Assessment Workflow
1. Fill the form (Identity, Credit Bureau, Income & Financial sections).
2. Click **"🔍 Assess Credit Risk"**.
3. View results in tabs: Application, Decision, Analysis, Audit.
4. If approved or under review, proceed to **Stage 2** via the buttons in the Decision tab.
5. In Stage 2, choose input method (Manual Entry, PDF Upload, Batch Analysis).
6. Download final PDF report containing Stage 1 & 2 results.

### OCR PDF Upload
- Go to Stage 2 → **PDF Upload**.
- Upload a CIBIL report PDF.
- The app extracts key fields using OCR and runs the deep dive analysis.
- Requires Tesseract and Python OCR packages (see Installation).

### Batch Processing
- Download the provided CSV template.
- Upload your file with required columns.
- Process and view decision distribution charts.
- Download results filtered by decision.

## 📁 Project Structure

```
credit-risk-engine/
├── notebooks/
│   ├── test.py                 # Main Streamlit app
│   ├── credit_risk_assets.pkl  # Stage 1 model
│   ├── stage2_cibil_model.pkl  # Stage 2 model (optional)
│   └── css_styles.py           # Custom CSS
├── utils/
│   └── pdf_generator.py        # PDF report generation
├── requirements.txt             # Python dependencies
├── packages.txt                 # System dependencies (for Tesseract)
└── README.md
```

## 📦 Dependencies

### Python Packages (requirements.txt)
```
joblib==1.3.2
streamlit==1.29.0
pandas==2.1.3
numpy==1.26.2
plotly==5.18.0
scikit-learn==1.3.2
reportlab==3.6.12
pytesseract
pdf2image
opencv-python
pillow
```

### System Dependencies (packages.txt for Streamlit Cloud)
```
tesseract-ocr
```

## ⚠️ Troubleshooting

### `st.set_page_config()` Error
- Ensure `st.set_page_config()` is called **only once** and **at the very top** of `test.py` (after imports, before any other Streamlit command).

### Duplicate Widget ID Error
- If you see "multiple identical st.button widgets", update unique `key` arguments. Our code uses `application_id` to generate keys in the Stage 2 navigation buttons.

### OCR Not Working
- Confirm Tesseract is installed (`tesseract --version`).
- Install Python packages: `pip install pytesseract pdf2image opencv-python pillow`.
- On Streamlit Cloud, ensure `packages.txt` exists with `tesseract-ocr`.

### Model Loading Fails
- Place `credit_risk_assets.pkl` in the same directory as `test.py`.
- For Stage 2, place `stage2_cibil_model.pkl` in the project root or `notebooks/`.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Zen Meraki**  
📅 January 2026  
📧 [your.email@example.com](mailto:your.email@example.com)

---

*This project is for educational and demonstration purposes.*
```

Just paste this into the GitHub editor and commit. Remember to replace the placeholder URL and author email if needed.
