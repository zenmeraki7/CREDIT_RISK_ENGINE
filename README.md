# 💳 Credit Risk Assessment System

A **hybrid ML + rule-based** loan decision engine built with Random Forest and deployed as a Streamlit web app.

---

## 🎯 What Does This Do?

Makes smart loan decisions using:
1. **Hard Rules** → Auto-reject risky applicants (bureau score < 550, severe delinquencies)
2. **ML Model** → Random Forest predicts APPROVE/REVIEW/REJECT
3. **Affordability Check** → Ensures debt-to-income ratio ≤ 45%

**Result:** Fast, accurate, and explainable lending decisions! ✅

---

## 📁 Project Structure

```
CREDIT_RISK_ENGINE/
├── 📊 DATA/
│   └── processed/
│       └── train_60k_rule_accepted.csv    # Your 60K training data
│
├── 📓 notebooks/
│   ├── loan_prediction.ipynb              # Training notebook
│   └── credit_risk_assets.pkl             # Trained model (generated)
│
├── 🐍 Python Scripts:
│   ├── app.py                             # 🌐 Streamlit web app (MAIN)
│   ├── extract_test_samples.py            # 🧪 Create test CSV
│   ├── validate_predictions.py            # ✅ Check accuracy
│   └── test_decision_engine.py            # 🔬 Unit tests
│
├── 📄 Generated Files:
│   ├── credit_risk_assets.pkl             # Trained model + encoders
│   ├── test_samples.csv                   # Test data (15-20 samples)
│   └── validation_results.csv             # Accuracy report
│
└── 📝 Documentation:
    ├── README.md                          # This file
    └── requirements.txt                   # Python dependencies
```

---

## 🚀 Quick Start (3 Steps)

### **Step 1: Train the Model**
```bash
# Open the notebook and run all cells
jupyter notebook notebooks/loan_prediction.ipynb
```
**Output:** Creates `credit_risk_assets.pkl` (your trained model)

---

### **Step 2: Test the System**
```bash
# Create test samples from your 60K dataset
python extract_test_samples.py

# Validate predictions (check accuracy)
python validate_predictions.py
```
**Output:** 
- `test_samples.csv` (ready to use)
- Shows accuracy % (should be >80%)

---

### **Step 3: Launch the Web App**
```bash
streamlit run app.py
```
**Opens in browser:** http://localhost:8501

---

## 🧪 How to Test If It's Working

### **Method 1: Quick Unit Test**
```bash
python test_decision_engine.py
```
**You should see:**
```
✅ Good Customer → APPROVE
❌ Bad Bureau Score → REJECT
❌ Severe Delinquency → REJECT
⚠️ High DTI → REVIEW
```

---

### **Method 2: Validate Against Real Data**
```bash
python validate_predictions.py
```
**You should see:**
```
📊 Overall Accuracy: 87.3% (14/16 correct)
✅ Correct Predictions: 14 cases
⚠️ Mismatches: 2 cases
```

**If accuracy < 70%:** Something's wrong with features!

---

### **Method 3: Test in Streamlit**
1. Run `streamlit run app.py`
2. Go to **"Batch Processing"** page
3. Upload `test_samples.csv`
4. Click **"Process All"**
5. Compare with `validation_results.csv`

**They should match!** ✅

---

## 📊 What the Streamlit App Can Do

### 🏠 **Home Page**
- Shows model info (60K training data, 15 features, accuracy)
- Lists all features the model uses

### 👤 **Single Assessment**
- Enter customer details manually
- Get instant decision (APPROVE/REVIEW/REJECT)
- See confidence score and reason

### 📊 **Batch Processing**
- Upload CSV with multiple customers
- Process hundreds at once
- Download results with predictions

### 📈 **Model Info**
- View feature importance
- Understand decision thresholds
- See model architecture

---

## 🔑 Key Features Your Model Uses

The model was trained to select the **top 15 most predictive features**. Common ones include:

- `bureau_score` - Credit score
- `dpd_90_count_6m` - Severe delinquencies (90+ days)
- `total_emi_monthly` - Total monthly loan payments
- `avg_salary_6m` - Average salary (6 months)
- `net_cash_surplus_6m` - Cash left after expenses
- `salary_stability_flag` - STABLE/MODERATE/UNSTABLE
- ... and 9 more

**To see YOUR exact features:**
```bash
python extract_test_samples.py
```

---

## 🛡️ Decision Rules

### **Hard Reject Rules (No ML needed)**
- Bureau score < 550 → **REJECT**
- Any 90+ day delinquency → **REJECT**

### **ML Model Prediction**
- Model analyzes all 15 features
- Outputs: APPROVE, REVIEW, or REJECT
- Includes confidence score (0-100%)

### **Affordability Overlay**
- Calculates DTI (Debt-to-Income) ratio
- If APPROVE but DTI > 45% → **REVIEW**

---

## 📝 Sample Test Cases

### ✅ **Should APPROVE**
```python
{
    'bureau_score': 720,
    'dpd_90_count_6m': 0,
    'total_emi_monthly': 15000,
    'avg_salary_6m': 50000,
    'salary_stability_flag': 'STABLE'
}
# DTI = 30% ✅
```

### ❌ **Should REJECT**
```python
{
    'bureau_score': 450,  # Below 550!
    'dpd_90_count_6m': 2  # Has severe delinquencies!
}
```

### ⚠️ **Should REVIEW**
```python
{
    'bureau_score': 720,
    'dpd_90_count_6m': 0,
    'total_emi_monthly': 30000,
    'avg_salary_6m': 50000  # DTI = 60% > 45%!
}
```

---

## 🔧 Troubleshooting

### **Problem: "credit_risk_assets.pkl not found"**
**Solution:**
```bash
# Make sure you ran the training notebook first!
# Then copy the file to the app directory
cp notebooks/credit_risk_assets.pkl .
```

---

### **Problem: "Low accuracy (<70%)"**
**Causes:**
- Feature names don't match between training and app
- Categorical encoding is wrong
- Missing important features

**Solution:**
```bash
# Check what features the model expects
python extract_test_samples.py

# Compare with your Streamlit form fields
# Update app.py to match exact feature names
```

---

### **Problem: "All predictions are REVIEW"**
**Cause:** Model is getting default values (0s or "Unknown")

**Solution:**
- Make sure your CSV has all required features
- Check feature names match exactly (case-sensitive!)
- Run `validate_predictions.py` to see which features are missing

---

## 📦 Installation

### **Requirements**
```bash
pip install -r requirements.txt
```

**Key packages:**
- `streamlit` - Web app framework
- `pandas` - Data processing
- `scikit-learn` - ML model
- `plotly` - Interactive charts
- `joblib` - Model saving/loading

---

## 🎓 Understanding the System

### **What is Hybrid Decision Engine?**
Instead of relying only on ML, we combine:
- **Rules** (regulatory compliance, common sense)
- **ML Model** (pattern recognition from 60K loans)
- **Business Logic** (affordability, risk tolerance)

### **Why 15 Features?**
More features ≠ better model! We selected the **top 15 most predictive** features to:
- Avoid overfitting
- Speed up predictions
- Reduce data collection burden

### **What Does Confidence Score Mean?**
- **80-100%**: Model is very sure about its decision
- **60-79%**: Model is moderately confident
- **<60%**: Model is uncertain (usually triggers REVIEW)

---

## 📊 Model Performance

**Trained on:** 60,000 loan applications  
**Features:** 15 (scientifically selected)  
**Algorithm:** Random Forest Classifier  
**Expected Accuracy:** 85-90%  

**Decision Breakdown:**
- ~60% APPROVE
- ~25% REVIEW  
- ~15% REJECT

---

## 🚨 Important Notes

1. **Feature Names Must Match Exactly**
   - Training CSV column names
   - Streamlit form field names
   - Test CSV column names
   - All must be identical!

2. **Categorical Features Need Encoding**
   - The model can't read "STABLE" directly
   - Must be converted to numbers (0, 1, 2)
   - This is handled automatically by `le_map`

3. **Missing Features = Bad Predictions**
   - If a feature is missing, it defaults to 0 or "Unknown"
   - This signals high risk to the model
   - Always provide all 15 features!

---

## 🤝 Contributing

Found a bug? Have suggestions?
1. Run `validate_predictions.py` first
2. Share the accuracy report
3. Describe what's not working
4. Include sample test case

---

## 📄 License

MIT License - Feel free to use for learning and commercial projects!

---

## 👨‍💻 Author

**Zen Meraki**  
January 2025

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Train model | Open `loan_prediction.ipynb` |
| Create test data | `python extract_test_samples.py` |
| Validate accuracy | `python validate_predictions.py` |
| Run unit tests | `python test_decision_engine.py` |
| Launch web app | `streamlit run app.py` |
| Check features | `python extract_test_samples.py` |

---

## ✨ That's It!

You now have a production-ready credit risk system. Happy lending! 🚀
