# Credit Risk Assessment System

A production-ready loan decision engine combining machine learning with rule-based logic for intelligent, explainable credit decisions.

## Overview

This system provides automated loan approval decisions using a hybrid approach that combines Random Forest classification with regulatory compliance rules and affordability checks. Built with Python and deployed as an interactive Streamlit web application.

### Key Features

- **Intelligent Decision Making**: Random Forest model trained on 60,000 loan applications
- **Rule-Based Safety**: Automatic rejection of high-risk applications based on credit bureau scores and delinquency history
- **Affordability Verification**: Debt-to-income ratio validation ensures responsible lending
- **Explainable Results**: Confidence scores and decision reasoning for every prediction
- **Batch Processing**: Evaluate hundreds of applications simultaneously
- **Web Interface**: User-friendly Streamlit dashboard for single and bulk assessments

## System Architecture

The system uses a three-layer decision framework:

1. **Hard Rules Layer**: Immediate rejection for applications that fail basic risk thresholds
2. **Machine Learning Layer**: Random Forest classifier analyzes 15 key financial indicators
3. **Affordability Overlay**: Post-approval DTI verification to ensure sustainable repayment

### Decision Outcomes

- **APPROVE**: Low-risk applicants with strong credit profiles and manageable debt levels
- **REVIEW**: Borderline cases requiring manual underwriter assessment
- **REJECT**: High-risk applicants who fail to meet minimum requirements

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

Key packages include Streamlit, pandas, scikit-learn, plotly, and joblib.

## Getting Started

### Training the Model

Open and execute the Jupyter notebook to train the Random Forest classifier:

```bash
jupyter notebook notebooks/loan_prediction.ipynb
```

This generates the trained model file (`credit_risk_assets.pkl`) containing the classifier and feature encoders.

### Launching the Application

Start the web interface:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Testing the System

Validate model accuracy:

```bash
python validate_predictions.py
```

Run unit tests:

```bash
python test_decision_engine.py
```

## Using the Application

### Single Application Assessment

1. Navigate to the "Single Assessment" page
2. Enter applicant financial information
3. Receive instant decision with confidence score and explanation

### Batch Processing

1. Navigate to the "Batch Processing" page
2. Upload CSV file containing applicant data
3. Process all applications simultaneously
4. Download results with predictions and reasoning

### Model Insights

View feature importance rankings, decision thresholds, and model architecture details in the "Model Info" section.

## Decision Logic

### Automatic Rejection Criteria

Applications are immediately rejected if:
- Credit bureau score below 550
- Any delinquency of 90+ days in the past 6 months

### Machine Learning Evaluation

The Random Forest model analyzes 15 carefully selected features including:
- Credit bureau score
- Recent delinquency counts
- Monthly debt obligations
- Income stability indicators
- Cash flow metrics

### Affordability Check

Approved applications undergo final DTI verification:
- DTI ratio above 45% triggers manual review
- Ensures borrower can sustainably manage additional debt

## Model Performance

- **Training Dataset**: 60,000 loan applications
- **Features**: 15 predictive indicators
- **Algorithm**: Random Forest Classifier
- **Expected Accuracy**: 85-90%

### Decision Distribution

Typical prediction breakdown:
- 60% Approve
- 25% Review
- 15% Reject

## Data Requirements

### Input Features

All applications must include the following information:
- Credit bureau score
- Income details (amount and stability)
- Existing debt obligations
- Recent payment behavior
- Delinquency history

### CSV Format for Batch Processing

Upload files must contain all required feature columns with exact naming conventions. Column names are case-sensitive and must match the training data format.

## Best Practices

### Ensuring Accuracy

- Provide complete data for all required features
- Use consistent categorical value formatting
- Verify feature names match exactly
- Test with sample data before production use

### Interpreting Results

- **Confidence 80-100%**: Model has high certainty
- **Confidence 60-79%**: Model is moderately confident
- **Confidence <60%**: Model is uncertain (typically triggers review)

## Technical Notes

### Feature Engineering

The system uses the top 15 most predictive features to balance accuracy with simplicity. Additional features provide diminishing returns and risk overfitting.

### Categorical Encoding

Non-numeric features (such as employment status) are automatically encoded using label encoders stored in the model file.

### Model Updates

To retrain with new data:
1. Update the training dataset
2. Re-run the Jupyter notebook
3. Replace the existing model file
4. Restart the application

## Troubleshooting

### Low Prediction Accuracy

Verify that:
- Feature names in input data match training data exactly
- All required features are present
- Categorical values use consistent formatting
- Model file is current and not corrupted

### Application Errors

Common issues:
- Missing model file: Ensure training notebook has been executed
- Import errors: Verify all dependencies are installed
- Data format errors: Check CSV structure matches requirements

## Support

For issues or questions:
1. Review the validation results from test scripts
2. Check that all features are correctly formatted
3. Verify model file is present and accessible

## License

MIT License - Free for educational and commercial use

## Author

Zen Meraki  
January 2026

---

**Note**: This system is designed for educational and demonstration purposes. For production deployment in regulated lending environments, ensure compliance with all applicable laws and regulations including fair lending practices and data privacy requirements.
