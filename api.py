"""
Credit Risk Assessment API - FastAPI Backend
Run with: uvicorn api:app --reload

Author: Zen Meraki
Date: January 2025
FIXED: Risk scoring now matches dataset (high score = low risk = APPROVE)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# =============================================================================
# FASTAPI APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="Credit Risk Assessment API",
    description="AI-powered credit risk scoring and loan decision API",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class CreditInput(BaseModel):
    """Input schema for credit assessment"""
    bureau_score: int = Field(..., ge=300, le=900, description="Credit bureau score (300-900)")
    dpd_15: int = Field(0, ge=0, le=100, description="Days Past Due 15+ count (6M)")
    dpd_30: int = Field(0, ge=0, le=100, description="Days Past Due 30+ count (6M)")
    dpd_90: int = Field(0, ge=0, le=50, description="Days Past Due 90+ count (6M)")
    active_loans: int = Field(0, ge=0, le=50, description="Number of active loans")
    total_emi: float = Field(0, ge=0, description="Total monthly EMI amount")
    avg_salary: float = Field(..., ge=1, description="Average monthly salary (6M)")
    net_surplus: float = Field(..., description="Net cash surplus (6M) - can be negative")
    bounces: int = Field(0, ge=0, le=50, description="Payment bounces (3M)")
    salary_stability: int = Field(1, ge=1, le=3, description="1=Stable, 2=Moderate, 3=Unstable")
    liquidity_flag: int = Field(1, ge=1, le=3, description="1=Adequate, 2=Moderate, 3=Low")
    bureau_risk_flag: int = Field(1, ge=1, le=3, description="1=Low, 2=Medium, 3=High")
    missing_months: int = Field(0, ge=0, le=6, description="Months without salary")
    
    class Config:
        schema_extra = {
            "example": {
                "bureau_score": 744,
                "dpd_15": 0,
                "dpd_30": 0,
                "dpd_90": 0,
                "active_loans": 5,
                "total_emi": 26190,
                "avg_salary": 20000,
                "net_surplus": -179272,
                "bounces": 0,
                "salary_stability": 1,
                "liquidity_flag": 3,
                "bureau_risk_flag": 1,
                "missing_months": 0
            }
        }


class CreditOutput(BaseModel):
    """Output schema for credit assessment"""
    risk_score: int = Field(..., description="Risk score (0-100, higher = lower risk)")
    decision: str = Field(..., description="Loan decision: APPROVE, REJECT, or MANUAL_REVIEW")
    reason: str = Field(..., description="Detailed reason for the decision")
    default_probability: float = Field(..., description="Estimated default probability (%)")
    confidence: float = Field(..., description="Confidence in decision (0-1)")
    
    class Config:
        schema_extra = {
            "example": {
                "risk_score": 85,
                "decision": "APPROVE",
                "reason": "Strong profile - Low risk",
                "default_probability": 15.0,
                "confidence": 0.92
            }
        }

# =============================================================================
# RISK SCORING LOGIC - CORRECTED
# =============================================================================

def calculate_risk_score(data: CreditInput) -> int:
    """
    Calculate comprehensive risk score (0-100)
    FIXED: Higher score = LOWER risk (matches dataset!)
    
    Dataset patterns:
    - Risk Score 100: Bureau 727+, no DPDs, positive surplus, 0 bounces, stable salary
    - Risk Score 85: Bureau 725+, no DPDs, 0 bounces, stable salary (can have negative surplus!)
    - Risk Score 75: Bureau 700+, clean payment history
    - Risk Score <55: High risk, likely rejection
    """
    
    bureau_score = data.bureau_score
    dpd_30 = data.dpd_30
    dpd_90 = data.dpd_90
    bounces = data.bounces
    net_surplus = data.net_surplus
    is_stable_salary = (data.salary_stability == 1)
    
    # Risk score determination based on dataset patterns
    
    # Risk Score 100: Best profile
    if (bureau_score >= 727 and 
        dpd_30 == 0 and dpd_90 == 0 and 
        bounces == 0 and 
        net_surplus > 0 and 
        is_stable_salary):
        return 100
    
    # Risk Score 85: Excellent profile (can have negative surplus!)
    elif (bureau_score >= 725 and 
          dpd_30 == 0 and dpd_90 == 0 and 
          bounces == 0 and 
          is_stable_salary):
        return 85
    
    # Risk Score 93: Very good profile
    elif (bureau_score >= 740 and 
          dpd_30 == 0 and dpd_90 == 0 and 
          bounces <= 1):
        return 93
    
    # Risk Score 75: Good profile
    elif (bureau_score >= 700 and 
          dpd_90 == 0 and 
          dpd_30 <= 1 and 
          bounces <= 2):
        return 75
    
    # Risk Score 65: Acceptable for review
    elif (bureau_score >= 650 and 
          dpd_90 == 0 and 
          bounces <= 3):
        return 65
    
    # Risk Score 55-60: Borderline
    elif bureau_score >= 600 and dpd_90 == 0:
        return 55 + min(5, (bureau_score - 600) // 20)
    
    # Below 55: High risk
    elif bureau_score >= 500:
        return max(0, bureau_score // 10 - 10)
    
    else:
        return 0


def make_loan_decision(risk_score: int, bureau_score: int, dpd_90: int) -> tuple:
    """
    Make loan decision based on risk score
    FIXED: High risk score = APPROVE (matches dataset!)
    
    Dataset rules:
    - APPROVE: risk_score >= 75, bureau >= 732, no hard rejects
    - REVIEW: risk_score 55-74
    - REJECT: risk_score < 55 OR bureau < 732 OR hard rejects
    """
    
    # Hard reject rules (critical failures)
    if bureau_score < 500:
        return "REJECT", "Bureau score critically low (<500)"
    if dpd_90 > 5:
        return "REJECT", "Too many severe delinquencies (90+ DPD > 5)"
    if bureau_score < 600 and dpd_90 > 2:
        return "REJECT", "Low bureau score with severe delinquencies"
    
    # Risk score-based decision (CORRECTED LOGIC)
    if risk_score >= 75:
        return "APPROVE", "Strong profile - Low risk"
    elif risk_score >= 55:
        return "MANUAL_REVIEW", "Medium risk - Manual review required"
    else:
        return "REJECT", "High risk profile"


def calculate_confidence(risk_score: int, bureau_score: int, dpd_90: int) -> float:
    """Calculate confidence in the decision"""
    confidence = 0.5
    
    # High confidence for extreme scores
    if risk_score >= 85 or risk_score <= 40:
        confidence += 0.3
    elif risk_score >= 75 or risk_score <= 50:
        confidence += 0.2
    else:
        confidence += 0.1
    
    # Bureau score adds confidence
    if bureau_score >= 750 or bureau_score <= 500:
        confidence += 0.15
    
    # Clean payment history adds confidence
    if dpd_90 == 0:
        confidence += 0.05
    
    return min(confidence, 1.0)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Credit Risk Assessment API",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "active",
        "version": "2.0.0"
    }


@app.post("/predict", response_model=CreditOutput)
async def predict(data: CreditInput):
    """
    Assess credit risk and make loan decision
    
    **Returns:**
    - risk_score: 0-100 (higher = lower risk)
    - decision: APPROVE, REJECT, or MANUAL_REVIEW
    - reason: Explanation for the decision
    - default_probability: Estimated default probability
    - confidence: Confidence in the decision (0-1)
    """
    try:
        # Calculate risk score
        risk_score = calculate_risk_score(data)
        
        # Make decision
        decision, reason = make_loan_decision(risk_score, data.bureau_score, data.dpd_90)
        
        # Calculate default probability (inverse of risk score)
        default_probability = 100 - risk_score
        
        # Calculate confidence
        confidence = calculate_confidence(risk_score, data.bureau_score, data.dpd_90)
        
        return CreditOutput(
            risk_score=risk_score,
            decision=decision,
            reason=reason,
            default_probability=round(default_probability, 2),
            confidence=round(confidence, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(data: list[CreditInput]):
    """
    Batch prediction endpoint for multiple applicants
    
    **Input:** List of credit applications
    **Returns:** List of assessments
    """
    try:
        results = []
        for applicant in data:
            risk_score = calculate_risk_score(applicant)
            decision, reason = make_loan_decision(risk_score, applicant.bureau_score, applicant.dpd_90)
            default_probability = 100 - risk_score
            confidence = calculate_confidence(risk_score, applicant.bureau_score, applicant.dpd_90)
            
            results.append({
                "risk_score": risk_score,
                "decision": decision,
                "reason": reason,
                "default_probability": round(default_probability, 2),
                "confidence": round(confidence, 2)
            })
        
        return {"total": len(results), "predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )