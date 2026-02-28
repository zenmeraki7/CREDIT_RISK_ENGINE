# """
# PDF Summary Generator
# Generates downloadable decision summary reports
# """

# from turtle import st
# from reportlab.lib.pagesizes import A4
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
# from reportlab.lib import colors
# from reportlab.lib.units import inch
# import io


# def generate_decision_pdf(decision_data, customer_data, affordability_data, reasons):
#     """
#     Generate PDF summary of credit decision
#     """
    
#     buffer = io.BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=A4)
#     elements = []
#     styles = getSampleStyleSheet()
    
#     # Title
#     title = Paragraph(f"<b>Credit Decision Summary</b>", styles['Title'])
#     elements.append(title)
#     elements.append(Spacer(1, 0.3*inch))
    
#     # Application details
#     app_info = f"""
#     <b>Application ID:</b> {decision_data['application_id']}<br/>
#     <b>Decision Date:</b> {decision_data['timestamp']}<br/>
#     <b>Decision:</b> {decision_data['decision']}<br/>
#     <b>Risk Score:</b> {decision_data['risk_score']}/1000<br/>
#     <b>PD:</b> {decision_data['pd_percentage']}%
#     """
#     elements.append(Paragraph(app_info, styles['Normal']))
#     elements.append(Spacer(1, 0.3*inch))
    
#     # Customer details table
#     customer_table_data = [
#         ['Field', 'Value'],
#         ['Name', customer_data.get('name', 'N/A')],
#         ['Age', str(customer_data.get('age', 'N/A'))],
#         ['Bureau Score', str(customer_data.get('bureau_score', 'N/A'))],
#         ['Monthly Income', f"₹{customer_data.get('avg_salary_6m', 0):,}"],
#     ]
    
#     customer_table = Table(customer_table_data, colWidths=[3*inch, 3*inch])
#     customer_table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
#         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (-1, 0), 12),
#         ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
#         ('GRID', (0, 0), (-1, -1), 1, colors.black)
#     ]))
    
#     elements.append(customer_table)
#     elements.append(Spacer(1, 0.3*inch))
    
#     # Affordability breakdown
#     affordability_text = f"""
#     <b>Affordability Assessment:</b><br/>
#     Monthly Income: ₹{affordability_data['monthly_income']:,}<br/>
#     Total EMI: ₹{affordability_data['total_emi']:,}<br/>
#     FOIR: {affordability_data['foir_percentage']}%<br/>
#     Net Disposable: ₹{affordability_data['net_disposable']:,}
#     """
#     elements.append(Paragraph(affordability_text, styles['Normal']))
#     elements.append(Spacer(1, 0.3*inch))
    
#     # Reason codes
#     reasons_text = "<b>Decision Reasons:</b><br/>"
#     for i, reason in enumerate(reasons, 1):
#         reasons_text += f"{i}. {reason}<br/>"
    
#     elements.append(Paragraph(reasons_text, styles['Normal']))
    
#     # Build PDF
#     doc.build(elements)
#     buffer.seek(0)
    
#     return buffer


# # Usage in Streamlit
# def add_download_button(decision_data, customer_data, affordability_data, reasons):
#     """Add download button to Streamlit page"""
    
#     pdf_buffer = generate_decision_pdf(
#         decision_data, customer_data, affordability_data, reasons
#     )
    
#     st.download_button(
#         label="📥 Download Decision Summary",
#         data=pdf_buffer,
#         file_name=f"credit_decision_{decision_data['application_id']}.pdf",
#         mime="application/pdf",
#         use_container_width=True
#     )


##################################################################################




"""
PDF Generation Utility for Credit Risk Assessment
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime


def generate_decision_pdf(decision_data, customer_data, affordability_data, reasons):
    """
    Generate professional decision report PDF
    
    Args:
        decision_data: Dictionary with decision, risk_score, pd_percentage, confidence, etc.
        customer_data: Dictionary with customer information
        affordability_data: Dictionary with EMI, FOIR calculations
        reasons: List of reason codes
    
    Returns:
        BytesIO buffer containing the PDF
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#587042'),
        spaceAfter=12,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#587042'),
        spaceAfter=6
    )
    
    # Build content
    story = []
    
    # Title
    story.append(Paragraph("CREDIT DECISION REPORT", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Header Info
    decision = decision_data.get('decision', 'ERROR')
    app_id = customer_data.get('application_id', 'N/A')
    timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    header_data = [
        ['Application ID:', app_id, 'Timestamp:', timestamp],
        ['Decision:', decision, 'Risk Score:', f"{decision_data.get('risk_score', 0)}/1000"],
        ['PD Score:', f"{decision_data.get('pd_percentage', 0)}%", 'Confidence:', f"{decision_data.get('confidence', 0):.1f}%"]
    ]
    
    header_table = Table(header_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f7fafc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Decision Status Box
    if decision == "APPROVE":
        decision_color = colors.HexColor('#48bb78')
        decision_icon = "✓ APPROVED"
    elif decision == "REJECT":
        decision_color = colors.HexColor('#f56565')
        decision_icon = "✗ REJECTED"
    else:
        decision_color = colors.HexColor('#ed8936')
        decision_icon = "⚠ REVIEW REQUIRED"
    
    decision_style = ParagraphStyle(
        'DecisionStatus',
        parent=styles['Normal'],
        fontSize=16,
        textColor=decision_color,
        alignment=1,
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    story.append(Paragraph(decision_icon, decision_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Customer Information
    story.append(Paragraph("CUSTOMER INFORMATION", heading_style))
    customer_info_data = [
        ['Age:', str(customer_data.get('age', 'N/A')), 'Employment:', customer_data.get('employment_type', 'N/A')],
        ['Monthly Income:', f"₹{customer_data.get('avg_salary_6m', 0):,}", 'Bureau Score:', str(customer_data.get('bureau_score', 0))],
        ['Loan Amount:', f"₹{customer_data.get('loan_amount', 0):,}", 'Tenure:', f"{customer_data.get('loan_tenure_months', 0)} months"],
        ['Interest Rate:', f"{customer_data.get('interest_rate', 0)}%", 'KYC Status:', 'Verified' if customer_data.get('kyc_verified', True) else 'Not Verified']
    ]
    
    customer_table = Table(customer_info_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
    customer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f7fafc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(customer_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Affordability Analysis
    story.append(Paragraph("AFFORDABILITY ANALYSIS", heading_style))
    foir = affordability_data.get('foir_percentage', 0)
    total_emi = affordability_data.get('total_emi', 0)
    net_disposable = affordability_data.get('net_disposable', 0)
    new_emi = affordability_data.get('new_emi', 0)
    existing_emi = affordability_data.get('existing_emi', 0)
    
    affordability_info = [
        ['New EMI:', f"₹{new_emi:,}", 'Existing EMI:', f"₹{existing_emi:,}"],
        ['Total EMI:', f"₹{total_emi:,}", 'FOIR:', f"{foir:.2f}%"],
        ['Net Disposable:', f"₹{net_disposable:,}", 'Status:', affordability_data.get('status', 'N/A')]
    ]
    
    affordability_table = Table(affordability_info, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
    affordability_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f7fafc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(affordability_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Decision Reasons
    story.append(Paragraph("DECISION REASONS", heading_style))
    reasons_list = []
    for i, reason in enumerate(reasons, 1):
        reasons_list.append([f"{i}.", reason])
    
    if reasons_list:
        reasons_table = Table(reasons_list, colWidths=[0.5*inch, 6.5*inch])
        reasons_table.setStyle(TableStyle([
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(reasons_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Risk Assessment
    story.append(Paragraph("RISK ASSESSMENT", heading_style))
    risk_data = [
        ['Risk Score:', f"{decision_data.get('risk_score', 0)}/1000"],
        ['PD (Probability of Default):', f"{decision_data.get('pd_percentage', 0)}%"],
        ['Model Confidence:', f"{decision_data.get('confidence', 0):.1f}%"],
        ['Bureau Score:', str(customer_data.get('bureau_score', 0))],
        ['DPD 90+ (6M):', str(customer_data.get('dpd_90_count_6m', 0))],
        ['Credit Utilization:', f"{customer_data.get('credit_utilization_pct', 0)}%"]
    ]
    
    risk_table = Table(risk_data, colWidths=[2.5*inch, 4.5*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 0.5*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Credit Risk Assessment System v8.3", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def generate_audit_pdf(audit_log):
    """
    Generate comprehensive audit trail PDF (Stage 1 + Stage 2)
    
    Args:
        audit_log: Dictionary containing complete audit information (Stage 1 and Stage 2)
    
    Returns:
        BytesIO buffer containing the PDF
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#587042'),
        spaceAfter=12,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#587042'),
        spaceAfter=6
    )
    
    story = []
    
    # Title
    story.append(Paragraph("AUDIT TRAIL REPORT", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Application Info (including both Stage 1 and Stage 2 decisions)
    app_id = audit_log.get('application_id', 'N/A')
    timestamp = audit_log.get('timestamp', 'N/A')
    stage1_decision = audit_log.get('decision', 'N/A')
    stage2_decision = audit_log.get('stage2_final_decision', 'Not Available')
    model_version = audit_log.get('model_version', 'N/A')
    
    header_data = [
        ['Application ID:', app_id],
        ['Timestamp:', timestamp],
        ['Stage 1 Decision:', stage1_decision],
        ['Stage 2 Decision:', stage2_decision],
        ['Model Version:', model_version]
    ]
    
    header_table = Table(header_data, colWidths=[2*inch, 5*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Policy Checks
    story.append(Paragraph("POLICY GATE CHECKS", heading_style))
    policy_checks = audit_log.get('policy_checks', {})
    policy_data = [[k, v] for k, v in policy_checks.items()]
    
    if policy_data:
        policy_table = Table(policy_data, colWidths=[2*inch, 5*inch])
        policy_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(policy_table)
    story.append(Spacer(1, 0.3*inch))
    
    # PD Calculation Breakdown
    story.append(Paragraph("PD CALCULATION FACTORS", heading_style))
    pd_factors = audit_log.get('pd_calculation_factors', {})
    pd_data = [
        ['Bureau Score:', str(pd_factors.get('bureau_score', 'N/A'))],
        ['Base PD:', f"{pd_factors.get('base_pd', 0):.2f}%"],
        ['DPD 90+ Count:', str(pd_factors.get('dpd_90', 0))],
        ['DPD 30+ Count:', str(pd_factors.get('dpd_30', 0))],
        ['Delinquency Multiplier:', f"{pd_factors.get('delinquency_multiplier', 1):.2f}x"],
        ['FOIR:', f"{pd_factors.get('foir', 0):.2f}%"],
        ['FOIR Adjustment:', f"{pd_factors.get('foir_adjustment', 0):.2f}%"],
        ['Employment Adjustment:', f"{pd_factors.get('employment_adjustment', 0):.2f}%"],
        ['ML Adjustment:', f"{pd_factors.get('ml_adjustment', 0):.2f}%"],
        ['FINAL PD:', f"{pd_factors.get('final_pd', 0):.2f}%"]
    ]
    
    pd_table = Table(pd_data, colWidths=[2.5*inch, 4.5*inch])
    pd_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#edf2f7')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(pd_table)
    story.append(Spacer(1, 0.3*inch))
    
    # ===== STAGE 2 DEEP DIVE RESULTS =====
    if 'stage2_final_decision' in audit_log:
        story.append(Paragraph("STAGE 2 DEEP DIVE RESULTS", heading_style))
        
        stage2_data = [
            ['Stage 2 Final Decision:', audit_log.get('stage2_final_decision', 'N/A')],
            ['Risk Tier:', audit_log.get('stage2_tier', 'N/A')],
            ['Interest Rate Range:', audit_log.get('stage2_interest_range', 'N/A')],
            ['Combined Risk Score:', str(audit_log.get('stage2_combined_risk_score', 'N/A'))],
            ['Stage 2 Confidence:', f"{audit_log.get('stage2_confidence', 0):.1f}%"],
            ['Stage 2 Reason:', audit_log.get('stage2_reason', 'N/A')]
        ]
        stage2_table = Table(stage2_data, colWidths=[2.5*inch, 4.5*inch])
        stage2_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(stage2_table)
        
        # Tier probabilities
        tier_probs = audit_log.get('stage2_tier_probabilities')
        if tier_probs and isinstance(tier_probs, dict):
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Tier Probabilities:", styles['Normal']))
            prob_data = [[tier, f"{prob:.1f}%"] for tier, prob in tier_probs.items()]
            prob_table = Table(prob_data, colWidths=[2.5*inch, 4.5*inch])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(prob_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Reason Codes
    story.append(Paragraph("DECISION REASONS", heading_style))
    reasons = audit_log.get('reason_codes', [])
    reasons_data = [[f"{i}.", reason] for i, reason in enumerate(reasons, 1)]
    
    if reasons_data:
        reasons_table = Table(reasons_data, colWidths=[0.5*inch, 6.5*inch])
        reasons_table.setStyle(TableStyle([
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(reasons_table)
    
    story.append(Spacer(1, 0.5*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph(f"Audit Trail Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
    story.append(Paragraph("This document serves as official record of the credit assessment decision process.", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer
