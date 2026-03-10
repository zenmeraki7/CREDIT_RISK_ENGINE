


# # # """
# # # PDF Generation Utility for Credit Risk Assessment
# # # """

# # # from reportlab.lib.pagesizes import letter
# # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
# # # from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# # # from reportlab.lib.units import inch
# # # from reportlab.lib import colors
# # # from reportlab.pdfgen import canvas
# # # from io import BytesIO
# # # from datetime import datetime


# # # def generate_decision_pdf(decision_data, customer_data, affordability_data, reasons):
# # #     """
# # #     Generate professional decision report PDF
    
# # #     Args:
# # #         decision_data: Dictionary with decision, risk_score, pd_percentage, confidence, etc.
# # #         customer_data: Dictionary with customer information
# # #         affordability_data: Dictionary with EMI, FOIR calculations
# # #         reasons: List of reason codes
    
# # #     Returns:
# # #         BytesIO buffer containing the PDF
# # #     """
# # #     buffer = BytesIO()
# # #     doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
# # #     # Get styles
# # #     styles = getSampleStyleSheet()
# # #     title_style = ParagraphStyle(
# # #         'CustomTitle',
# # #         parent=styles['Heading1'],
# # #         fontSize=20,
# # #         textColor=colors.HexColor('#587042'),
# # #         spaceAfter=12,
# # #         alignment=1  # Center
# # #     )
    
# # #     heading_style = ParagraphStyle(
# # #         'CustomHeading',
# # #         parent=styles['Heading2'],
# # #         fontSize=14,
# # #         textColor=colors.HexColor('#587042'),
# # #         spaceAfter=6
# # #     )
    
# # #     # Build content
# # #     story = []
    
# # #     # Title
# # #     story.append(Paragraph("CREDIT DECISION REPORT", title_style))
# # #     story.append(Spacer(1, 0.2*inch))
    
# # #     # Header Info
# # #     decision = decision_data.get('decision', 'ERROR')
# # #     app_id = customer_data.get('application_id', 'N/A')
# # #     timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
# # #     header_data = [
# # #         ['Application ID:', app_id, 'Timestamp:', timestamp],
# # #         ['Decision:', decision, 'Risk Score:', f"{decision_data.get('risk_score', 0)}/1000"],
# # #         ['PD Score:', f"{decision_data.get('pd_percentage', 0)}%", 'Confidence:', f"{decision_data.get('confidence', 0):.1f}%"]
# # #     ]
    
# # #     header_table = Table(header_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
# # #     header_table.setStyle(TableStyle([
# # #         ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
# # #         ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f7fafc')),
# # #         ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# # #         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
# # #         ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
# # #         ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
# # #         ('FONTSIZE', (0, 0), (-1, -1), 10),
# # #         ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
# # #         ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
# # #     ]))
# # #     story.append(header_table)
# # #     story.append(Spacer(1, 0.3*inch))
    
# # #     # Decision Status Box
# # #     if decision == "APPROVE":
# # #         decision_color = colors.HexColor('#48bb78')
# # #         decision_icon = "✓ APPROVED"
# # #     elif decision == "REJECT":
# # #         decision_color = colors.HexColor('#f56565')
# # #         decision_icon = "✗ REJECTED"
# # #     else:
# # #         decision_color = colors.HexColor('#ed8936')
# # #         decision_icon = "⚠ REVIEW REQUIRED"
    
# # #     decision_style = ParagraphStyle(
# # #         'DecisionStatus',
# # #         parent=styles['Normal'],
# # #         fontSize=16,
# # #         textColor=decision_color,
# # #         alignment=1,
# # #         spaceAfter=12,
# # #         fontName='Helvetica-Bold'
# # #     )
# # #     story.append(Paragraph(decision_icon, decision_style))
# # #     story.append(Spacer(1, 0.2*inch))
    
# # #     # Customer Information
# # #     story.append(Paragraph("CUSTOMER INFORMATION", heading_style))
# # #     customer_info_data = [
# # #         ['Age:', str(customer_data.get('age', 'N/A')), 'Employment:', customer_data.get('employment_type', 'N/A')],
# # #         ['Monthly Income:', f"₹{customer_data.get('avg_salary_6m', 0):,}", 'Bureau Score:', str(customer_data.get('bureau_score', 0))],
# # #         ['Loan Amount:', f"₹{customer_data.get('loan_amount', 0):,}", 'Tenure:', f"{customer_data.get('loan_tenure_months', 0)} months"],
# # #         ['Interest Rate:', f"{customer_data.get('interest_rate', 0)}%", 'KYC Status:', 'Verified' if customer_data.get('kyc_verified', True) else 'Not Verified']
# # #     ]
    
# # #     customer_table = Table(customer_info_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
# # #     customer_table.setStyle(TableStyle([
# # #         ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
# # #         ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f7fafc')),
# # #         ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# # #         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
# # #         ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
# # #         ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
# # #         ('FONTSIZE', (0, 0), (-1, -1), 9),
# # #         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# # #         ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
# # #     ]))
# # #     story.append(customer_table)
# # #     story.append(Spacer(1, 0.3*inch))
    
# # #     # Affordability Analysis
# # #     story.append(Paragraph("AFFORDABILITY ANALYSIS", heading_style))
# # #     foir = affordability_data.get('foir_percentage', 0)
# # #     total_emi = affordability_data.get('total_emi', 0)
# # #     net_disposable = affordability_data.get('net_disposable', 0)
# # #     new_emi = affordability_data.get('new_emi', 0)
# # #     existing_emi = affordability_data.get('existing_emi', 0)
    
# # #     affordability_info = [
# # #         ['New EMI:', f"₹{new_emi:,}", 'Existing EMI:', f"₹{existing_emi:,}"],
# # #         ['Total EMI:', f"₹{total_emi:,}", 'FOIR:', f"{foir:.2f}%"],
# # #         ['Net Disposable:', f"₹{net_disposable:,}", 'Status:', affordability_data.get('status', 'N/A')]
# # #     ]
    
# # #     affordability_table = Table(affordability_info, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
# # #     affordability_table.setStyle(TableStyle([
# # #         ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
# # #         ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f7fafc')),
# # #         ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# # #         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
# # #         ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
# # #         ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
# # #         ('FONTSIZE', (0, 0), (-1, -1), 9),
# # #         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# # #         ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
# # #     ]))
# # #     story.append(affordability_table)
# # #     story.append(Spacer(1, 0.3*inch))
    
# # #     # Decision Reasons
# # #     story.append(Paragraph("DECISION REASONS", heading_style))
# # #     reasons_list = []
# # #     for i, reason in enumerate(reasons, 1):
# # #         reasons_list.append([f"{i}.", reason])
    
# # #     if reasons_list:
# # #         reasons_table = Table(reasons_list, colWidths=[0.5*inch, 6.5*inch])
# # #         reasons_table.setStyle(TableStyle([
# # #             ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# # #             ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
# # #             ('ALIGN', (1, 0), (1, -1), 'LEFT'),
# # #             ('FONTSIZE', (0, 0), (-1, -1), 10),
# # #             ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# # #             ('VALIGN', (0, 0), (-1, -1), 'TOP')
# # #         ]))
# # #         story.append(reasons_table)
# # #     story.append(Spacer(1, 0.3*inch))
    
# # #     # Risk Assessment
# # #     story.append(Paragraph("RISK ASSESSMENT", heading_style))
# # #     risk_data = [
# # #         ['Risk Score:', f"{decision_data.get('risk_score', 0)}/1000"],
# # #         ['PD (Probability of Default):', f"{decision_data.get('pd_percentage', 0)}%"],
# # #         ['Model Confidence:', f"{decision_data.get('confidence', 0):.1f}%"],
# # #         ['Bureau Score:', str(customer_data.get('bureau_score', 0))],
# # #         ['DPD 90+ (6M):', str(customer_data.get('dpd_90_count_6m', 0))],
# # #         ['Credit Utilization:', f"{customer_data.get('credit_utilization_pct', 0)}%"]
# # #     ]
    
# # #     risk_table = Table(risk_data, colWidths=[2.5*inch, 4.5*inch])
# # #     risk_table.setStyle(TableStyle([
# # #         ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
# # #         ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# # #         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
# # #         ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
# # #         ('FONTSIZE', (0, 0), (-1, -1), 10),
# # #         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# # #         ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
# # #     ]))
# # #     story.append(risk_table)
# # #     story.append(Spacer(1, 0.5*inch))
    
# # #     # Footer
# # #     footer_style = ParagraphStyle(
# # #         'Footer',
# # #         parent=styles['Normal'],
# # #         fontSize=8,
# # #         textColor=colors.grey,
# # #         alignment=1
# # #     )
# # #     story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Credit Risk Assessment System v8.3", footer_style))
    
# # #     # Build PDF
# # #     doc.build(story)
# # #     buffer.seek(0)
# # #     return buffer


# # # def generate_audit_pdf(audit_log):
# # #     """
# # #     Generate comprehensive audit trail PDF (Stage 1 + Stage 2)
    
# # #     Args:
# # #         audit_log: Dictionary containing complete audit information (Stage 1 and Stage 2)
    
# # #     Returns:
# # #         BytesIO buffer containing the PDF
# # #     """
# # #     buffer = BytesIO()
# # #     doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
# # #     styles = getSampleStyleSheet()
# # #     title_style = ParagraphStyle(
# # #         'CustomTitle',
# # #         parent=styles['Heading1'],
# # #         fontSize=18,
# # #         textColor=colors.HexColor('#587042'),
# # #         spaceAfter=12,
# # #         alignment=1
# # #     )
    
# # #     heading_style = ParagraphStyle(
# # #         'CustomHeading',
# # #         parent=styles['Heading2'],
# # #         fontSize=12,
# # #         textColor=colors.HexColor('#587042'),
# # #         spaceAfter=6
# # #     )
    
# # #     story = []
    
# # #     # Title
# # #     story.append(Paragraph("AUDIT TRAIL REPORT", title_style))
# # #     story.append(Spacer(1, 0.2*inch))
    
# # #     # Application Info (including both Stage 1 and Stage 2 decisions)
# # #     app_id = audit_log.get('application_id', 'N/A')
# # #     timestamp = audit_log.get('timestamp', 'N/A')
# # #     stage1_decision = audit_log.get('decision', 'N/A')
# # #     stage2_decision = audit_log.get('stage2_final_decision', 'Not Available')
# # #     model_version = audit_log.get('model_version', 'N/A')
    
# # #     header_data = [
# # #         ['Application ID:', app_id],
# # #         ['Timestamp:', timestamp],
# # #         ['Stage 1 Decision:', stage1_decision],
# # #         ['Stage 2 Decision:', stage2_decision],
# # #         ['Model Version:', model_version]
# # #     ]
    
# # #     header_table = Table(header_data, colWidths=[2*inch, 5*inch])
# # #     header_table.setStyle(TableStyle([
# # #         ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
# # #         ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# # #         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
# # #         ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
# # #         ('FONTSIZE', (0, 0), (-1, -1), 10),
# # #         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# # #         ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
# # #     ]))
# # #     story.append(header_table)
# # #     story.append(Spacer(1, 0.3*inch))
    
# # #     # Policy Checks
# # #     story.append(Paragraph("POLICY GATE CHECKS", heading_style))
# # #     policy_checks = audit_log.get('policy_checks', {})
# # #     policy_data = [[k, v] for k, v in policy_checks.items()]
    
# # #     if policy_data:
# # #         policy_table = Table(policy_data, colWidths=[2*inch, 5*inch])
# # #         policy_table.setStyle(TableStyle([
# # #             ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
# # #             ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# # #             ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
# # #             ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
# # #             ('FONTSIZE', (0, 0), (-1, -1), 9),
# # #             ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# # #             ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
# # #         ]))
# # #         story.append(policy_table)
# # #     story.append(Spacer(1, 0.3*inch))
    
# # #     # PD Calculation Breakdown
# # #     story.append(Paragraph("PD CALCULATION FACTORS", heading_style))
# # #     pd_factors = audit_log.get('pd_calculation_factors', {})
# # #     pd_data = [
# # #         ['Bureau Score:', str(pd_factors.get('bureau_score', 'N/A'))],
# # #         ['Base PD:', f"{pd_factors.get('base_pd', 0):.2f}%"],
# # #         ['DPD 90+ Count:', str(pd_factors.get('dpd_90', 0))],
# # #         ['DPD 30+ Count:', str(pd_factors.get('dpd_30', 0))],
# # #         ['Delinquency Multiplier:', f"{pd_factors.get('delinquency_multiplier', 1):.2f}x"],
# # #         ['FOIR:', f"{pd_factors.get('foir', 0):.2f}%"],
# # #         ['FOIR Adjustment:', f"{pd_factors.get('foir_adjustment', 0):.2f}%"],
# # #         ['Employment Adjustment:', f"{pd_factors.get('employment_adjustment', 0):.2f}%"],
# # #         ['ML Adjustment:', f"{pd_factors.get('ml_adjustment', 0):.2f}%"],
# # #         ['FINAL PD:', f"{pd_factors.get('final_pd', 0):.2f}%"]
# # #     ]
    
# # #     pd_table = Table(pd_data, colWidths=[2.5*inch, 4.5*inch])
# # #     pd_table.setStyle(TableStyle([
# # #         ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
# # #         ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#edf2f7')),
# # #         ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# # #         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
# # #         ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
# # #         ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
# # #         ('FONTSIZE', (0, 0), (-1, -1), 9),
# # #         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# # #         ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
# # #     ]))
# # #     story.append(pd_table)
# # #     story.append(Spacer(1, 0.3*inch))
    
# # #     # ===== STAGE 2 DEEP DIVE RESULTS =====
# # #     if 'stage2_final_decision' in audit_log:
# # #         story.append(Paragraph("STAGE 2 DEEP DIVE RESULTS", heading_style))
        
# # #         stage2_data = [
# # #             ['Stage 2 Final Decision:', audit_log.get('stage2_final_decision', 'N/A')],
# # #             ['Risk Tier:', audit_log.get('stage2_tier', 'N/A')],
# # #             ['Interest Rate Range:', audit_log.get('stage2_interest_range', 'N/A')],
# # #             ['Combined Risk Score:', str(audit_log.get('stage2_combined_risk_score', 'N/A'))],
# # #             ['Stage 2 Confidence:', f"{audit_log.get('stage2_confidence', 0):.1f}%"],
# # #             ['Stage 2 Reason:', audit_log.get('stage2_reason', 'N/A')]
# # #         ]
# # #         stage2_table = Table(stage2_data, colWidths=[2.5*inch, 4.5*inch])
# # #         stage2_table.setStyle(TableStyle([
# # #             ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
# # #             ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# # #             ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
# # #             ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
# # #             ('FONTSIZE', (0, 0), (-1, -1), 9),
# # #             ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# # #             ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
# # #         ]))
# # #         story.append(stage2_table)
        
# # #         # Tier probabilities
# # #         tier_probs = audit_log.get('stage2_tier_probabilities')
# # #         if tier_probs and isinstance(tier_probs, dict):
# # #             story.append(Spacer(1, 0.1*inch))
# # #             story.append(Paragraph("Tier Probabilities:", styles['Normal']))
# # #             prob_data = [[tier, f"{prob:.1f}%"] for tier, prob in tier_probs.items()]
# # #             prob_table = Table(prob_data, colWidths=[2.5*inch, 4.5*inch])
# # #             prob_table.setStyle(TableStyle([
# # #                 ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
# # #                 ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# # #                 ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
# # #                 ('FONTSIZE', (0, 0), (-1, -1), 9),
# # #                 ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# # #                 ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
# # #             ]))
# # #             story.append(prob_table)
# # #         story.append(Spacer(1, 0.3*inch))
    
# # #     # Reason Codes
# # #     story.append(Paragraph("DECISION REASONS", heading_style))
# # #     reasons = audit_log.get('reason_codes', [])
# # #     reasons_data = [[f"{i}.", reason] for i, reason in enumerate(reasons, 1)]
    
# # #     if reasons_data:
# # #         reasons_table = Table(reasons_data, colWidths=[0.5*inch, 6.5*inch])
# # #         reasons_table.setStyle(TableStyle([
# # #             ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# # #             ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
# # #             ('ALIGN', (1, 0), (1, -1), 'LEFT'),
# # #             ('FONTSIZE', (0, 0), (-1, -1), 9),
# # #             ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# # #             ('VALIGN', (0, 0), (-1, -1), 'TOP')
# # #         ]))
# # #         story.append(reasons_table)
    
# # #     story.append(Spacer(1, 0.5*inch))
    
# # #     # Footer
# # #     footer_style = ParagraphStyle(
# # #         'Footer',
# # #         parent=styles['Normal'],
# # #         fontSize=8,
# # #         textColor=colors.grey,
# # #         alignment=1
# # #     )
# # #     story.append(Paragraph(f"Audit Trail Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
# # #     story.append(Paragraph("This document serves as official record of the credit assessment decision process.", footer_style))
    
# # #     # Build PDF
# # #     doc.build(story)
# # #     buffer.seek(0)
# # #     return buffer






# # # """
# # # PDF Summary Generator
# # # Generates downloadable decision summary reports
# # # """

# # # from turtle import st
# # # from reportlab.lib.pagesizes import A4
# # # from reportlab.lib.styles import getSampleStyleSheet
# # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
# # # from reportlab.lib import colors
# # # from reportlab.lib.units import inch
# # # import io


# # # def generate_decision_pdf(decision_data, customer_data, affordability_data, reasons):
# # #     """
# # #     Generate PDF summary of credit decision
# # #     """
    
# # #     buffer = io.BytesIO()
# # #     doc = SimpleDocTemplate(buffer, pagesize=A4)
# # #     elements = []
# # #     styles = getSampleStyleSheet()
    
# # #     # Title
# # #     title = Paragraph(f"<b>Credit Decision Summary</b>", styles['Title'])
# # #     elements.append(title)
# # #     elements.append(Spacer(1, 0.3*inch))
    
# # #     # Application details
# # #     app_info = f"""
# # #     <b>Application ID:</b> {decision_data['application_id']}<br/>
# # #     <b>Decision Date:</b> {decision_data['timestamp']}<br/>
# # #     <b>Decision:</b> {decision_data['decision']}<br/>
# # #     <b>Risk Score:</b> {decision_data['risk_score']}/1000<br/>
# # #     <b>PD:</b> {decision_data['pd_percentage']}%
# # #     """
# # #     elements.append(Paragraph(app_info, styles['Normal']))
# # #     elements.append(Spacer(1, 0.3*inch))
    
# # #     # Customer details table
# # #     customer_table_data = [
# # #         ['Field', 'Value'],
# # #         ['Name', customer_data.get('name', 'N/A')],
# # #         ['Age', str(customer_data.get('age', 'N/A'))],
# # #         ['Bureau Score', str(customer_data.get('bureau_score', 'N/A'))],
# # #         ['Monthly Income', f"₹{customer_data.get('avg_salary_6m', 0):,}"],
# # #     ]
    
# # #     customer_table = Table(customer_table_data, colWidths=[3*inch, 3*inch])
# # #     customer_table.setStyle(TableStyle([
# # #         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
# # #         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
# # #         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
# # #         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
# # #         ('FONTSIZE', (0, 0), (-1, 0), 12),
# # #         ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
# # #         ('GRID', (0, 0), (-1, -1), 1, colors.black)
# # #     ]))
    
# # #     elements.append(customer_table)
# # #     elements.append(Spacer(1, 0.3*inch))
    
# # #     # Affordability breakdown
# # #     affordability_text = f"""
# # #     <b>Affordability Assessment:</b><br/>
# # #     Monthly Income: ₹{affordability_data['monthly_income']:,}<br/>
# # #     Total EMI: ₹{affordability_data['total_emi']:,}<br/>
# # #     FOIR: {affordability_data['foir_percentage']}%<br/>
# # #     Net Disposable: ₹{affordability_data['net_disposable']:,}
# # #     """
# # #     elements.append(Paragraph(affordability_text, styles['Normal']))
# # #     elements.append(Spacer(1, 0.3*inch))
    
# # #     # Reason codes
# # #     reasons_text = "<b>Decision Reasons:</b><br/>"
# # #     for i, reason in enumerate(reasons, 1):
# # #         reasons_text += f"{i}. {reason}<br/>"
    
# # #     elements.append(Paragraph(reasons_text, styles['Normal']))
    
# # #     # Build PDF
# # #     doc.build(elements)
# # #     buffer.seek(0)
    
# # #     return buffer


# # # # Usage in Streamlit
# # # def add_download_button(decision_data, customer_data, affordability_data, reasons):
# # #     """Add download button to Streamlit page"""
    
# # #     pdf_buffer = generate_decision_pdf(
# # #         decision_data, customer_data, affordability_data, reasons
# # #     )
    
# # #     st.download_button(
# # #         label="📥 Download Decision Summary",
# # #         data=pdf_buffer,
# # #         file_name=f"credit_decision_{decision_data['application_id']}.pdf",
# # #         mime="application/pdf",
# # #         use_container_width=True
# # #     )


# # ##################################################################################



# # # """
# # # PDF Summary Generator
# # # Generates downloadable decision summary reports
# # # """

# # # from turtle import st
# # # from reportlab.lib.pagesizes import A4
# # # from reportlab.lib.styles import getSampleStyleSheet
# # # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
# # # from reportlab.lib import colors
# # # from reportlab.lib.units import inch
# # # import io


# # # def generate_decision_pdf(decision_data, customer_data, affordability_data, reasons):
# # #     """
# # #     Generate PDF summary of credit decision
# # #     """
    
# # #     buffer = io.BytesIO()
# # #     doc = SimpleDocTemplate(buffer, pagesize=A4)
# # #     elements = []
# # #     styles = getSampleStyleSheet()
    
# # #     # Title
# # #     title = Paragraph(f"<b>Credit Decision Summary</b>", styles['Title'])
# # #     elements.append(title)
# # #     elements.append(Spacer(1, 0.3*inch))
    
# # #     # Application details
# # #     app_info = f"""
# # #     <b>Application ID:</b> {decision_data['application_id']}<br/>
# # #     <b>Decision Date:</b> {decision_data['timestamp']}<br/>
# # #     <b>Decision:</b> {decision_data['decision']}<br/>
# # #     <b>Risk Score:</b> {decision_data['risk_score']}/1000<br/>
# # #     <b>PD:</b> {decision_data['pd_percentage']}%
# # #     """
# # #     elements.append(Paragraph(app_info, styles['Normal']))
# # #     elements.append(Spacer(1, 0.3*inch))
    
# # #     # Customer details table
# # #     customer_table_data = [
# # #         ['Field', 'Value'],
# # #         ['Name', customer_data.get('name', 'N/A')],
# # #         ['Age', str(customer_data.get('age', 'N/A'))],
# # #         ['Bureau Score', str(customer_data.get('bureau_score', 'N/A'))],
# # #         ['Monthly Income', f"₹{customer_data.get('avg_salary_6m', 0):,}"],
# # #     ]
    
# # #     customer_table = Table(customer_table_data, colWidths=[3*inch, 3*inch])
# # #     customer_table.setStyle(TableStyle([
# # #         ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
# # #         ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
# # #         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
# # #         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
# # #         ('FONTSIZE', (0, 0), (-1, 0), 12),
# # #         ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
# # #         ('GRID', (0, 0), (-1, -1), 1, colors.black)
# # #     ]))
    
# # #     elements.append(customer_table)
# # #     elements.append(Spacer(1, 0.3*inch))
    
# # #     # Affordability breakdown
# # #     affordability_text = f"""
# # #     <b>Affordability Assessment:</b><br/>
# # #     Monthly Income: ₹{affordability_data['monthly_income']:,}<br/>
# # #     Total EMI: ₹{affordability_data['total_emi']:,}<br/>
# # #     FOIR: {affordability_data['foir_percentage']}%<br/>
# # #     Net Disposable: ₹{affordability_data['net_disposable']:,}
# # #     """
# # #     elements.append(Paragraph(affordability_text, styles['Normal']))
# # #     elements.append(Spacer(1, 0.3*inch))
    
# # #     # Reason codes
# # #     reasons_text = "<b>Decision Reasons:</b><br/>"
# # #     for i, reason in enumerate(reasons, 1):
# # #         reasons_text += f"{i}. {reason}<br/>"
    
# # #     elements.append(Paragraph(reasons_text, styles['Normal']))
    
# # #     # Build PDF
# # #     doc.build(elements)
# # #     buffer.seek(0)
    
# # #     return buffer


# # # # Usage in Streamlit
# # # def add_download_button(decision_data, customer_data, affordability_data, reasons):
# # #     """Add download button to Streamlit page"""
    
# # #     pdf_buffer = generate_decision_pdf(
# # #         decision_data, customer_data, affordability_data, reasons
# # #     )
    
# # #     st.download_button(
# # #         label="📥 Download Decision Summary",
# # #         data=pdf_buffer,
# # #         file_name=f"credit_decision_{decision_data['application_id']}.pdf",
# # #         mime="application/pdf",
# # #         use_container_width=True
# # #     )


# # ##################################################################################




# # """
# # PDF Generation Utility for Credit Risk Assessment
# # Author: Zen Meraki
# # Version: 8.7 — Corrected field names, risk score /100, DPD tiers, v8.7 footer
# # """

# # from reportlab.lib.pagesizes import letter
# # from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
# #                                 Table, TableStyle, PageBreak)
# # from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# # from reportlab.lib.units import inch
# # from reportlab.lib import colors
# # from io import BytesIO
# # from datetime import datetime


# # # ---------------------------------------------------------------------------
# # # SHARED STYLE HELPERS
# # # ---------------------------------------------------------------------------
# # _BRAND   = colors.HexColor('#587042')
# # _LIGHT   = colors.HexColor('#f7fafc')
# # _GREY    = colors.HexColor('#e2e8f0')
# # _GREEN   = colors.HexColor('#48bb78')
# # _RED     = colors.HexColor('#f56565')
# # _ORANGE  = colors.HexColor('#ed8936')

# # def _styles():
# #     base = getSampleStyleSheet()
# #     title = ParagraphStyle('CRTitle', parent=base['Heading1'],
# #                            fontSize=20, textColor=_BRAND,
# #                            spaceAfter=10, alignment=1)
# #     heading = ParagraphStyle('CRHeading', parent=base['Heading2'],
# #                              fontSize=13, textColor=_BRAND,
# #                              spaceAfter=6, spaceBefore=10)
# #     small = ParagraphStyle('CRSmall', parent=base['Normal'],
# #                            fontSize=8, textColor=colors.grey, alignment=1)
# #     return base, title, heading, small


# # def _label_table(rows, col_widths, label_cols=(0,)):
# #     """Two-or-four column key-value table with shaded label cells."""
# #     t = Table(rows, colWidths=col_widths)
# #     style = [
# #         ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
# #         ('ALIGN',     (0, 0), (-1, -1), 'LEFT'),
# #         ('FONTSIZE',  (0, 0), (-1, -1), 9),
# #         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# #         ('GRID',      (0, 0), (-1, -1), 0.5, _GREY),
# #     ]
# #     for col in label_cols:
# #         style.append(('BACKGROUND', (col, 0), (col, -1), _LIGHT))
# #         style.append(('FONTNAME',   (col, 0), (col, -1), 'Helvetica-Bold'))
# #     t.setStyle(TableStyle(style))
# #     return t


# # def _safe_int(v, default=0):
# #     try:
# #         return int(round(float(v)))
# #     except (TypeError, ValueError):
# #         return default


# # def _safe_float(v, default=0.0):
# #     try:
# #         return float(v)
# #     except (TypeError, ValueError):
# #         return default


# # # ---------------------------------------------------------------------------
# # # DECISION REPORT  (Stage 1 — quick summary)
# # # ---------------------------------------------------------------------------
# # def generate_decision_pdf(decision_data, customer_data, affordability_data, reasons):
# #     """
# #     Generate the Stage 1 decision summary PDF.

# #     All numbers are taken directly from the dicts passed in — no re-calculation.
# #     Risk score is shown /100 (engine produces 0-100, not 0-1000).
# #     """
# #     buffer = BytesIO()
# #     doc = SimpleDocTemplate(buffer, pagesize=letter,
# #                             topMargin=0.5*inch, bottomMargin=0.5*inch,
# #                             leftMargin=0.6*inch, rightMargin=0.6*inch)
# #     base, title_style, heading_style, small_style = _styles()
# #     story = []

# #     # ── Title ────────────────────────────────────────────────────────────────
# #     story.append(Paragraph("CREDIT DECISION REPORT", title_style))
# #     story.append(Spacer(1, 0.15*inch))

# #     # ── Decision banner ───────────────────────────────────────────────────────
# #     decision   = decision_data.get('decision', 'ERROR')
# #     risk_score = _safe_int(decision_data.get('risk_score', 0))
# #     pd_pct     = _safe_float(decision_data.get('pd_percentage', 0))
# #     confidence = _safe_float(decision_data.get('confidence', 0))
# #     app_id     = customer_data.get('application_id', 'N/A')
# #     timestamp  = customer_data.get('timestamp',
# #                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# #     if decision == 'APPROVE':
# #         dec_color, dec_icon = _GREEN,  "APPROVED"
# #     elif decision == 'REJECT':
# #         dec_color, dec_icon = _RED,    "REJECTED"
# #     else:
# #         dec_color, dec_icon = _ORANGE, "REVIEW REQUIRED"

# #     dec_style = ParagraphStyle('DecBanner', parent=base['Normal'],
# #                                fontSize=16, textColor=dec_color,
# #                                fontName='Helvetica-Bold', alignment=1,
# #                                spaceAfter=8)
# #     story.append(Paragraph(dec_icon, dec_style))

# #     # ── Header table: App ID / Timestamp / Risk Score / PD / Confidence ──────
# #     story.append(_label_table([
# #         ['Application ID:', app_id,    'Timestamp:',   timestamp],
# #         ['Decision:',       decision,   'Risk Score:',  f"{risk_score}/100"],
# #         ['PD Score:',       f"{pd_pct:.2f}%", 'Confidence:', f"{confidence:.1f}%"],
# #     ], [1.3*inch, 2.2*inch, 1.3*inch, 2.2*inch], label_cols=(0, 2)))
# #     story.append(Spacer(1, 0.25*inch))

# #     # ── Customer information ──────────────────────────────────────────────────
# #     story.append(Paragraph("CUSTOMER INFORMATION", heading_style))

# #     age             = _safe_int(customer_data.get('age', 0))
# #     emp_type        = customer_data.get('employment_type', 'N/A')
# #     income          = _safe_int(customer_data.get('avg_salary_6m', 0))
# #     bureau_score    = _safe_int(customer_data.get('bureau_score', 0))
# #     loan_amount     = _safe_int(customer_data.get('loan_amount', 0))
# #     loan_tenure     = _safe_int(customer_data.get('loan_tenure_months', 0))
# #     interest_rate   = _safe_float(customer_data.get('interest_rate', 0))
# #     kyc             = 'Verified' if customer_data.get('kyc_verified', True) else 'Not Verified'
# #     gender          = customer_data.get('gender', 'N/A')
# #     city_tier       = customer_data.get('city_tier', 'N/A')
# #     rbi_consent     = 'Obtained' if customer_data.get('rbi_consent', False) else 'Not Obtained'
# #     dpd_90          = _safe_int(customer_data.get('dpd_90_count_6m', 0))
# #     dpd_30          = _safe_int(customer_data.get('dpd_30_count_6m', 0))
# #     credit_util     = _safe_float(customer_data.get('credit_utilization_pct', 0))
# #     active_loans    = _safe_int(customer_data.get('active_loans_count', 0))
# #     salary_stab     = customer_data.get('salary_stability_flag', 'N/A')
# #     pay_disc        = customer_data.get('payment_discipline_flag', 'N/A')

# #     story.append(_label_table([
# #         ['Age:',            str(age),                 'Employment:',      emp_type],
# #         ['Gender:',         gender,                   'City Tier:',       city_tier],
# #         ['Monthly Income:', f"Rs.{income:,}",         'Bureau Score:',    str(bureau_score)],
# #         ['Loan Amount:',    f"Rs.{loan_amount:,}",    'Tenure:',          f"{loan_tenure} months"],
# #         ['Interest Rate:',  f"{interest_rate:.2f}%",  'KYC Status:',      kyc],
# #         ['RBI Consent:',    rbi_consent,              'Active Loans:',    str(active_loans)],
# #         ['DPD 90+ (6M):',   str(dpd_90),             'DPD 30+ (6M):',   str(dpd_30)],
# #         ['Credit Util.:',   f"{credit_util:.1f}%",    'Salary Stability:', salary_stab],
# #         ['Payment Discipline:', pay_disc,             '', ''],
# #     ], [1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch], label_cols=(0, 2)))
# #     story.append(Spacer(1, 0.25*inch))

# #     # ── Affordability analysis ────────────────────────────────────────────────
# #     story.append(Paragraph("AFFORDABILITY ANALYSIS", heading_style))

# #     new_emi      = _safe_float(affordability_data.get('new_emi', 0))
# #     existing_emi = _safe_float(affordability_data.get('existing_emi', 0))
# #     total_emi    = _safe_float(affordability_data.get('total_emi', 0))
# #     foir         = _safe_float(affordability_data.get('foir_percentage', 0))
# #     net_disp     = _safe_float(affordability_data.get('net_disposable', 0))
# #     aff_status   = affordability_data.get('status', 'N/A')
# #     max_emi      = _safe_float(affordability_data.get('max_allowed_emi', 0))
# #     emi_headroom = _safe_float(affordability_data.get('emi_headroom', 0))

# #     story.append(_label_table([
# #         ['New EMI:',         f"Rs.{new_emi:,.0f}",      'Existing EMI:',   f"Rs.{existing_emi:,.0f}"],
# #         ['Total EMI:',       f"Rs.{total_emi:,.0f}",    'FOIR:',           f"{foir:.2f}%"],
# #         ['Net Disposable:',  f"Rs.{net_disp:,.0f}",     'Status:',         aff_status],
# #         ['Max Allowed EMI:', f"Rs.{max_emi:,.0f}",      'EMI Headroom:',   f"Rs.{emi_headroom:,.0f}"],
# #     ], [1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch], label_cols=(0, 2)))
# #     story.append(Spacer(1, 0.25*inch))

# #     # ── Decision reasons ──────────────────────────────────────────────────────
# #     story.append(Paragraph("DECISION REASONS", heading_style))
# #     if reasons:
# #         reason_rows = [[f"{i}.", r] for i, r in enumerate(reasons, 1)]
# #         rt = Table(reason_rows, colWidths=[0.4*inch, 6.6*inch])
# #         rt.setStyle(TableStyle([
# #             ('TEXTCOLOR',     (0, 0), (-1, -1), colors.black),
# #             ('ALIGN',         (0, 0), (0, -1),  'RIGHT'),
# #             ('ALIGN',         (1, 0), (1, -1),  'LEFT'),
# #             ('FONTSIZE',      (0, 0), (-1, -1), 10),
# #             ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# #             ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
# #         ]))
# #         story.append(rt)
# #     story.append(Spacer(1, 0.25*inch))

# #     # ── Risk assessment ───────────────────────────────────────────────────────
# #     story.append(Paragraph("RISK ASSESSMENT", heading_style))

# #     # DPD tier label
# #     if dpd_90 == 0:
# #         dpd_label = f"{dpd_90} (Clean)"
# #     elif dpd_90 == 1:
# #         dpd_label = f"{dpd_90} (Acceptable)"
# #     elif dpd_90 <= 5:
# #         dpd_label = f"{dpd_90} (Review Zone: 2-5)"
# #     else:
# #         dpd_label = f"{dpd_90} (REJECT: >5)"

# #     story.append(_label_table([
# #         ['Risk Score (0-100):',           f"{risk_score}/100"],
# #         ['PD (Probability of Default):',  f"{pd_pct:.2f}%"],
# #         ['Model Confidence:',             f"{confidence:.1f}%"],
# #         ['Bureau Score:',                 str(bureau_score)],
# #         ['DPD 90+ (6M):',                dpd_label],
# #         ['DPD 30+ (6M):',                str(dpd_30)],
# #         ['Credit Utilization:',           f"{credit_util:.1f}%"],
# #         ['Net Cash Surplus:',             f"Rs.{_safe_int(customer_data.get('net_cash_surplus_6m', 0)):,}"],
# #     ], [2.8*inch, 4.2*inch], label_cols=(0,)))
# #     story.append(Spacer(1, 0.4*inch))

# #     # ── Footer ─────────────────────────────────────────────────────────────────
# #     story.append(Paragraph(
# #         f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
# #         "Credit Risk Assessment System v8.7 | FOR INTERNAL USE ONLY",
# #         small_style))

# #     doc.build(story)
# #     buffer.seek(0)
# #     return buffer


# # # ---------------------------------------------------------------------------
# # # AUDIT TRAIL PDF  (Stage 1 + optional Stage 2)
# # # ---------------------------------------------------------------------------
# # def generate_audit_pdf(audit_log):
# #     """
# #     Generate comprehensive audit trail PDF covering Stage 1 policy gates,
# #     PD calculation breakdown, and — if present — Stage 2 deep-dive results.

# #     Key field mapping (audit_log dict):
# #         application_id, timestamp, model_version
# #         decision           — Stage 1 decision
# #         risk_score         — Stage 1 risk score  (0-100)
# #         pd_percentage      — Stage 1 final PD %
# #         confidence         — Stage 1 model confidence %
# #         policy_checks      — dict of gate results
# #         pd_calculation_factors  — dict with base_pd, multiplier, adjustments
# #         reason_codes       — list of strings
# #         stage2_final_decision, stage2_tier, stage2_interest_range,
# #         stage2_combined_risk_score, stage2_confidence, stage2_reason,
# #         stage2_tier_probabilities  — optional Stage 2 fields
# #     """
# #     buffer = BytesIO()
# #     doc = SimpleDocTemplate(buffer, pagesize=letter,
# #                             topMargin=0.5*inch, bottomMargin=0.5*inch,
# #                             leftMargin=0.6*inch, rightMargin=0.6*inch)
# #     base, title_style, heading_style, small_style = _styles()
# #     story = []

# #     # ── Title ─────────────────────────────────────────────────────────────────
# #     story.append(Paragraph("AUDIT TRAIL REPORT", title_style))
# #     story.append(Spacer(1, 0.15*inch))

# #     # ── Application header ────────────────────────────────────────────────────
# #     app_id           = audit_log.get('application_id', 'N/A')
# #     timestamp        = audit_log.get('timestamp', 'N/A')
# #     s1_decision      = audit_log.get('decision', 'N/A')
# #     s2_decision      = audit_log.get('stage2_final_decision', 'Not Run')
# #     model_version    = audit_log.get('model_version', '8.7')
# #     risk_score       = _safe_int(audit_log.get('risk_score', 0))
# #     pd_pct           = _safe_float(audit_log.get('pd_percentage', 0))
# #     confidence       = _safe_float(audit_log.get('confidence', 0))

# #     story.append(_label_table([
# #         ['Application ID:',    app_id,       'Timestamp:',        timestamp],
# #         ['Stage 1 Decision:',  s1_decision,  'Stage 2 Decision:', s2_decision],
# #         ['Risk Score (0-100):', f"{risk_score}/100", 'PD Score:',  f"{pd_pct:.2f}%"],
# #         ['Model Confidence:',  f"{confidence:.1f}%", 'Version:',  model_version],
# #     ], [1.7*inch, 2.1*inch, 1.7*inch, 1.5*inch], label_cols=(0, 2)))
# #     story.append(Spacer(1, 0.25*inch))

# #     # ── Policy gate checks ────────────────────────────────────────────────────
# #     story.append(Paragraph("POLICY GATE CHECKS", heading_style))
# #     policy_checks = audit_log.get('policy_checks', {})
# #     if policy_checks:
# #         pc_rows = [[k.replace('_', ' ').title(), str(v)]
# #                    for k, v in policy_checks.items()]
# #         story.append(_label_table(pc_rows, [2.0*inch, 5.0*inch], label_cols=(0,)))
# #     else:
# #         story.append(Paragraph("No policy check data available.", base['Normal']))
# #     story.append(Spacer(1, 0.25*inch))

# #     # ── PD calculation breakdown ──────────────────────────────────────────────
# #     story.append(Paragraph("PD CALCULATION FACTORS", heading_style))
# #     pd_f = audit_log.get('pd_calculation_factors', {})

# #     bureau_score       = _safe_int(pd_f.get('bureau_score', 0))
# #     base_pd            = _safe_float(pd_f.get('base_pd', 0))
# #     dpd_90             = _safe_int(pd_f.get('dpd_90', 0))
# #     dpd_30             = _safe_int(pd_f.get('dpd_30', 0))
# #     deliq_mult         = _safe_float(pd_f.get('delinquency_multiplier', 1.0))
# #     foir_val           = _safe_float(pd_f.get('foir', 0))
# #     foir_adj           = _safe_float(pd_f.get('foir_adjustment', 0))
# #     emp_adj            = _safe_float(pd_f.get('employment_adjustment', 0))
# #     inq_adj            = _safe_float(pd_f.get('inquiry_adjustment', 0))
# #     ml_adj             = _safe_float(pd_f.get('ml_adjustment', 0))
# #     final_pd           = _safe_float(pd_f.get('final_pd', pd_pct))

# #     # DPD tier label
# #     if dpd_90 == 0:
# #         dpd_tier = "0 (Clean — pass)"
# #     elif dpd_90 == 1:
# #         dpd_tier = "1 (Acceptable — pass)"
# #     elif dpd_90 <= 5:
# #         dpd_tier = f"{dpd_90} (Review zone 2-5)"
# #     else:
# #         dpd_tier = f"{dpd_90} (REJECT — exceeds 5)"

# #     pd_rows = [
# #         ['Bureau Score:',           str(bureau_score)],
# #         ['Base PD:',                f"{base_pd:.2f}%"],
# #         ['DPD 90+ Count:',          dpd_tier],
# #         ['DPD 30+ Count:',          str(dpd_30)],
# #         ['Delinquency Multiplier:', f"{deliq_mult:.2f}x"],
# #         ['FOIR:',                   f"{foir_val:.2f}%"],
# #         ['FOIR Adjustment:',        f"{foir_adj:+.2f}%"],
# #         ['Employment Adjustment:',  f"{emp_adj:+.2f}%"],
# #         ['Inquiry Adjustment:',     f"{inq_adj:+.2f}%"],
# #         ['ML Model Adjustment:',    f"{ml_adj:+.2f}%"],
# #         ['FINAL PD:',               f"{final_pd:.2f}%"],
# #     ]

# #     pd_table = Table(pd_rows, colWidths=[2.5*inch, 4.5*inch])
# #     pd_style = [
# #         ('BACKGROUND',    (0, 0),  (0, -1),  _LIGHT),
# #         ('BACKGROUND',    (0, -1), (-1, -1), colors.HexColor('#edf2f7')),
# #         ('TEXTCOLOR',     (0, 0),  (-1, -1), colors.black),
# #         ('ALIGN',         (0, 0),  (-1, -1), 'LEFT'),
# #         ('FONTNAME',      (0, 0),  (0, -1),  'Helvetica-Bold'),
# #         ('FONTNAME',      (0, -1), (-1, -1), 'Helvetica-Bold'),
# #         ('FONTSIZE',      (0, 0),  (-1, -1), 9),
# #         ('BOTTOMPADDING', (0, 0),  (-1, -1), 6),
# #         ('GRID',          (0, 0),  (-1, -1), 0.5, _GREY),
# #     ]
# #     pd_table.setStyle(TableStyle(pd_style))
# #     story.append(pd_table)
# #     story.append(Spacer(1, 0.25*inch))

# #     # ── Stage 2 results (if available) ───────────────────────────────────────
# #     if audit_log.get('stage2_final_decision') and \
# #        audit_log.get('stage2_final_decision') not in ('N/A', 'Not Run', None):

# #         story.append(Paragraph("STAGE 2 DEEP DIVE RESULTS", heading_style))

# #         s2_tier      = audit_log.get('stage2_tier', 'N/A')
# #         s2_int_range = audit_log.get('stage2_interest_range', 'N/A')
# #         s2_risk      = _safe_int(audit_log.get('stage2_combined_risk_score', 0))
# #         s2_conf      = _safe_float(audit_log.get('stage2_confidence', 0))
# #         s2_reason    = audit_log.get('stage2_reason', 'N/A')

# #         s2_rows = [
# #             ['Stage 2 Final Decision:',  s2_decision],
# #             ['Risk Tier:',               s2_tier],
# #             ['Interest Rate Range:',     s2_int_range],
# #             ['Combined Risk Score:',     str(s2_risk)],
# #             ['Stage 2 Confidence:',      f"{s2_conf:.1f}%"],
# #             ['Stage 2 Reason:',          str(s2_reason)],
# #         ]
# #         story.append(_label_table(s2_rows, [2.5*inch, 4.5*inch], label_cols=(0,)))

# #         # Tier probabilities
# #         tier_probs = audit_log.get('stage2_tier_probabilities')
# #         if tier_probs and isinstance(tier_probs, dict):
# #             story.append(Spacer(1, 0.1*inch))
# #             story.append(Paragraph("Tier Probabilities:", base['Normal']))
# #             tp_rows = [[tier, f"{prob:.1f}%"] for tier, prob in tier_probs.items()]
# #             story.append(_label_table(tp_rows, [2.5*inch, 4.5*inch], label_cols=(0,)))

# #         story.append(Spacer(1, 0.25*inch))

# #     # ── Decision reason codes ─────────────────────────────────────────────────
# #     story.append(Paragraph("DECISION REASONS", heading_style))
# #     reasons = audit_log.get('reason_codes', [])
# #     if reasons:
# #         r_rows = [[f"{i}.", str(r)] for i, r in enumerate(reasons, 1)]
# #         rt = Table(r_rows, colWidths=[0.4*inch, 6.6*inch])
# #         rt.setStyle(TableStyle([
# #             ('TEXTCOLOR',     (0, 0), (-1, -1), colors.black),
# #             ('ALIGN',         (0, 0), (0, -1),  'RIGHT'),
# #             ('ALIGN',         (1, 0), (1, -1),  'LEFT'),
# #             ('FONTSIZE',      (0, 0), (-1, -1), 9),
# #             ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
# #             ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
# #         ]))
# #         story.append(rt)
# #     else:
# #         story.append(Paragraph("No reason codes recorded.", base['Normal']))

# #     story.append(Spacer(1, 0.4*inch))

# #     # ── Footer ────────────────────────────────────────────────────────────────
# #     story.append(Paragraph(
# #         f"Audit Trail Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
# #         "Credit Risk Assessment System v8.7",
# #         small_style))
# #     story.append(Paragraph(
# #         "This document is an official record of the credit assessment decision process. "
# #         "FOR INTERNAL USE ONLY.",
# #         small_style))

# #     doc.build(story)
# #     buffer.seek(0)
# #     return buffer
# #     return buffer



# """
# PDF Generation Utility — Credit Risk Assessment System v8.7
# ===========================================================
# Two public functions:

#   generate_decision_pdf(decision_data, customer_data, affordability_data, reasons)
#       → BytesIO   Stage 1 quick-summary report (single page, letter size)

#   generate_audit_pdf(audit_log)
#       → BytesIO   Full audit trail: policy gates, PD breakdown, Stage 2 (optional)

# Both return a seeked BytesIO buffer ready for st.download_button / HTTP response.
# """

# # ---------------------------------------------------------------------------
# # Imports
# # ---------------------------------------------------------------------------
# from io import BytesIO
# from datetime import datetime

# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import (
#     SimpleDocTemplate, Paragraph, Spacer,
#     Table, TableStyle, HRFlowable,
# )
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import inch
# from reportlab.lib import colors

# # ---------------------------------------------------------------------------
# # Brand palette
# # ---------------------------------------------------------------------------
# _BRAND  = colors.HexColor('#1a365d')   # deep navy  — section headings
# _ACCENT = colors.HexColor('#2b6cb0')   # mid-blue   — heading background
# _LIGHT  = colors.HexColor('#ebf4ff')   # pale blue  — label cell background
# _GREY   = colors.HexColor('#cbd5e0')   # silver     — grid lines
# _GREEN  = colors.HexColor('#276749')   # forest green — APPROVE
# _RED    = colors.HexColor('#c53030')   # red        — REJECT
# _ORANGE = colors.HexColor('#c05621')   # burnt orange — REVIEW

# # ---------------------------------------------------------------------------
# # Internal helpers
# # ---------------------------------------------------------------------------

# def _safe_int(v, default: int = 0) -> int:
#     try:
#         return int(round(float(v)))
#     except (TypeError, ValueError):
#         return default


# def _safe_float(v, default: float = 0.0) -> float:
#     try:
#         return float(v)
#     except (TypeError, ValueError):
#         return default


# def _styles():
#     """Return (base, title_style, heading_style, sub_style, small_style)."""
#     base = getSampleStyleSheet()

#     title = ParagraphStyle(
#         'CRTitle', parent=base['Heading1'],
#         fontSize=18, textColor=_BRAND,
#         spaceAfter=6, spaceBefore=0,
#         alignment=1,
#         fontName='Helvetica-Bold',
#     )
#     heading = ParagraphStyle(
#         'CRHeading', parent=base['Heading2'],
#         fontSize=10.5, textColor=colors.white,
#         spaceAfter=4, spaceBefore=10,
#         fontName='Helvetica-Bold',
#         backColor=_ACCENT,
#         leftIndent=-4, rightIndent=-4,
#         borderPad=4,
#     )
#     sub = ParagraphStyle(
#         'CRSub', parent=base['Normal'],
#         fontSize=9, textColor=_BRAND,
#         fontName='Helvetica-Bold',
#         spaceAfter=2, spaceBefore=6,
#     )
#     small = ParagraphStyle(
#         'CRSmall', parent=base['Normal'],
#         fontSize=7.5, textColor=colors.grey,
#         alignment=1,
#     )
#     return base, title, heading, sub, small


# def _label_table(rows, col_widths, label_cols=(0,)):
#     """
#     Shaded key-value table.

#     rows        — list of lists
#     col_widths  — list of floats (already multiplied by inch)
#     label_cols  — column indices to shade as labels
#     """
#     t = Table(rows, colWidths=col_widths)
#     style = [
#         ('TEXTCOLOR',      (0, 0), (-1, -1), colors.black),
#         ('ALIGN',          (0, 0), (-1, -1), 'LEFT'),
#         ('VALIGN',         (0, 0), (-1, -1), 'MIDDLE'),
#         ('FONTSIZE',       (0, 0), (-1, -1), 8.5),
#         ('TOPPADDING',     (0, 0), (-1, -1), 5),
#         ('BOTTOMPADDING',  (0, 0), (-1, -1), 5),
#         ('LEFTPADDING',    (0, 0), (-1, -1), 6),
#         ('RIGHTPADDING',   (0, 0), (-1, -1), 6),
#         ('GRID',           (0, 0), (-1, -1), 0.4, _GREY),
#         ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f7fbff')]),
#     ]
#     for col in label_cols:
#         style += [
#             ('BACKGROUND', (col, 0), (col, -1), _LIGHT),
#             ('FONTNAME',   (col, 0), (col, -1), 'Helvetica-Bold'),
#             ('TEXTCOLOR',  (col, 0), (col, -1), _BRAND),
#         ]
#     t.setStyle(TableStyle(style))
#     return t


# def _banner(text: str, color) -> Table:
#     """Full-width coloured decision banner."""
#     t = Table([[text]], colWidths=[7.0 * inch])
#     t.setStyle(TableStyle([
#         ('BACKGROUND',    (0, 0), (-1, -1), color),
#         ('TEXTCOLOR',     (0, 0), (-1, -1), colors.white),
#         ('FONTNAME',      (0, 0), (-1, -1), 'Helvetica-Bold'),
#         ('FONTSIZE',      (0, 0), (-1, -1), 18),
#         ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
#         ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
#         ('TOPPADDING',    (0, 0), (-1, -1), 10),
#         ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
#     ]))
#     return t


# def _section_rule():
#     return HRFlowable(width='100%', thickness=0.5, color=_GREY,
#                       spaceAfter=4, spaceBefore=4)


# # ---------------------------------------------------------------------------
# # DECISION REPORT  (Stage 1 summary)
# # ---------------------------------------------------------------------------

# def generate_decision_pdf(decision_data: dict, customer_data: dict,
#                            affordability_data: dict, reasons: list) -> BytesIO:
#     """
#     Generate the Stage 1 credit decision summary PDF.

#     Parameters
#     ----------
#     decision_data : dict
#         decision           — 'APPROVE' | 'REJECT' | 'REVIEW'
#         risk_score         — int  (0-100 scale, Stage 1 engine output)
#         pd_percentage      — float  (%)
#         confidence         — float  (%)

#     customer_data : dict
#         application_id, timestamp,
#         age, employment_type, avg_salary_6m, bureau_score,
#         loan_amount, loan_tenure_months, interest_rate,
#         kyc_verified, gender, city_tier, rbi_consent,
#         dpd_90_count_6m, dpd_30_count_6m, credit_utilization_pct,
#         active_loans_count, salary_stability_flag, payment_discipline_flag,
#         net_cash_surplus_6m

#     affordability_data : dict
#         new_emi, existing_emi, total_emi, foir_percentage,
#         net_disposable, status, max_allowed_emi, emi_headroom

#     reasons : list[str]
#         Human-readable decision reason strings.

#     Returns
#     -------
#     BytesIO — seeked PDF buffer ready for download
#     """
#     buffer = BytesIO()
#     doc = SimpleDocTemplate(
#         buffer, pagesize=letter,
#         topMargin=0.45 * inch, bottomMargin=0.45 * inch,
#         leftMargin=0.65 * inch, rightMargin=0.65 * inch,
#     )
#     base, title_style, heading_style, sub_style, small_style = _styles()
#     story = []

#     # ── Title ─────────────────────────────────────────────────────────────────
#     story.append(Paragraph("CREDIT DECISION REPORT", title_style))
#     story.append(Spacer(1, 0.06 * inch))

#     # ── App ID / timestamp header bar ─────────────────────────────────────────
#     app_id    = customer_data.get('application_id', 'N/A')
#     timestamp = customer_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

#     story.append(_label_table(
#         [['Application ID:', app_id, 'Generated:', timestamp]],
#         [1.4 * inch, 2.4 * inch, 1.2 * inch, 2.0 * inch],
#         label_cols=(0, 2),
#     ))
#     story.append(Spacer(1, 0.12 * inch))

#     # ── Decision banner ───────────────────────────────────────────────────────
#     decision = decision_data.get('decision', 'ERROR')
#     if decision == 'APPROVE':
#         banner_color, banner_text = _GREEN,  'APPROVED'
#     elif decision == 'REJECT':
#         banner_color, banner_text = _RED,    'REJECTED'
#     else:
#         banner_color, banner_text = _ORANGE, 'REVIEW REQUIRED'

#     story.append(_banner(banner_text, banner_color))
#     story.append(Spacer(1, 0.14 * inch))

#     # ── Key decision metrics ──────────────────────────────────────────────────
#     risk_score   = _safe_int(decision_data.get('risk_score', 0))
#     pd_pct       = _safe_float(decision_data.get('pd_percentage', 0))
#     confidence   = _safe_float(decision_data.get('confidence', 0))
#     bureau_score = _safe_int(customer_data.get('bureau_score', 0))

#     story.append(Paragraph("DECISION METRICS", heading_style))
#     story.append(Spacer(1, 0.04 * inch))
#     story.append(_label_table([
#         ['Risk Score (0-100):',        f"{risk_score}/100",
#          'PD — Prob. of Default:',     f"{pd_pct:.2f}%"],
#         ['Model Confidence:',          f"{confidence:.1f}%",
#          'Bureau / CIBIL Score:',      str(bureau_score)],
#     ], [1.8 * inch, 1.7 * inch, 1.9 * inch, 1.6 * inch], label_cols=(0, 2)))
#     story.append(Spacer(1, 0.14 * inch))

#     # ── Customer information ──────────────────────────────────────────────────
#     story.append(Paragraph("CUSTOMER INFORMATION", heading_style))
#     story.append(Spacer(1, 0.04 * inch))

#     age          = _safe_int(customer_data.get('age', 0))
#     emp_type     = customer_data.get('employment_type', 'N/A')
#     income       = _safe_int(customer_data.get('avg_salary_6m', 0))
#     loan_amount  = _safe_int(customer_data.get('loan_amount', 0))
#     loan_tenure  = _safe_int(customer_data.get('loan_tenure_months', 0))
#     int_rate     = _safe_float(customer_data.get('interest_rate', 0))
#     kyc          = 'Verified' if customer_data.get('kyc_verified', True) else 'Not Verified'
#     gender       = customer_data.get('gender', 'N/A')
#     city_tier    = customer_data.get('city_tier', 'N/A')
#     rbi_consent  = 'Obtained' if customer_data.get('rbi_consent', False) else 'Not Obtained'
#     active_loans = _safe_int(customer_data.get('active_loans_count', 0))
#     dpd_90       = _safe_int(customer_data.get('dpd_90_count_6m', 0))
#     dpd_30       = _safe_int(customer_data.get('dpd_30_count_6m', 0))
#     credit_util  = _safe_float(customer_data.get('credit_utilization_pct', 0))
#     salary_stab  = customer_data.get('salary_stability_flag', 'N/A')
#     pay_disc     = customer_data.get('payment_discipline_flag', 'N/A')
#     net_surplus  = _safe_int(customer_data.get('net_cash_surplus_6m', 0))

#     story.append(_label_table([
#         ['Age:',              str(age),               'Employment Type:',    emp_type],
#         ['Gender:',           gender,                 'City Tier:',          city_tier],
#         ['Monthly Income:',   f"Rs.{income:,}",       'Bureau Score:',       str(bureau_score)],
#         ['Loan Amount:',      f"Rs.{loan_amount:,}",  'Tenure:',             f"{loan_tenure} months"],
#         ['Interest Rate:',    f"{int_rate:.2f}%",     'KYC Status:',         kyc],
#         ['RBI Consent:',      rbi_consent,            'Active Loans:',       str(active_loans)],
#         ['DPD 90+ (6M):',     str(dpd_90),            'DPD 30+ (6M):',       str(dpd_30)],
#         ['Credit Util.:',     f"{credit_util:.1f}%",  'Net Cash Surplus:',   f"Rs.{net_surplus:,}"],
#         ['Salary Stability:', salary_stab,            'Payment Discipline:', pay_disc],
#     ], [1.6 * inch, 1.9 * inch, 1.7 * inch, 1.8 * inch], label_cols=(0, 2)))
#     story.append(Spacer(1, 0.14 * inch))

#     # ── Affordability analysis ────────────────────────────────────────────────
#     story.append(Paragraph("AFFORDABILITY ANALYSIS", heading_style))
#     story.append(Spacer(1, 0.04 * inch))

#     new_emi      = _safe_float(affordability_data.get('new_emi', 0))
#     existing_emi = _safe_float(affordability_data.get('existing_emi', 0))
#     total_emi    = _safe_float(affordability_data.get('total_emi', 0))
#     foir         = _safe_float(affordability_data.get('foir_percentage', 0))
#     net_disp     = _safe_float(affordability_data.get('net_disposable', 0))
#     aff_status   = affordability_data.get('status', 'N/A')
#     max_emi      = _safe_float(affordability_data.get('max_allowed_emi', 0))
#     emi_headroom = _safe_float(affordability_data.get('emi_headroom', 0))

#     if foir <= 35:
#         foir_display = f"{foir:.2f}%  [Excellent — below 35%]"
#     elif foir <= 40:
#         foir_display = f"{foir:.2f}%  [Acceptable — 35-40%]"
#     elif foir <= 45:
#         foir_display = f"{foir:.2f}%  [High — Review, 40-45%]"
#     elif foir <= 50:
#         foir_display = f"{foir:.2f}%  [Elevated — 45-50%]"
#     else:
#         foir_display = f"{foir:.2f}%  [OVER-LEVERAGED > 50% — REJECT]"

#     story.append(_label_table([
#         ['New EMI:',          f"Rs.{new_emi:,.0f}",    'Existing EMI:',   f"Rs.{existing_emi:,.0f}"],
#         ['Total EMI:',        f"Rs.{total_emi:,.0f}",  'FOIR:',           foir_display],
#         ['Net Disposable:',   f"Rs.{net_disp:,.0f}",   'Aff. Status:',    aff_status],
#         ['Max Allowed EMI:',  f"Rs.{max_emi:,.0f}",    'EMI Headroom:',   f"Rs.{emi_headroom:,.0f}"],
#     ], [1.6 * inch, 1.9 * inch, 1.4 * inch, 2.1 * inch], label_cols=(0, 2)))
#     story.append(Spacer(1, 0.14 * inch))

#     # ── Risk indicators ───────────────────────────────────────────────────────
#     story.append(Paragraph("RISK INDICATORS", heading_style))
#     story.append(Spacer(1, 0.04 * inch))

#     if dpd_90 == 0:
#         dpd_label = "0  — Clean (passes hard gate)"
#     elif dpd_90 == 1:
#         dpd_label = "1  — Acceptable (passes hard gate)"
#     elif dpd_90 <= 5:
#         dpd_label = f"{dpd_90}  — Review zone (2-5)"
#     else:
#         dpd_label = f"{dpd_90}  — HARD REJECT (> 5)"

#     story.append(_label_table([
#         ['Risk Score (0-100):',           f"{risk_score}/100"],
#         ['PD (Probability of Default):',  f"{pd_pct:.2f}%"],
#         ['Model Confidence:',             f"{confidence:.1f}%"],
#         ['Bureau Score:',                 str(bureau_score)],
#         ['DPD 90+ (last 6M):',            dpd_label],
#         ['DPD 30+ (last 6M):',            str(dpd_30)],
#         ['Credit Utilisation:',           f"{credit_util:.1f}%"],
#         ['Net Cash Surplus (6M proxy):',  f"Rs.{net_surplus:,}"],
#     ], [2.8 * inch, 4.2 * inch], label_cols=(0,)))
#     story.append(Spacer(1, 0.14 * inch))

#     # ── Decision reasons ──────────────────────────────────────────────────────
#     story.append(Paragraph("DECISION REASONS", heading_style))
#     story.append(Spacer(1, 0.04 * inch))

#     if reasons:
#         r_rows = [[f"{i}.", str(r)] for i, r in enumerate(reasons, 1)]
#         rt = Table(r_rows, colWidths=[0.35 * inch, 6.65 * inch])
#         rt.setStyle(TableStyle([
#             ('TEXTCOLOR',      (0, 0), (-1, -1), colors.black),
#             ('ALIGN',          (0, 0), (0, -1),  'RIGHT'),
#             ('ALIGN',          (1, 0), (1, -1),  'LEFT'),
#             ('FONTSIZE',       (0, 0), (-1, -1), 9),
#             ('TOPPADDING',     (0, 0), (-1, -1), 4),
#             ('BOTTOMPADDING',  (0, 0), (-1, -1), 4),
#             ('LEFTPADDING',    (1, 0), (1, -1),  6),
#             ('VALIGN',         (0, 0), (-1, -1), 'TOP'),
#             ('ROWBACKGROUNDS', (0, 0), (-1, -1),
#              [colors.white, colors.HexColor('#f0f7ff')]),
#         ]))
#         story.append(rt)
#     else:
#         story.append(Paragraph("No reason codes recorded.", base['Normal']))

#     story.append(Spacer(1, 0.3 * inch))

#     # ── Footer ────────────────────────────────────────────────────────────────
#     story.append(_section_rule())
#     story.append(Paragraph(
#         f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; "
#         "Credit Risk Assessment System v8.7 &nbsp;|&nbsp; FOR INTERNAL USE ONLY",
#         small_style,
#     ))

#     doc.build(story)
#     buffer.seek(0)
#     return buffer


# # ---------------------------------------------------------------------------
# # AUDIT TRAIL PDF  (Stage 1 + optional Stage 2)
# # ---------------------------------------------------------------------------

# def generate_audit_pdf(audit_log: dict) -> BytesIO:
#     """
#     Generate the comprehensive audit trail PDF.

#     Parameters — audit_log dict keys
#     ---------------------------------
#     Core (Stage 1):
#         application_id, timestamp, model_version
#         decision            — Stage 1 decision string ('APPROVE'|'REJECT'|'REVIEW')
#         risk_score          — int, 0-100 scale
#         pd_percentage       — float %
#         confidence          — float %
#         policy_checks       — dict {gate_name: result_string}
#         pd_calculation_factors — dict with keys:
#                                    bureau_score, base_pd,
#                                    dpd_90, dpd_30, delinquency_multiplier,
#                                    foir, foir_adjustment, employment_adjustment,
#                                    inquiry_adjustment, ml_adjustment, final_pd
#         reason_codes        — list[str]

#     Stage 2 (optional — include only when Stage 2 ran):
#         stage2_final_decision     — str
#         stage2_tier               — 'P1' | 'P2' | 'P3' | 'P4'
#         stage2_interest_range     — str, e.g. '8.5-10%'
#         stage2_combined_risk_score — int, 0-1000 scale
#         stage2_confidence         — float %
#         stage2_reason             — str
#         stage2_tier_probabilities  — dict {tier: float}

#     Returns
#     -------
#     BytesIO — seeked PDF buffer ready for download
#     """
#     buffer = BytesIO()
#     doc = SimpleDocTemplate(
#         buffer, pagesize=letter,
#         topMargin=0.45 * inch, bottomMargin=0.45 * inch,
#         leftMargin=0.65 * inch, rightMargin=0.65 * inch,
#     )
#     base, title_style, heading_style, sub_style, small_style = _styles()
#     story = []

#     # ── Title ─────────────────────────────────────────────────────────────────
#     story.append(Paragraph("CREDIT RISK — AUDIT TRAIL REPORT", title_style))
#     story.append(Spacer(1, 0.06 * inch))

#     # ── Application header ────────────────────────────────────────────────────
#     app_id        = audit_log.get('application_id', 'N/A')
#     timestamp     = audit_log.get('timestamp', 'N/A')
#     s1_decision   = audit_log.get('decision', 'N/A')
#     s2_decision   = audit_log.get('stage2_final_decision', 'Not Run') or 'Not Run'
#     model_version = audit_log.get('model_version', '8.7')

#     s2_ran      = s2_decision not in ('Not Run', 'N/A', None, '')

#     s1_raw      = audit_log.get('risk_score', 0)
#     s2_raw      = audit_log.get('stage2_combined_risk_score', None)
#     risk_score  = _safe_int(s2_raw if (s2_ran and s2_raw) else s1_raw)
#     pd_pct      = _safe_float(audit_log.get('pd_percentage', 0))
#     confidence  = _safe_float(
#         audit_log.get('stage2_confidence', 0) if s2_ran
#         else audit_log.get('confidence', 0)
#     )
#     conf_label  = 'S2 Confidence:' if s2_ran else 'Model Confidence:'

#     if s2_ran:
#         score_label   = 'Combined Risk Score:'
#         score_display = f"{risk_score}  (0-1000 scale)"
#     else:
#         score_label   = 'Risk Score (0-100):'
#         score_display = f"{risk_score}/100"

#     story.append(_label_table([
#         ['Application ID:',    app_id,         'Timestamp:',         timestamp],
#         ['Stage 1 Decision:',  s1_decision,    'Stage 2 Decision:',  s2_decision],
#         [score_label,          score_display,  'PD Score:',          f"{pd_pct:.2f}%"],
#         [conf_label,           f"{confidence:.1f}%", 'Model Version:', model_version],
#     ], [1.8 * inch, 1.9 * inch, 1.7 * inch, 1.6 * inch], label_cols=(0, 2)))
#     story.append(Spacer(1, 0.14 * inch))

#     # ── Policy gate checks ────────────────────────────────────────────────────
#     story.append(Paragraph("POLICY GATE CHECKS", heading_style))
#     story.append(Spacer(1, 0.04 * inch))

#     policy_checks = audit_log.get('policy_checks', {})
#     if policy_checks:
#         pc_rows = []
#         for k, v in policy_checks.items():
#             label = k.replace('_', ' ').title()
#             val_str = str(v)
#             if val_str.upper() in ('PASS', 'TRUE', 'YES', '1', 'OK'):
#                 val_display = 'PASS  \u2713'
#             elif val_str.upper() in ('FAIL', 'FALSE', 'NO', '0', 'REJECT'):
#                 val_display = 'FAIL  \u2717'
#             else:
#                 val_display = val_str
#             pc_rows.append([label, val_display])
#         story.append(_label_table(pc_rows, [3.2 * inch, 3.8 * inch], label_cols=(0,)))
#     else:
#         story.append(Paragraph("No policy check data available.", base['Normal']))
#     story.append(Spacer(1, 0.14 * inch))

#     # ── PD calculation breakdown ──────────────────────────────────────────────
#     story.append(Paragraph("PD CALCULATION FACTORS", heading_style))
#     story.append(Spacer(1, 0.04 * inch))

#     pd_f = audit_log.get('pd_calculation_factors', {})

#     bureau_score = _safe_int(pd_f.get('bureau_score', 0))
#     base_pd      = _safe_float(pd_f.get('base_pd', 0))
#     dpd_90       = _safe_int(pd_f.get('dpd_90', 0))
#     dpd_30       = _safe_int(pd_f.get('dpd_30', 0))
#     deliq_mult   = _safe_float(pd_f.get('delinquency_multiplier', 1.0))
#     foir_val     = _safe_float(pd_f.get('foir', 0))
#     foir_adj     = _safe_float(pd_f.get('foir_adjustment', 0))
#     emp_adj      = _safe_float(pd_f.get('employment_adjustment', 0))
#     inq_adj      = _safe_float(pd_f.get('inquiry_adjustment', 0))
#     ml_adj       = _safe_float(pd_f.get('ml_adjustment', 0))
#     final_pd     = _safe_float(pd_f.get('final_pd', pd_pct))

#     # DPD 90 tier label mirrors hard-gate logic
#     if dpd_90 == 0:
#         dpd_tier = "0  — Clean (passes hard gate)"
#     elif dpd_90 == 1:
#         dpd_tier = "1  — Acceptable (passes hard gate)"
#     elif dpd_90 <= 5:
#         dpd_tier = f"{dpd_90}  — Review zone (2-5)"
#     else:
#         dpd_tier = f"{dpd_90}  — HARD REJECT (> 5)"

#     pd_rows = [
#         ['Bureau Score:',                str(bureau_score)],
#         ['Base PD (from score):',        f"{base_pd:.2f}%"],
#         ['DPD 90+ Count (6M):',          dpd_tier],
#         ['DPD 30+ Count (6M):',          str(dpd_30)],
#         ['Delinquency Multiplier:',      f"{deliq_mult:.2f}x"],
#         ['FOIR:',                        f"{foir_val:.2f}%"],
#         ['FOIR Adjustment:',             f"{foir_adj:+.2f}%"],
#         ['Employment Adjustment:',       f"{emp_adj:+.2f}%"],
#         ['Enquiry Adjustment:',          f"{inq_adj:+.2f}%"],
#         ['ML Model Adjustment:',         f"{ml_adj:+.2f}%"],
#         ['FINAL PD (capped 0.5-25%):',   f"{final_pd:.2f}%"],
#     ]

#     pd_table = Table(pd_rows, colWidths=[2.8 * inch, 4.2 * inch])
#     pd_table.setStyle(TableStyle([
#         ('BACKGROUND',    (0, 0),  (0, -1),  _LIGHT),
#         ('TEXTCOLOR',     (0, 0),  (-1, -1), colors.black),
#         ('ALIGN',         (0, 0),  (-1, -1), 'LEFT'),
#         ('FONTNAME',      (0, 0),  (0, -1),  'Helvetica-Bold'),
#         ('TEXTCOLOR',     (0, 0),  (0, -1),  _BRAND),
#         ('FONTSIZE',      (0, 0),  (-1, -1), 8.5),
#         ('TOPPADDING',    (0, 0),  (-1, -1), 5),
#         ('BOTTOMPADDING', (0, 0),  (-1, -1), 5),
#         ('LEFTPADDING',   (0, 0),  (-1, -1), 6),
#         ('GRID',          (0, 0),  (-1, -1), 0.4, _GREY),
#         # Highlight the final row
#         ('BACKGROUND',    (0, -1), (-1, -1), colors.HexColor('#dbeafe')),
#         ('FONTNAME',      (0, -1), (-1, -1), 'Helvetica-Bold'),
#     ]))
#     story.append(pd_table)
#     story.append(Spacer(1, 0.14 * inch))

#     # ── Stage 2 results (only when Stage 2 ran) ───────────────────────────────
#     if s2_ran:
#         story.append(Paragraph("STAGE 2 DEEP DIVE RESULTS", heading_style))
#         story.append(Spacer(1, 0.04 * inch))

#         s2_tier      = audit_log.get('stage2_tier', 'N/A')
#         s2_int_range = audit_log.get('stage2_interest_range', 'N/A')
#         s2_risk      = _safe_int(audit_log.get('stage2_combined_risk_score', 0))
#         s2_conf      = _safe_float(audit_log.get('stage2_confidence', 0))
#         s2_reason    = audit_log.get('stage2_reason', 'N/A')

#         s2_rows = [
#             ['Stage 2 Final Decision:',  str(s2_decision)],
#             ['Risk Tier:',               str(s2_tier)],
#             ['Interest Rate Range:',     str(s2_int_range)],
#             ['Combined Risk Score:',     f"{s2_risk}  (0-1000 scale)"],
#             ['Stage 2 Confidence:',      f"{s2_conf:.1f}%"],
#             ['Stage 2 Reason:',          str(s2_reason)],
#         ]
#         story.append(_label_table(s2_rows, [2.5 * inch, 4.5 * inch], label_cols=(0,)))

#         tier_probs = audit_log.get('stage2_tier_probabilities')
#         if tier_probs and isinstance(tier_probs, dict):
#             story.append(Spacer(1, 0.08 * inch))
#             story.append(Paragraph("Tier Probabilities:", sub_style))
#             tp_rows = [[tier, f"{prob:.1f}%"] for tier, prob in tier_probs.items()]
#             story.append(_label_table(tp_rows, [2.5 * inch, 4.5 * inch], label_cols=(0,)))

#         story.append(Spacer(1, 0.14 * inch))

#     # ── Decision reason codes ─────────────────────────────────────────────────
#     story.append(Paragraph("DECISION REASONS", heading_style))
#     story.append(Spacer(1, 0.04 * inch))

#     reasons_list = audit_log.get('reason_codes', [])
#     if reasons_list:
#         r_rows = [[f"{i}.", str(r)] for i, r in enumerate(reasons_list, 1)]
#         rt = Table(r_rows, colWidths=[0.35 * inch, 6.65 * inch])
#         rt.setStyle(TableStyle([
#             ('TEXTCOLOR',      (0, 0), (-1, -1), colors.black),
#             ('ALIGN',          (0, 0), (0, -1),  'RIGHT'),
#             ('ALIGN',          (1, 0), (1, -1),  'LEFT'),
#             ('FONTSIZE',       (0, 0), (-1, -1), 9),
#             ('TOPPADDING',     (0, 0), (-1, -1), 4),
#             ('BOTTOMPADDING',  (0, 0), (-1, -1), 4),
#             ('LEFTPADDING',    (1, 0), (1, -1),  6),
#             ('VALIGN',         (0, 0), (-1, -1), 'TOP'),
#             ('ROWBACKGROUNDS', (0, 0), (-1, -1),
#              [colors.white, colors.HexColor('#f0f7ff')]),
#         ]))
#         story.append(rt)
#     else:
#         story.append(Paragraph("No reason codes recorded.", base['Normal']))

#     story.append(Spacer(1, 0.3 * inch))

#     # ── Footer ────────────────────────────────────────────────────────────────
#     story.append(_section_rule())
#     story.append(Paragraph(
#         f"Audit Trail Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; "
#         "Credit Risk Assessment System v8.7",
#         small_style,
#     ))
#     story.append(Paragraph(
#         "This document is an official record of the credit assessment decision process. "
#         "FOR INTERNAL USE ONLY — NOT FOR DISTRIBUTION.",
#         small_style,
#     ))

#     doc.build(story)
#     buffer.seek(0)
#     return buffer









"""
PDF Generation Utility for Credit Risk Assessment
Author: Zen Meraki
Version: 8.7 — Corrected field names, risk score /100, DPD tiers, v8.7 footer
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime


# ---------------------------------------------------------------------------
# SHARED STYLE HELPERS
# ---------------------------------------------------------------------------
_BRAND   = colors.HexColor('#587042')
_LIGHT   = colors.HexColor('#f7fafc')
_GREY    = colors.HexColor('#e2e8f0')
_GREEN   = colors.HexColor('#48bb78')
_RED     = colors.HexColor('#f56565')
_ORANGE  = colors.HexColor('#ed8936')

def _styles():
    base = getSampleStyleSheet()
    title = ParagraphStyle('CRTitle', parent=base['Heading1'],
                           fontSize=20, textColor=_BRAND,
                           spaceAfter=10, alignment=1)
    heading = ParagraphStyle('CRHeading', parent=base['Heading2'],
                             fontSize=13, textColor=_BRAND,
                             spaceAfter=6, spaceBefore=10)
    small = ParagraphStyle('CRSmall', parent=base['Normal'],
                           fontSize=8, textColor=colors.grey, alignment=1)
    return base, title, heading, small


def _label_table(rows, col_widths, label_cols=(0,)):
    """Two-or-four column key-value table with shaded label cells."""
    t = Table(rows, colWidths=col_widths)
    style = [
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN',     (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE',  (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID',      (0, 0), (-1, -1), 0.5, _GREY),
    ]
    for col in label_cols:
        style.append(('BACKGROUND', (col, 0), (col, -1), _LIGHT))
        style.append(('FONTNAME',   (col, 0), (col, -1), 'Helvetica-Bold'))
    t.setStyle(TableStyle(style))
    return t


def _safe_int(v, default=0):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# DECISION REPORT  (Stage 1 — quick summary)
# ---------------------------------------------------------------------------
def generate_decision_pdf(decision_data, customer_data, affordability_data, reasons):
    """
    Generate the Stage 1 decision summary PDF.

    All numbers are taken directly from the dicts passed in — no re-calculation.
    Risk score is shown /100 (engine produces 0-100, not 0-1000).
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            topMargin=0.5*inch, bottomMargin=0.5*inch,
                            leftMargin=0.6*inch, rightMargin=0.6*inch)
    base, title_style, heading_style, small_style = _styles()
    story = []

    # ── Title ────────────────────────────────────────────────────────────────
    story.append(Paragraph("CREDIT DECISION REPORT", title_style))
    story.append(Spacer(1, 0.15*inch))

    # ── Decision banner ───────────────────────────────────────────────────────
    decision   = decision_data.get('decision', 'ERROR')
    risk_score = _safe_int(decision_data.get('risk_score', 0))
    pd_pct     = _safe_float(decision_data.get('pd_percentage', 0))
    confidence = _safe_float(decision_data.get('confidence', 0))
    app_id     = customer_data.get('application_id', 'N/A')
    timestamp  = customer_data.get('timestamp',
                                   datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    if decision == 'APPROVE':
        dec_color, dec_icon, dec_bg = _GREEN,  "✔  APPROVED",        colors.HexColor('#f0fff4')
    elif decision == 'REJECT':
        dec_color, dec_icon, dec_bg = _RED,    "✘  REJECTED",         colors.HexColor('#fff5f5')
    else:
        dec_color, dec_icon, dec_bg = _ORANGE, "⚑  REVIEW REQUIRED",  colors.HexColor('#fffbeb')

    # Full-width coloured banner table — dominant visual
    banner_tbl = Table([[dec_icon]], colWidths=[7.0 * inch])
    banner_tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), dec_color),
        ('TEXTCOLOR',     (0, 0), (-1, -1), colors.white),
        ('FONTNAME',      (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 28),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',    (0, 0), (-1, -1), 18),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 18),
    ]))
    story.append(banner_tbl)

    # ── Header table: App ID / Timestamp / Risk Score / PD / Confidence ──────
    story.append(_label_table([
        ['Application ID:', app_id,    'Timestamp:',   timestamp],
        ['Decision:',       decision,   'Risk Score:',  f"{risk_score}/100"],
        ['PD Score:',       f"{pd_pct:.2f}%", 'Confidence:', f"{confidence:.1f}%"],
    ], [1.3*inch, 2.2*inch, 1.3*inch, 2.2*inch], label_cols=(0, 2)))
    story.append(Spacer(1, 0.25*inch))

    # ── Customer information ──────────────────────────────────────────────────
    story.append(Paragraph("CUSTOMER INFORMATION", heading_style))

    age             = _safe_int(customer_data.get('age', 0))
    emp_type        = customer_data.get('employment_type', 'N/A')
    income          = _safe_int(customer_data.get('avg_salary_6m', 0))
    bureau_score    = _safe_int(customer_data.get('bureau_score', 0))
    loan_amount     = _safe_int(customer_data.get('loan_amount', 0))
    loan_tenure     = _safe_int(customer_data.get('loan_tenure_months', 0))
    interest_rate   = _safe_float(customer_data.get('interest_rate', 0))
    kyc             = 'Verified' if customer_data.get('kyc_verified', True) else 'Not Verified'
    gender          = customer_data.get('gender', 'N/A')
    city_tier       = customer_data.get('city_tier', 'N/A')
    rbi_consent     = 'Obtained' if customer_data.get('rbi_consent', False) else 'Not Obtained'
    dpd_90          = _safe_int(customer_data.get('dpd_90_count_6m', 0))
    dpd_30          = _safe_int(customer_data.get('dpd_30_count_6m', 0))
    credit_util     = _safe_float(customer_data.get('credit_utilization_pct', 0))
    active_loans    = _safe_int(customer_data.get('active_loans_count', 0))
    salary_stab     = customer_data.get('salary_stability_flag', 'N/A')
    pay_disc        = customer_data.get('payment_discipline_flag', 'N/A')

    story.append(_label_table([
        ['Age:',            str(age),                 'Employment:',      emp_type],
        ['Gender:',         gender,                   'City Tier:',       city_tier],
        ['Monthly Income:', f"Rs.{income:,}",         'Bureau Score:',    str(bureau_score)],
        ['Loan Amount:',    f"Rs.{loan_amount:,}",    'Tenure:',          f"{loan_tenure} months"],
        ['Interest Rate:',  f"{interest_rate:.2f}%",  'KYC Status:',      kyc],
        ['RBI Consent:',    rbi_consent,              'Active Loans:',    str(active_loans)],
        ['DPD 90+ (6M):',   str(dpd_90),             'DPD 30+ (6M):',   str(dpd_30)],
        ['Credit Util.:',   f"{credit_util:.1f}%",    'Salary Stability:', salary_stab],
        ['Payment Discipline:', pay_disc,             '', ''],
    ], [1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch], label_cols=(0, 2)))
    story.append(Spacer(1, 0.25*inch))

    # ── Affordability analysis ────────────────────────────────────────────────
    story.append(Paragraph("AFFORDABILITY ANALYSIS", heading_style))

    new_emi      = _safe_float(affordability_data.get('new_emi', 0))
    existing_emi = _safe_float(affordability_data.get('existing_emi', 0))
    total_emi    = _safe_float(affordability_data.get('total_emi', 0))
    foir         = _safe_float(affordability_data.get('foir_percentage', 0))
    net_disp     = _safe_float(affordability_data.get('net_disposable', 0))
    aff_status   = affordability_data.get('status', 'N/A')
    max_emi      = _safe_float(affordability_data.get('max_allowed_emi', 0))
    emi_headroom = _safe_float(affordability_data.get('emi_headroom', 0))

    story.append(_label_table([
        ['New EMI:',         f"Rs.{new_emi:,.0f}",      'Existing EMI:',   f"Rs.{existing_emi:,.0f}"],
        ['Total EMI:',       f"Rs.{total_emi:,.0f}",    'FOIR:',           f"{foir:.2f}%"],
        ['Net Disposable:',  f"Rs.{net_disp:,.0f}",     'Status:',         aff_status],
        ['Max Allowed EMI:', f"Rs.{max_emi:,.0f}",      'EMI Headroom:',   f"Rs.{emi_headroom:,.0f}"],
    ], [1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch], label_cols=(0, 2)))
    story.append(Spacer(1, 0.25*inch))

    # ── Decision reasons ──────────────────────────────────────────────────────
    story.append(Paragraph("DECISION REASONS", heading_style))
    if reasons:
        reason_rows = [[f"{i}.", r] for i, r in enumerate(reasons, 1)]
        rt = Table(reason_rows, colWidths=[0.4*inch, 6.6*inch])
        rt.setStyle(TableStyle([
            ('TEXTCOLOR',     (0, 0), (-1, -1), colors.black),
            ('ALIGN',         (0, 0), (0, -1),  'RIGHT'),
            ('ALIGN',         (1, 0), (1, -1),  'LEFT'),
            ('FONTSIZE',      (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(rt)
    story.append(Spacer(1, 0.25*inch))

    # ── Risk assessment ───────────────────────────────────────────────────────
    story.append(Paragraph("RISK ASSESSMENT", heading_style))

    # DPD tier label
    if dpd_90 == 0:
        dpd_label = f"{dpd_90} (Clean)"
    elif dpd_90 == 1:
        dpd_label = f"{dpd_90} (Acceptable)"
    elif dpd_90 <= 5:
        dpd_label = f"{dpd_90} (Review Zone: 2-5)"
    else:
        dpd_label = f"{dpd_90} (REJECT: >5)"

    story.append(_label_table([
        ['Risk Score (0-100):',           f"{risk_score}/100"],
        ['PD (Probability of Default):',  f"{pd_pct:.2f}%"],
        ['Model Confidence:',             f"{confidence:.1f}%"],
        ['Bureau Score:',                 str(bureau_score)],
        ['DPD 90+ (6M):',                dpd_label],
        ['DPD 30+ (6M):',                str(dpd_30)],
        ['Credit Utilization:',           f"{credit_util:.1f}%"],
        ['Net Cash Surplus:',             f"Rs.{_safe_int(customer_data.get('net_cash_surplus_6m', 0)):,}"],
    ], [2.8*inch, 4.2*inch], label_cols=(0,)))
    story.append(Spacer(1, 0.4*inch))

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "Credit Risk Assessment System v8.7 | FOR INTERNAL USE ONLY",
        small_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ---------------------------------------------------------------------------
# AUDIT TRAIL PDF  (Stage 1 + optional Stage 2)
# ---------------------------------------------------------------------------
def generate_audit_pdf(audit_log):
    """
    Generate comprehensive audit trail PDF covering Stage 1 policy gates,
    PD calculation breakdown, and — if present — Stage 2 deep-dive results.

    Key field mapping (audit_log dict):
        application_id, timestamp, model_version
        decision           — Stage 1 decision
        risk_score         — Stage 1 risk score  (0-100)
        pd_percentage      — Stage 1 final PD %
        confidence         — Stage 1 model confidence %
        policy_checks      — dict of gate results
        pd_calculation_factors  — dict with base_pd, multiplier, adjustments
        reason_codes       — list of strings
        stage2_final_decision, stage2_tier, stage2_interest_range,
        stage2_combined_risk_score, stage2_confidence, stage2_reason,
        stage2_tier_probabilities  — optional Stage 2 fields
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            topMargin=0.5*inch, bottomMargin=0.5*inch,
                            leftMargin=0.6*inch, rightMargin=0.6*inch)
    base, title_style, heading_style, small_style = _styles()
    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("AUDIT TRAIL REPORT", title_style))
    story.append(Spacer(1, 0.15*inch))

    # ── Application header ────────────────────────────────────────────────────
    app_id           = audit_log.get('application_id', 'N/A')
    timestamp        = audit_log.get('timestamp', 'N/A')
    s1_decision      = audit_log.get('decision', 'N/A')
    s2_decision      = audit_log.get('stage2_final_decision', 'Not Run')
    model_version    = audit_log.get('model_version', '8.7')
    # If Stage 2 ran, show Stage 2 values in the header — they are the final binding numbers.
    # Stage 1 values are kept as fallback if Stage 2 was not run.
    s2_ran           = s2_decision not in ('Not Run', 'N/A', None, '')
    s2_raw_score     = audit_log.get('stage2_combined_risk_score', None)
    s1_raw_score     = audit_log.get('risk_score', 0)
    # Use Stage 2 combined score when available and non-zero; else Stage 1 score
    risk_score       = _safe_int(s2_raw_score if (s2_ran and s2_raw_score) else s1_raw_score)
    pd_pct        = _safe_float(audit_log.get('pd_percentage', 0))
    confidence    = _safe_float(audit_log.get('stage2_confidence', 0)
                                if s2_ran else audit_log.get('confidence', 0))
    conf_label    = 'S2 Confidence:' if s2_ran else 'Model Confidence:'
    # FIX P-1: When Stage 2 has run, combined_risk_score is on a 0-1000 scale
    # (Stage 1 normalised × 0.4 + Stage 2 tier score × 0.6). The old code always
    # appended '/100', which was misleading for the combined score.
    # Now: Stage 1 only → show '/100'; Stage 2 combined → show '(0-1000 scale)'.
    if s2_ran:
        score_label       = 'Combined Risk Score:'
        risk_score_display = f"{risk_score} (0-1000 scale)"
    else:
        score_label       = 'Risk Score (0-100):'
        risk_score_display = f"{risk_score}/100"

    story.append(_label_table([
        ['Application ID:',    app_id,       'Timestamp:',        timestamp],
        ['Stage 1 Decision:',  s1_decision,  'Stage 2 Decision:', s2_decision],
        [score_label,          risk_score_display, 'PD Score:',   f"{pd_pct:.2f}%"],
        [conf_label,           f"{confidence:.1f}%", 'Version:',  model_version],
    ], [1.7*inch, 2.1*inch, 1.7*inch, 1.5*inch], label_cols=(0, 2)))
    story.append(Spacer(1, 0.14*inch))

    # ── Prominent final decision banner ──────────────────────────────────────
    _final_dec = s2_decision if s2_ran else s1_decision
    if _final_dec == 'APPROVE':
        _bc, _bt = _GREEN,  '✔  APPROVED'
    elif _final_dec == 'REJECT':
        _bc, _bt = _RED,    '✘  REJECTED'
    else:
        _bc, _bt = _ORANGE, '⚑  REVIEW REQUIRED'
    _suffix = ' (Stage 2 Final)' if s2_ran else ' (Stage 1)'
    _audit_banner = Table([[_bt + _suffix]], colWidths=[7.0 * inch])
    _audit_banner.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), _bc),
        ('TEXTCOLOR',     (0, 0), (-1, -1), colors.white),
        ('FONTNAME',      (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 24),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',    (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(_audit_banner)
    story.append(Spacer(1, 0.2*inch))

    # ── Policy gate checks ────────────────────────────────────────────────────
    story.append(Paragraph("POLICY GATE CHECKS", heading_style))
    policy_checks = audit_log.get('policy_checks', {})
    if policy_checks:
        pc_rows = [[k.replace('_', ' ').title(), str(v)]
                   for k, v in policy_checks.items()]
        story.append(_label_table(pc_rows, [2.0*inch, 5.0*inch], label_cols=(0,)))
    else:
        story.append(Paragraph("No policy check data available.", base['Normal']))
    story.append(Spacer(1, 0.25*inch))

    # ── PD calculation breakdown ──────────────────────────────────────────────
    story.append(Paragraph("PD CALCULATION FACTORS", heading_style))
    pd_f = audit_log.get('pd_calculation_factors', {})

    bureau_score       = _safe_int(pd_f.get('bureau_score', 0))
    base_pd            = _safe_float(pd_f.get('base_pd', 0))
    dpd_90             = _safe_int(pd_f.get('dpd_90', 0))
    dpd_30             = _safe_int(pd_f.get('dpd_30', 0))
    deliq_mult         = _safe_float(pd_f.get('delinquency_multiplier', 1.0))
    foir_val           = _safe_float(pd_f.get('foir', 0))
    foir_adj           = _safe_float(pd_f.get('foir_adjustment', 0))
    emp_adj            = _safe_float(pd_f.get('employment_adjustment', 0))
    inq_adj            = _safe_float(pd_f.get('inquiry_adjustment', 0))
    ml_adj             = _safe_float(pd_f.get('ml_adjustment', 0))
    final_pd           = _safe_float(pd_f.get('final_pd', pd_pct))

    # DPD tier label
    if dpd_90 == 0:
        dpd_tier = "0 (Clean — pass)"
    elif dpd_90 == 1:
        dpd_tier = "1 (Acceptable — pass)"
    elif dpd_90 <= 5:
        dpd_tier = f"{dpd_90} (Review zone 2-5)"
    else:
        dpd_tier = f"{dpd_90} (REJECT — exceeds 5)"

    pd_rows = [
        ['Bureau Score:',           str(bureau_score)],
        ['Base PD:',                f"{base_pd:.2f}%"],
        ['DPD 90+ Count:',          dpd_tier],
        ['DPD 30+ Count:',          str(dpd_30)],
        ['Delinquency Multiplier:', f"{deliq_mult:.2f}x"],
        ['FOIR:',                   f"{foir_val:.2f}%"],
        ['FOIR Adjustment:',        f"{foir_adj:+.2f}%"],
        ['Employment Adjustment:',  f"{emp_adj:+.2f}%"],
        ['Inquiry Adjustment:',     f"{inq_adj:+.2f}%"],
        ['ML Model Adjustment:',    f"{ml_adj:+.2f}%"],
        ['FINAL PD:',               f"{final_pd:.2f}%"],
    ]

    pd_table = Table(pd_rows, colWidths=[2.5*inch, 4.5*inch])
    pd_style = [
        ('BACKGROUND',    (0, 0),  (0, -1),  _LIGHT),
        ('BACKGROUND',    (0, -1), (-1, -1), colors.HexColor('#edf2f7')),
        ('TEXTCOLOR',     (0, 0),  (-1, -1), colors.black),
        ('ALIGN',         (0, 0),  (-1, -1), 'LEFT'),
        ('FONTNAME',      (0, 0),  (0, -1),  'Helvetica-Bold'),
        ('FONTNAME',      (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0),  (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0),  (-1, -1), 6),
        ('GRID',          (0, 0),  (-1, -1), 0.5, _GREY),
    ]
    pd_table.setStyle(TableStyle(pd_style))
    story.append(pd_table)
    story.append(Spacer(1, 0.25*inch))

    # ── Stage 2 results (if available) ───────────────────────────────────────
    if audit_log.get('stage2_final_decision') and \
       audit_log.get('stage2_final_decision') not in ('N/A', 'Not Run', None):

        story.append(Paragraph("STAGE 2 DEEP DIVE RESULTS", heading_style))

        s2_tier      = audit_log.get('stage2_tier', 'N/A')
        s2_int_range = audit_log.get('stage2_interest_range', 'N/A')
        s2_risk      = _safe_int(audit_log.get('stage2_combined_risk_score', 0))
        s2_conf      = _safe_float(audit_log.get('stage2_confidence', 0))
        s2_reason    = audit_log.get('stage2_reason', 'N/A')

        s2_rows = [
            ['Stage 2 Final Decision:',  s2_decision],
            ['Risk Tier:',               s2_tier],
            ['Interest Rate Range:',     s2_int_range],
            ['Combined Risk Score:',     str(s2_risk)],
            ['Stage 2 Confidence:',      f"{s2_conf:.1f}%"],
            ['Stage 2 Reason:',          str(s2_reason)],
        ]
        story.append(_label_table(s2_rows, [2.5*inch, 4.5*inch], label_cols=(0,)))

        # Tier probabilities
        tier_probs = audit_log.get('stage2_tier_probabilities')
        if tier_probs and isinstance(tier_probs, dict):
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Tier Probabilities:", base['Normal']))
            tp_rows = [[tier, f"{prob:.1f}%"] for tier, prob in tier_probs.items()]
            story.append(_label_table(tp_rows, [2.5*inch, 4.5*inch], label_cols=(0,)))

        story.append(Spacer(1, 0.25*inch))

    # ── Decision reason codes ─────────────────────────────────────────────────
    story.append(Paragraph("DECISION REASONS", heading_style))
    reasons = audit_log.get('reason_codes', [])
    if reasons:
        r_rows = [[f"{i}.", str(r)] for i, r in enumerate(reasons, 1)]
        rt = Table(r_rows, colWidths=[0.4*inch, 6.6*inch])
        rt.setStyle(TableStyle([
            ('TEXTCOLOR',     (0, 0), (-1, -1), colors.black),
            ('ALIGN',         (0, 0), (0, -1),  'RIGHT'),
            ('ALIGN',         (1, 0), (1, -1),  'LEFT'),
            ('FONTSIZE',      (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(rt)
    else:
        story.append(Paragraph("No reason codes recorded.", base['Normal']))

    story.append(Spacer(1, 0.4*inch))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Paragraph(
        f"Audit Trail Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "Credit Risk Assessment System v8.7",
        small_style))
    story.append(Paragraph(
        "This document is an official record of the credit assessment decision process. "
        "FOR INTERNAL USE ONLY.",
        small_style))

    doc.build(story)
    buffer.seek(0)
    return buffer
