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
PDF Summary Generator
Generates downloadable decision summary reports with proper formatting
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import io


def generate_decision_pdf(decision_data, customer_data, affordability_data, reasons):
    """Generate a professional PDF report for credit decision"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#587042'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#587042'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.black,
        alignment=TA_LEFT
    )
    
    # -----------------------------
    # Title
    # -----------------------------
    elements.append(Paragraph("Credit Decision Summary", title_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # -----------------------------
    # Application Details
    # -----------------------------
    app_id = decision_data.get('application_id', 'N/A')
    timestamp = decision_data.get('timestamp', 'N/A')
    decision = decision_data.get('decision', 'N/A')
    risk_score = decision_data.get('risk_score', 'N/A')
    pd_percentage = decision_data.get('pd_percentage', 'N/A')
    
    app_info = f"""
    <b>Application ID:</b> {app_id}<br/>
    <b>Decision Date:</b> {timestamp}<br/>
    <b>Decision:</b> {decision}<br/>
    <b>Risk Score:</b> {risk_score} / 1000<br/>
    <b>PD:</b> {pd_percentage}%
    """
    elements.append(Paragraph(app_info, normal_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # -----------------------------
    # Customer Details Table
    # -----------------------------
    customer_table_data = [
        ['Field', 'Value'],
        ['Age', str(customer_data.get('age', 'N/A'))],
        ['Employment Type', str(customer_data.get('employment_type', 'N/A'))],
        ['Bureau Score', str(customer_data.get('bureau_score', 'N/A'))],
        ['Monthly Income', f"Rs.{customer_data.get('avg_salary_6m', 0):,}"],
        ['Loan Amount', f"Rs.{customer_data.get('loan_amount', 0):,}"],
    ]
    
    customer_table = Table(customer_table_data, colWidths=[3 * inch, 3 * inch])
    customer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#587042')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FAF7E6')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(customer_table)
    elements.append(Spacer(1, 0.3 * inch))
    
    # -----------------------------
    # Affordability Section
    # -----------------------------
    elements.append(Paragraph("Affordability Assessment", heading_style))
    
    monthly_income = affordability_data.get('monthly_income', 0)
    total_emi = affordability_data.get('total_emi', 0)
    foir = affordability_data.get('foir_percentage', 'N/A')
    net_disposable = affordability_data.get('net_disposable', 0)
    
    affordability_text = f"""
    Monthly Income: Rs.{monthly_income:,}<br/>
    Total EMI: Rs.{total_emi:,}<br/>
    FOIR: {foir}%<br/>
    Net Disposable: Rs.{net_disposable:,}
    """
    elements.append(Paragraph(affordability_text, normal_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # -----------------------------
    # Reason Codes
    # -----------------------------
    elements.append(Paragraph("Decision Reasons", heading_style))
    
    for i, reason in enumerate(reasons, 1):
        # Clean the reason text - remove special characters that might cause issues
        clean_reason = str(reason).replace('₹', 'Rs.')
        elements.append(Paragraph(f"{i}. {clean_reason}", normal_style))
        elements.append(Spacer(1, 0.1 * inch))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer
