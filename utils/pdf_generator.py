"""
PDF Summary Generator
Generates downloadable decision summary reports
"""

from turtle import st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io


def generate_decision_pdf(decision_data, customer_data, affordability_data, reasons):
    """
    Generate PDF summary of credit decision
    """
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph(f"<b>Credit Decision Summary</b>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    # Application details
    app_info = f"""
    <b>Application ID:</b> {decision_data['application_id']}<br/>
    <b>Decision Date:</b> {decision_data['timestamp']}<br/>
    <b>Decision:</b> {decision_data['decision']}<br/>
    <b>Risk Score:</b> {decision_data['risk_score']}/1000<br/>
    <b>PD:</b> {decision_data['pd_percentage']}%
    """
    elements.append(Paragraph(app_info, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Customer details table
    customer_table_data = [
        ['Field', 'Value'],
        ['Name', customer_data.get('name', 'N/A')],
        ['Age', str(customer_data.get('age', 'N/A'))],
        ['Bureau Score', str(customer_data.get('bureau_score', 'N/A'))],
        ['Monthly Income', f"₹{customer_data.get('avg_salary_6m', 0):,}"],
    ]
    
    customer_table = Table(customer_table_data, colWidths=[3*inch, 3*inch])
    customer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(customer_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Affordability breakdown
    affordability_text = f"""
    <b>Affordability Assessment:</b><br/>
    Monthly Income: ₹{affordability_data['monthly_income']:,}<br/>
    Total EMI: ₹{affordability_data['total_emi']:,}<br/>
    FOIR: {affordability_data['foir_percentage']}%<br/>
    Net Disposable: ₹{affordability_data['net_disposable']:,}
    """
    elements.append(Paragraph(affordability_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Reason codes
    reasons_text = "<b>Decision Reasons:</b><br/>"
    for i, reason in enumerate(reasons, 1):
        reasons_text += f"{i}. {reason}<br/>"
    
    elements.append(Paragraph(reasons_text, styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return buffer


# Usage in Streamlit
def add_download_button(decision_data, customer_data, affordability_data, reasons):
    """Add download button to Streamlit page"""
    
    pdf_buffer = generate_decision_pdf(
        decision_data, customer_data, affordability_data, reasons
    )
    
    st.download_button(
        label="📥 Download Decision Summary",
        data=pdf_buffer,
        file_name=f"credit_decision_{decision_data['application_id']}.pdf",
        mime="application/pdf",
        use_container_width=True
    )