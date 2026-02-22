# report_utils.py
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def generate_pdf(filename, prediction):
    doc = SimpleDocTemplate(filename)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("<b>Supermarket Sales AI Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph(f"Predicted Total Sales: ${prediction:,.2f}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    #elements.append(Paragraph(f"Model Confidence: {confidence}%", styles["Normal"]))

    doc.build(elements)