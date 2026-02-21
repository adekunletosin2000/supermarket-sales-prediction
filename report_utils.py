from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Spacer
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Spacer
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.platypus import Spacer

def generate_pdf(filename, prediction, confidence):
    doc = SimpleDocTemplate(filename)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Supermarket Sales AI Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph(f"Predicted Total Sales: ${prediction:,.2f}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Model Confidence: {confidence}%", styles["Normal"]))

    doc.build(elements)