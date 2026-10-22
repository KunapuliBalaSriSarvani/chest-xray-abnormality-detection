from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

def generate_pdf(patient, severity, findings, original_img, detected_img, pdf_path):

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # ===== HEADER =====
    c.setFont("Helvetica-Bold",20)
    c.drawString(50,height-50,"AI Chest Imaging Center")

    c.setFont("Helvetica",11)
    c.drawString(50,height-70,"Specialized Center for Chest X-ray Abnormality Detection")

    c.line(40,height-90,width-40,height-90)

    # ===== PATIENT DETAILS =====
    c.setFont("Helvetica-Bold",12)
    c.drawString(50,height-120,"Patient Information")

    c.setFont("Helvetica",11)

    c.drawString(60,height-150,f"Patient Name : {patient}")
    c.drawString(60,height-170,"Examination : Chest X-ray")

    c.drawString(350,height-150,f"Severity Level : {severity}")
    c.drawString(350,height-170,f"Report Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    c.line(40,height-200,width-40,height-200)

    # ===== IMAGES =====
    c.setFont("Helvetica-Bold",12)
    c.drawString(50,height-230,"Radiology Visualization")

    c.setFont("Helvetica",10)

    c.drawString(90,height-250,"Original X-ray")
    import os
    import os

    # Convert to absolute paths
    original_img = os.path.abspath(original_img)
    detected_img = os.path.abspath(detected_img)

    # Draw images safely
    if os.path.exists(original_img):
        c.drawImage(original_img, 50, height - 450, width=220, height=180)

    if os.path.exists(detected_img):
        c.drawImage(detected_img, 320, height - 450, width=220, height=180)
    c.drawString(360,height-250,"Detected Regions")
    c.drawImage(detected_img,320,height-450,width=220,height=180)

    # ===== FINDINGS =====
    c.setFont("Helvetica-Bold",12)
    c.drawString(50,height-480,"Detected Abnormalities")

    y = height-510
    c.setFont("Helvetica",11)

    if "No abnormalities detected" in findings:
        c.drawString(70,y,"• No abnormalities detected")
    else:
        for f in findings:
            c.drawString(70,y,f"• {f}")
            y -= 20

    # ===== IMPRESSION =====
    c.setFont("Helvetica-Bold",12)
    c.drawString(50,y-20,"Impression")

    c.setFont("Helvetica",11)
    c.drawString(70,y-50,f"AI Severity Assessment : {severity}")

    # ===== RECOMMENDATION =====
    c.setFont("Helvetica-Bold",12)
    c.drawString(50,y-90,"Clinical Recommendation")

    c.setFont("Helvetica",11)
    c.drawString(
        70,
        y-120,
        "Radiological correlation advised. Clinical evaluation recommended."
    )

    # ===== SIGNATURE =====
    c.line(50,120,250,120)
    c.drawString(90,100,"Radiology Technologist")

    c.line(320,120,520,120)
    c.drawString(380,100,"AI Diagnostic System")

    # ===== FOOTER =====
    c.line(40,80,width-40,80)

    c.setFont("Helvetica",9)
    c.drawString(
        50,
        60,
        "Disclaimer: AI generated report for demonstration purposes. Radiologist confirmation required."
    )

    c.save()