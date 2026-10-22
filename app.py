from flask import Flask, render_template, request, send_file
import os
from inference import analyze_image
from generate_pdf import generate_pdf
from evaluation_utils import read_training_metrics

app = Flask(__name__)

# Ensure required folders exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/reports", exist_ok=True)
os.makedirs("static/results", exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        patient = request.form.get("patient")
        image = request.files.get("image")

        # Save uploaded image
        upload_path = "static/uploads/upload.png"
        image.save(upload_path)

        # Run AI detection
        results = analyze_image(upload_path)

        # Generate PDF report
        pdf_path = "static/reports/report.pdf"

        generate_pdf(
            patient,
            results["severity"],
            results["findings"],
            upload_path,                 # original uploaded image
            results["detected_image"],   # detected regions image
            pdf_path
        )

        return render_template(
            "result.html",
            patient=patient,
            original=upload_path,
            severity=results["severity"],
            findings=results["findings"],
            detected=results["detected_image"],
            heatmap=results["heatmap_image"]
        )

    return render_template("index.html")


@app.route("/evaluation")
def evaluation():

    metrics = read_training_metrics()

    return render_template(
        "evaluation.html",
        metrics=metrics
    )


@app.route("/download")
def download():

    pdf_path = "static/reports/report.pdf"

    return send_file(
        pdf_path,
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)