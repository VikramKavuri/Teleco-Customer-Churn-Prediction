from __future__ import annotations

import io
import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from modeling import (
    InputValidationError,
    METRICS_PATH,
    build_single_record,
    ensure_model_artifact,
    parse_uploaded_file,
    predict_dataframe,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".pdf"}

artifact_bundle = ensure_model_artifact()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "telco-churn-demo-secret")
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

    @app.route("/", methods=["GET"])
    def home():
        return render_dashboard()

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok", "model": artifact_bundle["deployed_model_name"]}, 200

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            record = build_single_record(request.form)
            result = predict_dataframe(record, artifact_bundle).iloc[0].to_dict()
            return render_dashboard(single_result=result, form_state=request.form.to_dict())
        except InputValidationError as exc:
            flash(str(exc), "error")
            return render_dashboard(form_state=request.form.to_dict()), 400

    @app.route("/batch", methods=["POST"])
    def batch_predict():
        uploaded_file = request.files.get("customer_file")
        if not uploaded_file or not uploaded_file.filename:
            flash("Please choose a file to score.", "error")
            return redirect(url_for("home"))

        file_suffix = Path(uploaded_file.filename).suffix.lower()
        if file_suffix not in ALLOWED_EXTENSIONS:
            flash("Supported file types are CSV, XLSX, JSON, and PDF.", "error")
            return redirect(url_for("home"))

        file_name = f"{uuid4().hex}_{secure_filename(uploaded_file.filename)}"
        saved_path = UPLOAD_DIR / file_name
        uploaded_file.save(saved_path)

        try:
            uploaded_df = parse_uploaded_file(saved_path)
            result_df = predict_dataframe(uploaded_df, artifact_bundle)
        except InputValidationError as exc:
            flash(str(exc), "error")
            return redirect(url_for("home"))

        result_path = UPLOAD_DIR / f"{saved_path.stem}_predictions.csv"
        result_df.to_csv(result_path, index=False)
        summary = {
            "rows": int(result_df.shape[0]),
            "average_probability": round(float(result_df["churn_probability"].mean()), 4),
            "high_risk_customers": int((result_df["churn_probability"] >= 0.6).sum()),
        }
        preview_html = result_df.head(25).to_html(
            classes="table table-striped",
            index=False,
            float_format=lambda value: f"{value:.2%}" if value <= 1 else f"{value:.2f}",
        )
        return render_dashboard(
            batch_preview=preview_html,
            batch_summary=summary,
            download_name=result_path.name,
        )

    @app.route("/download/<path:file_name>", methods=["GET"])
    def download(file_name: str):
        file_path = UPLOAD_DIR / secure_filename(file_name)
        if not file_path.exists():
            flash("That results file is no longer available.", "error")
            return redirect(url_for("home"))
        return send_file(file_path, as_attachment=True, download_name=file_path.name)

    return app


def render_dashboard(
    single_result: dict | None = None,
    batch_preview: str | None = None,
    batch_summary: dict | None = None,
    download_name: str | None = None,
    form_state: dict | None = None,
):
    ui_metadata = artifact_bundle["ui_metadata"]
    metrics = artifact_bundle["metrics"][artifact_bundle["deployed_model_name"]]
    model_display_names = {
        "xgboost": "XGBoost",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
    }
    metrics["model_name"] = model_display_names.get(
        artifact_bundle["deployed_model_name"],
        artifact_bundle["deployed_model_name"].replace("_", " ").title(),
    )
    defaults = dict(ui_metadata["defaults"])
    if form_state:
        defaults.update(form_state)
    return render_template(
        "index.html",
        single_result=single_result,
        batch_preview=batch_preview,
        batch_summary=batch_summary,
        download_name=download_name,
        defaults=defaults,
        categorical_options=ui_metadata["categorical_options"],
        numeric_ranges=ui_metadata["numeric_ranges"],
        metrics=metrics,
        dataset_rows=artifact_bundle["dataset_rows"],
        class_balance=artifact_bundle["class_balance"],
        feature_list=ui_metadata["features"],
    )


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
