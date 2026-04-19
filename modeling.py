from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import pdfplumber


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Telco_customer_churn.xlsx"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_ONNX_PATH = ARTIFACT_DIR / "xgboost_model.onnx"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

TARGET_COLUMN = "churn_label"
IDENTIFIER_COLUMN = "customerid"
DEPLOYED_MODEL_NAME = "xgboost"
ONNX_INPUT_NAME = "float_input"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

CANONICAL_CATEGORY_MAPS: dict[str, dict[str, str]] = {
    "gender": {"male": "Male", "female": "Female"},
    "senior_citizen": {
        "yes": "Yes",
        "no": "No",
        "1": "Yes",
        "0": "No",
        "true": "Yes",
        "false": "No",
    },
    "partner": {"yes": "Yes", "no": "No"},
    "dependents": {"yes": "Yes", "no": "No"},
    "phone_service": {"yes": "Yes", "no": "No"},
    "multiple_lines": {
        "yes": "Yes",
        "no": "No",
        "no phone service": "No phone service",
    },
    "internet_service": {
        "dsl": "DSL",
        "fiber optic": "Fiber optic",
        "fiber": "Fiber optic",
        "no": "No",
    },
    "online_security": {
        "yes": "Yes",
        "no": "No",
        "no internet service": "No internet service",
    },
    "online_backup": {
        "yes": "Yes",
        "no": "No",
        "no internet service": "No internet service",
    },
    "device_protection": {
        "yes": "Yes",
        "no": "No",
        "no internet service": "No internet service",
    },
    "tech_support": {
        "yes": "Yes",
        "no": "No",
        "no internet service": "No internet service",
    },
    "streaming_tv": {
        "yes": "Yes",
        "no": "No",
        "no internet service": "No internet service",
    },
    "streaming_movies": {
        "yes": "Yes",
        "no": "No",
        "no internet service": "No internet service",
    },
    "contract": {
        "month-to-month": "Month-to-month",
        "one year": "One year",
        "two year": "Two year",
    },
    "paperless_billing": {"yes": "Yes", "no": "No"},
    "payment_method": {
        "electronic check": "Electronic check",
        "mailed check": "Mailed check",
        "bank transfer (automatic)": "Bank transfer (automatic)",
        "credit card (automatic)": "Credit card (automatic)",
    },
    TARGET_COLUMN: {"yes": "Yes", "no": "No", "1": "Yes", "0": "No"},
}


class InputValidationError(ValueError):
    pass


@dataclass
class TrainingResult:
    metadata: dict[str, Any]
    metrics: dict[str, Any]


def clean_column_name(name: Any) -> str:
    sanitized = str(name).strip().replace(" ", "_").replace("/", "_")
    return "".join(ch.lower() for ch in sanitized if ch.isalnum() or ch == "_")


def _normalize_category(series: pd.Series, column: str) -> pd.Series:
    mapping = CANONICAL_CATEGORY_MAPS.get(column)
    if not mapping:
        return series
    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(mapping)
    )
    return normalized.fillna(series.astype("string").str.strip())


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [clean_column_name(column) for column in normalized.columns]

    alias_map = {
        "customer_id": IDENTIFIER_COLUMN,
        "customerid": IDENTIFIER_COLUMN,
        "monthlycharge": "monthly_charges",
        "monthlycharges": "monthly_charges",
        "totalcharge": "total_charges",
        "totalcharges": "total_charges",
        "tenure": "tenure_months",
        "tenuremonths": "tenure_months",
        "paymentmethod": "payment_method",
        "phoneservice": "phone_service",
        "paperlessbilling": "paperless_billing",
        "multiplelines": "multiple_lines",
        "internetservice": "internet_service",
        "onlinesecurity": "online_security",
        "onlinebackup": "online_backup",
        "deviceprotection": "device_protection",
        "techsupport": "tech_support",
        "streamingtv": "streaming_tv",
        "streamingmovies": "streaming_movies",
        "seniorcitizen": "senior_citizen",
        "churnlabel": TARGET_COLUMN,
    }
    normalized = normalized.rename(
        columns={column: alias_map.get(column, column) for column in normalized.columns}
    )

    for column in NUMERIC_FEATURES:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    for column in CATEGORICAL_FEATURES + [TARGET_COLUMN]:
        if column in normalized.columns:
            normalized[column] = _normalize_category(normalized[column], column)

    if "multiple_lines" in normalized.columns and "phone_service" in normalized.columns:
        no_phone = normalized["phone_service"].astype("string").str.lower() == "no"
        normalized.loc[no_phone, "multiple_lines"] = "No phone service"

    internet_dependent_columns = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]
    if "internet_service" in normalized.columns:
        no_internet = normalized["internet_service"].astype("string").str.lower() == "no"
        for column in internet_dependent_columns:
            if column in normalized.columns:
                normalized.loc[no_internet, column] = "No internet service"

    if "total_charges" in normalized.columns:
        missing_total = normalized["total_charges"].isna()
        if {"monthly_charges", "tenure_months"}.issubset(normalized.columns):
            normalized.loc[missing_total, "total_charges"] = (
                normalized.loc[missing_total, "monthly_charges"].fillna(0)
                * normalized.loc[missing_total, "tenure_months"].fillna(0)
            )

    return normalized


def validate_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in MODEL_FEATURES if column not in df.columns]
    if missing_columns:
        raise InputValidationError(
            "Input data is missing required columns: " + ", ".join(missing_columns)
        )
    return df[MODEL_FEATURES].copy()


def load_training_dataframe() -> pd.DataFrame:
    raw_df = pd.read_excel(DATA_PATH)
    normalized = normalize_dataframe(raw_df)
    if TARGET_COLUMN not in normalized.columns:
        raise InputValidationError("Training dataset is missing the churn label column.")
    dataset = normalized[MODEL_FEATURES + [TARGET_COLUMN, IDENTIFIER_COLUMN]].copy()
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    dataset = dataset.dropna(subset=[TARGET_COLUMN])
    dataset = dataset.drop_duplicates(subset=[IDENTIFIER_COLUMN], keep="first")
    return dataset


def build_ui_metadata(dataset: pd.DataFrame) -> dict[str, Any]:
    options = {
        feature: sorted(dataset[feature].dropna().astype(str).unique().tolist())
        for feature in CATEGORICAL_FEATURES
    }
    numeric_ranges = {}
    for feature in NUMERIC_FEATURES:
        series = dataset[feature].dropna()
        numeric_ranges[feature] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }
    defaults = {
        feature: options[feature][0] if options[feature] else ""
        for feature in CATEGORICAL_FEATURES
    }
    defaults.update(
        {feature: round(values["median"], 2) for feature, values in numeric_ranges.items()}
    )
    return {
        "categorical_options": options,
        "numeric_ranges": numeric_ranges,
        "defaults": defaults,
        "features": MODEL_FEATURES,
    }


def _build_encoding_metadata(dataset: pd.DataFrame) -> dict[str, Any]:
    numeric_medians = {
        feature: float(dataset[feature].median())
        for feature in NUMERIC_FEATURES
    }
    categorical_fill_values = {}
    category_levels = {}
    for feature in CATEGORICAL_FEATURES:
        mode = dataset[feature].mode(dropna=True)
        fill_value = str(mode.iloc[0]) if not mode.empty else ""
        categorical_fill_values[feature] = fill_value
        category_levels[feature] = sorted(dataset[feature].dropna().astype(str).unique().tolist())
    return {
        "numeric_medians": numeric_medians,
        "categorical_fill_values": categorical_fill_values,
        "category_levels": category_levels,
    }


def encode_features(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    working = validate_feature_frame(df).copy()

    for feature in NUMERIC_FEATURES:
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
        working[feature] = working[feature].fillna(metadata["numeric_medians"][feature])

    for feature in CATEGORICAL_FEATURES:
        working[feature] = (
            working[feature]
            .astype("string")
            .str.strip()
            .replace({"<NA>": None})
            .fillna(metadata["categorical_fill_values"][feature])
        )
        working[feature] = pd.Categorical(
            working[feature],
            categories=metadata["category_levels"][feature],
        )

    encoded = pd.get_dummies(working, columns=CATEGORICAL_FEATURES, dtype=float)
    feature_columns = metadata.get("encoded_feature_columns")
    if feature_columns is not None:
        encoded = encoded.reindex(columns=feature_columns, fill_value=0.0)
    runtime_columns = metadata.get("runtime_feature_names")
    if runtime_columns is not None:
        encoded.columns = runtime_columns
    return encoded


def _evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, predictions)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions)), 4),
        "recall": round(float(recall_score(y_true, predictions)), 4),
        "f1": round(float(f1_score(y_true, predictions)), 4),
        "confusion_matrix": matrix.tolist(),
        "positive_rate": round(float(predictions.mean()), 4),
    }


def train_and_persist_model() -> TrainingResult:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    dataset = load_training_dataframe()
    x = dataset[MODEL_FEATURES].copy()
    y = dataset[TARGET_COLUMN].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    encoding_metadata = _build_encoding_metadata(x_train)
    x_train_encoded = encode_features(x_train, encoding_metadata)
    raw_encoded_columns = x_train_encoded.columns.tolist()
    encoding_metadata["encoded_feature_columns"] = raw_encoded_columns
    encoding_metadata["runtime_feature_names"] = [f"f{i}" for i in range(len(raw_encoded_columns))]
    x_train_encoded.columns = encoding_metadata["runtime_feature_names"]
    x_test_encoded = encode_features(x_test, encoding_metadata)

    candidate_models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
        ),
    }

    metrics_by_model: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}
    for model_name, estimator in candidate_models.items():
        estimator.fit(x_train_encoded, y_train)
        trained_models[model_name] = estimator
        probabilities = estimator.predict_proba(x_test_encoded)[:, 1]
        metrics_by_model[model_name] = _evaluate_predictions(y_test, probabilities)

    full_encoding_metadata = _build_encoding_metadata(dataset[MODEL_FEATURES])
    x_full_encoded = encode_features(dataset[MODEL_FEATURES], full_encoding_metadata)
    raw_full_columns = x_full_encoded.columns.tolist()
    full_encoding_metadata["encoded_feature_columns"] = raw_full_columns
    full_encoding_metadata["runtime_feature_names"] = [f"f{i}" for i in range(len(raw_full_columns))]
    x_full_encoded.columns = full_encoding_metadata["runtime_feature_names"]

    deployed_model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
    )
    deployed_model.fit(x_full_encoded, y)

    initial_type = [(ONNX_INPUT_NAME, FloatTensorType([None, x_full_encoded.shape[1]]))]
    onnx_model = convert_xgboost(deployed_model, initial_types=initial_type)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "deployed_model_name": DEPLOYED_MODEL_NAME,
        "best_model_name": max(
            metrics_by_model,
            key=lambda name: (metrics_by_model[name]["roc_auc"], metrics_by_model[name]["f1"]),
        ),
        "metrics": metrics_by_model,
        "ui_metadata": build_ui_metadata(dataset),
        "dataset_rows": int(dataset.shape[0]),
        "class_balance": {
            "churn": int(y.sum()),
            "retain": int((1 - y).sum()),
        },
        "numeric_medians": full_encoding_metadata["numeric_medians"],
        "categorical_fill_values": full_encoding_metadata["categorical_fill_values"],
        "category_levels": full_encoding_metadata["category_levels"],
        "encoded_feature_columns": full_encoding_metadata["encoded_feature_columns"],
        "runtime_feature_names": full_encoding_metadata["runtime_feature_names"],
        "feature_order": MODEL_FEATURES,
        "onnx_input_name": ONNX_INPUT_NAME,
        "onnx_output_name": "probabilities",
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ONNX_PATH.write_bytes(onnx_model.SerializeToString())
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics_by_model, indent=2), encoding="utf-8")
    return TrainingResult(metadata=metadata, metrics=metrics_by_model)


def ensure_model_artifact() -> dict[str, Any]:
    if not MODEL_ONNX_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Model artifacts are missing. Run `python train_model.py` locally before starting the app."
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    session = ort.InferenceSession(
        str(MODEL_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    return {"session": session, **metadata}


def predict_dataframe(df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    normalized = normalize_dataframe(df)
    encoded = encode_features(normalized, artifact)
    raw_outputs = artifact["session"].run(
        None,
        {artifact["onnx_input_name"]: encoded.to_numpy(dtype=np.float32)},
    )
    probabilities = np.asarray(raw_outputs[1])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result = pd.DataFrame()
    if IDENTIFIER_COLUMN in normalized.columns:
        result[IDENTIFIER_COLUMN] = normalized[IDENTIFIER_COLUMN]
    else:
        result[IDENTIFIER_COLUMN] = [f"row_{index + 1}" for index in range(len(normalized))]
    result["churn_probability"] = np.round(probabilities, 4)
    result["predicted_churn"] = np.where(predictions == 1, "Yes", "No")
    return result


def build_single_record(form_values: dict[str, str]) -> pd.DataFrame:
    record: dict[str, Any] = {}
    for feature in CATEGORICAL_FEATURES:
        record[feature] = form_values.get(feature, "").strip()
    for feature in NUMERIC_FEATURES:
        raw_value = form_values.get(feature, "").strip()
        if raw_value == "":
            raise InputValidationError(f"Please provide a value for {feature}.")
        record[feature] = float(raw_value)
    if record["total_charges"] < 0 or record["monthly_charges"] < 0 or record["tenure_months"] < 0:
        raise InputValidationError("Numeric inputs must be non-negative.")
    return pd.DataFrame([record])


def parse_uploaded_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".json":
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InputValidationError("JSON upload could not be parsed.") from exc
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
            return pd.DataFrame(data["records"])
        raise InputValidationError("JSON upload must be a list of records or contain a 'records' list.")
    if suffix == ".pdf":
        return parse_pdf_table(file_path)
    raise InputValidationError("Supported upload formats are CSV, XLSX, JSON, and PDF.")


def parse_pdf_table(file_path: Path) -> pd.DataFrame:
    extracted_tables: list[pd.DataFrame] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table or len(table) < 2:
                    continue
                header = [clean_column_name(cell or "") for cell in table[0]]
                rows = table[1:]
                frame = pd.DataFrame(rows, columns=header)
                extracted_tables.append(frame)

    if not extracted_tables:
        raise InputValidationError(
            "No structured table was found in the PDF. Upload CSV, XLSX, or JSON if possible."
        )

    best_table = max(
        extracted_tables,
        key=lambda frame: len(set(frame.columns).intersection(set(MODEL_FEATURES + [TARGET_COLUMN, IDENTIFIER_COLUMN]))),
    )
    if not set(MODEL_FEATURES).intersection(set(best_table.columns)):
        raise InputValidationError(
            "The PDF table does not include enough model columns. Use the sample schema from the app."
        )
    return best_table
