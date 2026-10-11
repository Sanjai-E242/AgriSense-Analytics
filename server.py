"""
Flask Application Server for Crop Recommendation System — AgriSense Analytics
Principles of Data Science (PoDS) Project
"""

import os
import io
import json
import sqlite3
import datetime
from typing import Dict, Any, List
from flask import Flask, send_from_directory, request, jsonify
import numpy as np
import pandas as pd
from validator import validate_csv_content

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "prediction_history.db")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15 MB Max Upload Limit

def init_history_db():
    """Initializes local SQLite database for prediction audit trails."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                n REAL NOT NULL,
                p REAL NOT NULL,
                k REAL NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                ph REAL NOT NULL,
                rainfall REAL NOT NULL,
                predicted_crop TEXT NOT NULL,
                champion_model TEXT NOT NULL,
                dataset_name TEXT,
                dataset_fingerprint TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()

def record_prediction_history(inputs: Dict[str, Any], predicted_crop: str, champion_model: str, dataset_name: str, dataset_fingerprint: str):
    """Appends an authentic inference record to SQLite audit trail."""
    init_history_db()
    now_ts = datetime.datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_history (
                timestamp, n, p, k, temperature, humidity, ph, rainfall,
                predicted_crop, champion_model, dataset_name, dataset_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_ts,
            float(inputs.get("N", 0.0)),
            float(inputs.get("P", 0.0)),
            float(inputs.get("K", 0.0)),
            float(inputs.get("temperature", 0.0)),
            float(inputs.get("humidity", 0.0)),
            float(inputs.get("ph", 0.0)),
            float(inputs.get("rainfall", 0.0)),
            str(predicted_crop),
            str(champion_model),
            str(dataset_name or ""),
            str(dataset_fingerprint or "")
        ))
        conn.commit()
    finally:
        conn.close()

def get_all_prediction_history() -> List[Dict[str, Any]]:
    """Retrieves all logged inference records in reverse chronological order."""
    if not os.path.exists(DB_PATH):
        return []
    try:
        init_history_db()
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, n, p, k, temperature, humidity, ph, rainfall,
                       predicted_crop, champion_model, dataset_name, dataset_fingerprint
                FROM prediction_history
                ORDER BY id DESC
            """)
            rows = cursor.fetchall()
            records = []
            for r in rows:
                d = dict(r)
                d['N'] = d.get('n')
                d['P'] = d.get('p')
                d['K'] = d.get('k')
                d['model_identifier'] = d.get('champion_model')
                records.append(d)
            return records
        finally:
            conn.close()
    except Exception:
        return []

def clear_all_prediction_history() -> bool:
    """Safely clears all logged inference records from SQLite audit trail."""
    if not os.path.exists(DB_PATH):
        return True
    try:
        init_history_db()
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prediction_history")
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception:
        return False

def sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NumPy scalars and arrays to native Python types for clean JSON serialization."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj

@app.errorhandler(400)
def handle_bad_request(error):
    if request.path.startswith('/api/'):
        return jsonify({
            "success": False,
            "error_type": "BAD_REQUEST",
            "message": str(getattr(error, 'description', error))
        }), 400
    return jsonify({"success": False, "error_type": "BAD_REQUEST", "message": "Bad request."}), 400

@app.errorhandler(404)
def handle_not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({
            "success": False,
            "error_type": "NOT_FOUND",
            "message": f"API endpoint '{request.path}' not found."
        }), 404
    return jsonify({"success": False, "error_type": "NOT_FOUND", "message": f"Resource '{request.path}' not found."}), 404

@app.errorhandler(405)
def handle_method_not_allowed(error):
    if request.path.startswith('/api/'):
        return jsonify({
            "success": False,
            "error_type": "METHOD_NOT_ALLOWED",
            "message": f"HTTP method '{request.method}' not allowed for '{request.path}'."
        }), 405
    return jsonify({"success": False, "error_type": "METHOD_NOT_ALLOWED", "message": "Method not allowed."}), 405

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error_type": "FILE_TOO_LARGE",
        "message": "File exceeds maximum permitted size of 15 MB."
    }), 413

@app.errorhandler(500)
@app.errorhandler(Exception)
def handle_internal_error(error):
    app.logger.error(f"Internal error on {request.path}: {str(error)}", exc_info=True)
    if request.path.startswith('/api/'):
        return jsonify({
            "success": False,
            "error_type": "INTERNAL_SERVER_ERROR",
            "message": f"Internal server error: {str(error)}"
        }), 500
    return jsonify({
        "success": False,
        "error_type": "INTERNAL_SERVER_ERROR",
        "message": f"An unexpected server error occurred: {str(error)}"
    }), 500

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

@app.route('/')
def index():
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return send_from_directory(BASE_DIR, "index.html")

@app.route('/<path:path>')
def static_proxy(path):
    if path.startswith('api/') or path == 'api':
        return jsonify({
            "success": False,
            "error_type": "NOT_FOUND",
            "message": f"API endpoint '/{path}' not found."
        }), 404
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    if os.path.exists(os.path.join(BASE_DIR, path)):
        return send_from_directory(BASE_DIR, path)
    return jsonify({
        "success": False,
        "error_type": "NOT_FOUND",
        "message": f"Resource '{path}' not found."
    }), 404

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "PoDS Crop Recommendation Server",
        "version": "0.5.1"
    })

@app.route('/api/validate-dataset', methods=['POST'])
def validate_dataset():
    """
    Validates the uploaded CSV dataset on the server against the canonical 8-column schema,
    documented aliases, data quality rules, and 15MB limit. If valid, persists the active
    dataset and invalidates any stale model artifacts to ensure single-source-of-truth consistency.
    """
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "error_type": "NO_FILE",
            "message": "No file uploaded in the request."
        }), 400

    uploaded_file = request.files['file']
    filename = uploaded_file.filename

    if not filename:
        return jsonify({
            "success": False,
            "error_type": "EMPTY_FILENAME",
            "message": "No file selected."
        }), 400

    if not filename.lower().endswith('.csv'):
        return jsonify({
            "success": False,
            "error_type": "INVALID_FORMAT",
            "message": f"Unsupported file type for '{filename}'. Only CSV (.csv) files are permitted.",
            "filename": filename
        }), 400

    try:
        content_bytes = uploaded_file.read()
        result = validate_csv_content(content_bytes, filename)
        
        if result.get("success"):
            # Persist active dataset locally in data/
            os.makedirs(DATA_DIR, exist_ok=True)
            active_csv_path = os.path.join(DATA_DIR, "active_dataset.csv")
            active_meta_path = os.path.join(DATA_DIR, "active_dataset.json")
            
            # Save normalized CSV if available so subsequent pipelines receive canonical structure
            normalized_csv = result.get("normalized_csv")
            if normalized_csv:
                with open(active_csv_path, "w", encoding="utf-8") as f:
                    f.write(normalized_csv)
            else:
                with open(active_csv_path, "wb") as f:
                    f.write(content_bytes)
                
            classes_list = result.get("classes") or result.get("target_summary", {}).get("classes", [])
            with open(active_meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "filename": filename,
                    "rows": result.get("rows"),
                    "columns": result.get("columns"),
                    "classes": classes_list,
                    "class_count": len(classes_list),
                    "dataset_fingerprint": result.get("dataset_fingerprint"),
                    "provenance_status": result.get("provenance_status", "Source not verified"),
                    "warnings": result.get("warnings", []),
                    "range_warnings": result.get("range_warnings", []),
                    "uploaded_at": datetime.datetime.now().isoformat()
                }, f, indent=2)
            result["persisted"] = True

            # Invalidate older models when a new dataset is uploaded to prevent state inconsistency
            for mf in ["best_crop_model.json", "model_metadata.json"]:
                mp = os.path.join(MODELS_DIR, mf)
                if os.path.exists(mp):
                    try:
                        os.remove(mp)
                    except Exception:
                        pass
            
        status_code = 200 if result.get("success") else 400
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({
            "success": False,
            "error_type": "SERVER_ERROR",
            "message": f"Server encountered an error while validating the dataset: {str(e)}"
        }), 500

@app.route('/api/dataset-status', methods=['GET'])
def dataset_status():
    """Returns the status and metadata of the currently active ingested dataset."""
    active_csv = os.path.join(DATA_DIR, "active_dataset.csv")
    active_meta = os.path.join(DATA_DIR, "active_dataset.json")
    if os.path.exists(active_csv) and os.path.exists(active_meta):
        try:
            with open(active_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return jsonify({
                "has_dataset": True,
                "metadata": meta
            })
        except Exception:
            pass
    return jsonify({
        "has_dataset": False,
        "message": "No active dataset ingested on the server."
    })

@app.route('/api/dataset-reset', methods=['POST'])
def dataset_reset():
    """Resets the currently active dataset and clears dependent model status."""
    active_csv = os.path.join(DATA_DIR, "active_dataset.csv")
    active_meta = os.path.join(DATA_DIR, "active_dataset.json")
    for p in [active_csv, active_meta]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    for mf in ["best_crop_model.json", "model_metadata.json"]:
        mp = os.path.join(MODELS_DIR, mf)
        if os.path.exists(mp):
            try:
                os.remove(mp)
            except Exception:
                pass
    return jsonify({"success": True, "message": "Active dataset reset successfully. Dependent model artifacts cleared."})

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """
    Checks whether a trained model artifact exists, whether it is valid,
    and whether it is compatible with the currently active dataset.
    """
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    model_path = os.path.join(MODELS_DIR, "best_crop_model.json")
    active_meta_path = os.path.join(DATA_DIR, "active_dataset.json")
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return jsonify({
            "trained": False,
            "artifact_present": False,
            "artifact_valid": False,
            "message": "No trained model artifacts found."
        })
    
    try:
        from pipeline import load_production_model
        pipeline, meta = load_production_model()

        # Check compatibility with active dataset fingerprint
        model_compatible = True
        compatibility_msg = "Model matches active dataset."
        if os.path.exists(active_meta_path):
            try:
                with open(active_meta_path, "r", encoding="utf-8") as f:
                    d_meta = json.load(f)
                active_fp = d_meta.get("dataset_fingerprint")
                model_fp = meta.get("dataset_fingerprint")
                if active_fp and model_fp and active_fp != model_fp:
                    model_compatible = False
                    compatibility_msg = "Model was trained on a different dataset. Re-training required."
            except Exception:
                pass
        else:
            # If no active dataset is on server, model is considered unlinked/stale
            model_compatible = False
            compatibility_msg = "No active dataset ingested on the server."

        return jsonify({
            "trained": model_compatible,
            "artifact_present": True,
            "artifact_valid": True,
            "model_compatible": model_compatible,
            "message": compatibility_msg,
            "champion_model": meta.get("champion_model", pipeline.name),
            "metadata": meta
        })
    except Exception as e:
        return jsonify({
            "trained": False,
            "artifact_present": True,
            "artifact_valid": False,
            "error": f"Model artifact found but corrupted/invalid: {str(e)}"
        })

@app.route('/api/train-models', methods=['GET', 'POST'])
def train_models():
    """
    Executes reproducible ML training across Decision Tree, Random Forest, and KNN.
    Requires explicit user approval.
    Production API strictly rejects fixture bypass (allow_fixture_for_testing is ALWAYS False).
    """
    if request.method != 'POST':
        return jsonify({
            "success": False,
            "error_type": "METHOD_NOT_ALLOWED",
            "message": f"HTTP method '{request.method}' not allowed for '/api/train-models'. Only POST is supported."
        }), 405
    try:
        data = request.get_json(silent=True) or {}
        req_dataset_name = data.get("dataset_name")
        dataset_name = req_dataset_name or "uploaded_dataset.csv"
        approved_by_user = bool(data.get("approved_by_user", False))
        allow_fixture_for_testing = False
        csv_text = data.get("csv_text", "")

        if not approved_by_user:
            return jsonify({
                "success": False,
                "error_type": "APPROVAL_REQUIRED",
                "message": "Project owner confirmation is required. You must check the authorization box before training."
            }), 400

        dataset_fingerprint = data.get("dataset_fingerprint")
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        active_csv = os.path.join(DATA_DIR, "active_dataset.csv")
        active_meta = os.path.join(DATA_DIR, "active_dataset.json")

        # Check if client-provided csv_text has valid data records
        csv_has_rows = False
        if csv_text and isinstance(csv_text, str) and len(csv_text.strip()) > 0:
            lines = [l for l in csv_text.strip().splitlines() if l.strip()]
            if len(lines) >= 2:
                csv_has_rows = True

        # If client csv_text is missing or lacks data rows, fall back to persisted active dataset
        if not csv_has_rows:
            if os.path.exists(active_csv):
                try:
                    with open(active_csv, "r", encoding="utf-8") as f:
                        csv_text = f.read()
                    lines = [l for l in csv_text.strip().splitlines() if l.strip()]
                    if len(lines) >= 2:
                        csv_has_rows = True
                except Exception as e:
                    return jsonify({
                        "success": False,
                        "error_type": "READ_ERROR",
                        "message": f"Could not read active dataset from server: {str(e)}"
                    }), 500

        # Ensure dataset_fingerprint is sourced from active dataset metadata if available
        if os.path.exists(active_meta):
            try:
                with open(active_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if not req_dataset_name:
                        dataset_name = meta.get("filename", dataset_name)
                    if not dataset_fingerprint:
                        dataset_fingerprint = meta.get("dataset_fingerprint")
            except Exception:
                pass

        if not csv_has_rows:
            return jsonify({
                "success": False,
                "error_type": "NO_DATASET_CONTENT",
                "message": "No dataset records provided. Please ingest and inspect a dataset first in Dataset Explorer."
            }), 400

        from pipeline import train_and_evaluate_pipeline

        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as pe:
            return jsonify({
                "success": False,
                "error_type": "MALFORMED_CSV",
                "message": f"Could not parse CSV content: {str(pe)}"
            }), 400

        result = train_and_evaluate_pipeline(
            df=df,
            dataset_name=dataset_name,
            approved_by_user=approved_by_user,
            allow_fixture_for_testing=allow_fixture_for_testing,
            dataset_fingerprint=dataset_fingerprint
        )

        if result.get("success"):
            # Verify artifact persistence completeness
            model_path = os.path.join(MODELS_DIR, "best_crop_model.json")
            meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
            if not (os.path.exists(model_path) and os.path.getsize(model_path) > 0 and
                    os.path.exists(meta_path) and os.path.getsize(meta_path) > 0):
                return jsonify({
                    "success": False,
                    "error_type": "ARTIFACT_PERSISTENCE_FAILED",
                    "message": "Model training finished but artifact persistence verification failed. Model artifacts are incomplete."
                }), 500

            # Ensure dataset files are persisted locally if not already saved
            if not os.path.exists(active_csv) or os.path.getsize(active_csv) == 0:
                try:
                    with open(active_csv, "w", encoding="utf-8") as f:
                        f.write(csv_text)
                except Exception:
                    pass
            if not os.path.exists(active_meta):
                try:
                    classes_list = result.get("classes", [])
                    with open(active_meta, "w", encoding="utf-8") as f:
                        json.dump({
                            "filename": dataset_name,
                            "rows": len(df),
                            "columns": list(df.columns),
                            "classes": classes_list,
                            "class_count": len(classes_list),
                            "dataset_fingerprint": result.get("dataset_fingerprint", dataset_fingerprint),
                            "provenance_status": "Source verified (Trained Model)",
                            "uploaded_at": datetime.datetime.now().isoformat()
                        }, f, indent=2)
                except Exception:
                    pass

        sanitized_result = sanitize_for_json(result)
        status_code = 200 if result.get("success") else 400
        return jsonify(sanitized_result), status_code

    except Exception as e:
        return jsonify({
            "success": False,
            "error_type": "PIPELINE_ERROR",
            "message": f"ML pipeline error: {str(e)}"
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Receives 7 agronomic features, validates inputs and model readiness,
    runs inference through the production ModelPipeline, logs the prediction
    to the persistent SQLite audit trail, and returns recommendation results.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "success": False,
            "error_type": "INVALID_JSON",
            "message": "Request body must be valid JSON containing the 7 input features."
        }), 400

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    model_path = os.path.join(MODELS_DIR, "best_crop_model.json")
    active_meta_path = os.path.join(DATA_DIR, "active_dataset.json")

    # Verify model artifact exists
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return jsonify({
            "success": False,
            "error_type": "NO_TRAINED_MODEL",
            "message": "No valid trained model found. You must train an approved model before inference."
        }), 400

    # Verify active dataset exists on the server
    if not os.path.exists(active_meta_path):
        return jsonify({
            "success": False,
            "error_type": "NO_ACTIVE_DATASET",
            "message": "No active dataset found on the server. Please upload and train on an authorized dataset before running inference."
        }), 400

    # Verify model compatibility with active dataset fingerprint
    try:
        with open(active_meta_path, "r", encoding="utf-8") as f:
            d_meta = json.load(f)
        with open(metadata_path, "r", encoding="utf-8") as f:
            m_meta = json.load(f)
        active_fp = d_meta.get("dataset_fingerprint")
        model_fp = m_meta.get("dataset_fingerprint")
        if active_fp and model_fp and active_fp != model_fp:
            return jsonify({
                "success": False,
                "error_type": "MODEL_INCOMPATIBLE",
                "message": "The active model was trained on a different dataset. Please retrain for the current dataset."
            }), 400
    except Exception:
        pass

    try:
        from pipeline import predict_crop
        result = predict_crop(data)

        # Record prediction into persistent SQLite audit trail
        record_prediction_history(
            inputs=result.get("inputs", {}),
            predicted_crop=result.get("predicted_crop", ""),
            champion_model=result.get("champion_model", ""),
            dataset_name=result.get("dataset_name", ""),
            dataset_fingerprint=result.get("dataset_fingerprint", "")
        )

        return jsonify(result), 200
    except FileNotFoundError:
        return jsonify({
            "success": False,
            "error_type": "NO_TRAINED_MODEL",
            "message": "No valid trained model found. You must train an approved model before inference."
        }), 400
    except ValueError as ve:
        return jsonify({
            "success": False,
            "error_type": "INVALID_INFERENCE_INPUT",
            "message": str(ve)
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error_type": "INFERENCE_ERROR",
            "message": f"Inference failed: {str(e)}"
        }), 500

@app.route('/api/prediction-history', methods=['GET'])
def get_prediction_history_route():
    """Returns persistent prediction history records from SQLite audit trail."""
    records = get_all_prediction_history()
    return jsonify({
        "success": True,
        "count": len(records),
        "history": records
    })

@app.route('/api/prediction-history/clear', methods=['POST'])
def clear_prediction_history_route():
    """Clears all logged inference records from SQLite audit trail upon user confirmation."""
    success = clear_all_prediction_history()
    return jsonify({
        "success": success,
        "message": "Prediction history audit trail cleared successfully." if success else "Failed to clear history."
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting PoDS Crop Recommendation Server on http://localhost:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=False)
