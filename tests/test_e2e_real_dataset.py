import os
import sys
import time
import json
import urllib.request
import urllib.parse
import mimetypes

BASE_URL = os.environ.get("AGRISENSE_TEST_URL", "http://127.0.0.1:5000")
CANDIDATE_PATHS = [
    r"C:\Users\jmsar\.gemini\antigravity-ide\brain\c6163f69-3785-45f3-b171-6f4be900f9ba\.user_uploaded\media_1791729624499.csv",
    os.path.join(os.path.dirname(__file__), "..", "data", "active_dataset.csv")
]

def load_or_create_dataset_bytes():
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return f.read()
    # Generate 2,200-sample synthetic dataset if artifact path is not present
    import numpy as np
    import pandas as pd
    np.random.seed(42)
    classes = [
        "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
        "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
        "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
        "coconut", "cotton", "jute", "coffee"
    ]
    rows = []
    for cls in classes:
        for _ in range(100):
            rows.append({
                "N": np.random.uniform(0, 140),
                "P": np.random.uniform(5, 145),
                "K": np.random.uniform(5, 205),
                "temperature": np.random.uniform(8, 44),
                "humidity": np.random.uniform(14, 100),
                "ph": np.random.uniform(3.5, 9.9),
                "rainfall": np.random.uniform(20, 299),
                "label": cls
            })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode('utf-8')

def run_test():
    print(f"=== Starting Real E2E Test on {BASE_URL} ===")
    time.sleep(1.0)  # wait for server to bind

    # 1. Homepage & Static Assets
    print("\n--- 1. Testing Homepage & Static Assets ---")
    req = urllib.request.urlopen(f"{BASE_URL}/")
    assert req.status == 200
    html = req.read().decode('utf-8')
    assert "AgriSense Analytics" in html
    print(f"Homepage OK: HTTP 200, length {len(html)} bytes")

    css_req = urllib.request.urlopen(f"{BASE_URL}/css/tokens.css")
    assert css_req.status == 200
    print(f"tokens.css OK: HTTP 200, length {len(css_req.read())} bytes")

    js_req = urllib.request.urlopen(f"{BASE_URL}/js/app.js")
    assert js_req.status == 200
    print(f"app.js OK: HTTP 200, length {len(js_req.read())} bytes")

    # 2. Health Endpoint
    print("\n--- 2. Testing /api/health ---")
    h_req = urllib.request.urlopen(f"{BASE_URL}/api/health")
    h_data = json.loads(h_req.read().decode('utf-8'))
    print("Health response:", h_data)
    assert h_data.get("status") == "healthy"

    # 3. Dataset Validation (multipart upload of 2,200 rows)
    print("\n--- 3. Testing /api/validate-dataset with 2,200 samples ---")
    file_bytes = load_or_create_dataset_bytes()

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="Crop_recommendation 2200 samples.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode('utf-8') + file_bytes + f"\r\n--{boundary}--\r\n".encode('utf-8')

    v_req = urllib.request.Request(
        f"{BASE_URL}/api/validate-dataset",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(v_req) as resp:
        v_data = json.loads(resp.read().decode('utf-8'))
    t_val = time.perf_counter() - t0
    rows = v_data.get('rows')
    cols = v_data.get('columns')
    classes = v_data.get('classes') or v_data.get('target_summary', {}).get('classes', [])
    print(f"Validation finished in {t_val:.3f}s: success={v_data.get('success')}, rows={rows}, cols={cols}, classes={len(classes)}")
    assert v_data.get("success") is True
    assert rows == 2200
    assert (len(cols) == 8 if isinstance(cols, list) else cols == 8)
    assert len(classes) == 22

    # 4. Dataset Status
    print("\n--- 4. Testing /api/dataset-status ---")
    with urllib.request.urlopen(f"{BASE_URL}/api/dataset-status") as resp:
        ds_data = json.loads(resp.read().decode('utf-8'))
    print(f"Dataset status: has_dataset={ds_data.get('has_dataset')}, filename={ds_data.get('filename')}")
    assert ds_data.get("has_dataset") is True

    # 5. Model Training (Synchronous 2,200 samples across DT, RF, KNN + 5-Fold Stratified CV)
    print("\n--- 5. Testing /api/train-models (Training DT, RF, KNN on 2,200 samples) ---")
    csv_text = file_bytes.decode('utf-8')
    train_payload = json.dumps({
        "approved_by_user": True,
        "dataset_name": "Crop_recommendation 2200 samples.csv",
        "csv_text": csv_text
    }).encode('utf-8')

    tr_req = urllib.request.Request(
        f"{BASE_URL}/api/train-models",
        data=train_payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    t_train_start = time.perf_counter()
    with urllib.request.urlopen(tr_req) as resp:
        tr_data = json.loads(resp.read().decode('utf-8'))
    t_train_total = time.perf_counter() - t_train_start
    print(f"Training completed successfully in {t_train_total:.3f} seconds!")
    print(f"Champion Model: {tr_data.get('champion_model')}")
    print(f"Dataset Fingerprint: {tr_data.get('dataset_fingerprint')}")
    assert tr_data.get("success") is True
    assert tr_data.get("champion_model") in ["Random Forest", "Decision Tree", "K-Nearest Neighbors"]

    benchmarks = tr_data.get("benchmark_results", {})
    for m_name, m_info in benchmarks.items():
        cv = m_info.get("cross_validation", {})
        print(f"\nModel: {m_name}")
        print(f"  Train Acc: {m_info.get('train_accuracy')}, Train Macro-F1: {m_info.get('train_macro_f1')}")
        print(f"  Test Acc: {m_info.get('test_accuracy')}, Test Macro-F1: {m_info.get('macro_f1')}")
        print(f"  CV 5-Fold Mean Acc: {cv.get('accuracy_mean')} (+/- {cv.get('accuracy_std')})")
        print(f"  CV 5-Fold Mean Macro-F1: {cv.get('macro_f1_mean')} (+/- {cv.get('macro_f1_std')})")
        print(f"  Selection Source: {m_info.get('selection_source')}")

    # 6. Model Status
    print("\n--- 6. Testing /api/model-status ---")
    with urllib.request.urlopen(f"{BASE_URL}/api/model-status") as resp:
        ms_data = json.loads(resp.read().decode('utf-8'))
    print(f"Model status: trained={ms_data.get('trained')}, valid={ms_data.get('artifact_valid')}, champion={ms_data.get('champion_model')}")
    assert ms_data.get("trained") is True
    assert ms_data.get("artifact_valid") is True

    # 7. Real Crop Prediction
    print("\n--- 7. Testing /api/predict (Inference) ---")
    pred_payload = json.dumps({
        "N": 90, "P": 42, "K": 43,
        "temperature": 20.879744, "humidity": 82.002744,
        "ph": 6.502985, "rainfall": 202.935536
    }).encode('utf-8')

    p_req = urllib.request.Request(
        f"{BASE_URL}/api/predict",
        data=pred_payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    t_pred_start = time.perf_counter()
    with urllib.request.urlopen(p_req) as resp:
        p_data = json.loads(resp.read().decode('utf-8'))
    t_pred = time.perf_counter() - t_pred_start
    print(f"Inference finished in {t_pred:.4f}s: predicted_crop={p_data.get('predicted_crop')}, champion_model={p_data.get('champion_model')}")
    assert p_data.get("success") is True
    assert p_data.get("predicted_crop") == "rice"

    # 8. Prediction History
    print("\n--- 8. Testing /api/prediction-history ---")
    with urllib.request.urlopen(f"{BASE_URL}/api/prediction-history") as resp:
        hist_data = json.loads(resp.read().decode('utf-8'))
    print(f"History count: {hist_data.get('count')}")
    assert hist_data.get("count") >= 1
    last_record = hist_data["history"][0]
    print(f"Last record: timestamp={last_record.get('timestamp')}, crop={last_record.get('predicted_crop')}, model={last_record.get('champion_model')}")
    assert last_record.get("predicted_crop") == "rice"

    print("\n=== ALL REAL E2E TESTS PASSED WITH 100% SUCCESS! ===")

if __name__ == "__main__":
    run_test()
