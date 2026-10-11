# Data Directory (`data/`)

This directory is reserved for runtime data storage in the AgriSense Analytics platform:

1. **Active Ingested Dataset**:
   - `active_dataset.csv`: Ingested and canonicalized CSV dataset.
   - `active_dataset.json`: Validated metadata containing schema verification, row count, target class list, and SHA-256 canonical dataset fingerprint.

2. **Inference Audit Log**:
   - `prediction_history.db`: Local SQLite database storing authentic inference logs (`timestamp, n, p, k, temperature, humidity, ph, rainfall, predicted_crop, champion_model, dataset_fingerprint`).

### Clean Startup Policy
In adherence to data integrity standards, production releases start with **no active dataset**. The application automatically generates active dataset and database files upon user upload and prediction.
