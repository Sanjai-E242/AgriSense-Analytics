"""
Dataset Validator and Ingestion Engine — AgriSense Analytics
Principles of Data Science (PoDS) Project

Universal Ingestion & Validation Rules:
1. File Size: Maximum 15 MB enforced.
2. Encoding Support: Robust UTF-8 with BOM (utf-8-sig), standard UTF-8, and Latin-1 fallback.
3. Universal Column Normalization & Canonical Schema:
   Normalizes recognized columns to:
   ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
   Supports column names with leading/trailing whitespace, capitalization differences,
   and arbitrary column ordering.
4. Documented Column Aliases:
   - N: nitrogen, soil_nitrogen, n_ratio, n_content, n_level
   - P: phosphorus, phosphorous, soil_phosphorus, p_ratio, p_content, p_level
   - K: potassium, soil_potassium, k_ratio, k_content, k_level
   - temperature: temp, ambient_temperature, air_temp, temperature_c, temp_c
   - humidity: relative_humidity, rel_humidity, humid, humidity_percent, rh
   - ph: soil_ph, ph_level, ph_value
   - rainfall: precipitation, rain, rainfall_mm, precip
   - label: crop, crop_name, crop_label, target, crop_type, recommended_crop
5. Ambiguity & Duplicate Detection:
   - Detects duplicate column names in the CSV header.
   - Flags collisions if multiple source columns map to the same canonical field.
6. Additional Unrelated Columns:
   - Safely extracts canonical features while logging informative warnings for additional columns.
   - Supports strict validation mode when explicitly requested.
7. Data Quality & Scientific Integrity:
   - Rejects null, NaN, empty strings, and whitespace-only entries.
   - Rejects non-numeric, positive infinity, negative infinity, or unparseable feature values.
   - Rejects blank target labels.
   - Requires at least 2 distinct valid crop classes.
   - Accurately tracks duplicate rows without silently mutating or deleting them.
   - Non-fatal physical agricultural range warnings for extreme values.
   - Generates deterministic dataset fingerprints (SHA-256).
   - Distinguishes schema validity from provenance status ('Source not verified').
"""

import io
import math
import hashlib
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Exact standard schema required for model inputs and target
REQUIRED_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"
REQUIRED_SCHEMA = REQUIRED_FEATURES + [TARGET_COLUMN]

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

# Documented aliases (lowercase stripped alias -> canonical field)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "N": ["n", "nitrogen", "soil_nitrogen", "n_ratio", "n_content", "n_level"],
    "P": ["p", "phosphorus", "phosphorous", "soil_phosphorus", "p_ratio", "p_content", "p_level"],
    "K": ["k", "potassium", "soil_potassium", "k_ratio", "k_content", "k_level"],
    "temperature": ["temperature", "temp", "ambient_temperature", "air_temp", "temperature_c", "temp_c"],
    "humidity": ["humidity", "relative_humidity", "rel_humidity", "humid", "humidity_percent", "rh"],
    "ph": ["ph", "soil_ph", "ph_level", "ph_value"],
    "rainfall": ["rainfall", "precipitation", "rain", "rainfall_mm", "precip"],
    "label": ["label", "crop", "crop_name", "crop_label", "target", "crop_type", "recommended_crop"]
}

# Physical bounds for configurable non-fatal agricultural warnings
PHYSICAL_BOUNDS: Dict[str, Dict[str, float]] = {
    "N": {"min": 0.0, "max": 300.0, "unit": "index ratio / mg/kg"},
    "P": {"min": 0.0, "max": 300.0, "unit": "index ratio / mg/kg"},
    "K": {"min": 0.0, "max": 300.0, "unit": "index ratio / mg/kg"},
    "temperature": {"min": -10.0, "max": 60.0, "unit": "°C"},
    "humidity": {"min": 0.0, "max": 100.0, "unit": "%"},
    "ph": {"min": 0.0, "max": 14.0, "unit": "pH scale"},
    "rainfall": {"min": 0.0, "max": 3000.0, "unit": "mm"}
}


def normalize_column_name(raw_name: str) -> Optional[str]:
    """
    Maps a raw column name to its canonical feature name using documented aliases.
    Case-insensitive, whitespace-trimmed.
    """
    cleaned = str(raw_name).strip().lower()
    for canon, aliases in COLUMN_ALIASES.items():
        if cleaned in aliases:
            return canon
    return None


def validate_csv_content(
    csv_bytes: bytes,
    filename: str,
    strict_columns: bool = False
) -> Dict[str, Any]:
    """
    Authoritatively validates, normalizes, and inspects an uploaded CSV file.
    Does not silently mutate source rows or drop records.
    Returns structured validation telemetry or explicit failure reasons.
    """
    # 1. File size check (15 MB maximum)
    file_size_bytes = len(csv_bytes)
    if file_size_bytes == 0:
        return {
            "success": False,
            "error_type": "EMPTY_FILE",
            "message": "The selected file is empty (0 bytes). Please upload a valid CSV file with data.",
            "filename": filename,
            "file_size": 0
        }

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        return {
            "success": False,
            "error_type": "FILE_TOO_LARGE",
            "message": f"File size ({file_size_bytes / (1024 * 1024):.2f} MB) exceeds the 15 MB limit.",
            "filename": filename,
            "file_size": file_size_bytes
        }

    # Deterministic dataset fingerprint will be computed from canonical schema data
    dataset_fingerprint = ""

    # 2. Text decoding (UTF-8 BOM utf-8-sig first, then utf-8, then latin-1 fallback)
    csv_text = ""
    for enc in ['utf-8-sig', 'utf-8', 'latin-1']:
        try:
            csv_text = csv_bytes.decode(enc)
            break
        except Exception:
            continue

    if not csv_text:
        return {
            "success": False,
            "error_type": "DECODE_ERROR",
            "message": "Unable to decode file content as UTF-8, UTF-8 BOM, or Latin-1 text.",
            "filename": filename,
            "file_size": file_size_bytes
        }

    # 3. Parse CSV via pandas with strict line validation
    try:
        df = pd.read_csv(io.StringIO(csv_text), on_bad_lines='error')
    except Exception as e:
        return {
            "success": False,
            "error_type": "PARSE_ERROR",
            "message": f"Malformed CSV structure: {str(e)}",
            "filename": filename,
            "file_size": file_size_bytes
        }

    if df.empty or len(df) == 0:
        return {
            "success": False,
            "error_type": "EMPTY_DATASET",
            "message": "The CSV file contains a header but no data records (0 rows).",
            "filename": filename,
            "file_size": file_size_bytes
        }

    row_count, col_count = df.shape
    raw_columns = [str(c) for c in df.columns]

    # 4. Check for duplicate raw column names in header before pandas alters them
    import csv
    try:
        header_line = csv_text.strip().splitlines()[0]
        raw_header_tokens = [tok.strip().lower() for tok in next(csv.reader([header_line]))]
        if len(set(raw_header_tokens)) != len(raw_header_tokens):
            dup_names = [c for c in raw_header_tokens if raw_header_tokens.count(c) > 1]
            return {
                "success": False,
                "error_type": "DUPLICATE_COLUMNS",
                "message": f"Duplicate column names detected in CSV header: {list(set(dup_names))}.",
                "actual_columns": raw_header_tokens,
                "filename": filename,
                "file_size": file_size_bytes
            }
    except Exception:
        pass

    # 5. Schema Normalization & Alias Resolution
    mapping: Dict[str, List[str]] = {canon: [] for canon in REQUIRED_SCHEMA}
    extra_cols: List[str] = []

    for raw_col in raw_columns:
        canon = normalize_column_name(raw_col)
        if canon:
            mapping[canon].append(raw_col)
        else:
            extra_cols.append(raw_col)

    # Check for ambiguous multiple mappings to the same canonical column
    ambiguous_mappings = {canon: cols for canon, cols in mapping.items() if len(cols) > 1}
    if ambiguous_mappings:
        ambig_desc = [f"'{canon}': {cols}" for canon, cols in ambiguous_mappings.items()]
        return {
            "success": False,
            "error_type": "AMBIGUOUS_COLUMN_MAPPING",
            "message": f"Multiple columns map to the same canonical feature: {'; '.join(ambig_desc)}. Please provide an unambiguous CSV.",
            "ambiguous_mappings": ambiguous_mappings,
            "actual_columns": raw_columns,
            "filename": filename,
            "file_size": file_size_bytes
        }

    # Check for missing required canonical columns
    missing_required = [canon for canon in REQUIRED_SCHEMA if len(mapping[canon]) == 0]
    if missing_required:
        alias_hints = {canon: COLUMN_ALIASES[canon] for canon in missing_required}
        return {
            "success": False,
            "error_type": "MISSING_REQUIRED_COLUMNS",
            "message": f"Dataset is missing required column(s): {', '.join(missing_required)}. Accepted aliases for missing fields: {alias_hints}.",
            "missing_columns": missing_required,
            "actual_columns": raw_columns,
            "accepted_aliases": alias_hints,
            "filename": filename,
            "file_size": file_size_bytes
        }

    # Additional unrelated columns policy
    warnings: List[str] = []
    if extra_cols:
        if strict_columns:
            return {
                "success": False,
                "error_type": "UNEXPECTED_EXTRA_COLUMNS",
                "message": f"Dataset contains unexpected extra column(s): {', '.join(extra_cols)}. Strict mode requires strictly the 8 canonical columns.",
                "extra_columns": extra_cols,
                "actual_columns": raw_columns,
                "filename": filename,
                "file_size": file_size_bytes
            }
        warnings.append(
            f"Additional column(s) detected: {extra_cols}. The canonical 8 features were safely extracted and mapped."
        )

    # 6. Extract Normalized DataFrame in Exact Canonical Order
    norm_df = pd.DataFrame()
    for canon in REQUIRED_SCHEMA:
        source_col = mapping[canon][0]
        norm_df[canon] = df[source_col]

    # 7. Missing / Null / Blank / Whitespace Audit
    null_counts: Dict[str, int] = {}
    for col in REQUIRED_SCHEMA:
        null_count = int(norm_df[col].isnull().sum())
        blank_mask = norm_df[col].astype(str).str.strip().isin(['', 'nan', 'null', 'none', 'NA', 'N/A', '.'])
        null_count = max(null_count, int(blank_mask.sum()))
        null_counts[col] = null_count

    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        failing_cols = [c for c, cnt in null_counts.items() if cnt > 0]
        return {
            "success": False,
            "error_type": "NULL_VALUES_DETECTED",
            "message": f"Dataset contains {total_nulls} missing/blank value(s) in column(s): {', '.join(failing_cols)}. The dataset must be complete without missing values.",
            "null_counts": null_counts,
            "total_nulls": total_nulls,
            "filename": filename,
            "file_size": file_size_bytes
        }

    # 8. Feature Numeric & Finite Validity Check
    non_numeric_features: List[str] = []
    infinite_features: List[str] = []
    numeric_df = pd.DataFrame()

    for col in REQUIRED_FEATURES:
        try:
            converted = pd.to_numeric(norm_df[col], errors='coerce')
            if converted.isnull().any():
                non_numeric_features.append(col)
                continue
            if np.isinf(converted).any():
                infinite_features.append(col)
                continue
            numeric_df[col] = converted
        except Exception:
            non_numeric_features.append(col)

    if non_numeric_features:
        return {
            "success": False,
            "error_type": "NON_NUMERIC_FEATURES",
            "message": f"Feature column(s) contain invalid non-numeric values: {', '.join(non_numeric_features)}.",
            "non_numeric_features": non_numeric_features,
            "filename": filename,
            "file_size": file_size_bytes
        }

    if infinite_features:
        return {
            "success": False,
            "error_type": "INFINITE_VALUES_DETECTED",
            "message": f"Feature column(s) contain infinite (inf / -inf) values: {', '.join(infinite_features)}.",
            "infinite_features": infinite_features,
            "filename": filename,
            "file_size": file_size_bytes
        }

    # 9. Target Label Validity & Class Distribution Check
    target_series = norm_df[TARGET_COLUMN].astype(str).str.strip()
    blank_targets = int((target_series == '').sum())
    if blank_targets > 0:
        return {
            "success": False,
            "error_type": "BLANK_TARGET_LABEL",
            "message": f"Found {blank_targets} blank or empty target label(s) in '{TARGET_COLUMN}'.",
            "filename": filename,
            "file_size": file_size_bytes
        }

    class_counts = target_series.value_counts().to_dict()
    unique_classes = sorted(list(class_counts.keys()))

    if len(unique_classes) < 2:
        return {
            "success": False,
            "error_type": "INSUFFICIENT_CLASSES",
            "message": f"Target column '{TARGET_COLUMN}' contains only {len(unique_classes)} class ('{unique_classes[0] if unique_classes else 'None'}'). Classification requires at least 2 distinct crop species classes.",
            "class_counts": class_counts,
            "filename": filename,
            "file_size": file_size_bytes
        }

    # Deterministic dataset fingerprint (SHA-256 computed on normalized canonical data)
    dataset_fingerprint = hashlib.sha256(norm_df[REQUIRED_SCHEMA].to_csv(index=False).encode('utf-8')).hexdigest()[:16]

    # 10. Rare Classes Warning (Check minimum samples for cross-validation)
    min_class_samples = min(class_counts.values())
    if min_class_samples < 5:
        rare_classes = [c for c, count in class_counts.items() if count < 5]
        warnings.append(
            f"Class(es) {rare_classes} contain fewer than 5 samples (min count: {min_class_samples}). "
            f"Stratified 5-fold cross-validation will fall back to training-set selection."
        )

    # 11. Duplicate Rows Scan
    duplicate_count = int(norm_df.duplicated().sum())
    duplicate_percentage = round((duplicate_count / row_count) * 100, 2)
    if duplicate_count > 0:
        warnings.append(
            f"Detected {duplicate_count} duplicate row(s) ({duplicate_percentage}%). "
            f"Duplicate records are tracked without silent modification."
        )

    # 12. Physical Agricultural Range Inspection (Non-fatal warnings)
    range_warnings: List[str] = []
    for col, bounds in PHYSICAL_BOUNDS.items():
        col_min = float(numeric_df[col].min())
        col_max = float(numeric_df[col].max())
        if col_min < bounds["min"] or col_max > bounds["max"]:
            range_warnings.append(
                f"Feature '{col}' values [{col_min:.2f}, {col_max:.2f}] exceed documented standard "
                f"agronomic bounds [{bounds['min']}, {bounds['max']}] {bounds['unit']}."
            )

    # 13. Summary Statistics for Continuous Features
    summary_stats = {}
    for col in REQUIRED_FEATURES:
        col_vals = numeric_df[col]
        summary_stats[col] = {
            "count": int(col_vals.count()),
            "mean": round(float(col_vals.mean()), 2) if not math.isnan(col_vals.mean()) else 0.0,
            "std": round(float(col_vals.std()), 2) if not math.isnan(col_vals.std()) else 0.0,
            "min": round(float(col_vals.min()), 2) if not math.isnan(col_vals.min()) else 0.0,
            "q25": round(float(col_vals.quantile(0.25)), 2) if not math.isnan(col_vals.quantile(0.25)) else 0.0,
            "median": round(float(col_vals.median()), 2) if not math.isnan(col_vals.median()) else 0.0,
            "q75": round(float(col_vals.quantile(0.75)), 2) if not math.isnan(col_vals.quantile(0.75)) else 0.0,
            "max": round(float(col_vals.max()), 2) if not math.isnan(col_vals.max()) else 0.0
        }

    # 14. Pearson Correlation Matrix
    corr_df = numeric_df.corr(method="pearson").round(3)
    correlation_matrix = {
        "columns": REQUIRED_FEATURES,
        "matrix": corr_df.to_dict()
    }

    # 15. Target Summary
    min_freq = min(class_counts.values())
    max_freq = max(class_counts.values())
    target_summary = {
        "target_column": TARGET_COLUMN,
        "unique_classes_count": len(unique_classes),
        "classes": unique_classes,
        "class_frequencies": {str(k): int(v) for k, v in class_counts.items()},
        "is_balanced": bool((max_freq - min_freq) / max(1, max_freq) < 0.20)
    }

    # 16. Inferred Data Types
    dtypes_inferred = {col: "Numeric (Continuous Float)" for col in REQUIRED_FEATURES}
    dtypes_inferred[TARGET_COLUMN] = "Categorical (String/Class)"

    # 17. Quality Checklist
    checklist = {
        "required_columns_present": {
            "passed": True,
            "details": f"All {len(REQUIRED_SCHEMA)} canonical features mapped and validated."
        },
        "numeric_types_valid": {
            "passed": True,
            "details": "All 7 agronomic inputs contain strictly finite numeric values."
        },
        "missing_values_checked": {
            "passed": total_nulls == 0,
            "details": "0 missing or blank values detected across all columns."
        },
        "duplicate_rows_checked": {
            "passed": duplicate_count == 0,
            "details": "0 duplicate sample rows detected." if duplicate_count == 0 else f"Detected {duplicate_count} duplicate rows ({duplicate_percentage}%)."
        },
        "target_labels_inspected": {
            "passed": len(unique_classes) >= 2,
            "details": f"Found {len(unique_classes)} distinct valid crop species classes."
        },
        "ranges_reviewed": {
            "passed": len(range_warnings) == 0,
            "details": "Physical agricultural measurement bounds verified." if len(range_warnings) == 0 else f"{len(range_warnings)} parameter range warnings reported."
        }
    }

    # Return structured telemetry — includes top-level classes to prevent metadata bugs
    return {
        "success": True,
        "filename": filename,
        "file_size": file_size_bytes,
        "dataset_fingerprint": dataset_fingerprint,
        "provenance_status": "Source not verified (User Upload)",
        "rows": row_count,
        "columns": REQUIRED_SCHEMA,
        "column_count": len(REQUIRED_SCHEMA),
        "schema_matched": True,
        "missing_expected": [],
        "extra_cols": extra_cols,
        "extra_columns": extra_cols,
        "column_mappings": {mapping[c][0]: c for c in REQUIRED_SCHEMA},
        "warnings": warnings,
        "range_warnings": range_warnings,
        "dtypes": dtypes_inferred,
        "null_counts": null_counts,
        "null_percentages": {col: 0.0 for col in REQUIRED_SCHEMA},
        "total_nulls": 0,
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
        "classes": unique_classes,
        "target_classes": unique_classes,
        "target_summary": target_summary,
        "summary_stats": summary_stats,
        "correlation_matrix": correlation_matrix,
        "checklist": checklist,
        "normalized_csv": norm_df.to_csv(index=False)
    }
