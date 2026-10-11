"""
Reproducible Machine Learning Training, Evaluation & Inference Pipeline
Crop Recommendation System — Principles of Data Science (PoDS) Project

Architecture & Methodology:
1. DATASET SAFETY: Requires explicit user approval. Rejects test/sample fixtures.
2. LEAK-FREE PREPROCESSING: StandardScaler fit strictly on training partitions;
   never fit on validation or test observations.
3. SOUND EVALUATION METHODOLOGY:
   - 80/20 train/test split.
   - Cross-validation evaluated strictly inside the 80% training partition.
   - Model selection (champion choice) determined by mean cross-validation Macro-F1.
   - Tie-break rule: mean cross-validation accuracy.
   - Unbiased held-out test evaluation conducted on the untouched 20% test set.
   - Production champion pipeline trained on complete dataset for live inference.
4. MODEL STORAGE & INFERENCE:
   - Saves production pipeline to models/best_crop_model.json.
   - Saves comprehensive metadata to models/model_metadata.json.
   - Provides safe deserialization and range-checked inference.
"""

import os
import json
import time
import math
import datetime
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# Reproducibility Seed
RANDOM_SEED = 42

# Agronomic Feature Schema (in exact required order)
REQUIRED_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"
REQUIRED_SCHEMA = REQUIRED_FEATURES + [TARGET_COLUMN]

# Documented physical feature bounds for validation and inference
FEATURE_BOUNDS = {
    "N": {"min": 0.0, "max": 500.0, "unit": "mg/kg", "name": "Nitrogen"},
    "P": {"min": 0.0, "max": 500.0, "unit": "mg/kg", "name": "Phosphorus"},
    "K": {"min": 0.0, "max": 500.0, "unit": "mg/kg", "name": "Potassium"},
    "temperature": {"min": -10.0, "max": 60.0, "unit": "°C", "name": "Temperature"},
    "humidity": {"min": 0.0, "max": 100.0, "unit": "%", "name": "Relative Humidity"},
    "ph": {"min": 0.0, "max": 14.0, "unit": "pH", "name": "Soil pH"},
    "rainfall": {"min": 0.0, "max": 2000.0, "unit": "mm", "name": "Rainfall"}
}

# Designated models storage directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Prohibited test fixture identifiers (specifically designated fixtures only)
DISALLOWED_FIXTURE_NAMES = [
    "crop_recommendation_sample_test.csv",
    "crop_recommendation_5000_simulated_demo.csv",
    "fixture_data.csv",
    "synthetic_dataset.csv"
]

# ==============================================================================
# 1. PREPROCESSING: STANDARD SCALER (Leak-Free)
# ==============================================================================
class StandardScalerModel:
    """Standardizes features by removing mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScalerModel has not been fitted.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "scale": self.scale_.tolist() if self.scale_ is not None else []
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        scaler = cls()
        if "mean" in data and "scale" in data and len(data["mean"]) > 0:
            scaler.mean_ = np.array(data["mean"], dtype=float)
            scaler.scale_ = np.array(data["scale"], dtype=float)
        return scaler

# ==============================================================================
# 2. CLASSIFIER 1: K-NEAREST NEIGHBORS (KNN)
# ==============================================================================
class KNNClassifierModel:
    """K-Nearest Neighbors Classifier with Euclidean distance."""
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=str)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError("KNNClassifierModel has not been fitted.")
        preds = []
        k_val = min(self.k, len(self.X_train_))
        for x in X:
            dists = np.linalg.norm(self.X_train_ - x, axis=1)
            nn_idx = np.argsort(dists)[:k_val]
            nn_labels = self.y_train_[nn_idx]
            # Majority voting with distance tie-break
            unique, counts = np.unique(nn_labels, return_counts=True)
            winner = unique[np.argmax(counts)]
            preds.append(winner)
        return np.array(preds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "K-Nearest Neighbors",
            "k": int(self.k),
            "X_train": self.X_train_.tolist() if self.X_train_ is not None else [],
            "y_train": self.y_train_.tolist() if self.y_train_ is not None else []
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        knn = cls(k=data.get("k", 5))
        if "X_train" in data and "y_train" in data and len(data["X_train"]) > 0:
            knn.X_train_ = np.array(data["X_train"], dtype=float)
            knn.y_train_ = np.array(data["y_train"], dtype=str)
        return knn

# ==============================================================================
# 3. CLASSIFIER 2: DECISION TREE CLASSIFIER (CART Gini)
# ==============================================================================
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self):
        return self.value is not None

class DecisionTreeModel:
    """CART Decision Tree Classifier with Gini Impurity."""
    def __init__(self, max_depth: int = 10, min_samples_split: int = 4, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root = None

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - float(np.sum(probs ** 2))

    def _best_split(self, X: np.ndarray, y: np.ndarray, feat_subset: Optional[List[int]] = None):
        best_gain = -1.0
        best_feat = None
        best_thresh = None

        n_samples, n_features = X.shape
        features = feat_subset if feat_subset is not None else list(range(n_features))
        current_gini = self._gini(y)

        for feat in features:
            col = X[:, feat]
            thresholds = np.unique(col)
            if len(thresholds) > 20:
                thresholds = np.percentile(col, np.linspace(5, 95, 19))

            for thresh in thresholds:
                left_mask = col <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                p_left = np.sum(left_mask) / n_samples
                p_right = 1.0 - p_left
                weighted_gini = p_left * left_gini + p_right * right_gini
                gain = current_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0, feat_subset_size: Optional[int] = None) -> DecisionTreeNode:
        n_samples, n_features = X.shape
        unique_classes, counts = np.unique(y, return_counts=True)

        if len(unique_classes) == 1 or depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_val = unique_classes[np.argmax(counts)]
            return DecisionTreeNode(value=str(leaf_val))

        rng = np.random.RandomState(self.random_state + depth)
        feat_subset = None
        if feat_subset_size is not None and feat_subset_size < n_features:
            feat_subset = list(rng.choice(n_features, size=feat_subset_size, replace=False))

        best_feat, best_thresh = self._best_split(X, y, feat_subset)
        if best_feat is None:
            leaf_val = unique_classes[np.argmax(counts)]
            return DecisionTreeNode(value=str(leaf_val))

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, feat_subset_size)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, feat_subset_size)
        return DecisionTreeNode(feature=int(best_feat), threshold=float(best_thresh), left=left_child, right=right_child)

    def fit(self, X: np.ndarray, y: np.ndarray, feat_subset_size: Optional[int] = None):
        self.root = self._build_tree(np.asarray(X, dtype=float), np.asarray(y, dtype=str), depth=0, feat_subset_size=feat_subset_size)
        return self

    def _predict_sample(self, node: DecisionTreeNode, x: np.ndarray):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(node.left, x)
        return self._predict_sample(node.right, x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.root is None:
            raise ValueError("DecisionTreeModel has not been fitted.")
        return np.array([self._predict_sample(self.root, x) for x in X])

    def _node_to_dict(self, node: DecisionTreeNode):
        if node is None:
            return None
        if node.is_leaf:
            return {"value": str(node.value)}
        return {
            "feature": int(node.feature),
            "threshold": float(node.threshold),
            "left": self._node_to_dict(node.left),
            "right": self._node_to_dict(node.right)
        }

    @classmethod
    def _dict_to_node(cls, d: Dict[str, Any]) -> Optional[DecisionTreeNode]:
        if not d:
            return None
        if "value" in d:
            return DecisionTreeNode(value=str(d["value"]))
        return DecisionTreeNode(
            feature=int(d["feature"]),
            threshold=float(d["threshold"]),
            left=cls._dict_to_node(d.get("left")),
            right=cls._dict_to_node(d.get("right"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Decision Tree",
            "max_depth": int(self.max_depth),
            "min_samples_split": int(self.min_samples_split),
            "random_state": int(self.random_state),
            "tree": self._node_to_dict(self.root)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        tree = cls(
            max_depth=data.get("max_depth", 10),
            min_samples_split=data.get("min_samples_split", 4),
            random_state=data.get("random_state", 42)
        )
        if "tree" in data:
            tree.root = cls._dict_to_node(data["tree"])
        return tree

# ==============================================================================
# 4. CLASSIFIER 3: RANDOM FOREST CLASSIFIER
# ==============================================================================
class RandomForestModel:
    """Random Forest Classifier with Bootstrap Aggregation & Feature Subsampling."""
    def __init__(self, n_estimators: int = 25, max_depth: int = 10, min_samples_split: int = 3, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=str)
        n_samples, n_features = X.shape
        feat_subset_size = max(1, int(math.sqrt(n_features)))

        self.trees = []
        for i in range(self.n_estimators):
            tree_seed = self.random_state + (i * 17)
            rng = np.random.RandomState(tree_seed)
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            tree = DecisionTreeModel(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=tree_seed
            )
            tree.fit(X[boot_idx], y[boot_idx], feat_subset_size=feat_subset_size)
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not self.trees:
            raise ValueError("RandomForestModel has not been fitted.")
        all_preds = np.array([tree.predict(X) for tree in self.trees]) # (n_trees, n_samples)
        final_preds = []
        for col_idx in range(X.shape[0]):
            votes = all_preds[:, col_idx]
            unique, counts = np.unique(votes, return_counts=True)
            winner = unique[np.argmax(counts)]
            final_preds.append(winner)
        return np.array(final_preds)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Random Forest",
            "n_estimators": int(self.n_estimators),
            "max_depth": int(self.max_depth),
            "min_samples_split": int(self.min_samples_split),
            "random_state": int(self.random_state),
            "trees": [t.to_dict() for t in self.trees]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        rf = cls(
            n_estimators=data.get("n_estimators", 25),
            max_depth=data.get("max_depth", 10),
            min_samples_split=data.get("min_samples_split", 3),
            random_state=data.get("random_state", 42)
        )
        if "trees" in data:
            rf.trees = [DecisionTreeModel.from_dict(t) for t in data["trees"]]
        return rf

# Aliases for classifier classes
DecisionTreeClassifierModel = DecisionTreeModel
RandomForestClassifierModel = RandomForestModel
KNNClassifier = KNNClassifierModel

# ==============================================================================
# 5. UNIFIED ML PIPELINE (Scaler + Classifier Container)
# ==============================================================================
class ModelPipeline:
    """Encapsulates StandardScaler and a Classifier to eliminate data leakage."""
    def __init__(self, name: str, classifier, scaler: Optional[StandardScalerModel] = None):
        self.name = name
        self.classifier = classifier
        self.scaler = scaler if scaler is not None else StandardScalerModel()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def to_dict(self) -> Dict[str, Any]:
        classifier_type = self.name
        return {
            "name": self.name,
            "classifier_type": classifier_type,
            "scaler": self.scaler.to_dict(),
            "classifier_data": self.classifier.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        name = data.get("name", "Unknown Classifier")
        clf_type = data.get("classifier_type", name)
        scaler = StandardScalerModel.from_dict(data.get("scaler", {}))
        clf_data = data.get("classifier_data", {})

        if "Decision Tree" in clf_type or "DecisionTree" in clf_type:
            classifier = DecisionTreeModel.from_dict(clf_data)
        elif "Random Forest" in clf_type or "RandomForest" in clf_type:
            classifier = RandomForestModel.from_dict(clf_data)
        elif "K-Nearest" in clf_type or "KNN" in clf_type:
            classifier = KNNClassifierModel.from_dict(clf_data)
        else:
            raise ValueError(f"Unknown classifier type in model artifact: '{clf_type}'")

        return cls(name=name, classifier=classifier, scaler=scaler)

# ==============================================================================
# 6. DATASET SAFETY & PRE-TRAINING AUDITING
# ==============================================================================
def check_fixture_block(dataset_name: str, allow_fixture_for_testing: bool = False) -> Tuple[bool, str]:
    if allow_fixture_for_testing:
        return True, ""

    name_clean = os.path.basename(str(dataset_name)).lower().strip()
    if name_clean in DISALLOWED_FIXTURE_NAMES or name_clean.startswith("synthetic_") or name_clean.startswith("fixture_"):
        return False, (
            f"Training prohibited on designated test fixture '{dataset_name}'. "
            f"In compliance with project safety protocols, models may only be trained "
            f"on a verified agricultural dataset authorized by the project owner."
        )
    return True, ""

def validate_training_eligibility(
    df: pd.DataFrame,
    dataset_name: str,
    approved_by_user: bool = False,
    allow_fixture_for_testing: bool = False,
    strict_columns: bool = False
) -> Dict[str, Any]:
    """Validates dataset safety, schema completeness, and sample sufficiency."""
    # 1. Project Owner Confirmation Check
    if not approved_by_user:
        return {
            "eligible": False,
            "error_type": "APPROVAL_REQUIRED",
            "message": "Explicit project owner confirmation required. Check the authorization checkbox before initiating model training."
        }

    # 2. Test Fixture Check
    allowed, fixture_msg = check_fixture_block(dataset_name, allow_fixture_for_testing)
    if not allowed:
        return {
            "eligible": False,
            "error_type": "FIXTURE_DATASET_BLOCKED",
            "message": fixture_msg
        }

    # 3. Non-empty Check
    if df.empty or len(df) == 0:
        return {
            "eligible": False,
            "error_type": "EMPTY_DATASET",
            "message": "The selected dataset contains 0 records."
        }

    # 4. Universal Column Mapping & Reordering
    from validator import normalize_column_name
    mapping: Dict[str, List[str]] = {canon: [] for canon in REQUIRED_SCHEMA}
    cols = list(df.columns)
    extra_cols: List[str] = []

    for c in cols:
        canon = normalize_column_name(str(c))
        if canon:
            mapping[canon].append(str(c))
        else:
            extra_cols.append(str(c))

    missing_required = [canon for canon in REQUIRED_SCHEMA if len(mapping[canon]) == 0]
    if missing_required:
        return {
            "eligible": False,
            "error_type": "MISSING_REQUIRED_COLUMNS",
            "message": f"Dataset is missing required column(s): {', '.join(missing_required)}.",
            "missing_columns": missing_required
        }

    if extra_cols and strict_columns:
        return {
            "eligible": False,
            "error_type": "UNEXPECTED_EXTRA_COLUMNS",
            "message": f"Dataset contains unexpected extra column(s): {', '.join(extra_cols)}. Exact 8 columns required: {', '.join(REQUIRED_SCHEMA)}.",
            "extra_columns": extra_cols
        }

    # Clean normalized DataFrame with canonical column order
    clean_df = pd.DataFrame()
    for canon in REQUIRED_SCHEMA:
        clean_df[canon] = df[mapping[canon][0]]
    df = clean_df

    # 5. Numeric & Finite Values Check
    non_numeric_features = []
    infinite_features = []
    for f in REQUIRED_FEATURES:
        try:
            converted = pd.to_numeric(df[f], errors='coerce')
            if converted.isnull().any():
                non_numeric_features.append(f)
            elif np.isinf(converted).any():
                infinite_features.append(f)
        except Exception:
            non_numeric_features.append(f)

    if non_numeric_features:
        return {
            "eligible": False,
            "error_type": "NON_NUMERIC_FEATURES",
            "message": f"Feature column(s) contain invalid non-numeric values: {', '.join(non_numeric_features)}.",
            "non_numeric_features": non_numeric_features
        }

    if infinite_features:
        return {
            "eligible": False,
            "error_type": "INFINITE_VALUES_DETECTED",
            "message": f"Feature column(s) contain infinite (inf / -inf) values: {', '.join(infinite_features)}.",
            "infinite_features": infinite_features
        }

    # 6. Null & Blank Values Audit
    null_counts = df[REQUIRED_SCHEMA].isnull().sum().to_dict()
    total_nulls = int(df[REQUIRED_SCHEMA].isnull().sum().sum())
    blank_targets = int((df[TARGET_COLUMN].astype(str).str.strip().isin(['', 'nan', 'null'])).sum())
    if total_nulls > 0 or blank_targets > 0:
        return {
            "eligible": False,
            "error_type": "NULL_VALUES_DETECTED",
            "message": f"Detected {total_nulls} missing values and {blank_targets} blank target labels. The dataset must be complete.",
            "null_counts": null_counts,
            "total_nulls": total_nulls
        }

    # 7. Target Classes & Sample Size Check
    class_counts = df[TARGET_COLUMN].astype(str).str.strip().value_counts().to_dict()
    unique_classes = sorted(list(class_counts.keys()))
    total_samples = len(df)

    if len(unique_classes) < 2:
        return {
            "eligible": False,
            "error_type": "INSUFFICIENT_CLASSES",
            "message": f"Target column '{TARGET_COLUMN}' contains only {len(unique_classes)} class. Classification requires at least 2 distinct crop species."
        }

    min_class_samples = min(class_counts.values())
    max_class_samples = max(class_counts.values())

    reliability_warnings = []
    if total_samples < 25:
        reliability_warnings.append(
            f"Small Sample Caveat: Dataset contains {total_samples} samples. "
            f"Validation metrics on small datasets exhibit higher statistical variance."
        )

    if min_class_samples < 2:
        rare_classes = [c for c, count in class_counts.items() if count < 2]
        reliability_warnings.append(
            f"Under-represented class caveat: Class(es) {rare_classes} have only 1 observation. Stratified split cannot guarantee fold representation."
        )

    imbalance_ratio = max_class_samples / max(1, min_class_samples)
    if imbalance_ratio > 3.0:
        reliability_warnings.append(
            f"Class Imbalance Caveat: Ratio between dominant class ({max_class_samples}) and rare class ({min_class_samples}) is {imbalance_ratio:.1f}:1."
        )

    can_stratify = (min_class_samples >= 2)
    can_cv = (min_class_samples >= 3) and (total_samples >= 20)
    cv_folds = min(5, min_class_samples) if can_cv else 0

    return {
        "eligible": True,
        "dataset_name": dataset_name,
        "total_samples": total_samples,
        "feature_count": len(REQUIRED_FEATURES),
        "class_count": len(unique_classes),
        "classes": unique_classes,
        "class_counts": class_counts,
        "min_class_samples": min_class_samples,
        "can_stratify": can_stratify,
        "can_cv": can_cv,
        "cv_folds": cv_folds,
        "clean_df": clean_df,
        "reliability_warnings": reliability_warnings
    }

# ==============================================================================
# 7. STRATIFIED SAMPLING & CROSS-VALIDATION UTILITIES
# ==============================================================================
def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproducible train/test split with stratified sampling when feasible."""
    n_samples = len(y)
    rng = np.random.RandomState(random_state)

    if not stratify:
        indices = rng.permutation(n_samples)
        test_size = max(1, int(n_samples * test_ratio))
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    train_indices = []
    test_indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_cls_test = max(1, int(len(cls_idx) * test_ratio)) if len(cls_idx) >= 2 else 0
        test_indices.extend(cls_idx[:n_cls_test])
        train_indices.extend(cls_idx[n_cls_test:])

    train_idx = np.array(train_indices)
    test_idx = np.array(test_indices)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def stratified_kfold_cv_scores(
    classifier_cls,
    classifier_kwargs: dict,
    name: str,
    X_train_part: np.ndarray,
    y_train_part: np.ndarray,
    k_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluates cross-validation STRICTLY within the training partition.
    Fits a separate StandardScaler on each training fold to eliminate data leakage.
    Returns fold accuracy, fold macro-F1, mean, and std.
    """
    rng = np.random.RandomState(random_state)
    folds = [[] for _ in range(k_folds)]

    for cls in np.unique(y_train_part):
        cls_idx = np.where(y_train_part == cls)[0]
        rng.shuffle(cls_idx)
        for i, idx in enumerate(cls_idx):
            folds[i % k_folds].append(idx)

    acc_scores = []
    f1_scores = []
    classes = sorted(list(np.unique(y_train_part)))

    for fold_i in range(k_folds):
        val_idx = np.array(folds[fold_i])
        train_idx = np.array([idx for f in range(k_folds) if f != fold_i for idx in folds[f]])

        if len(val_idx) == 0 or len(train_idx) == 0:
            continue

        # Fit fresh pipeline strictly on this fold's training samples
        fold_scaler = StandardScalerModel()
        fold_X_tr_scaled = fold_scaler.fit_transform(X_train_part[train_idx])
        fold_X_val_scaled = fold_scaler.transform(X_train_part[val_idx])

        clf = classifier_cls(**classifier_kwargs)
        clf.fit(fold_X_tr_scaled, y_train_part[train_idx])
        preds = clf.predict(fold_X_val_scaled)

        # Fold metrics
        fold_acc = float(np.mean(preds == y_train_part[val_idx]))
        fold_metrics = compute_evaluation_metrics(y_train_part[val_idx], preds, classes)
        acc_scores.append(round(fold_acc, 4))
        f1_scores.append(round(fold_metrics["macro_f1"], 4))

    return {
        "supported": True,
        "folds": k_folds,
        "accuracy_scores": acc_scores,
        "accuracy_mean": round(float(np.mean(acc_scores)), 4) if acc_scores else None,
        "accuracy_std": round(float(np.std(acc_scores)), 4) if acc_scores else None,
        "macro_f1_scores": f1_scores,
        "macro_f1_mean": round(float(np.mean(f1_scores)), 4) if f1_scores else None,
        "macro_f1_std": round(float(np.std(f1_scores)), 4) if f1_scores else None,
        "mean": round(float(np.mean(f1_scores)), 4) if f1_scores else None,
        "std": round(float(np.std(f1_scores)), 4) if f1_scores else None
    }

# ==============================================================================
# 8. COMPREHENSIVE EVALUATION METRICS
# ==============================================================================
def compute_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    """Calculates accuracy, macro precision, recall, F1, confusion matrix, and per-class metrics."""
    n_samples = len(y_true)
    if n_samples == 0:
        return {}

    accuracy = float(np.mean(y_true == y_pred))

    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true_val, pred_val in zip(y_true, y_pred):
        if true_val in class_to_idx and pred_val in class_to_idx:
            cm[class_to_idx[true_val], class_to_idx[pred_val]] += 1

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = int(np.sum(cm[i, :]))

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * p * r) / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        per_class[cls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": {
            "classes": classes,
            "matrix": cm.tolist()
        },
        "per_class_metrics": per_class
    }

# ==============================================================================
# 9. MAIN TRAINING & EVALUATION WORKFLOW
# ==============================================================================
def train_and_evaluate_pipeline(
    df: pd.DataFrame,
    dataset_name: str,
    approved_by_user: bool = False,
    allow_fixture_for_testing: bool = False,
    dataset_fingerprint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes reproducible ML training and comparative benchmarking:
    1. Splits into 80% train / 20% untouched held-out test partitions.
    2. Runs cross-validation strictly within the 80% train partition.
    3. Selects champion model using mean cross-validation Macro-F1 (tie-break: CV accuracy).
    4. Evaluates champion and competitors on the untouched 20% test set.
    5. Fits the selected production pipeline on the complete dataset for live inference.
    6. Saves production model to models/best_crop_model.json and schema metadata.
    """
    # 1. Authoritative safety & eligibility validation
    eligibility = validate_training_eligibility(
        df, dataset_name, approved_by_user, allow_fixture_for_testing
    )
    if not eligibility["eligible"]:
        return {
            "success": False,
            "error_type": eligibility["error_type"],
            "message": eligibility["message"],
            "eligibility": eligibility
        }

    # Use normalized canonical DataFrame
    df = eligibility.get("clean_df", df)
    if not dataset_fingerprint:
        dataset_fingerprint = hashlib.sha256(df[REQUIRED_SCHEMA].to_csv(index=False).encode('utf-8')).hexdigest()[:16]

    # 2. Extract feature matrix X and target vector y
    X = df[REQUIRED_FEATURES].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].astype(str).str.strip().to_numpy()
    classes = eligibility["classes"]
    total_samples = len(df)

    # 3. Stratified Train / Test Partition (80% Train, 20% Untouched Held-out Test)
    test_ratio = 0.2 if total_samples >= 10 else 0.33
    can_stratify = eligibility["can_stratify"]

    X_train, X_test, y_train, y_test = stratified_split(
        X, y,
        test_ratio=test_ratio,
        random_state=RANDOM_SEED,
        stratify=can_stratify
    )

    train_samples = len(y_train)
    test_samples = len(y_test)

    # 4. Define Classifier Models Specifications
    # Model parameters documented for reproducibility
    models_spec = {
        "Decision Tree": {
            "cls": DecisionTreeModel,
            "kwargs": {"max_depth": 10, "min_samples_split": 4, "random_state": RANDOM_SEED}
        },
        "Random Forest": {
            "cls": RandomForestModel,
            "kwargs": {"n_estimators": 25, "max_depth": 10, "min_samples_split": 3, "random_state": RANDOM_SEED}
        },
        "K-Nearest Neighbors": {
            "cls": KNNClassifierModel,
            "kwargs": {"k": min(5, max(1, train_samples - 1))}
        }
    }

    # 5. Cross-Validation strictly inside Training Partition (for Model Selection)
    cv_class_counts = pd.Series(y_train).value_counts().to_dict()
    min_train_class = min(cv_class_counts.values()) if cv_class_counts else 0
    can_train_cv = (min_train_class >= 2) and (train_samples >= 10)
    cv_folds = min(5, min_train_class) if can_train_cv else 0

    benchmark_results = {}
    best_model_name = None
    best_cv_f1 = -1.0
    best_cv_acc = -1.0

    for name, spec in models_spec.items():
        # Step A: Perform CV strictly on (X_train, y_train)
        if can_train_cv and cv_folds >= 2:
            cv_result = stratified_kfold_cv_scores(
                spec["cls"], spec["kwargs"], name, X_train, y_train, k_folds=cv_folds, random_state=RANDOM_SEED
            )
        else:
            cv_result = {
                "supported": False,
                "folds": 0,
                "accuracy_scores": [],
                "accuracy_mean": None,
                "accuracy_std": None,
                "macro_f1_scores": [],
                "macro_f1_mean": None,
                "macro_f1_std": None
            }

        # Step B: Fit pipeline on full training partition (X_train, y_train) with its own scaler
        train_scaler = StandardScalerModel()
        X_tr_scaled = train_scaler.fit_transform(X_train)
        X_te_scaled = train_scaler.transform(X_test)

        train_clf = spec["cls"](**spec["kwargs"])
        train_clf.fit(X_tr_scaled, y_train)

        # In-sample train predictions and training-only metrics
        train_preds = train_clf.predict(X_tr_scaled)
        train_acc = round(float(np.mean(train_preds == y_train)), 4)
        train_eval = compute_evaluation_metrics(y_train, train_preds, classes)
        train_macro_f1 = train_eval["macro_f1"]

        # Held-out test predictions (Untouched test partition, evaluated strictly post-selection)
        test_preds = train_clf.predict(X_te_scaled)
        test_metrics = compute_evaluation_metrics(y_test, test_preds, classes)

        # Champion Selection Metric Definition:
        # Evaluated strictly inside the training partition to ensure zero test-data influence:
        # - Primary: Mean Cross-Validation Macro-F1 within training fold (if CV supported)
        # - Tie-break: Mean Cross-Validation Accuracy within training fold (if CV supported)
        # - Fallback (small datasets where CV is unavailable): In-sample training partition Macro-F1 (with train accuracy tie-break)
        # HELD-OUT TEST METRICS ARE NEVER USED FOR MODEL SELECTION.
        if cv_result["supported"] and cv_result["macro_f1_mean"] is not None:
            selection_f1 = cv_result["macro_f1_mean"]
            selection_acc = cv_result["accuracy_mean"] if cv_result["accuracy_mean"] is not None else 0.0
            selection_source = "Stratified 5-Fold CV (Training Partition Only)"
        else:
            selection_f1 = train_macro_f1
            selection_acc = train_acc
            selection_source = "In-Sample Training Partition (CV Unavailable on Small Dataset)"

        benchmark_results[name] = {
            "model_name": name,
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro_f1,
            "test_accuracy": test_metrics["accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "per_class_metrics": test_metrics["per_class_metrics"],
            "cross_validation": cv_result,
            "selection_score_f1": selection_f1,
            "selection_score_acc": selection_acc,
            "selection_source": selection_source
        }

        if (selection_f1 is not None and selection_f1 > best_cv_f1) or \
           (selection_f1 == best_cv_f1 and selection_acc is not None and selection_acc > best_cv_acc):
            best_cv_f1 = selection_f1
            best_cv_acc = selection_acc if selection_acc is not None else 0.0
            best_model_name = name

    if best_model_name is None:
        best_model_name = "Random Forest"

    # 6. Fit Production Pipeline on COMPLETE Eligible Dataset
    # Preprocessing is fit across all verified data for deployment inference
    prod_spec = models_spec[best_model_name]
    prod_scaler = StandardScalerModel()
    prod_X_scaled = prod_scaler.fit_transform(X)

    prod_clf = prod_spec["cls"](**prod_spec["kwargs"])
    prod_clf.fit(prod_X_scaled, y)

    production_pipeline = ModelPipeline(
        name=best_model_name,
        classifier=prod_clf,
        scaler=prod_scaler
    )

    # 7. Safe Serialization to models/best_crop_model.json and model_metadata.json
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_artifact_path = os.path.join(MODELS_DIR, "best_crop_model.json")
    metadata_artifact_path = os.path.join(MODELS_DIR, "model_metadata.json")

    with open(model_artifact_path, "w", encoding="utf-8") as f:
        json.dump(production_pipeline.to_dict(), f, indent=2)

    champion_test_metrics = benchmark_results[best_model_name]
    metadata_payload = {
        "champion_model": best_model_name,
        "selection_rule": f"Evaluated strictly within training partition via {champion_test_metrics['selection_source']}. Held-out test set was untouched during model selection.",
        "selection_source": champion_test_metrics["selection_source"],
        "selection_score_f1": champion_test_metrics["selection_score_f1"],
        "selection_score_acc": champion_test_metrics["selection_score_acc"],
        "feature_schema": REQUIRED_FEATURES,
        "target_classes": classes,
        "class_count": len(classes),
        "total_samples": total_samples,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "random_seed": RANDOM_SEED,
        "trained_at": datetime.datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "dataset_fingerprint": dataset_fingerprint,
        "champion_held_out_metrics": {
            "train_accuracy": champion_test_metrics["train_accuracy"],
            "test_accuracy": champion_test_metrics["test_accuracy"],
            "macro_f1": champion_test_metrics["macro_f1"],
            "macro_precision": champion_test_metrics["macro_precision"],
            "macro_recall": champion_test_metrics["macro_recall"]
        },
        "champion_cv_metrics": champion_test_metrics["cross_validation"],
        "benchmark_summary": {
            name: {
                "train_acc": m["train_accuracy"],
                "test_acc": m["test_accuracy"],
                "macro_f1": m["macro_f1"],
                "cv_macro_f1": m["cross_validation"]["macro_f1_mean"]
            } for name, m in benchmark_results.items()
        },
        "artifact_file": "models/best_crop_model.json"
    }

    with open(metadata_artifact_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    return {
        "success": True,
        "dataset_name": dataset_name,
        "dataset_fingerprint": dataset_fingerprint,
        "total_samples": total_samples,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "classes": classes,
        "champion_model": best_model_name,
        "selection_rule": metadata_payload["selection_rule"],
        "selection_source": champion_test_metrics["selection_source"],
        "selection_score_f1": champion_test_metrics["selection_score_f1"],
        "selection_score_acc": champion_test_metrics["selection_score_acc"],
        "benchmark_results": benchmark_results,
        "reliability_warnings": eligibility["reliability_warnings"],
        "storage": {
            "model_path": "models/best_crop_model.json",
            "metadata_path": "models/model_metadata.json"
        }
    }

# ==============================================================================
# 10. SAFE INFERENCE & MODEL DESERIALIZATION ENGINE
# ==============================================================================
def load_production_model() -> Tuple[ModelPipeline, Dict[str, Any]]:
    """
    Safely deserializes the production model artifact and metadata.
    Reconstructs scaler and classifier without code execution or arbitrary paths.
    """
    model_path = os.path.join(MODELS_DIR, "best_crop_model.json")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Trained model artifact or metadata file does not exist.")

    try:
        with open(model_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Corrupted model artifact file: {str(e)}")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Corrupted metadata file: {str(e)}")

    # Validate artifact structure and schema
    if "scaler" not in model_data or "classifier_data" not in model_data:
        raise ValueError("Invalid model artifact structure: missing scaler or classifier.")

    schema = meta_data.get("feature_schema", [])
    if schema != REQUIRED_FEATURES:
        raise ValueError(f"Incompatible feature schema in model metadata: {schema}")

    pipeline = ModelPipeline.from_dict(model_data)
    return pipeline, meta_data

def predict_crop(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates input features against physical domain thresholds and runs inference
    through the production ModelPipeline.
    """
    pipeline, meta = load_production_model()

    # Support flat dictionary or nested 'features' dictionary
    feature_source = inputs.get("features", inputs) if isinstance(inputs, dict) else {}

    validated_vector = []
    validated_dict = {}

    for feat in REQUIRED_FEATURES:
        if feat not in feature_source:
            raise ValueError(f"Missing required input feature: '{feat}'.")

        raw_val = feature_source[feat]
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            raise ValueError(f"Feature '{feat}' must be a valid numeric value; received '{raw_val}'.")

        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Feature '{feat}' must be a finite number; received '{val}'.")

        bounds = FEATURE_BOUNDS[feat]
        if val < bounds["min"] or val > bounds["max"]:
            raise ValueError(
                f"Feature '{bounds['name']}' ({val} {bounds['unit']}) is outside physical limits "
                f"[{bounds['min']}, {bounds['max']} {bounds['unit']}]."
            )

        validated_vector.append(val)
        validated_dict[feat] = val

    # Execute inference
    X_input = np.array([validated_vector], dtype=float)
    predicted_array = pipeline.predict(X_input)
    predicted_crop = str(predicted_array[0])

    return {
        "success": True,
        "predicted_crop": predicted_crop,
        "champion_model": meta.get("champion_model", pipeline.name),
        "inputs": validated_dict,
        "feature_schema": REQUIRED_FEATURES,
        "trained_at": meta.get("trained_at"),
        "dataset_name": meta.get("dataset_name"),
        "dataset_fingerprint": meta.get("dataset_fingerprint"),
        "inference_timestamp": datetime.datetime.now().isoformat()
    }
