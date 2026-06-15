"""
EHR Nexus - ML Preprocessing Pipeline
=======================================
Handles data loading, cleaning, normalization, and feature engineering
for training ML models on patient records.

Supports:
- CSV, JSON, and JSONL input formats
- Automated missing value handling
- Text vectorization (via sentence-transformers)
- ICD-10 code encoding
- Lab value normalization
- Train/validation/test splitting
"""

from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class PreprocessingPipeline:
    """
    Complete data preprocessing pipeline for EHR Nexus ML training.

    Usage:
        pipe = PreprocessingPipeline()
        X_train, X_val, X_test, y_train, y_val, y_test = pipe.load_and_split("patients.csv")
    """

    # ICD-10 code validation pattern
    ICD10_PATTERN = re.compile(r"^[A-TV-Z][0-9]{2}(\.?[0-9A-Z]{1,4})?$", re.IGNORECASE)

    # Common lab parameter normalizations
    LAB_UNIT_MAP = {
        "mg/dL": "mg_dl", "mg/dl": "mg_dl", "mg%": "mg_dl",
        "g/dL": "g_dl", "g/dl": "g_dl", "gm/dL": "g_dl",
        "mmol/L": "mmol_l",
        "cells/uL": "cells_ul", "cells/mcL": "cells_ul",
        "U/L": "u_l",
        "ng/mL": "ng_ml",
        "pg/mL": "pg_ml",
    }

    # Severity encoding
    SEVERITY_MAP = {"mild": 1, "moderate": 2, "severe": 3, "critical": 4}

    def __init__(self, config: Any = None):
        """
        Args:
            config: MLConfig instance or None (uses defaults)
        """
        if config is not None:
            from .config import MLConfig
            config = config
        else:
            from .config import MLConfig
            config = MLConfig()

        self.config = config
        self.random_state = config.random_state
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.standard_scalers: Dict[str, StandardScaler] = {}
        self.imputers: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self._embedding_model = None

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def load_dataset(self, path: str) -> pd.DataFrame:
        """
        Load a dataset from CSV, JSON, or JSONL file.

        Args:
            path: Path to the dataset file

        Returns:
            pandas DataFrame
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        ext = path_obj.suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext == ".jsonl":
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            df = pd.DataFrame(records)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .csv, .json, or .jsonl")

        print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        return df

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standard cleaning operations to a DataFrame.

        Operations:
        - Drop fully empty rows/columns
        - Fill missing text with empty strings
        - Remove duplicate rows
        - Standardize text (lowercase, strip)
        - Validate ICD-10 codes where present
        - Clip extreme numeric outliers

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Drop rows/columns that are completely empty
        df = df.dropna(how="all", axis=0)
        df = df.dropna(how="all", axis=1)

        # Drop duplicate rows
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            print(f"Removed {before - after} duplicate row(s)")

        # Standardize text columns
        for col in df.select_dtypes(include=["object"]).columns:
            # Lowercase and strip whitespace
            df[col] = df[col].astype(str).str.lower().str.strip()
            # Replace multiple spaces
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)
            # Replace NaN-like strings with actual NaN
            df[col] = df[col].replace(["nan", "none", "null", "na", ""], np.nan)

        # Validate ICD-10 codes if column exists
        code_cols = [c for c in df.columns if "code" in c.lower() or "icd" in c.lower()]
        for col in code_cols:
            valid_mask = df[col].apply(
                lambda x: bool(self.ICD10_PATTERN.match(str(x))) if pd.notna(x) else False
            )
            invalid_count = (~valid_mask & df[col].notna()).sum()
            if invalid_count > 0:
                print(f"Warning: {invalid_count} invalid ICD-10 codes in '{col}' column")
                df.loc[~valid_mask, col] = np.nan

        # Clip extreme numeric outliers (99.9th percentile)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ("patient_id", "consultation_id"):
                continue
            upper = df[col].quantile(0.999)
            lower = df[col].quantile(0.001)
            # Only clip if the data has meaningful spread
            if upper != lower:
                df[col] = df[col].clip(lower, upper)

        return df

    def extract_labels(self, df: pd.DataFrame, task: str = "auto") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract feature matrix X and label vector y from DataFrame.

        Supports multiple task types:
        - "symptom_correlation": labels in 'label' or 'is_correlated' column
        - "diagnosis_prediction": labels in 'confirmed_diagnosis' column
        - "lab_anomaly": labels in 'is_abnormal' column
        - "auto": tries to infer label column from common names

        Args:
            df: Cleaned DataFrame
            task: Task type string

        Returns:
            (X_df, y_series) tuple
        """
        # Common label column names
        label_candidates = {
            "symptom_correlation": ["label", "is_correlated", "correlated", "match"],
            "diagnosis_prediction": ["confirmed_diagnosis", "diagnosis_label", "icd10_target"],
            "lab_anomaly": ["is_abnormal", "abnormal", "anomaly", "label"],
        }

        if task == "auto":
            # Try all candidates
            for task_type, candidates in label_candidates.items():
                for col in candidates:
                    if col in df.columns:
                        task = task_type
                        label_col = col
                        break
                if task != "auto":
                    break
            if task == "auto":
                raise ValueError(
                    "Could not auto-detect label column. "
                    "Expected one of: label, is_correlated, confirmed_diagnosis, is_abnormal"
                )
        else:
            candidates = label_candidates.get(task, ["label"])
            label_col = None
            for col in candidates:
                if col in df.columns:
                    label_col = col
                    break
            if label_col is None:
                raise ValueError(f"Could not find label column for task '{task}'")

        y = df[label_col].copy()
        X = df.drop(columns=[label_col])

        # Encode labels if they are strings
        if y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=label_col)
            self.label_encoders["label"] = le
            print(f"Encoded labels: {len(le.classes_)} unique classes -> {dict(zip(le.classes_, le.transform(le.classes_)))}")

        print(f"Features: {X.shape[1]} columns | Labels: {y.name} ({y.nunique()} unique values)")
        return X, y

    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from raw data.

        This includes:
        - ICD-10 category extraction (first 3 chars)
        - Severity numeric encoding
        - Age binning
        - Lab value statistical features

        Args:
            X: Feature DataFrame

        Returns:
            Enhanced feature DataFrame
        """
        df = X.copy()

        # ---- ICD-10 Category Features ----
        code_cols = [c for c in df.columns if "code" in c.lower() or "icd" in c.lower()]
        for col in code_cols:
            if col in df.columns:
                cat_col = f"{col}_category"
                df[cat_col] = df[col].apply(self._icd10_category)
                # One-hot encode categories if there are enough unique values
                if df[cat_col].nunique() > 1 and df[cat_col].nunique() <= 25:
                    dummies = pd.get_dummies(df[cat_col], prefix=col.replace("_code", ""), dummy_na=False)
                    df = pd.concat([df, dummies], axis=1)
                # Keep the raw category too for label encoding
                df[cat_col] = df[cat_col].fillna("unknown")

        # ---- Severity Encoding ----
        severity_cols = [c for c in df.columns if "severity" in c.lower()]
        for col in severity_cols:
            df[f"{col}_numeric"] = df[col].map(self.SEVERITY_MAP).fillna(0)

        # ---- Age Binning ----
        age_cols = [c for c in df.columns if c.lower() in ("age", "patient_age")]
        for col in age_cols:
            if col in df.columns and df[col].dtype in (np.int64, np.float64):
                df[f"{col}_group"] = pd.cut(
                    df[col],
                    bins=[0, 18, 35, 50, 65, 80, 200],
                    labels=["child", "young_adult", "adult", "middle_age", "senior", "elderly"]
                ).astype(str)

        # ---- Symptom Name Features ----
        symptom_cols = [c for c in df.columns if "symptom" in c.lower() or "symptom" in c.lower()]
        for col in symptom_cols:
            if col in df.columns and df[col].dtype == "object":
                df[f"{col}_length"] = df[col].fillna("").str.len()
                df[f"{col}_word_count"] = df[col].fillna("").str.split().str.len()

        return df

    def vectorize_text(
        self,
        df: pd.DataFrame,
        text_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert text columns to numerical embeddings using sentence-transformers.

        Args:
            df: Feature DataFrame
            text_columns: Columns to vectorize. If None, auto-detect text columns.

        Returns:
            DataFrame with text columns replaced by embedding vectors
        """
        if text_columns is None:
            text_columns = [
                c for c in df.columns
                if df[c].dtype == "object"
                and c not in ("patient_id", "consultation_id")
            ]

        if not text_columns:
            return df

        # Lazy-load the embedding model
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Model loaded.")

        df = df.copy()
        for col in text_columns:
            texts = df[col].fillna("").tolist()
            if len(texts) > 1000:
                # Batch process large datasets
                embeddings = self._embedding_model.encode(texts, batch_size=128, show_progress_bar=True)
            else:
                embeddings = self._embedding_model.encode(texts)

            # Add embedding dimensions as columns
            embed_dim = embeddings.shape[1]
            for i in range(min(embed_dim, 64)):  # Keep top 64 dims for performance
                df[f"{col}_emb_{i}"] = embeddings[:, i]

            # Drop original text column (embeddings replace it)
            df = df.drop(columns=[col])

            print(f"Vectorized '{col}': {embed_dim} dims -> reduced to 64 dims")

        return df

    def normalize_features(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.

        Args:
            X: Feature DataFrame
            fit: If True, fit the scaler. If False, use previously fitted scaler.

        Returns:
            Normalized DataFrame
        """
        df = X.copy()

        # Select numerical columns (exclude IDs and categorical)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_patterns = ["patient_id", "consultation_id", "emb_"]
        num_cols = [
            c for c in num_cols
            if not any(p in c for p in exclude_patterns)
        ]

        if not num_cols:
            return df

        if fit:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols].fillna(0))
            self.standard_scalers["features"] = scaler
            print(f"Normalized {len(num_cols)} numerical features")
        else:
            if "features" not in self.standard_scalers:
                raise ValueError("No fitted scaler found. Call with fit=True first.")
            df[num_cols] = self.standard_scalers["features"].transform(
                df[num_cols].fillna(0)
            )

        return df

    def load_and_split(
        self,
        path: str,
        task: str = "auto",
        normalize: bool = True,
        vectorize: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Complete end-to-end pipeline: load, clean, feature engineer, and split.

        Args:
            path: Path to dataset file
            task: Task type ("symptom_correlation", "diagnosis_prediction", "lab_anomaly", "auto")
            normalize: Whether to normalize numerical features
            vectorize: Whether to vectorize text columns

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print(f"\n{'='*60}")
        print(f"Starting preprocessing pipeline for: {path}")
        print(f"{'='*60}")

        # 1. Load
        df = self.load_dataset(path)

        # 2. Clean
        df = self.clean_dataframe(df)
        print(f"After cleaning: {len(df)} rows")

        # 3. Extract labels
        X, y = self.extract_labels(df, task=task)

        # 4. Engineer features
        X = self.engineer_features(X)
        print(f"After feature engineering: {X.shape[1]} columns")

        # 5. Vectorize text (optional)
        if vectorize and self.config.use_text_embeddings:
            X = self.vectorize_text(X)
            print(f"After vectorization: {X.shape[1]} columns")

        # 6. Handle remaining categorical columns
        X = self._encode_categoricals(X)

        # 7. Normalize (optional)
        if normalize:
            X = self.normalize_features(X, fit=True)

        # 8. Train/Val/Test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_split_ratio,
            random_state=self.random_state,
            stratify=y if y.nunique() <= 10 else None,
        )

        val_ratio_adjusted = self.config.val_split_ratio / (1 - self.config.test_split_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=self.random_state,
            stratify=y_temp if y_temp.nunique() <= 10 else None,
        )

        self.feature_columns = X_train.columns.tolist()

        print(f"\nFinal split sizes:")
        print(f"  Train: {len(X_train)} rows, {X_train.shape[1]} features")
        print(f"  Val:   {len(X_val)} rows")
        print(f"  Test:  {len(X_test)} rows")
        print(f"{'='*60}\n")

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ----------------------------------------------------------------
    # Internal Methods
    # ----------------------------------------------------------------

    def _icd10_category(self, code: Any) -> str:
        """Extract the broad ICD-10 category from a code."""
        if pd.isna(code) or not isinstance(code, str):
            return "unknown"

        code = code.strip().upper()
        match = self.ICD10_PATTERN.match(code)
        if not match:
            return "invalid"

        # First letter + first digit gives broad category
        letter = code[0]
        return f"{letter}00-{letter}99"

    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Label encode remaining categorical columns."""
        df = X.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_cols:
            # Skip ID columns
            if col in ("patient_id", "consultation_id"):
                df[col] = df[col].astype("category").cat.codes
                continue

            # Label encode
            le = LabelEncoder()
            df[col] = df[col].fillna("missing")
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        if cat_cols:
            print(f"Encoded {len(cat_cols)} categorical columns")

        return df

    def get_feature_importance(
        self,
        model: Any,
        top_n: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Extract feature importance from a trained model.

        Args:
            model: Trained sklearn model with feature_importances_ or coef_
            top_n: Number of top features to return

        Returns:
            List of {"feature": name, "importance": score} dicts
        """
        if not self.feature_columns:
            return []

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return []

        # Handle dimension mismatch
        if len(importances) != len(self.feature_columns):
            return []

        indices = np.argsort(importances)[::-1][:top_n]
        return [
            {
                "feature": self.feature_columns[i],
                "importance": round(float(importances[i]), 4)
            }
            for i in indices
        ]

    def preview_dataset(self, path: str, n: int = 5) -> None:
        """
        Preview a dataset to understand its structure before full processing.

        Args:
            path: Path to dataset file
            n: Number of rows to display
        """
        df = self.load_dataset(path)
        print(f"\nDataset Preview ({len(df)} total rows):")
        print(f"{'-'*60}")
        print(f"Columns ({len(df.columns)}):")
        for col in df.columns:
            dtype = df[col].dtype
            nunique = df[col].nunique()
            missing = df[col].isna().sum()
            sample = df[col].dropna().iloc[0] if len(df) > 0 else "N/A"
            print(f"  - {col:30s} | {str(dtype):10s} | {nunique:5d} unique | {missing:4d} missing")
            print(f"    Sample: {str(sample)[:60]}")

        print(f"\nFirst {n} rows:")
        print(df.head(n).to_string())
        print(f"{'-'*60}\n")
