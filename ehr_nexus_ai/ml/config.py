"""
EHR Nexus - ML Configuration
==============================
Central configuration for all ML models and training parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class MLConfig:
    """Configuration for the ML training pipeline."""

    # --- Data Processing ---
    test_split_ratio: float = 0.15
    val_split_ratio: float = 0.15
    random_state: int = 42

    # --- Symptom Correlator Model ---
    symptom_model_type: str = "gradient_boosting"  # or "random_forest", "logistic_regression"
    symptom_embedding_dim: int = 384  # all-MiniLM-L6-v2 output
    symptom_n_estimators: int = 200
    symptom_max_depth: int = 10
    symptom_learning_rate: float = 0.1

    # --- Diagnosis Correlator Model ---
    diagnosis_model_type: str = "gradient_boosting"
    diagnosis_n_estimators: int = 300
    diagnosis_max_depth: int = 8
    diagnosis_learning_rate: float = 0.05

    # --- Lab Anomaly Detector ---
    lab_model_type: str = "isolation_forest"
    lab_contamination: float = 0.05  # expected proportion of anomalies
    lab_n_estimators: int = 100
    lab_max_samples: int = 256

    # --- Training ---
    cv_folds: int = 5  # cross-validation folds
    early_stopping_rounds: int = 20
    use_class_weights: bool = True  # handle imbalanced data

    # --- Feature Engineering ---
    use_text_embeddings: bool = True
    use_icd10_features: bool = True
    use_demographics: bool = True
    use_lab_statistics: bool = True

    # --- Paths ---
    model_save_dir: str = "ehr_nexus_ai/ml/saved_models"
    dataset_schema_path: str = "ehr_nexus_ai/ml/dataset_schema.md"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def small_dataset(cls) -> "MLConfig":
        """Configuration optimized for small datasets (< 1000 records)."""
        return cls(
            symptom_n_estimators=100,
            symptom_max_depth=5,
            diagnosis_n_estimators=100,
            diagnosis_max_depth=5,
            cv_folds=3,
            use_text_embeddings=False,
        )

    @classmethod
    def large_dataset(cls) -> "MLConfig":
        """Configuration optimized for large datasets (> 10000 records)."""
        return cls(
            symptom_n_estimators=500,
            symptom_max_depth=15,
            diagnosis_n_estimators=500,
            diagnosis_max_depth=12,
            cv_folds=10,
            use_text_embeddings=True,
        )