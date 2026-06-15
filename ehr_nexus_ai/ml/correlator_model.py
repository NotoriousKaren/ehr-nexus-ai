"""
EHR Nexus - ML Correlation Models
===================================
ML models that replace or augment the rule-based correlation engines.
Each model is designed to be trained on labeled data and can be
used as a drop-in replacement for the corresponding rule-based engine.

Models:
- SymptomCorrelatorModel: Binary classifier for symptom matching
- DiagnosisCorrelatorModel: Multi-class classifier for diagnosis prediction
- LabAnomalyDetector: Unsupervised anomaly detection for lab values
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import pickle
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV


class BaseCorrelatorModel:
    """
    Base class for all ML correlation models.
    Provides common save/load, predict, and evaluation methods.
    """

    def __init__(self, config: Any = None):
        self.model = None
        self.is_trained = False
        self.feature_columns: List[str] = []
        self.label_encoder: Any = None
        self.training_metrics: Dict[str, float] = {}
        self._setup_config(config)

    def _setup_config(self, config: Any) -> None:
        """Set configuration from MLConfig or defaults."""
        if config is None:
            from .config import MLConfig
            config = MLConfig()
        self.config = config

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels for input features."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities for input features."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("This model does not support predict_proba")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Returns:
            Dict with accuracy, precision, recall, f1, and (if binary) roc_auc
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        y_pred = self.predict(X_test)
        n_classes = len(np.unique(y_test))

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        }

        # ROC-AUC only for binary classification
        if n_classes == 2 and hasattr(self.model, "predict_proba"):
            try:
                y_prob = self.model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)
            except Exception:
                pass

        metrics["grade"] = self._grade_metrics(metrics)
        self.training_metrics = metrics
        return metrics

    def save(self, filepath: str) -> str:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model .pkl file

        Returns:
            The path the model was saved to
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "label_encoder": self.label_encoder,
            "training_metrics": self.training_metrics,
            "model_type": self.__class__.__name__,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {path}")
        return str(path)

    def load(self, filepath: str) -> "BaseCorrelatorModel":
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model .pkl file

        Returns:
            self with loaded model
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_columns = model_data.get("feature_columns", [])
        self.label_encoder = model_data.get("label_encoder")
        self.training_metrics = model_data.get("training_metrics", {})
        self.is_trained = True

        print(f"Model loaded from: {path}")
        print(f"  Type: {model_data.get('model_type', 'Unknown')}")
        if self.training_metrics:
            print(f"  Training metrics: {self.training_metrics}")

        return self

    def get_feature_importance(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get feature importance from the trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            List of {"feature": name, "importance": score} dicts
        """
        if not self.is_trained or self.model is None:
            return []

        if not self.feature_columns:
            return []

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_).flatten()
            if self.model.coef_.ndim > 1:
                importances = np.mean(importances, axis=0)
        else:
            return []

        if len(importances) != len(self.feature_columns):
            return []

        indices = np.argsort(importances)[::-1][:top_n]
        return [
            {"feature": self.feature_columns[i], "importance": round(float(importances[i]), 4)}
            for i in indices
        ]

    def _grade_metrics(self, metrics: Dict[str, float]) -> str:
        """Assign a simple letter grade based on weighted F1 score."""
        f1 = metrics.get("f1_score", 0)
        if f1 >= 0.90:
            return "A+"
        if f1 >= 0.85:
            return "A"
        if f1 >= 0.80:
            return "B+"
        if f1 >= 0.75:
            return "B"
        if f1 >= 0.70:
            return "C+"
        if f1 >= 0.60:
            return "C"
        if f1 >= 0.50:
            return "D"
        return "F"


class SymptomCorrelatorModel(BaseCorrelatorModel):
    """
    ML model for symptom correlation.

    Trained on labeled pairs of symptoms to learn which symptoms
    are semantically equivalent or clinically correlated.

    Usage:
        model = SymptomCorrelatorModel()
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        model.save("saved_models/symptom_model.pkl")
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.model_type = self.config.symptom_model_type

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train the symptom correlation model.

        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_val: Optional validation feature matrix
            y_val: Optional validation labels

        Returns:
            Training metrics dict
        """
        n_classes = len(np.unique(y_train))
        class_weight = "balanced" if self.config.use_class_weights else None

        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.symptom_n_estimators,
                max_depth=self.config.symptom_max_depth,
                learning_rate=self.config.symptom_learning_rate,
                random_state=self.config.random_state,
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config.symptom_n_estimators,
                max_depth=self.config.symptom_max_depth,
                class_weight=class_weight,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight=class_weight,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Train
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate on train set
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print(f"Validation metrics: {val_metrics}")
            return val_metrics

        self.training_metrics = {"train_accuracy": round(train_acc, 4)}
        return self.training_metrics


class DiagnosisCorrelatorModel(BaseCorrelatorModel):
    """
    ML model for diagnosis correlation and prediction.

    Trains on patient features (symptoms, demographics, labs) to
    predict the most likely diagnosis (ICD-10 code or category).

    Usage:
        model = DiagnosisCorrelatorModel()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.model_type = self.config.diagnosis_model_type

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train the diagnosis correlation model.

        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_val: Optional validation feature matrix
            y_val: Optional validation labels

        Returns:
            Training metrics dict
        """
        n_classes = len(np.unique(y_train))
        class_weight = "balanced" if self.config.use_class_weights else None

        if n_classes > 50:
            print(f"Warning: {n_classes} classes detected. Consider grouping diagnoses.")

        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.diagnosis_n_estimators,
                max_depth=self.config.diagnosis_max_depth,
                learning_rate=self.config.diagnosis_learning_rate,
                random_state=self.config.random_state,
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.config.diagnosis_n_estimators,
                max_depth=self.config.diagnosis_max_depth,
                class_weight=class_weight,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training accuracy: {train_acc:.4f} ({n_classes} classes)")

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            print(f"Validation metrics: {val_metrics}")
            return val_metrics

        self.training_metrics = {"train_accuracy": round(train_acc, 4)}
        return self.training_metrics


class LabAnomalyDetector(BaseCorrelatorModel):
    """
    Unsupervised anomaly detection model for laboratory values.

    Uses Isolation Forest to detect abnormal lab results based on
    learned patterns rather than fixed reference ranges.

    Usage:
        detector = LabAnomalyDetector()
        detector.fit(X_train)  # Unsupervised - no labels needed
        anomalies = detector.predict(X_test)  # -1 = anomaly, 1 = normal
    """

    def __init__(self, config: Any = None):
        super().__init__(config)

    def train(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train the anomaly detector (unsupervised - no labels needed).

        Args:
            X_train: Training feature matrix
            y_train: Ignored (unsupervised)
            X_val: Optional validation data
            y_val: Optional validation labels

        Returns:
            Training metrics dict
        """
        self.model = IsolationForest(
            n_estimators=self.config.lab_n_estimators,
            max_samples=self.config.lab_max_samples,
            contamination=self.config.lab_contamination,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        self.model.fit(X_train)
        self.is_trained = True

        # Count detected anomalies
        predictions = self.model.predict(X_train)
        n_anomalies = np.sum(predictions == -1)
        n_total = len(predictions)
        anomaly_rate = n_anomalies / n_total * 100

        print(f"Training complete. Detected {n_anomalies}/{n_total} ({anomaly_rate:.1f}%) anomalies in training data")

        # If validation labels exist, evaluate
        if X_val is not None and y_val is not None:
            y_pred = self.model.predict(X_val)
            # Convert: Isolation Forest returns -1=anomaly, 1=normal
            # Map to binary: anomaly=1, normal=0 for evaluation
            y_pred_binary = np.where(y_pred == -1, 1, 0)
            metrics = {
                "precision": round(precision_score(y_val, y_pred_binary, zero_division=0), 4),
                "recall": round(recall_score(y_val, y_pred_binary, zero_division=0), 4),
                "f1_score": round(f1_score(y_val, y_pred_binary, zero_division=0), 4),
            }
            self.training_metrics = metrics
            return metrics

        self.training_metrics = {"anomaly_rate": round(anomaly_rate, 2)}
        return self.training_metrics

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict anomalies. Returns -1 for anomaly, 1 for normal.

        This overrides the parent predict to return interpretable values.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)

    def anomaly_score(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get anomaly scores for each sample.
        Lower scores = more anomalous.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.score_samples(X)

    def get_normal_range(
        self, X: np.ndarray, confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Estimate the 'normal range' for parameters based on training data.

        Args:
            X: Feature matrix
            confidence: Confidence interval (0.0-1.0)

        Returns:
            Dict with 'lower', 'upper', and 'contamination_rate'
        """
        scores = self.anomaly_score(X)
        threshold = np.percentile(scores, (1 - confidence) * 100)

        return {
            "score_threshold": round(float(threshold), 4),
            "contamination_rate": round(1 - confidence, 4),
            "n_samples": len(X),
        }
