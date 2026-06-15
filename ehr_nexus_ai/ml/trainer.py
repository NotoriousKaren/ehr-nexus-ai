"""
EHR Nexus - Model Trainer
===========================
Complete training pipeline for all ML models.
This is the main entry point that users interact with to train models.

Usage:
    # Quick training on a CSV dataset
    from ehr_nexus_ai.ml import ModelTrainer
    trainer = ModelTrainer()
    results = trainer.train("patient_data.csv", task="auto")

    # Preview dataset first
    trainer.preview("patient_data.csv")

    # Train specific model
    trainer.train("symptom_pairs.csv", task="symptom_correlation")

    # Compare ML vs rule-based
    trainer.compare_with_rules("test_data.csv")
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from .preprocessing import PreprocessingPipeline
from .correlator_model import (
    SymptomCorrelatorModel,
    DiagnosisCorrelatorModel,
    LabAnomalyDetector,
)
from .evaluator import ModelEvaluator
from .config import MLConfig


class ModelTrainer:
    """
    Complete training pipeline for EHR Nexus ML models.

    Handles:
    - Dataset preview and validation
    - Full preprocessing (cleaning, feature engineering, vectorization)
    - Model training with cross-validation
    - Comprehensive evaluation
    - Model persistence (save/load)
    - Comparison against rule-based baselines
    """

    def __init__(self, config: Optional[MLConfig] = None):
        """
        Args:
            config: MLConfig instance. If None, uses default config.
                    Use MLConfig.small_dataset() for <1000 records.
                    Use MLConfig.large_dataset() for >10000 records.
        """
        self.config = config or MLConfig()
        self.preprocessor = PreprocessingPipeline(self.config)
        self.evaluator = ModelEvaluator()
        self.trained_models: Dict[str, Any] = {}
        self.training_history: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------
    # Main Public API
    # ----------------------------------------------------------------

    def preview(self, dataset_path: str, n: int = 5) -> None:
        """
        Preview a dataset to understand its structure before training.

        Args:
            dataset_path: Path to CSV, JSON, or JSONL file
            n: Number of rows to display
        """
        self.preprocessor.preview_dataset(dataset_path, n=n)

    def train(
        self,
        dataset_path: str,
        task: str = "auto",
        save_model: bool = True,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete training pipeline: load, preprocess, train, evaluate, and save.

        Args:
            dataset_path: Path to dataset file (CSV, JSON, JSONL)
            task: Task type:
                - "auto": auto-detect from dataset columns
                - "symptom_correlation": train symptom matching model
                - "diagnosis_prediction": train diagnosis classifier
                - "lab_anomaly": train lab anomaly detector
            save_model: Whether to save the trained model to disk
            model_name: Optional name for the saved model

        Returns:
            Dict with training results including metrics and model path

        Example:
            >>> trainer = ModelTrainer()
            >>> results = trainer.train("symptom_pairs.csv", task="symptom_correlation")
            >>> print(f"Accuracy: {results['test_metrics']['accuracy']}")
        """
        print(f"\n{'='*60}")
        print(f"EHR Nexus ML - Training Pipeline")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_path}")
        print(f"Task:    {task}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Step 1: Preprocess data
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = (
                self.preprocessor.load_and_split(
                    dataset_path, task=task,
                    normalize=True, vectorize=True
                )
            )
        except Exception as e:
            print(f"ERROR during preprocessing: {e}")
            print("\nTip: Use trainer.preview('dataset.csv') first to check your dataset structure.")
            raise

        # Convert to numpy arrays for model training
        X_train_np = X_train.values.astype(np.float64)
        X_val_np = X_val.values.astype(np.float64) if len(X_val) > 0 else None
        X_test_np = X_test.values.astype(np.float64)
        y_train_np = y_train.values
        y_val_np = y_val.values if y_val is not None and len(y_val) > 0 else None
        y_test_np = y_test.values

        # Step 2: Determine task if auto
        if task == "auto":
            task = self._infer_task(X_train, y_train)

        # Step 3: Train model
        print(f"\nTraining {task} model...")
        model = self._create_model(task)
        model.feature_columns = self.preprocessor.feature_columns
        model.label_encoder = self.preprocessor.label_encoders.get("label")

        metrics = model.train(X_train_np, y_train_np, X_val_np, y_val_np)

        # Step 4: Evaluate on test set
        print(f"\nEvaluating on test set ({len(X_test_np)} samples)...")
        test_metrics = model.evaluate(X_test_np, y_test_np)
        print(f"Test metrics: {test_metrics}")

        if test_metrics.get("roc_auc"):
            print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")

        # Step 5: Cross-validation
        print(f"\nRunning {self.config.cv_folds}-fold cross-validation...")
        X_full = np.vstack([X_train_np, X_val_np]) if X_val_np is not None else X_train_np
        y_full = np.concatenate([y_train_np, y_val_np]) if y_val_np is not None else y_train_np

        cv_folds = min(self.config.cv_folds, len(np.unique(y_full)))
        cv_results = self.evaluator.cross_validate(
            model.model, X_full, y_full, cv=cv_folds
        )

        # Store trained model
        model_name = model_name or f"{task}_model_{int(time.time())}"
        self.trained_models[model_name] = model

        # Step 6: Save model
        saved_path = None
        if save_model:
            save_dir = Path(self.config.model_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            saved_path = str(save_dir / f"{model_name}.pkl")
            model.save(saved_path)

        # Step 7: Feature importance
        feature_importance = model.get_feature_importance(top_n=15)

        elapsed = time.time() - start_time

        # Compile results
        results = {
            "model_name": model_name,
            "task": task,
            "dataset": dataset_path,
            "dataset_size": len(X_train_np) + (len(X_val_np) or 0) + len(X_test_np),
            "n_features": X_train_np.shape[1],
            "training_time_seconds": round(elapsed, 2),
            "train_metrics": metrics,
            "test_metrics": test_metrics,
            "cross_validation": {
                "n_folds": cv_folds,
                "accuracy_mean": cv_results["results"].get("accuracy", {}).get("mean", 0),
                "accuracy_std": cv_results["results"].get("accuracy", {}).get("std", 0),
                "f1_mean": cv_results["results"].get("f1_weighted", {}).get("mean", 0),
            },
            "feature_importance_top": feature_importance[:10],
            "grade": test_metrics.get("grade", "N/A"),
            "saved_path": saved_path,
        }

        self.training_history.append(results)

        # Print summary
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Model:      {model_name}")
        print(f"  Task:       {task}")
        print(f"  Dataset:    {results['dataset_size']} rows, {results['n_features']} features")
        print(f"  Time:       {results['training_time_seconds']}s")
        print(f"  Test Acc:   {test_metrics.get('accuracy', 'N/A'):.2%}")
        print(f"  Test F1:    {test_metrics.get('f1_score', 'N/A'):.4f}")
        print(f"  Grade:      {test_metrics.get('grade', 'N/A')}")
        print(f"  CV Acc:     {results['cross_validation']['accuracy_mean']:.4f} +/- {results['cross_validation']['accuracy_std']:.4f}")
        if saved_path:
            print(f"  Saved to:   {saved_path}")

        if feature_importance:
            print(f"\n  Top Features:")
            for i, feat in enumerate(feature_importance[:5], 1):
                print(f"    {i}. {feat['feature']}: {feat['importance']:.4f}")

        print(f"{'='*60}\n")

        return results

    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to .pkl model file

        Returns:
            Loaded model instance
        """
        # Determine model type from file
        data = None
        import pickle
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        model_type = data.get("model_type", "SymptomCorrelatorModel")

        if "Symptom" in model_type:
            model = SymptomCorrelatorModel(self.config)
        elif "Diagnosis" in model_type:
            model = DiagnosisCorrelatorModel(self.config)
        elif "Lab" in model_type:
            model = LabAnomalyDetector(self.config)
        else:
            model = SymptomCorrelatorModel(self.config)

        model.load(model_path)
        self.trained_models[Path(model_path).stem] = model
        return model

    def evaluate_model(
        self,
        model_name: str,
        test_data_path: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a previously trained model on new test data.

        Args:
            model_name: Name of the trained model
            test_data_path: Path to test dataset

        Returns:
            Evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found. Train or load it first.")

        model = self.trained_models[model_name]

        # Preprocess test data using same pipeline
        X_test, y_test = self.preprocessor.extract_labels(
            self.preprocessor.clean_dataframe(self.preprocessor.load_dataset(test_data_path)),
            task="auto",
        )

        X_test = self.preprocessor.engineer_features(X_test)
        if self.config.use_text_embeddings:
            X_test = self.preprocessor.vectorize_text(X_test)
        X_test = self.preprocessor._encode_categoricals(X_test)
        X_test = self.preprocessor.normalize_features(X_test, fit=False)

        X_test_np = X_test.values.astype(np.float64)
        y_test_np = y_test.values

        metrics = model.evaluate(X_test_np, y_test_np)
        return metrics

    def compare_with_rules(
        self,
        test_data_path: str,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare ML model against rule-based baseline.

        Args:
            test_data_path: Path to test dataset with ground truth labels
            model_name: Name of trained ML model. Uses last trained if None.

        Returns:
            Comparison results
        """
        if not self.trained_models:
            raise ValueError("No trained models available. Train a model first.")

        if model_name is None:
            model_name = list(self.trained_models.keys())[-1]

        model = self.trained_models[model_name]

        # Load and preprocess test data
        df = self.preprocessor.clean_dataframe(
            self.preprocessor.load_dataset(test_data_path)
        )
        X, y = self.preprocessor.extract_labels(df, task="auto")
        X = self.preprocessor.engineer_features(X)
        if self.config.use_text_embeddings:
            X = self.preprocessor.vectorize_text(X)
        X = self.preprocessor._encode_categoricals(X)
        X = self.preprocessor.normalize_features(X, fit=False)

        X_np = X.values.astype(np.float64)
        y_np = y.values

        # Rule-based "predictions" - use majority class as baseline
        from collections import Counter
        most_common = Counter(y_np).most_common(1)[0][0]
        rule_pred = np.full_like(y_np, most_common)

        comparison = self.evaluator.compare_to_baseline(
            model, rule_pred, X_np, y_np
        )

        print(f"\n{'='*60}")
        print(f"ML vs Rule-Based Comparison")
        print(f"{'='*60}")
        print(f"  ML Accuracy:      {comparison['ml_accuracy']:.4f}")
        print(f"  Rule Accuracy:    {comparison['rule_based_accuracy']:.4f}")
        print(f"  Improvement:      {comparison['absolute_improvement']:+.4f}")
        print(f"  Relative Imprv:   {comparison['relative_improvement_pct']:+.2f}%")
        print(f"  ML Better:        {comparison['ml_better']}")
        print(f"{'='*60}\n")

        return comparison

    def training_summary(self) -> str:
        """
        Generate a summary of all training runs.

        Returns:
            Formatted string
        """
        if not self.training_history:
            return "No training runs completed yet."

        lines = ["=" * 60]
        lines.append("EHR Nexus ML - Training History Summary")
        lines.append("=" * 60)

        for i, run in enumerate(self.training_history, 1):
            lines.append(f"\n--- Run #{i}: {run['model_name']} ---")
            lines.append(f"  Task:      {run['task']}")
            lines.append(f"  Dataset:   {run['dataset_size']} rows x {run['n_features']} features")
            lines.append(f"  Test Acc:  {run['test_metrics'].get('accuracy', 'N/A'):.4f}")
            lines.append(f"  Test F1:   {run['test_metrics'].get('f1_score', 'N/A'):.4f}")
            lines.append(f"  Grade:     {run.get('grade', 'N/A')}")
            lines.append(f"  Saved:     {run.get('saved_path', 'Not saved')}")

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Internal Methods
    # ----------------------------------------------------------------

    def _create_model(self, task: str) -> Any:
        """Create the appropriate model for the task."""
        task_lower = task.lower()

        if "symptom" in task_lower or "correlation" in task_lower:
            model = SymptomCorrelatorModel(self.config)
        elif "diagnosis" in task_lower:
            model = DiagnosisCorrelatorModel(self.config)
        elif "lab" in task_lower or "anomaly" in task_lower:
            model = LabAnomalyDetector(self.config)
        else:
            raise ValueError(f"Unknown task type: '{task}'. "
                           "Use 'symptom_correlation', 'diagnosis_prediction', "
                           "or 'lab_anomaly'")

        return model

    def _infer_task(self, X: pd.DataFrame, y: pd.Series) -> str:
        """
        Infer the task type from the dataset features.

        Heuristics:
        - If columns include 'symptom' -> symptom_correlation
        - If columns include 'icd10' or 'diagnosis' -> diagnosis_prediction
        - If columns include 'lab_value' or 'parameter' -> lab_anomaly
        - Default: symptom_correlation
        """
        col_names = " ".join(X.columns.str.lower())

        if "symptom" in col_names:
            return "symptom_correlation"
        elif "diagnosis" in col_names or "icd" in col_names:
            return "diagnosis_prediction"
        elif "lab" in col_names or "parameter" in col_names:
            return "lab_anomaly"
        else:
            print("Could not auto-detect task from column names. "
                  "Defaulting to 'symptom_correlation'.")
            return "symptom_correlation"
