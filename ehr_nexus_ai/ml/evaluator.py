"""
EHR Nexus - Model Evaluator
=============================
Comprehensive evaluation metrics for trained ML models.
Provides accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
cross-validation scores, and comparison against rule-based baselines.
"""

from typing import Dict, List, Optional, Any, Tuple
import json

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score,
)


class ModelEvaluator:
    """
    Evaluates ML models with comprehensive metrics and cross-validation.

    Usage:
        evaluator = ModelEvaluator()
        results = evaluator.full_evaluation(model, X_test, y_test)
        cv_scores = evaluator.cross_validate(model, X_train, y_train, cv=5)
    """

    def __init__(self):
        self.evaluation_history: List[Dict[str, Any]] = []

    def full_evaluation(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on a trained model.

        Args:
            model: Trained model with predict() and optionally predict_proba()
            X_test: Test feature matrix
            y_test: Test labels
            model_name: Identifier for this evaluation

        Returns:
            Dict with all computed metrics
        """
        y_pred = model.predict(X_test)
        n_classes = len(np.unique(y_test))

        metrics: Dict[str, Any] = {
            "model_name": model_name,
            "n_test_samples": len(y_test),
            "n_classes": n_classes,
            "class_distribution": self._class_distribution(y_test),
        }

        # Core classification metrics
        metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
        metrics["precision"] = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        metrics["f1_score"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        metrics["matthews_corrcoef"] = round(float(matthews_corrcoef(y_test, y_pred)), 4)
        metrics["cohen_kappa"] = round(float(cohen_kappa_score(y_test, y_pred)), 4)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics["per_class"] = {
            str(k): {kk: round(float(vv), 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        }

        # ROC-AUC (binary only)
        if n_classes == 2 and hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
            except Exception:
                metrics["roc_auc"] = None

        # Determine grade
        metrics["grade"] = self._grade_model(metrics)

        self.evaluation_history.append(metrics)
        return metrics

    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run k-fold cross-validation.

        Args:
            model: Untrained model instance
            X: Full feature matrix
            y: Full label vector
            cv: Number of folds
            scoring: List of metrics to compute. Defaults to accuracy, f1, roc_auc

        Returns:
            Dict with mean and std for each metric
        """
        if scoring is None:
            scoring = ["accuracy", "f1_weighted", "roc_auc_ovr"]

        results = {}
        for metric in scoring:
            try:
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring=metric, n_jobs=-1,
                    error_score="raise"
                )
                results[metric] = {
                    "mean": round(float(np.mean(scores)), 4),
                    "std": round(float(np.std(scores)), 4),
                    "scores": [round(float(s), 4) for s in scores],
                }
            except Exception as e:
                results[metric] = {
                    "error": str(e),
                    "mean": 0.0,
                    "std": 0.0,
                    "scores": [],
                }

        cv_result = {
            "n_folds": cv,
            "n_samples": len(y),
            "results": results,
        }

        return cv_result

    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test set.

        Args:
            models: Dict of {model_name: trained_model}
            X_test: Test feature matrix
            y_test: Test labels

        Returns:
            Dict with comparison table sorted by F1 score
        """
        comparisons = []
        for name, model in models.items():
            metrics = self.full_evaluation(model, X_test, y_test, model_name=name)
            comparisons.append({
                "name": name,
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "roc_auc": metrics.get("roc_auc", "N/A"),
                "grade": metrics["grade"],
            })

        # Sort by F1 descending
        comparisons.sort(key=lambda x: x["f1_score"], reverse=True)

        return {
            "comparisons": comparisons,
            "best_model": comparisons[0]["name"] if comparisons else None,
            "worst_model": comparisons[-1]["name"] if comparisons else None,
        }

    def compare_to_baseline(
        self,
        ml_model: Any,
        rule_based_predictions: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compare ML model against rule-based baseline.

        Args:
            ml_model: Trained ML model
            rule_based_predictions: Predictions from rule-based engine
            X_test: Test features
            y_test: Test labels

        Returns:
            Dict with comparison metrics
        """
        ml_pred = ml_model.predict(X_test)

        ml_accuracy = accuracy_score(y_test, ml_pred)
        rule_accuracy = accuracy_score(y_test, rule_based_predictions)

        improvement = ml_accuracy - rule_accuracy
        improvement_pct = (improvement / max(rule_accuracy, 0.001)) * 100

        return {
            "ml_accuracy": round(float(ml_accuracy), 4),
            "rule_based_accuracy": round(float(rule_accuracy), 4),
            "absolute_improvement": round(float(improvement), 4),
            "relative_improvement_pct": round(float(improvement_pct), 2),
            "ml_better": ml_accuracy > rule_accuracy,
        }

    def _grade_model(self, metrics: Dict[str, Any]) -> str:
        """Assign a letter grade based on F1 score."""
        f1 = metrics.get("f1_score", 0)
        if f1 >= 0.90:
            return "A+"
        elif f1 >= 0.85:
            return "A"
        elif f1 >= 0.80:
            return "B+"
        elif f1 >= 0.75:
            return "B"
        elif f1 >= 0.70:
            return "C+"
        elif f1 >= 0.60:
            return "C"
        elif f1 >= 0.50:
            return "D"
        else:
            return "F"

    def _class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get class distribution from labels."""
        unique, counts = np.unique(y, return_counts=True)
        return {str(k): int(v) for k, v in zip(unique, counts)}

    def summary_report(self) -> str:
        """
        Generate a text summary of all evaluations performed.

        Returns:
            Formatted string with evaluation history
        """
        if not self.evaluation_history:
            return "No evaluations have been performed yet."

        lines = ["=" * 60]
        lines.append("EHR Nexus ML - Evaluation Summary")
        lines.append("=" * 60)

        for i, eval_result in enumerate(self.evaluation_history, 1):
            lines.append(f"\n--- Evaluation #{i}: {eval_result['model_name']} ---")
            lines.append(f"  Test samples: {eval_result['n_test_samples']}")
            lines.append(f"  Classes: {eval_result['n_classes']}")
            lines.append(f"  Accuracy:  {eval_result['accuracy']:.4f}")
            lines.append(f"  Precision: {eval_result['precision']:.4f}")
            lines.append(f"  Recall:    {eval_result['recall']:.4f}")
            lines.append(f"  F1 Score:  {eval_result['f1_score']:.4f}")
            if eval_result.get("roc_auc"):
                lines.append(f"  ROC-AUC:   {eval_result['roc_auc']:.4f}")
            lines.append(f"  Grade:     {eval_result['grade']}")
            lines.append(f"  MCC:       {eval_result.get('matthews_corrcoef', 'N/A')}")
            lines.append(f"  Kappa:     {eval_result.get('cohen_kappa', 'N/A')}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)