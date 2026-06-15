"""
EHR Nexus - Machine Learning Module
=====================================
Provides ML-enhanced correlation models that can be trained on labeled
patient data to produce more accurate comparisons than the rule-based engines.

This module is OPTIONAL - the system works without it using rule-based methods.
When trained with sufficient data, it significantly improves correlation accuracy.

Components:
- PreprocessingPipeline: Data cleaning, normalization, feature extraction
- SymptomCorrelatorModel: ML model for symptom matching
- DiagnosisCorrelatorModel: ML model for diagnosis correlation
- LabAnomalyDetector: ML model for lab value anomaly detection
- ModelTrainer: Complete training pipeline with cross-validation
- ModelEvaluator: Accuracy, precision, recall, F1, ROC-AUC metrics

Usage:
    from ehr_nexus_ai.ml import ModelTrainer
    trainer = ModelTrainer()
    trainer.train(csv_path="patient_data.csv")
    trainer.evaluate()
"""

from .trainer import ModelTrainer
from .preprocessing import PreprocessingPipeline
from .correlator_model import SymptomCorrelatorModel, DiagnosisCorrelatorModel, LabAnomalyDetector
from .evaluator import ModelEvaluator
from .config import MLConfig

__all__ = [
    "ModelTrainer",
    "PreprocessingPipeline",
    "SymptomCorrelatorModel",
    "DiagnosisCorrelatorModel",
    "LabAnomalyDetector",
    "ModelEvaluator",
    "MLConfig",
]
