"""
EHR Nexus - Main AI Orchestrator
==================================
The central entry point for the Diagnostic Correlation Engine.
Aggregates results from all sub-engines and produces a unified,
explainable correlation report.

Usage:
    orchestrator = DiagnosticCorrelationOrchestrator()
    report = orchestrator.analyze(patient_record)
    print(report.to_json())
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

from ..core.data_models import PatientRecord, Consultation
from ..engines.symptom_correlation_engine import (
    SymptomCorrelationEngine, SymptomCorrelationResult
)
from ..engines.diagnosis_correlation_engine import (
    DiagnosisCorrelationEngine, DiagnosisCorrelationResult
)
from ..engines.lab_trend_analysis_engine import (
    LabTrendAnalysisEngine, LabTrendAnalysisResult
)
from ..engines.duplicate_lab_detector import (
    DuplicateLabDetector, DuplicateLabDetectionResult
)
from ..engines.medication_reminder_engine import (
    MedicationReminderEngine, MedicationReminderResult
)

if TYPE_CHECKING:
    from ..ml import MLConfig, ModelTrainer
from ..core.explainable_engine import (
    ExplainableEngine, ExplainableCorrelationReport
)


class DiagnosticCorrelationOrchestrator:
    """
    Main orchestrator for the EHR Nexus Diagnostic Correlation Engine.

    Coordinates all sub-engines, aggregates their outputs, and produces
    a unified explainable correlation report for clinical decision support.
    """

    def __init__(
        self,
        symptom_engine: Optional[SymptomCorrelationEngine] = None,
        diagnosis_engine: Optional[DiagnosisCorrelationEngine] = None,
        lab_engine: Optional[LabTrendAnalysisEngine] = None,
        duplicate_detector: Optional[DuplicateLabDetector] = None,
        medication_engine: Optional[MedicationReminderEngine] = None,
        use_ml: bool = False,
        ml_model_path: Optional[str] = None,
        ml_config: Optional["MLConfig"] = None,
    ):
        """
        Initialize the orchestrator with custom or default engine instances.

        Args:
            symptom_engine: SymptomCorrelationEngine instance.
            diagnosis_engine: DiagnosisCorrelationEngine instance.
            lab_engine: LabTrendAnalysisEngine instance.
            duplicate_detector: DuplicateLabDetector instance.
            medication_engine: MedicationReminderEngine instance.
            use_ml: If True, use ML-enhanced models instead of rule-based.
            ml_model_path: Path to a pre-trained ML model .pkl file.
            ml_config: MLConfig for training new models.
        """
        self.symptom_engine = symptom_engine or SymptomCorrelationEngine()
        self.diagnosis_engine = diagnosis_engine or DiagnosisCorrelationEngine()
        self.lab_engine = lab_engine or LabTrendAnalysisEngine()
        self.duplicate_detector = duplicate_detector or DuplicateLabDetector()
        self.medication_engine = medication_engine or MedicationReminderEngine()

        # --- ML Integration ---
        self.use_ml = use_ml
        self.ml_trainer: Optional["ModelTrainer"] = None
        self.ml_config = ml_config

        if use_ml:
            from ..ml import MLConfig, ModelTrainer

            self.ml_trainer = ModelTrainer(self.ml_config or MLConfig())
            if ml_model_path:
                self.ml_trainer.load_model(ml_model_path)
                print(f"[EHR Nexus AI] ML model loaded from: {ml_model_path}")
            else:
                print("[EHR Nexus AI] ML mode enabled (no model loaded). Call train_ml() to train.")

    def analyze(
        self,
        record: PatientRecord,
        current_consultation: Optional[Consultation] = None,
        new_lab_uploads: Optional[list] = None
    ) -> ExplainableCorrelationReport:
        """
        Perform full diagnostic correlation analysis on a patient record.

        Args:
            record: Complete patient record with historical consultations.
            current_consultation: Current consultation (optional, auto-detected).
            new_lab_uploads: List of newly uploaded lab results for duplicate
                            detection (optional).

        Returns:
            ExplainableCorrelationReport with all findings and provenance.

        Example:
            >>> orchestrator = DiagnosticCorrelationOrchestrator()
            >>> report = orchestrator.analyze(patient_record)
            >>> print(report.executive_summary)
        """
        if current_consultation is None:
            current_consultation = record.get_latest_consultation()

        # Run all engines in parallel (conceptually)
        symptom_result = self.symptom_engine.correlate(
            record, current_consultation
        )
        diagnosis_result = self.diagnosis_engine.correlate(
            record, current_consultation
        )
        lab_result = self.lab_engine.analyze(
            record, current_consultation
        )
        medication_result = self.medication_engine.analyze(
            record, current_consultation
        )

        # Duplicate detection for newly uploaded labs
        duplicate_result = None
        if new_lab_uploads is not None:
            duplicate_result = self.duplicate_detector.detect(
                record, new_lab_uploads,
                current_consultation.consultation_id
                if current_consultation else None
            )
        elif current_consultation and current_consultation.laboratory_results:
            # Also check current consultation's labs against history
            duplicate_result = self.duplicate_detector.detect(
                record, current_consultation.laboratory_results,
                current_consultation.consultation_id
            )

        # Build explainable report
        explainable = ExplainableEngine(record)
        report = explainable.build_report(
            symptom_result=symptom_result,
            diagnosis_result=diagnosis_result,
            lab_result=lab_result,
            duplicate_result=duplicate_result,
            medication_result=medication_result,
            current_consultation=current_consultation
        )

        return report

    def analyze_symptoms_only(
        self,
        record: PatientRecord,
        current_consultation: Optional[Consultation] = None
    ) -> SymptomCorrelationResult:
        """Run only the symptom correlation engine."""
        return self.symptom_engine.correlate(record, current_consultation)

    def analyze_diagnoses_only(
        self,
        record: PatientRecord,
        current_consultation: Optional[Consultation] = None
    ) -> DiagnosisCorrelationResult:
        """Run only the diagnosis correlation engine."""
        return self.diagnosis_engine.correlate(record, current_consultation)

    def analyze_labs_only(
        self,
        record: PatientRecord,
        current_consultation: Optional[Consultation] = None
    ) -> LabTrendAnalysisResult:
        """Run only the laboratory trend analysis engine."""
        return self.lab_engine.analyze(record, current_consultation)

    def check_duplicates_only(
        self,
        record: PatientRecord,
        new_labs: list
    ) -> DuplicateLabDetectionResult:
        """Run only the duplicate lab detection."""
        return self.duplicate_detector.detect(record, new_labs)

    def check_medications_only(
        self,
        record: PatientRecord,
        current_consultation: Optional[Consultation] = None
    ) -> MedicationReminderResult:
        """Run only the medication reminder engine."""
        return self.medication_engine.analyze(record, current_consultation)

    def train_ml(
        self,
        dataset_path: str,
        task: str = "auto",
        save_model: bool = True,
    ) -> Dict[str, Any]:
        """
        Train an ML model on a labeled dataset.

        After training, the ML model will be used for correlations
        in subsequent analyze() calls.

        Args:
            dataset_path: Path to CSV/JSON/JSONL dataset
            task: Task type ("auto", "symptom_correlation", "diagnosis_prediction", "lab_anomaly")
            save_model: Whether to save the trained model to disk

        Returns:
            Training results dict with metrics

        Example:
            >>> orchestrator = DiagnosticCorrelationOrchestrator(use_ml=True)
            >>> results = orchestrator.train_ml("ehr_nexus_ai/data/symptom_pairs.csv")
            >>> print(f"Accuracy: {results['test_metrics']['accuracy']:.2%}")
        """
        if self.ml_trainer is None:
            from ..ml import MLConfig, ModelTrainer

            self.ml_trainer = ModelTrainer(self.ml_config or MLConfig())

        print(f"\n{'='*60}")
        print(f"Training ML Model for: {task}")
        print(f"{'='*60}")

        results = self.ml_trainer.train(
            dataset_path=dataset_path,
            task=task,
            save_model=save_model,
            model_name=f"ehr_nexus_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # Enable ML mode automatically after training
        self.use_ml = True
        print(f"[EHR Nexus AI] ML mode activated with {results['test_metrics'].get('accuracy', 0):.2%} accuracy.")

        return results

    def evaluate_ml(
        self,
        test_data_path: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained ML model on test data.

        Args:
            test_data_path: Path to test dataset

        Returns:
            Evaluation metrics
        """
        if not self.ml_trainer or not self.ml_trainer.trained_models:
            raise RuntimeError(
                "No trained ML model available. "
                "Call train_ml() first or pass ml_model_path to the constructor."
            )

        return self.ml_trainer.evaluate_model(
            model_name=list(self.ml_trainer.trained_models.keys())[-1],
            test_data_path=test_data_path,
        )

    def get_pipeline_description(self) -> Dict[str, Any]:
        """
        Return a description of the analysis pipeline configuration.
        Useful for system logging and transparency.
        """
        ml_info = {}
        if self.use_ml and self.ml_trainer:
            if self.ml_trainer.trained_models:
                model_name = list(self.ml_trainer.trained_models.keys())[-1]
                model = self.ml_trainer.trained_models[model_name]
                ml_info = {
                    "model_name": model_name,
                    "metrics": model.training_metrics,
                    "n_features": len(model.feature_columns) if model.feature_columns else 0,
                }

        return {
            "orchestrator": "DiagnosticCorrelationOrchestrator",
            "version": "1.0.0",
            "ml_enabled": self.use_ml,
            "ml_info": ml_info,
            "engines": [
                {
                    "name": "SymptomCorrelationEngine",
                    "parameters": {
                        "similarity_threshold":
                            self.symptom_engine.similarity_threshold
                    }
                },
                {
                    "name": "DiagnosisCorrelationEngine",
                    "parameters": {
                        "similarity_threshold":
                            self.diagnosis_engine.similarity_threshold
                    }
                },
                {
                    "name": "LabTrendAnalysisEngine",
                    "parameters": {
                        "significant_change_threshold":
                            self.lab_engine.significant_change_threshold,
                        "min_data_points":
                            self.lab_engine.min_data_points
                    }
                },
                {
                    "name": "DuplicateLabDetector",
                    "parameters": {
                        "time_window_hours":
                            self.duplicate_detector.time_window_hours,
                        "value_similarity_threshold":
                            self.duplicate_detector.value_similarity_threshold
                    }
                },
                {
                    "name": "MedicationReminderEngine",
                    "parameters": {
                        "refill_reminder_days_before":
                            self.medication_engine.refill_reminder_days_before,
                        "overdue_grace_days":
                            self.medication_engine.overdue_grace_days,
                        "order_pending_threshold_days":
                            self.medication_engine.order_pending_threshold_days
                    }
                }
            ],
            "output_format": "ExplainableCorrelationReport",
            "disclaimer": (
                "This system is a Clinical Decision Support System (CDSS). "
                "All outputs are correlations and recommendations for "
                "physician review. No output constitutes a medical diagnosis "
                "or treatment recommendation."
            )
        }
