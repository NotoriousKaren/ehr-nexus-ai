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

from typing import Optional, Dict, Any
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
        medication_engine: Optional[MedicationReminderEngine] = None
    ):
        """
        Initialize the orchestrator with custom or default engine instances.

        Args:
            symptom_engine: SymptomCorrelationEngine instance.
            diagnosis_engine: DiagnosisCorrelationEngine instance.
            lab_engine: LabTrendAnalysisEngine instance.
            duplicate_detector: DuplicateLabDetector instance.
            medication_engine: MedicationReminderEngine instance.
        """
        self.symptom_engine = symptom_engine or SymptomCorrelationEngine()
        self.diagnosis_engine = diagnosis_engine or DiagnosisCorrelationEngine()
        self.lab_engine = lab_engine or LabTrendAnalysisEngine()
        self.duplicate_detector = duplicate_detector or DuplicateLabDetector()
        self.medication_engine = medication_engine or MedicationReminderEngine()

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

    def get_pipeline_description(self) -> Dict[str, Any]:
        """
        Return a description of the analysis pipeline configuration.
        Useful for system logging and transparency.
        """
        return {
            "orchestrator": "DiagnosticCorrelationOrchestrator",
            "version": "1.0.0",
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