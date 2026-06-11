"""
EHR Nexus - Diagnostic Correlation Engine
===========================================
A Clinical Decision Support System (CDSS) that assists healthcare
professionals by analyzing and correlating historical patient records
with current consultations and laboratory findings.

This module is designed as a CDSS — it does NOT diagnose, predict,
or prescribe treatments. All outputs are correlations and
recommendations for physician review.

Components:
- SymptomCorrelationEngine: Compares current vs historical symptoms
- DiagnosisCorrelationEngine: Correlates current vs historical diagnoses
- LabTrendAnalysisEngine: Analyzes lab value trends and abnormalities
- DuplicateLabDetector: Detects duplicate lab uploads
- MedicationReminderEngine: Monitors medications and orders
- ExplainableEngine: Produces traceable, explainable outputs
- DiagnosticCorrelationOrchestrator: Main entry point

Usage:
    from ehr_nexus_ai import DiagnosticCorrelationOrchestrator
    from ehr_nexus_ai.core.data_models import PatientRecord

    orchestrator = DiagnosticCorrelationOrchestrator()
    report = orchestrator.analyze(patient_record)
    print(report.executive_summary)
"""

from .core.orchestrator import DiagnosticCorrelationOrchestrator
from .core.data_models import (
    PatientRecord, Patient, Consultation, Symptom,
    ClinicalDiagnosis, LaboratoryResult, Medication, DoctorOrder
)
from .utils.config import EHRNexusAIConfig, DEFAULT_CONFIG

__version__ = "1.0.0"
__author__ = "EHR Nexus AI Team"

__all__ = [
    "DiagnosticCorrelationOrchestrator",
    "PatientRecord",
    "Patient",
    "Consultation",
    "Symptom",
    "ClinicalDiagnosis",
    "LaboratoryResult",
    "Medication",
    "DoctorOrder",
    "EHRNexusAIConfig",
    "DEFAULT_CONFIG",
]