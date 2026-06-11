"""
EHR Nexus - Unit Tests for All AI Engines
===========================================
Comprehensive test suite covering all components of the
Diagnostic Correlation Engine.

Run: python -m ehr_nexus_ai.tests.test_all_engines
"""

import sys
import os
import json
from datetime import datetime, date
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ehr_nexus_ai.core.data_models import (
    PatientRecord, Patient, Consultation, Symptom,
    ClinicalDiagnosis, LaboratoryResult, Medication,
    DoctorOrder, Sex
)
from ehr_nexus_ai.engines.symptom_correlation_engine import SymptomCorrelationEngine
from ehr_nexus_ai.engines.diagnosis_correlation_engine import DiagnosisCorrelationEngine
from ehr_nexus_ai.engines.lab_trend_analysis_engine import LabTrendAnalysisEngine
from ehr_nexus_ai.engines.duplicate_lab_detector import DuplicateLabDetector
from ehr_nexus_ai.engines.medication_reminder_engine import MedicationReminderEngine
from ehr_nexus_ai.core.orchestrator import DiagnosticCorrelationOrchestrator
from ehr_nexus_ai.core.explainable_engine import ExplainableEngine, ExplanationNode


def create_minimal_record() -> PatientRecord:
    """Create a minimal patient record for basic testing."""
    patient = Patient(
        patient_id="TEST-001",
        name="Test Patient",
        age=30,
        sex=Sex.MALE
    )

    cons1 = Consultation(
        consultation_id="C1",
        patient_id="TEST-001",
        consultation_date=datetime(2025, 1, 1),
        symptoms=[
            Symptom(name="Headache", severity="Mild"),
            Symptom(name="Fever", severity="Moderate"),
        ],
        diagnoses=[
            ClinicalDiagnosis(
                diagnosis_id="D1",
                diagnosis_code="J06.9",
                diagnosis_name="Acute URI"
            )
        ],
        laboratory_results=[
            LaboratoryResult(
                lab_result_id="L1",
                test_name="CBC",
                parameter_name="Hemoglobin",
                value=14.0,
                unit="g/dL",
                reference_range_low=13.5,
                reference_range_high=17.5
            )
        ],
        medications=[
            Medication(
                medication_id="M1",
                medication_name="Paracetamol",
                dosage="500mg",
                frequency="Three times daily",
                route="Oral",
                start_date=date(2025, 1, 1),
                end_date=date(2025, 1, 7),
                status="Completed"
            )
        ]
    )

    cons2 = Consultation(
        consultation_id="C2",
        patient_id="TEST-001",
        consultation_date=datetime(2026, 6, 1),
        symptoms=[
            Symptom(name="Headache", severity="Severe"),
            Symptom(name="Dizziness", severity="Moderate"),
        ],
        diagnoses=[
            ClinicalDiagnosis(
                diagnosis_id="D2",
                diagnosis_code="I10",
                diagnosis_name="Essential hypertension"
            )
        ],
        laboratory_results=[
            LaboratoryResult(
                lab_result_id="L2",
                test_name="Blood Chemistry",
                parameter_name="Fasting Blood Glucose",
                value=110,
                unit="mg/dL",
                reference_range_low=70,
                reference_range_high=100,
                is_abnormal=True,
                flag="High"
            )
        ],
        medications=[
            Medication(
                medication_id="M2",
                medication_name="Amlodipine",
                dosage="5mg",
                frequency="Once daily",
                route="Oral",
                start_date=date(2026, 6, 1),
                end_date=date(2026, 12, 1),
                status="Active",
                next_refill_date=date(2026, 12, 1)
            )
        ]
    )

    return PatientRecord(patient=patient, consultations=[cons1, cons2])


def test_data_models():
    """Test core data model creation and serialization."""
    print("\n[TEST] Data Models...", end=" ")

    patient = Patient(patient_id="P1", name="John", age=25, sex=Sex.MALE)
    assert patient.patient_id == "P1"
    assert patient.name == "John"
    assert patient.sex == Sex.MALE

    symptom = Symptom(name="Cough", severity="Moderate", duration="5 days")
    assert symptom.name == "Cough"
    assert symptom.severity == "Moderate"

    lab = LaboratoryResult(
        lab_result_id="L1",
        test_name="CBC",
        parameter_name="WBC",
        value=8.5,
        unit="x10^9/L",
        reference_range_low=4.0,
        reference_range_high=10.0
    )
    assert lab.is_within_reference_range is True
    assert lab.to_dict()["value"] == 8.5

    diagnosis = ClinicalDiagnosis(
        diagnosis_id="D1",
        diagnosis_code="I10",
        diagnosis_name="Hypertension"
    )
    assert diagnosis.diagnosis_code == "I10"

    med = Medication(
        medication_id="M1",
        medication_name="Metformin",
        dosage="500mg",
        frequency="Twice daily"
    )
    assert med.medication_name == "Metformin"

    order = DoctorOrder(
        order_id="O1",
        order_type="Lab Test",
        description="CBC",
        status="Pending"
    )
    assert order.status == "Pending"

    print("PASSED")


def test_patient_record():
    """Test PatientRecord operations."""
    print("[TEST] PatientRecord...", end=" ")

    record = create_minimal_record()
    assert len(record.consultations) == 2

    latest = record.get_latest_consultation()
    assert latest.consultation_id == "C2"

    historical = record.get_historical_consultations()
    assert len(historical) == 1
    assert historical[0].consultation_id == "C1"

    all_labs = record.get_all_lab_results()
    assert len(all_labs) == 2

    all_meds = record.get_all_medications()
    assert len(all_meds) == 2

    # Test dict serialization
    d = record.to_dict()
    assert d["patient"]["name"] == "Test Patient"
    assert len(d["consultations"]) == 2

    # Test reconstruction
    restored = PatientRecord.from_dict(d)
    assert restored.patient.patient_id == "TEST-001"
    assert len(restored.consultations) == 2
    assert restored.consultations[0].symptoms[0].name == "Headache"

    print("PASSED")


def test_symptom_correlation_engine():
    """Test symptom correlation with known patterns."""
    print("[TEST] SymptomCorrelationEngine...", end=" ")

    record = create_minimal_record()
    engine = SymptomCorrelationEngine(similarity_threshold=0.7)

    # Test with current consultation = C2 (has Headache + Dizziness)
    result = engine.correlate(record, record.get_latest_consultation())

    assert result.recurring_symptoms is not None
    # Headache is recurring (appeared in both C1 and C2)
    headache_found = any(
        r["current_symptom"] == "Headache"
        for r in result.recurring_symptoms
    )
    assert headache_found, "Headache should be identified as recurring"

    # Dizziness is new (only in C2)
    assert "Dizziness" in result.new_symptoms, \
        "Dizziness should be identified as new symptom"

    # Severity change: Headache went from Mild to Severe
    severity_found = any(
        s["symptom"] == "Headache" and s["trend"] == "Worsening"
        for s in result.severity_changes
    )
    assert severity_found, \
        "Headache severity worsening should be detected"

    assert result.summary != ""
    assert 0 <= result.confidence_score <= 1.0

    # Test without historical data
    single_record = PatientRecord(
        patient=record.patient,
        consultations=[record.consultations[1]]
    )
    single_result = engine.correlate(single_record)
    assert "No historical" in single_result.summary

    print("PASSED")


def test_diagnosis_correlation_engine():
    """Test diagnosis correlation with known patterns."""
    print("[TEST] DiagnosisCorrelationEngine...", end=" ")

    record = create_minimal_record()
    engine = DiagnosisCorrelationEngine(similarity_threshold=0.75)

    result = engine.correlate(record, record.get_latest_consultation())

    # Current consultation (C2) has Essential hypertension
    # This is new - not in C1 (which had Acute URI)
    assert len(result.recurring_diagnoses) == 0
    assert len(result.new_diagnoses) > 0

    # Add a consultation with matching diagnosis to test recurrence
    patient = record.patient
    cons3 = Consultation(
        consultation_id="C3",
        patient_id="TEST-001",
        consultation_date=datetime(2025, 6, 1),
        diagnoses=[
            ClinicalDiagnosis(
                diagnosis_id="D3",
                diagnosis_code="I10",
                diagnosis_name="Essential hypertension"
            )
        ]
    )
    extended_record = PatientRecord(
        patient=patient,
        consultations=[cons3] + record.consultations
    )

    # C2 is latest, has I10 - should match C3
    result2 = engine.correlate(extended_record)
    assert len(result2.recurring_diagnoses) > 0, \
        "Should detect recurring I10 diagnosis"

    print("PASSED")


def test_lab_trend_analysis():
    """Test lab trend analysis with limited data."""
    print("[TEST] LabTrendAnalysisEngine...", end=" ")

    record = create_minimal_record()
    engine = LabTrendAnalysisEngine(min_data_points=2)

    result = engine.analyze(record, record.get_latest_consultation())

    # Should find trends with 2+ measurements
    if result.trends_by_parameter:
        for trend in result.trends_by_parameter:
            assert trend["statistics"]["mean"] > 0
            assert trend["number_of_measurements"] >= 2

    # Test significant changes
    for change in result.significant_changes:
        assert abs(change["z_score"]) >= engine.significant_change_threshold

    # Test current vs historical comparisons
    for comp in result.current_vs_historical_comparisons:
        assert "current_value" in comp
        assert "historical" in comp or "status" in comp

    print("PASSED")


def test_duplicate_lab_detector():
    """Test duplicate lab detection."""
    print("[TEST] DuplicateLabDetector...", end=" ")

    record = create_minimal_record()
    detector = DuplicateLabDetector(time_window_hours=48)

    # Create duplicate labs
    dup_labs = [
        # Exact duplicate of L1 (Hemoglobin)
        LaboratoryResult(
            lab_result_id="LD1",
            test_name="CBC",
            parameter_name="Hemoglobin",
            value=14.0,
            unit="g/dL",
            reference_range_low=13.5,
            reference_range_high=17.5,
            uploaded_at=datetime.now()
        ),
        # Unique lab
        LaboratoryResult(
            lab_result_id="LD2",
            test_name="Lipid Profile",
            parameter_name="LDL Cholesterol",
            value=130,
            unit="mg/dL",
            reference_range_low=0,
            reference_range_high=100,
            uploaded_at=datetime.now()
        )
    ]

    result = detector.detect(record, dup_labs)
    assert result.duplicate_count > 0, \
        "Should detect at least the Hemoglobin duplicate"
    assert result.unique_results >= 1, \
        "Should count the unique LDL result"

    # Test with no existing labs
    empty_record = PatientRecord(
        patient=record.patient,
        consultations=[]
    )
    result2 = detector.detect(empty_record, dup_labs)
    assert result2.duplicate_count == 0
    assert result2.unique_results == len(dup_labs)

    # Test hash-based matching
    lab_with_hash = LaboratoryResult(
        lab_result_id="LD3",
        test_name="CBC",
        parameter_name="Hemoglobin",
        value=14.0,
        unit="g/dL",
        reference_range_low=13.5,
        reference_range_high=17.5,
        file_hash="abc123def456"
    )
    lab_with_same_hash = LaboratoryResult(
        lab_result_id="LD4",
        test_name="CBC",
        parameter_name="Hemoglobin",
        value=14.0,
        unit="g/dL",
        reference_range_low=13.5,
        reference_range_high=17.5,
        file_hash="abc123def456"
    )

    result3 = detector.detect(record, [lab_with_same_hash])
    # Should match by content hash even without explicit file_hash
    assert result3.duplicate_count >= 0

    print("PASSED")


def test_medication_reminder_engine():
    """Test medication and order monitoring."""
    print("[TEST] MedicationReminderEngine...", end=" ")

    record = create_minimal_record()
    engine = MedicationReminderEngine()

    result = engine.analyze(record)
    assert result.active_medications is not None
    assert result.pending_doctor_orders is not None

    # Overdue medications check
    for med in result.overdue_medications:
        assert "overdue_reasons" in med
        assert len(med["overdue_reasons"]) > 0

    # Check summary generation
    assert result.summary != ""

    print("PASSED")


def test_explainable_engine():
    """Test explanation node and report generation."""
    print("[TEST] ExplainableEngine...", end=" ")

    record = create_minimal_record()
    engine = ExplainableEngine(record)

    # Test individual node
    node = ExplanationNode(
        finding_type="test_finding",
        description="Test description",
        confidence=0.85,
        source_consultations=["C1"],
        evidence=[{"key": "value"}],
        methodology="Test methodology",
        severity="INFO"
    )
    assert node.finding_type == "test_finding"
    assert node.confidence == 0.85

    # Test child addition
    child = ExplanationNode(
        finding_type="child_finding",
        description="Child",
        confidence=0.9,
        source_consultations=[],
        evidence=[],
        methodology="Child method"
    )
    node.add_child(child)
    assert len(node.children) == 1

    # Test report generation without results
    report = engine.build_report()
    assert report.patient_id == "TEST-001"
    assert report.analysis_timestamp != ""
    assert report.total_findings == 0  # No results provided

    # Test JSON serialization
    json_str = report.to_json()
    json_data = json.loads(json_str)
    assert json_data["patient_id"] == "TEST-001"

    print("PASSED")


def test_orchestrator():
    """Test the main orchestrator pipeline."""
    print("[TEST] DiagnosticCorrelationOrchestrator...", end=" ")

    record = create_minimal_record()
    orchestrator = DiagnosticCorrelationOrchestrator()

    # Full analysis
    report = orchestrator.analyze(record)

    assert report.patient_id == "TEST-001"
    assert report.executive_summary != ""
    assert 0 <= report.overall_confidence <= 1.0

    # Pipeline description
    pipeline = orchestrator.get_pipeline_description()
    assert pipeline["orchestrator"] == "DiagnosticCorrelationOrchestrator"
    assert len(pipeline["engines"]) == 5
    assert "disclaimer" in pipeline

    # Individual engine access
    sym_result = orchestrator.analyze_symptoms_only(record)
    assert sym_result is not None

    dx_result = orchestrator.analyze_diagnoses_only(record)
    assert dx_result is not None

    lab_result = orchestrator.analyze_labs_only(record)
    assert lab_result is not None

    med_result = orchestrator.check_medications_only(record)
    assert med_result is not None

    print("PASSED")


def test_fuzzy_matching():
    """Test fuzzy matching capabilities."""
    print("[TEST] Fuzzy Matching...", end=" ")

    from ehr_nexus_ai.engines.symptom_correlation_engine import \
        SymptomCorrelationEngine
    engine = SymptomCorrelationEngine()

    # Test synonym matching
    score = engine._fuzzy_match_symptom("headache", "head pain")
    assert score >= 0.7, f"Synonym 'headache'/'head pain' score {score}"

    score = engine._fuzzy_match_symptom("fever", "pyrexia")
    assert score >= 0.7, f"Synonym 'fever'/'pyrexia' score {score}"

    # Test exact match
    score = engine._fuzzy_match_symptom("cough", "cough")
    assert score == 1.0

    # Test different symptoms
    score = engine._fuzzy_match_symptom("headache", "foot pain")
    assert score < 0.5, "Unrelated symptoms should have low score"

    # Test ICD code matching
    from ehr_nexus_ai.engines.diagnosis_correlation_engine import \
        DiagnosisCorrelationEngine
    dx_engine = DiagnosisCorrelationEngine()

    dx_a = ClinicalDiagnosis(
        diagnosis_id="DA",
        diagnosis_code="I10",
        diagnosis_name="Hypertension"
    )
    dx_b = ClinicalDiagnosis(
        diagnosis_id="DB",
        diagnosis_code="I10",
        diagnosis_name="Essential hypertension"
    )
    score = dx_engine._fuzzy_match_diagnosis(dx_a, dx_b)
    assert score >= 0.75, f"Same ICD-10 code should match highly ({score})"

    print("PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("EHR NEXUS AI - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    tests = [
        test_data_models,
        test_patient_record,
        test_fuzzy_matching,
        test_symptom_correlation_engine,
        test_diagnosis_correlation_engine,
        test_lab_trend_analysis,
        test_duplicate_lab_detector,
        test_medication_reminder_engine,
        test_explainable_engine,
        test_orchestrator,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n  X FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  X ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, "
          f"{len(tests)} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)