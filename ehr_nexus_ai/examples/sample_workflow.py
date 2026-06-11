"""
EHR Nexus - Sample Workflow & Example Input/Output
=====================================================
Demonstrates how to use the Diagnostic Correlation Engine with
realistic patient data. This script shows:
1. Creating sample patient data
2. Running the full analysis pipeline
3. Viewing results for each engine
4. Understanding the explainable output

Run: python -m ehr_nexus_ai.examples.sample_workflow
"""

import sys
import os
from datetime import datetime, date, timedelta
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ehr_nexus_ai.core.data_models import (
    PatientRecord, Patient, Consultation, Symptom,
    ClinicalDiagnosis, LaboratoryResult, Medication,
    DoctorOrder, Sex
)
from ehr_nexus_ai.core.orchestrator import DiagnosticCorrelationOrchestrator


def create_sample_patient_record() -> PatientRecord:
    """
    Create a realistic sample patient with 5 consultations over 2 years.
    This demonstrates recurring symptoms, diagnoses, lab trends, etc.
    """
    patient = Patient(
        patient_id="PT-1001",
        name="Juan Dela Cruz",
        age=45,
        sex=Sex.MALE,
        created_at=datetime(2024, 1, 1)
    )

    consultations = []

    # === Consultation 1: Initial Visit (18 months ago) ===
    cons1 = Consultation(
        consultation_id="C001",
        patient_id="PT-1001",
        consultation_date=datetime(2024, 6, 15, 10, 30),
        chief_complaint="Fever and headache for 3 days",
        visit_type="Initial",
        symptoms=[
            Symptom(name="Fever", severity="Moderate", duration="3 days"),
            Symptom(name="Headache", severity="Mild", duration="3 days"),
            Symptom(name="Body aches", severity="Moderate", duration="2 days"),
        ],
        diagnoses=[
            ClinicalDiagnosis(
                diagnosis_id="D001",
                diagnosis_code="J06.9",
                diagnosis_name="Acute upper respiratory infection, unspecified",
                diagnosis_type="Primary"
            )
        ],
        doctor_orders=[
            DoctorOrder(
                order_id="O001",
                order_type="Lab Test",
                description="Complete Blood Count",
                status="Completed",
                created_at=datetime(2024, 6, 15, 10, 30),
                completed_at=datetime(2024, 6, 16, 14, 0)
            ),
            DoctorOrder(
                order_id="O002",
                order_type="Medication",
                description="Paracetamol 500mg",
                status="Completed",
                created_at=datetime(2024, 6, 15, 10, 30),
                completed_at=datetime(2024, 6, 20, 10, 0)
            )
        ],
        medications=[
            Medication(
                medication_id="M001",
                medication_name="Paracetamol",
                dosage="500mg",
                frequency="Three times daily",
                route="Oral",
                start_date=date(2024, 6, 15),
                end_date=date(2024, 6, 22),
                status="Completed"
            )
        ],
        laboratory_results=[
            LaboratoryResult(
                lab_result_id="L001",
                test_name="Complete Blood Count",
                parameter_name="Hemoglobin",
                value=14.2,
                unit="g/dL",
                reference_range_low=13.5,
                reference_range_high=17.5,
                is_abnormal=False,
                flag="Normal",
                uploaded_at=datetime(2024, 6, 16, 14, 0)
            ),
            LaboratoryResult(
                lab_result_id="L002",
                test_name="Complete Blood Count",
                parameter_name="White Blood Cell Count",
                value=11.5,
                unit="x10^9/L",
                reference_range_low=4.0,
                reference_range_high=10.0,
                is_abnormal=True,
                flag="High",
                uploaded_at=datetime(2024, 6, 16, 14, 0)
            )
        ]
    )
    consultations.append(cons1)

    # === Consultation 2: Follow-up (12 months ago) ===
    cons2 = Consultation(
        consultation_id="C002",
        patient_id="PT-1001",
        consultation_date=datetime(2025, 1, 10, 9, 0),
        chief_complaint="Recurring headache and dizziness",
        visit_type="Follow-up",
        symptoms=[
            Symptom(name="Headache", severity="Moderate", duration="1 week"),
            Symptom(name="Dizziness", severity="Mild", duration="3 days"),
            Symptom(name="Fatigue", severity="Moderate", duration="2 weeks"),
        ],
        diagnoses=[
            ClinicalDiagnosis(
                diagnosis_id="D002",
                diagnosis_code="G43.909",
                diagnosis_name="Migraine, unspecified, not intractable",
                diagnosis_type="Primary"
            ),
            ClinicalDiagnosis(
                diagnosis_id="D003",
                diagnosis_code="I10",
                diagnosis_name="Essential hypertension",
                diagnosis_type="Secondary"
            )
        ],
        doctor_orders=[
            DoctorOrder(
                order_id="O003",
                order_type="Lab Test",
                description="Blood Glucose Fasting",
                status="Completed",
                created_at=datetime(2025, 1, 10, 9, 0),
                completed_at=datetime(2025, 1, 11, 8, 0)
            ),
            DoctorOrder(
                order_id="O004",
                order_type="Medication",
                description="Amlodipine 5mg daily",
                status="Active",
                created_at=datetime(2025, 1, 10, 9, 0)
            ),
            DoctorOrder(
                order_id="O005",
                order_type="Imaging",
                description="CT Scan of Head",
                status="Pending",
                created_at=datetime(2025, 1, 10, 9, 0)
            )
        ],
        medications=[
            Medication(
                medication_id="M002",
                medication_name="Amlodipine",
                dosage="5mg",
                frequency="Once daily",
                route="Oral",
                start_date=date(2025, 1, 10),
                end_date=date(2025, 7, 10),
                status="Active",
                next_refill_date=date(2025, 7, 10)
            ),
            Medication(
                medication_id="M003",
                medication_name="Sumatriptan",
                dosage="50mg",
                frequency="As needed",
                route="Oral",
                start_date=date(2025, 1, 10),
                end_date=date(2025, 4, 10),
                status="Completed"
            )
        ],
        laboratory_results=[
            LaboratoryResult(
                lab_result_id="L003",
                test_name="Blood Chemistry",
                parameter_name="Fasting Blood Glucose",
                value=125,
                unit="mg/dL",
                reference_range_low=70,
                reference_range_high=100,
                is_abnormal=True,
                flag="High",
                uploaded_at=datetime(2025, 1, 11, 8, 0)
            ),
            LaboratoryResult(
                lab_result_id="L004",
                test_name="Complete Blood Count",
                parameter_name="Hemoglobin",
                value=13.8,
                unit="g/dL",
                reference_range_low=13.5,
                reference_range_high=17.5,
                is_abnormal=False,
                flag="Normal",
                uploaded_at=datetime(2025, 1, 11, 8, 0)
            )
        ]
    )
    consultations.append(cons2)

    # === Consultation 3: Follow-up (8 months ago) ===
    cons3 = Consultation(
        consultation_id="C003",
        patient_id="PT-1001",
        consultation_date=datetime(2025, 4, 5, 11, 0),
        chief_complaint="Follow-up on blood pressure and headaches",
        visit_type="Follow-up",
        symptoms=[
            Symptom(name="Headache", severity="Moderate", duration="2 weeks"),
            Symptom(name="Dizziness", severity="Mild", duration="1 week"),
        ],
        diagnoses=[
            ClinicalDiagnosis(
                diagnosis_id="D004",
                diagnosis_code="I10",
                diagnosis_name="Essential hypertension",
                diagnosis_type="Primary"
            ),
            ClinicalDiagnosis(
                diagnosis_id="D005",
                diagnosis_code="R73.09",
                diagnosis_name="Impaired fasting glucose",
                diagnosis_type="Secondary"
            )
        ],
        doctor_orders=[
            DoctorOrder(
                order_id="O006",
                order_type="Lab Test",
                description="Lipid Profile",
                status="Completed",
                created_at=datetime(2025, 4, 5, 11, 0),
                completed_at=datetime(2025, 4, 6, 9, 0)
            )
        ],
        medications=[
            Medication(
                medication_id="M002",
                medication_name="Amlodipine",
                dosage="5mg",
                frequency="Once daily",
                route="Oral",
                start_date=date(2025, 1, 10),
                end_date=date(2025, 7, 10),
                status="Active",
                next_refill_date=date(2025, 7, 10)
            )
        ],
        laboratory_results=[
            LaboratoryResult(
                lab_result_id="L005",
                test_name="Blood Chemistry",
                parameter_name="Fasting Blood Glucose",
                value=132,
                unit="mg/dL",
                reference_range_low=70,
                reference_range_high=100,
                is_abnormal=True,
                flag="High",
                uploaded_at=datetime(2025, 4, 6, 9, 0)
            ),
            LaboratoryResult(
                lab_result_id="L006",
                test_name="Lipid Profile",
                parameter_name="LDL Cholesterol",
                value=145,
                unit="mg/dL",
                reference_range_low=0,
                reference_range_high=100,
                is_abnormal=True,
                flag="High",
                uploaded_at=datetime(2025, 4, 6, 9, 0)
            )
        ]
    )
    consultations.append(cons3)

    # === Consultation 4: Follow-up (3 months ago) ===
    cons4 = Consultation(
        consultation_id="C004",
        patient_id="PT-1001",
        consultation_date=datetime(2026, 1, 20, 14, 0),
        chief_complaint="Recurring headaches, checking labs",
        visit_type="Follow-up",
        symptoms=[
            Symptom(name="Headache", severity="Moderate", duration="3 weeks"),
            Symptom(name="Fatigue", severity="Moderate", duration="1 month"),
        ],
        diagnoses=[
            ClinicalDiagnosis(
                diagnosis_id="D006",
                diagnosis_code="I10",
                diagnosis_name="Essential hypertension",
                diagnosis_type="Primary"
            )
        ],
        doctor_orders=[
            DoctorOrder(
                order_id="O007",
                order_type="Lab Test",
                description="Complete Blood Count, Blood Chemistry",
                status="Completed",
                created_at=datetime(2026, 1, 20, 14, 0),
                completed_at=datetime(2026, 1, 21, 10, 0)
            )
        ],
        medications=[
            Medication(
                medication_id="M004",
                medication_name="Amlodipine",
                dosage="10mg",
                frequency="Once daily",
                route="Oral",
                start_date=date(2026, 1, 20),
                end_date=date(2026, 7, 20),
                status="Active",
                next_refill_date=date(2026, 7, 20)
            ),
            Medication(
                medication_id="M005",
                medication_name="Metformin",
                dosage="500mg",
                frequency="Twice daily",
                route="Oral",
                start_date=date(2026, 1, 20),
                end_date=date(2026, 7, 20),
                status="Active",
                next_refill_date=date(2026, 7, 20)
            )
        ],
        laboratory_results=[
            LaboratoryResult(
                lab_result_id="L007",
                test_name="Complete Blood Count",
                parameter_name="Hemoglobin",
                value=13.5,
                unit="g/dL",
                reference_range_low=13.5,
                reference_range_high=17.5,
                is_abnormal=False,
                flag="Normal",
                uploaded_at=datetime(2026, 1, 21, 10, 0)
            ),
            LaboratoryResult(
                lab_result_id="L008",
                test_name="Blood Chemistry",
                parameter_name="Fasting Blood Glucose",
                value=128,
                unit="mg/dL",
                reference_range_low=70,
                reference_range_high=100,
                is_abnormal=True,
                flag="High",
                uploaded_at=datetime(2026, 1, 21, 10, 0)
            ),
            LaboratoryResult(
                lab_result_id="L009",
                test_name="Lipid Profile",
                parameter_name="LDL Cholesterol",
                value=138,
                unit="mg/dL",
                reference_range_low=0,
                reference_range_high=100,
                is_abnormal=True,
                flag="High",
                uploaded_at=datetime(2026, 1, 21, 10, 0)
            )
        ]
    )
    consultations.append(cons4)

    # === Consultation 5: CURRENT Visit (today) ===
    today = datetime.now()
    cons5 = Consultation(
        consultation_id="C005",
        patient_id="PT-1001",
        consultation_date=today,
        chief_complaint="Severe headache with blurred vision",
        visit_type="Follow-up",
        symptoms=[
            Symptom(name="Headache", severity="Severe", duration="1 week"),
            Symptom(name="Blurred vision", severity="Moderate", duration="3 days"),
            Symptom(name="Dizziness", severity="Moderate", duration="1 week"),
            Symptom(name="Fatigue", severity="Severe", duration="2 weeks"),
        ],
        diagnoses=[
            ClinicalDiagnosis(
                diagnosis_id="D007",
                diagnosis_code="I10",
                diagnosis_name="Essential hypertension",
                diagnosis_type="Primary"
            ),
            ClinicalDiagnosis(
                diagnosis_id="D008",
                diagnosis_code="E11.9",
                diagnosis_name="Type 2 diabetes mellitus without complications",
                diagnosis_type="Secondary"
            )
        ],
        doctor_orders=[
            DoctorOrder(
                order_id="O008",
                order_type="Lab Test",
                description="HbA1c, Lipid Profile, Creatinine",
                status="Pending",
                created_at=today
            ),
            DoctorOrder(
                order_id="O009",
                order_type="Imaging",
                description="CT Scan of Head",
                status="Pending",
                created_at=today
            )
        ],
        medications=[
            Medication(
                medication_id="M004",
                medication_name="Amlodipine",
                dosage="10mg",
                frequency="Once daily",
                route="Oral",
                start_date=date(2026, 1, 20),
                end_date=date(2026, 7, 20),
                status="Active",
                next_refill_date=date(2026, 7, 20)
            ),
            Medication(
                medication_id="M005",
                medication_name="Metformin",
                dosage="500mg",
                frequency="Twice daily",
                route="Oral",
                start_date=date(2026, 1, 20),
                end_date=date(2026, 7, 20),
                status="Active",
                next_refill_date=date(2026, 7, 20)
            )
        ],
        laboratory_results=[
            LaboratoryResult(
                lab_result_id="L010",
                test_name="Complete Blood Count",
                parameter_name="Hemoglobin",
                value=13.2,
                unit="g/dL",
                reference_range_low=13.5,
                reference_range_high=17.5,
                is_abnormal=True,
                flag="Low",
                uploaded_at=datetime.now()
            ),
            LaboratoryResult(
                lab_result_id="L011",
                test_name="Blood Chemistry",
                parameter_name="Fasting Blood Glucose",
                value=145,
                unit="mg/dL",
                reference_range_low=70,
                reference_range_high=100,
                is_abnormal=True,
                flag="High",
                uploaded_at=datetime.now()
            )
        ]
    )
    consultations.append(cons5)

    return PatientRecord(patient=patient, consultations=consultations)


def demonstrate_orchestrator():
    """Demonstrate the full AI pipeline."""
    print("=" * 80)
    print("EHR NEXUS - DIAGNOSTIC CORRELATION ENGINE")
    print("Sample Workflow Demonstration")
    print("=" * 80)

    # Create sample data
    print("\n[1] Creating sample patient record with 5 consultations over 2 years...")
    record = create_sample_patient_record()
    patient = record.patient
    print(f"    Patient: {patient.name} (ID: {patient.patient_id}, Age: {patient.age})")
    print(f"    Total consultations: {len(record.consultations)}")
    print(f"    Date range: {record.consultations[0].consultation_date.date()} "
          f"to {record.consultations[-1].consultation_date.date()}")

    # Initialize orchestrator
    print("\n[2] Initializing Diagnostic Correlation Orchestrator...")
    orchestrator = DiagnosticCorrelationOrchestrator()

    # Get current consultation
    current = record.get_latest_consultation()
    print(f"    Current consultation: {current.consultation_id}")
    print(f"    Chief complaint: {current.chief_complaint}")
    print(f"    Current symptoms: {[s.name for s in current.symptoms]}")
    print(f"    Current diagnoses: {[d.diagnosis_name for d in current.diagnoses]}")

    # Run full analysis
    print("\n[3] Running full diagnostic correlation analysis...")
    print("    (Analyzing symptoms, diagnoses, labs, medications, and orders)")
    report = orchestrator.analyze(record)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\n[Executive Summary]")
    print(f"   {report.executive_summary}")

    print(f"\n[Overall Confidence: {report.overall_confidence:.0%}]")
    print(f"[Total Findings: {report.total_findings}]")
    print(f"[Analysis Timestamp: {report.analysis_timestamp}]")

    # Print individual findings
    print("\n" + "-" * 80)
    print("DETAILED FINDINGS")
    print("-" * 80)

    for i, node in enumerate(report.nodes, 1):
        print(f"\n--- Finding #{i}: {node.finding_type} ({node.severity}) ---")
        print(f"  Description: {node.description}")
        print(f"  Confidence: {node.confidence:.0%}")
        print(f"  Methodology: {node.methodology[:100]}...")

        if node.evidence:
            print(f"  Evidence:")
            for ev in node.evidence[:3]:
                print(f"    * {json.dumps(ev)[:120]}")

        if node.children:
            for child in node.children:
                print(f"  -> Sub-finding: {child.description[:120]}")

    # Demonstrate run-mode specific engines
    print("\n" + "=" * 80)
    print("INDIVIDUAL ENGINE OUTPUTS")
    print("=" * 80)

    # Symptom correlation only
    print("\n[SYMPTOM CORRELATION ENGINE]")
    sym_result = orchestrator.analyze_symptoms_only(record)
    print(f"  Recurring symptoms: {len(sym_result.recurring_symptoms)}")
    for r in sym_result.recurring_symptoms[:2]:
        print(f"    * {r['current_symptom']} (matched {r['match_count']}x in history)")
    print(f"  New symptoms: {sym_result.new_symptoms}")
    print(f"  Severity changes: {len(sym_result.severity_changes)}")
    for s in sym_result.severity_changes:
        print(f"    * {s['symptom']}: {s['trend']}")
    print(f"  Summary: {sym_result.summary}")

    # Diagnosis correlation only
    print("\n[DIAGNOSIS CORRELATION ENGINE]")
    dx_result = orchestrator.analyze_diagnoses_only(record)
    print(f"  Recurring diagnoses: {len(dx_result.recurring_diagnoses)}")
    for r in dx_result.recurring_diagnoses:
        print(f"    * {r['current_diagnosis']} ({r['current_code']}) - {r['category']}")
        print(f"      Occurred {r['match_count']}x in {r['unique_consultations']} visits")
    print(f"  New diagnoses: {dx_result.new_diagnoses}")
    print(f"  Summary: {dx_result.summary}")

    # Lab trend analysis only
    print("\n[LAB TREND ANALYSIS ENGINE]")
    lab_result = orchestrator.analyze_labs_only(record)
    print(f"  Parameters analyzed: {len(lab_result.trends_by_parameter)}")
    for t in lab_result.trends_by_parameter:
        print(f"    * {t['parameter']}: {t['trend']['direction']} trend "
              f"(mean={t['statistics']['mean']}, n={t['number_of_measurements']})")
    print(f"  Significant changes: {len(lab_result.significant_changes)}")
    for s in lab_result.significant_changes:
        print(f"    * {s['parameter']}: {s['direction']} (z={s['z_score']})")
    print(f"  Summary: {lab_result.summary}")

    # Duplicate detection (simulate new uploads)
    print("\n[DUPLICATE LAB DETECTOR]")
    # Create a duplicate of the last glucose result
    dup_labs = [
        LaboratoryResult(
            lab_result_id="L012",
            test_name="Blood Chemistry",
            parameter_name="Fasting Blood Glucose",
            value=145,
            unit="mg/dL",
            reference_range_low=70,
            reference_range_high=100,
            is_abnormal=True,
            flag="High",
            uploaded_at=datetime.now()
        ),
        LaboratoryResult(
            lab_result_id="L013",
            test_name="Lipid Profile",
            parameter_name="Total Cholesterol",
            value=200,
            unit="mg/dL",
            reference_range_low=0,
            reference_range_high=200,
            is_abnormal=True,
            flag="High",
            uploaded_at=datetime.now()
        )
    ]
    dup_result = orchestrator.check_duplicates_only(record, dup_labs)
    print(f"  Duplicates found: {dup_result.duplicate_count}/{len(dup_labs)}")
    print(f"  Unique results: {dup_result.unique_results}")
    if dup_result.exact_duplicates:
        for d in dup_result.exact_duplicates:
            print(f"    * EXACT: {d['test_name']}/{d['parameter']} = {d['value']}")
    print(f"  Summary: {dup_result.summary}")

    # Medication reminders
    print("\n[MEDICATION REMINDER ENGINE]")
    med_result = orchestrator.check_medications_only(record)
    print(f"  Active medications: {len(med_result.active_medications)}")
    for m in med_result.active_medications:
        print(f"    * {m['medication_name']} {m['dosage']}, {m['frequency']}")
    print(f"  Overdue medications: {len(med_result.overdue_medications)}")
    for m in med_result.overdue_medications:
        print(f"    * {m['medication_name']}: {m['overdue_reasons'][:1]}")
    print(f"  Pending doctor orders: {len(med_result.pending_doctor_orders)}")
    for o in med_result.pending_doctor_orders:
        print(f"    * {o['description']} ({o['days_pending']} days, severity={o['severity']})")
    print(f"  Summary: {med_result.summary}")

    # JSON output example
    print("\n[FULL REPORT AS JSON (first 2000 chars)]")
    json_output = report.to_json()
    print(json_output[:2000] + "\n... (truncated)")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nAll findings above are CORRELATIONS and RECOMMENDATIONS")
    print("for physician review. They do NOT constitute medical")
    print("diagnoses or treatment recommendations.")
    print()


if __name__ == "__main__":
    demonstrate_orchestrator()