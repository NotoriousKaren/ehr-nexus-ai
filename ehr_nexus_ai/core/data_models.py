"""
EHR Nexus - Core Data Models
=================================
Defines the fundamental data structures used across all AI engine components.
These models represent patient data, consultations, lab results, medications, etc.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum


class Sex(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"


@dataclass
class Patient:
    """Represents a patient in the EHR system."""
    patient_id: str
    name: str
    age: int
    sex: Sex
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat() if self.created_at else None
        return d


@dataclass
class Symptom:
    """A single symptom reported during a consultation."""
    symptom_id: str = ""
    name: str = ""
    description: str = ""
    severity: str = ""       # e.g., "Mild", "Moderate", "Severe"
    duration: str = ""       # e.g., "3 days", "2 weeks"
    body_location: str = ""  # e.g., "Chest", "Abdomen", "Head"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalDiagnosis:
    """A diagnosis made during a consultation."""
    diagnosis_id: str = ""
    diagnosis_code: str = ""   # e.g., ICD-10 code
    diagnosis_name: str = ""
    diagnosis_type: str = ""   # e.g., "Primary", "Secondary", "Provisional"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorOrder:
    """An order or plan issued by a doctor."""
    order_id: str = ""
    order_type: str = ""       # e.g., "Lab Test", "Imaging", "Procedure", "Referral"
    description: str = ""
    status: str = ""           # e.g., "Pending", "Completed", "Cancelled"
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    is_flagged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        if self.completed_at:
            d["completed_at"] = self.completed_at.isoformat()
        return d


@dataclass
class Medication:
    """A medication prescribed or taken by the patient."""
    medication_id: str = ""
    medication_name: str = ""
    dosage: str = ""
    frequency: str = ""        # e.g., "Once daily", "Twice daily"
    route: str = ""            # e.g., "Oral", "IV", "Topical"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: str = ""           # e.g., "Active", "Completed", "Discontinued"
    is_overdue: bool = False
    last_refill_date: Optional[date] = None
    next_refill_date: Optional[date] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.start_date:
            d["start_date"] = self.start_date.isoformat()
        if self.end_date:
            d["end_date"] = self.end_date.isoformat()
        if self.last_refill_date:
            d["last_refill_date"] = self.last_refill_date.isoformat()
        if self.next_refill_date:
            d["next_refill_date"] = self.next_refill_date.isoformat()
        return d


@dataclass
class LaboratoryResult:
    """A single laboratory test result."""
    lab_result_id: str = ""
    test_name: str = ""         # e.g., "Complete Blood Count", "Blood Glucose"
    test_category: str = ""     # e.g., "Hematology", "Chemistry", "Microbiology"
    parameter_name: str = ""    # e.g., "Hemoglobin", "WBC", "Glucose"
    value: float = 0.0
    unit: str = ""              # e.g., "g/dL", "cells/uL", "mg/dL"
    reference_range_low: float = 0.0
    reference_range_high: float = 0.0
    is_abnormal: bool = False
    flag: str = ""              # e.g., "High", "Low", "Critical High", "Normal"
    uploaded_at: Optional[datetime] = None
    file_hash: str = ""         # MD5/SHA256 hash for duplicate detection
    notes: str = ""

    @property
    def is_within_reference_range(self) -> bool:
        return self.reference_range_low <= self.value <= self.reference_range_high

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.uploaded_at:
            d["uploaded_at"] = self.uploaded_at.isoformat()
        return d


@dataclass
class Consultation:
    """
    Represents a single patient consultation / visit.
    This is the central unit of analysis for the correlation engines.
    """
    consultation_id: str
    patient_id: str
    consultation_date: datetime
    symptoms: List[Symptom] = field(default_factory=list)
    diagnoses: List[ClinicalDiagnosis] = field(default_factory=list)
    doctor_orders: List[DoctorOrder] = field(default_factory=list)
    medications: List[Medication] = field(default_factory=list)
    laboratory_results: List[LaboratoryResult] = field(default_factory=list)
    doctor_notes: str = ""
    visit_type: str = ""          # e.g., "Initial", "Follow-up", "Emergency"
    chief_complaint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.consultation_date:
            d["consultation_date"] = self.consultation_date.isoformat()
        return d


@dataclass
class PatientRecord:
    """
    Complete historical record for a patient spanning N years.
    This is the primary input to the AI correlation engine.
    """
    patient: Patient
    consultations: List[Consultation] = field(default_factory=list)

    def get_consultations_sorted(self) -> List[Consultation]:
        return sorted(self.consultations,
                      key=lambda c: c.consultation_date, reverse=True)

    def get_latest_consultation(self) -> Optional[Consultation]:
        sorted_cons = self.get_consultations_sorted()
        return sorted_cons[0] if sorted_cons else None

    def get_historical_consultations(self, exclude_latest: bool = True) -> List[Consultation]:
        sorted_cons = self.get_consultations_sorted()
        if exclude_latest and len(sorted_cons) > 0:
            return sorted_cons[1:]
        return sorted_cons

    def get_all_lab_results(self) -> List[LaboratoryResult]:
        labs = []
        for cons in self.consultations:
            labs.extend(cons.laboratory_results)
        return labs

    def get_all_medications(self) -> List[Medication]:
        meds = []
        for cons in self.consultations:
            meds.extend(cons.medications)
        return meds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient": self.patient.to_dict(),
            "consultations": [c.to_dict() for c in self.consultations]
        }

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        """Safely parse a datetime from either string or datetime object."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return None

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        """Safely parse a date from either string or date object."""
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            return date.fromisoformat(value)
        return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientRecord":
        """Reconstruct a PatientRecord from a dictionary (e.g., from JSON API)."""
        patient_data = data.get("patient", {})
        patient = Patient(
            patient_id=patient_data.get("patient_id", ""),
            name=patient_data.get("name", ""),
            age=patient_data.get("age", 0),
            sex=Sex(patient_data.get("sex", "Other")),
            created_at=cls._parse_datetime(patient_data.get("created_at"))
                       or datetime.now()
        )

        consultations = []
        for cons_data in data.get("consultations", []):
            symptoms = [
                Symptom(**s) for s in cons_data.get("symptoms", [])
            ]
            diagnoses = [
                ClinicalDiagnosis(**d) for d in cons_data.get("diagnoses", [])
            ]
            orders = [
                DoctorOrder(
                    order_id=o.get("order_id", ""),
                    order_type=o.get("order_type", ""),
                    description=o.get("description", ""),
                    status=o.get("status", ""),
                    created_at=cls._parse_datetime(o.get("created_at")),
                    completed_at=cls._parse_datetime(o.get("completed_at")),
                    is_flagged=o.get("is_flagged", False)
                )
                for o in cons_data.get("doctor_orders", [])
            ]
            medications = [
                Medication(
                    medication_id=m.get("medication_id", ""),
                    medication_name=m.get("medication_name", ""),
                    dosage=m.get("dosage", ""),
                    frequency=m.get("frequency", ""),
                    route=m.get("route", ""),
                    start_date=cls._parse_date(m.get("start_date")),
                    end_date=cls._parse_date(m.get("end_date")),
                    status=m.get("status", ""),
                    is_overdue=m.get("is_overdue", False),
                    last_refill_date=cls._parse_date(m.get("last_refill_date")),
                    next_refill_date=cls._parse_date(m.get("next_refill_date")),
                    notes=m.get("notes", "")
                )
                for m in cons_data.get("medications", [])
            ]
            lab_results = [
                LaboratoryResult(
                    lab_result_id=lr.get("lab_result_id", ""),
                    test_name=lr.get("test_name", ""),
                    test_category=lr.get("test_category", ""),
                    parameter_name=lr.get("parameter_name", ""),
                    value=lr.get("value", 0.0),
                    unit=lr.get("unit", ""),
                    reference_range_low=lr.get("reference_range_low", 0.0),
                    reference_range_high=lr.get("reference_range_high", 0.0),
                    is_abnormal=lr.get("is_abnormal", False),
                    flag=lr.get("flag", ""),
                    uploaded_at=cls._parse_datetime(lr.get("uploaded_at")),
                    file_hash=lr.get("file_hash", ""),
                    notes=lr.get("notes", "")
                )
                for lr in cons_data.get("laboratory_results", [])
            ]

            consultation = Consultation(
                consultation_id=cons_data.get("consultation_id", ""),
                patient_id=cons_data.get("patient_id", ""),
                consultation_date=cls._parse_datetime(
                    cons_data.get("consultation_date")
                ) or datetime.now(),
                symptoms=symptoms,
                diagnoses=diagnoses,
                doctor_orders=orders,
                medications=medications,
                laboratory_results=lab_results,
                doctor_notes=cons_data.get("doctor_notes", ""),
                visit_type=cons_data.get("visit_type", ""),
                chief_complaint=cons_data.get("chief_complaint", "")
            )
            consultations.append(consultation)

        return cls(patient=patient, consultations=consultations)