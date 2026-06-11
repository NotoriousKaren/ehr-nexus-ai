"""
EHR Nexus - Diagnosis Correlation Engine
==========================================
Correlates current diagnoses with previous diagnoses and medical history.
Identifies:
- Recurring diagnoses and diagnostic patterns
- Diagnosis-symptom associations
- Diagnosis progression or stability over time
- New diagnoses not previously recorded
- Correlations between diagnoses and laboratory findings
"""

from typing import List, Dict, Optional, Any, Tuple
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from datetime import datetime

from ..core.data_models import (
    PatientRecord, Consultation, ClinicalDiagnosis,
    Symptom, LaboratoryResult
)


class DiagnosisCorrelationResult:
    """
    Encapsulates the output of the diagnosis correlation engine.
    Provides explainable results showing which historical records
    contributed to each correlation finding.
    """
    def __init__(self):
        self.recurring_diagnoses: List[Dict[str, Any]] = []
        self.diagnosis_symptom_associations: List[Dict[str, Any]] = []
        self.diagnosis_progression: List[Dict[str, Any]] = []
        self.new_diagnoses: List[str] = []
        self.diagnosis_lab_correlations: List[Dict[str, Any]] = []
        self.summary: str = ""
        self.confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recurring_diagnoses": self.recurring_diagnoses,
            "diagnosis_symptom_associations": self.diagnosis_symptom_associations,
            "diagnosis_progression": self.diagnosis_progression,
            "new_diagnoses": self.new_diagnoses,
            "diagnosis_lab_correlations": self.diagnosis_lab_correlations,
            "summary": self.summary,
            "confidence_score": self.confidence_score
        }


class DiagnosisCorrelationEngine:
    """
    Rule-based engine for correlating current diagnoses with historical
    patient diagnoses. Uses fuzzy matching, frequency analysis, and
    temporal trend detection.
    """

    # ICD-10 code groupings for diagnostic categories
    DIAGNOSTIC_CATEGORIES = {
        "Infectious/Parasitic": ("A00", "B99"),
        "Neoplasms": ("C00", "D49"),
        "Blood/Immune": ("D50", "D89"),
        "Endocrine/Metabolic": ("E00", "E89"),
        "Mental/Behavioral": ("F00", "F99"),
        "Nervous System": ("G00", "G99"),
        "Eye/Ear": ("H00", "H95"),
        "Circulatory": ("I00", "I99"),
        "Respiratory": ("J00", "J99"),
        "Digestive": ("K00", "K95"),
        "Skin": ("L00", "L99"),
        "Musculoskeletal": ("M00", "M99"),
        "Genitourinary": ("N00", "N99"),
        "Pregnancy/Childbirth": ("O00", "O9A"),
        "Perinatal": ("P00", "P96"),
        "Congenital": ("Q00", "Q99"),
        "Symptoms NEC": ("R00", "R99"),
        "Injury/Poisoning": ("S00", "T88"),
        "External Causes": ("V00", "Y99"),
        "Health Status": ("Z00", "Z99"),
    }

    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold

    def correlate(
        self,
        record: PatientRecord,
        current_consultation: Optional[Consultation] = None
    ) -> DiagnosisCorrelationResult:
        """
        Main entry point: correlates current diagnoses with historical records.

        Args:
            record: Complete patient record with all consultations.
            current_consultation: Current consultation to analyze.
                                  If None, uses latest.

        Returns:
            DiagnosisCorrelationResult with all findings.
        """
        result = DiagnosisCorrelationResult()

        if current_consultation is None:
            current_consultation = record.get_latest_consultation()

        if current_consultation is None:
            result.summary = "No consultation data available for correlation."
            return result

        historical = record.get_historical_consultations(exclude_latest=True)

        if not historical:
            result.summary = (
                "No historical diagnosis data available. "
                "This appears to be the patient's first recorded diagnosis."
            )
            result.confidence_score = 1.0
            return result

        current_diagnoses = current_consultation.diagnoses

        if not current_diagnoses:
            result.summary = (
                "No diagnoses recorded in the current consultation."
            )
            return result

        # 1. Detect recurring diagnoses
        result.recurring_diagnoses = self._find_recurring_diagnoses(
            current_diagnoses, historical
        )

        # 2. Analyze diagnosis-symptom associations
        result.diagnosis_symptom_associations = self._analyze_symptom_associations(
            current_diagnoses, current_consultation.symptoms, historical
        )

        # 3. Track diagnosis progression
        result.diagnosis_progression = self._track_diagnosis_progression(
            current_diagnoses, historical
        )

        # 4. Identify new diagnoses
        result.new_diagnoses = self._find_new_diagnoses(
            current_diagnoses, historical
        )

        # 5. Correlate diagnoses with lab findings
        result.diagnosis_lab_correlations = self._correlate_with_labs(
            current_diagnoses, current_consultation.laboratory_results,
            historical
        )

        # 6. Generate summary
        result.summary = self._generate_summary(current_diagnoses, result)

        # 7. Compute confidence
        result.confidence_score = self._compute_confidence(
            current_diagnoses, historical, result
        )

        return result

    def _fuzzy_match_diagnosis(
        self, dx_a: ClinicalDiagnosis, dx_b: ClinicalDiagnosis
    ) -> float:
        """
        Match two diagnoses using:
        - Exact ICD-10 code match (highest confidence)
        - Partial ICD-10 code prefix match (same category)
        - Fuzzy name matching via SequenceMatcher
        """
        # Exact ICD-10 code match
        if dx_a.diagnosis_code and dx_b.diagnosis_code:
            if dx_a.diagnosis_code == dx_b.diagnosis_code:
                return 1.0

            # Same category (first 3 characters of ICD-10)
            if dx_a.diagnosis_code[:3] == dx_b.diagnosis_code[:3]:
                return 0.85

        # Fuzzy name matching
        a_lower = dx_a.diagnosis_name.lower().strip()
        b_lower = dx_b.diagnosis_name.lower().strip()
        if a_lower == b_lower:
            return 1.0

        return SequenceMatcher(None, a_lower, b_lower).ratio()

    def _categorize_diagnosis(self, code: str) -> str:
        """Categorize an ICD-10 code into a diagnostic category."""
        if not code or len(code) < 3:
            return "Uncategorized"
        prefix = code[:3]
        # Check if prefix falls within any category range
        for category, (start, end) in self.DIAGNOSTIC_CATEGORIES.items():
            if start[:3] <= prefix <= end[:3]:
                return category
        return "Uncategorized"

    def _find_recurring_diagnoses(
        self,
        current_diagnoses: List[ClinicalDiagnosis],
        historical: List[Consultation]
    ) -> List[Dict[str, Any]]:
        """Find diagnoses from current consultation that recur in history."""
        recurring = []

        for dx in current_diagnoses:
            matches = []
            for hist_cons in historical:
                for hist_dx in hist_cons.diagnoses:
                    score = self._fuzzy_match_diagnosis(dx, hist_dx)
                    if score >= self.similarity_threshold:
                        matches.append({
                            "consultation_id": hist_cons.consultation_id,
                            "consultation_date":
                                hist_cons.consultation_date.isoformat(),
                            "historical_diagnosis": hist_dx.diagnosis_name,
                            "historical_code": hist_dx.diagnosis_code,
                            "historical_type": hist_dx.diagnosis_type,
                            "match_score": round(score, 3),
                            "historical_notes": hist_dx.notes
                        })

            if matches:
                category = self._categorize_diagnosis(dx.diagnosis_code)
                recurring.append({
                    "current_diagnosis": dx.diagnosis_name,
                    "current_code": dx.diagnosis_code,
                    "current_type": dx.diagnosis_type,
                    "category": category,
                    "historical_matches": matches,
                    "match_count": len(matches),
                    "unique_consultations": len(set(
                        m["consultation_id"] for m in matches
                    ))
                })

        recurring.sort(key=lambda x: x["match_count"], reverse=True)
        return recurring

    def _analyze_symptom_associations(
        self,
        current_diagnoses: List[ClinicalDiagnosis],
        current_symptoms: List[Symptom],
        historical: List[Consultation]
    ) -> List[Dict[str, Any]]:
        """
        For each current diagnosis, find which symptoms have historically
        been associated with it. This provides insight into the typical
        clinical presentation of the diagnosis for this patient.
        """
        associations = []

        for dx in current_diagnoses:
            # Collect all historical consultations with matching diagnoses
            symptom_counter: Dict[str, int] = Counter()
            symptom_details: List[Dict[str, Any]] = []

            for hist_cons in historical:
                for hist_dx in hist_cons.diagnoses:
                    score = self._fuzzy_match_diagnosis(dx, hist_dx)
                    if score >= self.similarity_threshold:
                        # Record all symptoms from this consultation
                        for hist_symptom in hist_cons.symptoms:
                            symptom_counter[hist_symptom.name] += 1
                            symptom_details.append({
                                "symptom": hist_symptom.name,
                                "severity": hist_symptom.severity,
                                "consultation_date":
                                    hist_cons.consultation_date.isoformat(),
                                "consultation_id": hist_cons.consultation_id
                            })

            if symptom_counter:
                # Most common associated symptoms
                common_symptoms = symptom_counter.most_common(10)
                # Check which current symptoms align
                current_symptom_names = {s.name.lower() for s in current_symptoms}
                aligned_symptoms = [
                    s_name for s_name in current_symptom_names
                    if any(s_name in hist.lower()
                           for hist, _ in common_symptoms)
                ]

                associations.append({
                    "diagnosis": dx.diagnosis_name,
                    "diagnosis_code": dx.diagnosis_code,
                    "historically_associated_symptoms": [
                        {"symptom": name, "frequency": count}
                        for name, count in common_symptoms
                    ],
                    "currently_aligned_symptoms": aligned_symptoms,
                    "total_historical_instances": sum(symptom_counter.values())
                })

        return associations

    def _track_diagnosis_progression(
        self,
        current_diagnoses: List[ClinicalDiagnosis],
        historical: List[Consultation]
    ) -> List[Dict[str, Any]]:
        """
        Track how each diagnosis has changed or evolved over time.
        Identifies if diagnoses are persistent, evolving, or newly emerged.
        """
        progression = []

        for dx in current_diagnoses:
            timeline = []
            for hist_cons in historical:
                for hist_dx in hist_cons.diagnoses:
                    score = self._fuzzy_match_diagnosis(dx, hist_dx)
                    if score >= self.similarity_threshold:
                        timeline.append({
                            "date": hist_cons.consultation_date.isoformat(),
                            "diagnosis_name": hist_dx.diagnosis_name,
                            "diagnosis_code": hist_dx.diagnosis_code,
                            "diagnosis_type": hist_dx.diagnosis_type,
                            "notes": hist_dx.notes
                        })

            if timeline:
                # Sort by date
                timeline.sort(key=lambda x: x["date"])

                # Determine progression pattern
                unique_codes = set(t["diagnosis_code"]
                                   for t in timeline if t["diagnosis_code"])
                unique_names = list(set(t["diagnosis_name"] for t in timeline))

                if len(unique_codes) <= 1 and len(unique_names) <= 1:
                    pattern = "Persistent - Same diagnosis across visits"
                elif len(unique_names) > 1:
                    pattern = ("Evolving - Diagnosis description changed "
                               "over time")
                else:
                    pattern = "Recurrent - Same diagnosis with gaps"

                progression.append({
                    "current_diagnosis": dx.diagnosis_name,
                    "current_code": dx.diagnosis_code,
                    "timeline": timeline,
                    "first_recorded_date": timeline[0]["date"],
                    "latest_historical_date": timeline[-1]["date"],
                    "total_occurrences": len(timeline),
                    "progression_pattern": pattern,
                    "prior_diagnosis_names": unique_names
                })

        progression.sort(key=lambda x: x["total_occurrences"], reverse=True)
        return progression

    def _find_new_diagnoses(
        self,
        current_diagnoses: List[ClinicalDiagnosis],
        historical: List[Consultation]
    ) -> List[str]:
        """
        Identify diagnoses in the current consultation that have never
        appeared in any historical consultation.
        """
        new_diagnoses = []

        for dx in current_diagnoses:
            is_new = True
            for hist_cons in historical:
                for hist_dx in hist_cons.diagnoses:
                    score = self._fuzzy_match_diagnosis(dx, hist_dx)
                    if score >= self.similarity_threshold:
                        is_new = False
                        break
                if not is_new:
                    break
            if is_new:
                new_diagnoses.append(
                    f"{dx.diagnosis_name} ({dx.diagnosis_code})"
                )

        return new_diagnoses

    def _correlate_with_labs(
        self,
        current_diagnoses: List[ClinicalDiagnosis],
        current_labs: List[LaboratoryResult],
        historical: List[Consultation]
    ) -> List[Dict[str, Any]]:
        """
        Find correlations between diagnoses and laboratory findings.
        For example, a diagnosis of Diabetes Mellitus historically
        associated with elevated blood glucose.
        """
        lab_correlations = []

        for dx in current_diagnoses:
            # Collect all labs from consultations with this diagnosis
            associated_labs: Dict[str, List[float]] = defaultdict(list)
            lab_details: List[Dict[str, Any]] = []

            for hist_cons in historical:
                for hist_dx in hist_cons.diagnoses:
                    score = self._fuzzy_match_diagnosis(dx, hist_dx)
                    if score >= self.similarity_threshold:
                        for lab in hist_cons.laboratory_results:
                            associated_labs[lab.parameter_name].append(
                                lab.value
                            )
                            lab_details.append({
                                "consultation_date":
                                    hist_cons.consultation_date.isoformat(),
                                "test_name": lab.test_name,
                                "parameter": lab.parameter_name,
                                "value": lab.value,
                                "unit": lab.unit,
                                "flag": lab.flag,
                                "reference_range": (
                                    f"{lab.reference_range_low}-"
                                    f"{lab.reference_range_high}"
                                )
                            })

            if associated_labs:
                # Summarize lab patterns
                lab_summaries = []
                for param, values in associated_labs.items():
                    if values:
                        avg_value = sum(values) / len(values)
                        lab_summaries.append({
                            "parameter": param,
                            "historical_average": round(avg_value, 2),
                            "min_value": round(min(values), 2),
                            "max_value": round(max(values), 2),
                            "num_measurements": len(values)
                        })

                lab_correlations.append({
                    "diagnosis": dx.diagnosis_name,
                    "diagnosis_code": dx.diagnosis_code,
                    "lab_parameter_summaries": lab_summaries,
                    "all_lab_details": lab_details,
                    "total_associated_lab_records": len(lab_details)
                })

        return lab_correlations

    def _generate_summary(
        self,
        current_diagnoses: List[ClinicalDiagnosis],
        result: DiagnosisCorrelationResult
    ) -> str:
        """Generate a human-readable summary."""
        parts = []

        if result.recurring_diagnoses:
            names = [d["current_diagnosis"]
                     for d in result.recurring_diagnoses[:3]]
            parts.append(
                f"Found {len(result.recurring_diagnoses)} recurring "
                f"diagnosis(es): {', '.join(names)}. "
                f"These diagnoses have been recorded in previous consultations."
            )

        if result.new_diagnoses:
            parts.append(
                f"New diagnosis(es) detected: "
                f"{', '.join(result.new_diagnoses)}. "
                f"These have not been previously recorded and may "
                f"represent a new clinical development."
            )

        if result.diagnosis_progression:
            persistent = [d for d in result.diagnosis_progression
                          if "Persistent" in d["progression_pattern"]]
            evolving = [d for d in result.diagnosis_progression
                        if "Evolving" in d["progression_pattern"]]
            if persistent:
                parts.append(
                    f"{len(persistent)} diagnosis(es) show a persistent "
                    f"pattern, indicating a chronic condition."
                )
            if evolving:
                names = [d["current_diagnosis"] for d in evolving]
                parts.append(
                    f"Diagnosis evolution noted for: {', '.join(names)}."
                )

        if result.diagnosis_lab_correlations:
            parts.append(
                f"Laboratory correlations found for "
                f"{len(result.diagnosis_lab_correlations)} diagnosis(es), "
                f"linking current diagnoses with historical lab findings."
            )

        if not parts:
            parts.append(
                "No significant diagnosis correlations found between "
                "current and historical consultations."
            )

        return " ".join(parts)

    def _compute_confidence(
        self,
        current_diagnoses: List[ClinicalDiagnosis],
        historical: List[Consultation],
        result: DiagnosisCorrelationResult
    ) -> float:
        """Compute confidence score for diagnosis correlation results."""
        if not current_diagnoses or not historical:
            return 0.0

        # Factor 1: Historical depth (30%)
        history_factor = min(len(historical) / 8.0, 1.0) * 0.3

        # Factor 2: Match rate (40%)
        matched = len(result.recurring_diagnoses)
        match_rate = matched / max(len(current_diagnoses), 1)
        match_factor = match_rate * 0.4

        # Factor 3: Data quality - ICD-10 codes present (30%)
        has_codes = sum(1 for d in current_diagnoses if d.diagnosis_code)
        code_factor = (has_codes / max(len(current_diagnoses), 1)) * 0.3

        return round(min(history_factor + match_factor + code_factor, 1.0), 2)