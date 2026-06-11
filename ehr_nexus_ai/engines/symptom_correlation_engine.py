"""
EHR Nexus - Symptom Correlation Engine
========================================
Compares a patient's current symptoms with symptoms documented in
previous consultations to identify:
- Recurring symptoms
- Recurring symptom clusters (combinations that appear together)
- Progression or changes in symptom severity over time
- Symptom patterns that correlate with specific diagnoses
"""

from typing import List, Tuple, Dict, Optional, Any
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import re

from ..core.data_models import (
    PatientRecord, Consultation, Symptom, ClinicalDiagnosis
)


class SymptomCorrelationResult:
    """
    Encapsulates the output of the symptom correlation engine.
    Provides explainable results showing which historical records
    contributed to each correlation finding.
    """
    def __init__(self):
        # List of (symptom_name, match_count, match_details)
        self.recurring_symptoms: List[Dict[str, Any]] = []
        # List of recurring symptom clusters (groups of symptoms appearing together)
        self.recurring_clusters: List[Dict[str, Any]] = []
        # Severity progression tracking
        self.severity_changes: List[Dict[str, Any]] = []
        # New symptoms not seen in history
        self.new_symptoms: List[str] = []
        # Overall correlation summary
        self.summary: str = ""
        # Confidence score (0.0 - 1.0)
        self.confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recurring_symptoms": self.recurring_symptoms,
            "recurring_clusters": self.recurring_clusters,
            "severity_changes": self.severity_changes,
            "new_symptoms": self.new_symptoms,
            "summary": self.summary,
            "confidence_score": self.confidence_score
        }


class SymptomCorrelationEngine:
    """
    Rule-based engine for correlating current symptoms with historical
    patient symptoms. Uses string similarity, frequency analysis, and
    temporal pattern detection to generate explainable correlations.
    """

    # Common symptom synonyms and related terms for fuzzy matching
    SYMPTOM_SYNONYMS = {
        "headache": ["head pain", "cephalgia", "migraine", "head ache"],
        "fever": ["pyrexia", "high temperature", "elevated temp", "febrile"],
        "cough": ["coughing", "tussis", "hacking cough"],
        "nausea": ["sick to stomach", "feeling sick", "queasy"],
        "vomiting": ["throwing up", "emesis", "puking"],
        "dizziness": ["vertigo", "lightheaded", "spinning sensation"],
        "fatigue": ["tiredness", "exhaustion", "weakness", "lethargy"],
        "chest pain": ["chest discomfort", "angina", "chest tightness"],
        "shortness of breath": ["dyspnea", "SOB", "difficulty breathing"],
        "abdominal pain": ["stomach ache", "belly pain", "abdominal discomfort"],
        "back pain": ["backache", "lumbago", "spinal pain"],
        "sore throat": ["pharyngitis", "throat pain", "scratchy throat"],
        "rash": ["skin eruption", "hives", "dermatitis", "skin lesion"],
        "joint pain": ["arthralgia", "joint ache", "arthritis symptoms"],
        "swelling": ["edema", "inflammation", "puffiness"],
    }

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Args:
            similarity_threshold: Minimum similarity score (0.0-1.0)
                                  to consider two symptoms as matching.
        """
        self.similarity_threshold = similarity_threshold

    def correlate(
        self,
        record: PatientRecord,
        current_consultation: Optional[Consultation] = None
    ) -> SymptomCorrelationResult:
        """
        Main entry point: correlates current symptoms with historical symptoms.

        Args:
            record: The complete patient record with all consultations.
            current_consultation: The most recent consultation to analyze.
                                  If None, uses the latest consultation in the record.

        Returns:
            SymptomCorrelationResult with all findings.
        """
        result = SymptomCorrelationResult()

        if current_consultation is None:
            current_consultation = record.get_latest_consultation()

        if current_consultation is None:
            result.summary = "No consultation data available for correlation."
            return result

        historical = record.get_historical_consultations(exclude_latest=True)

        if not historical:
            result.summary = (
                "No historical consultation data available. "
                "This appears to be the patient's first visit."
            )
            result.confidence_score = 1.0
            return result

        current_symptoms = current_consultation.symptoms

        if not current_symptoms:
            result.summary = (
                "No symptoms recorded in the current consultation."
            )
            return result

        # 1. Detect recurring symptoms
        result.recurring_symptoms = self._find_recurring_symptoms(
            current_symptoms, historical
        )

        # 2. Detect recurring symptom clusters
        result.recurring_clusters = self._find_recurring_clusters(
            current_symptoms, historical
        )

        # 3. Detect severity changes
        result.severity_changes = self._track_severity_changes(
            current_symptoms, historical
        )

        # 4. Identify new symptoms
        result.new_symptoms = self._find_new_symptoms(
            current_symptoms, historical
        )

        # 5. Generate summary
        result.summary = self._generate_summary(
            current_symptoms, result
        )

        # 6. Compute confidence score
        result.confidence_score = self._compute_confidence(
            current_symptoms, historical, result
        )

        return result

    def _fuzzy_match_symptom(self, symptom_name_a: str,
                              symptom_name_b: str) -> float:
        """
        Fuzzy match two symptom names using:
        - Exact match (fast)
        - Synonym lookup
        - SequenceMatcher for partial matching
        """
        a_lower = symptom_name_a.lower().strip()
        b_lower = symptom_name_b.lower().strip()

        # Exact match
        if a_lower == b_lower:
            return 1.0

        # Synonym match
        for canonical, synonyms in self.SYMPTOM_SYNONYMS.items():
            if (a_lower == canonical or a_lower in synonyms) and \
               (b_lower == canonical or b_lower in synonyms):
                return 1.0

        # Sequence matcher
        return SequenceMatcher(None, a_lower, b_lower).ratio()

    def _match_symptom_in_list(
        self, symptom: Symptom, symptom_list: List[Symptom]
    ) -> List[Tuple[Symptom, float]]:
        """
        Find all matches for a symptom within a list of symptoms.
        Returns list of (matched_symptom, similarity_score).
        """
        matches = []
        for hist_symptom in symptom_list:
            score = self._fuzzy_match_symptom(
                symptom.name, hist_symptom.name
            )
            if score >= self.similarity_threshold:
                matches.append((hist_symptom, score))
        return matches

    def _find_recurring_symptoms(
        self,
        current_symptoms: List[Symptom],
        historical: List[Consultation]
    ) -> List[Dict[str, Any]]:
        """
        For each current symptom, check if it appeared in past consultations.
        Returns detailed match information.
        """
        recurring = []

        for symptom in current_symptoms:
            # Collect all historical matches
            matches = []  # list of (consultation_date, symptom, score)
            for hist_cons in historical:
                for hist_symptom in hist_cons.symptoms:
                    score = self._fuzzy_match_symptom(
                        symptom.name, hist_symptom.name
                    )
                    if score >= self.similarity_threshold:
                        matches.append({
                            "consultation_id": hist_cons.consultation_id,
                            "consultation_date": (
                                hist_cons.consultation_date.isoformat()
                            ),
                            "historical_symptom": hist_symptom.name,
                            "historical_severity": hist_symptom.severity,
                            "match_score": round(score, 3)
                        })

            if matches:
                recurring.append({
                    "current_symptom": symptom.name,
                    "current_severity": symptom.severity,
                    "current_duration": symptom.duration,
                    "historical_matches": matches,
                    "match_count": len(matches),
                    "unique_consultations": len(set(
                        m["consultation_id"] for m in matches
                    ))
                })

        # Sort by match count (most recurrent first)
        recurring.sort(key=lambda x: x["match_count"], reverse=True)
        return recurring

    def _find_recurring_clusters(
        self,
        current_symptoms: List[Symptom],
        historical: List[Consultation]
    ) -> List[Dict[str, Any]]:
        """
        Identify clusters of symptoms that appear together in current and
        historical consultations. A cluster is 2+ symptoms that co-occur.
        """
        current_names = set(s.name.lower().strip() for s in current_symptoms)
        cluster_counts = Counter()

        # Map cluster to the consultations where it appeared
        cluster_consultations: Dict[Tuple[str, ...], List[str]] = defaultdict(list)

        for hist_cons in historical:
            hist_names = set(s.name.lower().strip()
                             for s in hist_cons.symptoms)
            # Find overlap between current and historical symptoms
            overlap = current_names & hist_names
            if len(overlap) >= 2:
                cluster_key = tuple(sorted(overlap))
                cluster_counts[cluster_key] += 1
                cluster_consultations[cluster_key].append(
                    hist_cons.consultation_id
                )

        recurring_clusters = []
        for cluster, count in cluster_counts.most_common(10):
            if count >= 1:
                recurring_clusters.append({
                    "symptoms_in_cluster": list(cluster),
                    "co_occurrence_count": count,
                    "consultation_ids": list(set(
                        cluster_consultations[cluster]
                    ))
                })

        return recurring_clusters

    def _track_severity_changes(
        self,
        current_symptoms: List[Symptom],
        historical: List[Consultation]
    ) -> List[Dict[str, Any]]:
        """
        Track how symptom severity has changed across visits.
        """
        severity_map = {"Mild": 1, "Moderate": 2, "Severe": 3}
        changes = []

        for symptom in current_symptoms:
            # Get all historical occurrences with severity
            history = []  # (date, severity_score)
            for hist_cons in historical:
                for hist_symptom in hist_cons.symptoms:
                    score = self._fuzzy_match_symptom(
                        symptom.name, hist_symptom.name
                    )
                    if score >= self.similarity_threshold and hist_symptom.severity:
                        sev_score = severity_map.get(hist_symptom.severity, 0)
                        history.append({
                            "date": hist_cons.consultation_date.isoformat(),
                            "severity": hist_symptom.severity,
                            "severity_score": sev_score
                        })

            if history:
                # Sort by date
                history.sort(key=lambda x: x["date"])
                current_sev = severity_map.get(symptom.severity, 0)

                # Determine trend (compare current to most recent historical)
                if len(history) >= 1:
                    last = history[-1]["severity_score"]
                    if current_sev > last:
                        trend = "Worsening"
                    elif current_sev < last:
                        trend = "Improving"
                    else:
                        trend = "Stable"
                else:
                    trend = "Insufficient data"

                changes.append({
                    "symptom": symptom.name,
                    "current_severity": symptom.severity or "Not specified",
                    "current_severity_score": current_sev,
                    "historical_severity_timeline": history,
                    "trend": trend
                })

        changes.sort(key=lambda x: x["current_severity_score"], reverse=True)
        return changes

    def _find_new_symptoms(
        self,
        current_symptoms: List[Symptom],
        historical: List[Consultation]
    ) -> List[str]:
        """
        Identify symptoms in the current consultation that have never
        appeared in any historical consultation.
        """
        # Collect all historical symptom names
        all_historical_names = set()
        for hist_cons in historical:
            for hist_symptom in hist_cons.symptoms:
                all_historical_names.add(hist_symptom.name.lower().strip())

        new_symptoms = []
        for symptom in current_symptoms:
            is_new = True
            for hist_name in all_historical_names:
                score = self._fuzzy_match_symptom(symptom.name, hist_name)
                if score >= self.similarity_threshold:
                    is_new = False
                    break
            if is_new:
                new_symptoms.append(symptom.name)

        return new_symptoms

    def _generate_summary(
        self,
        current_symptoms: List[Symptom],
        result: SymptomCorrelationResult
    ) -> str:
        """Generate a human-readable summary of correlation findings."""
        parts = []

        if result.recurring_symptoms:
            top_recurring = result.recurring_symptoms[:3]
            recurring_names = [r["current_symptom"]
                               for r in top_recurring]
            parts.append(
                f"Found {len(result.recurring_symptoms)} recurring symptom(s): "
                f"{', '.join(recurring_names)}. "
                f"These symptoms have appeared in previous consultations, "
                f"suggesting a chronic or recurrent pattern."
            )

        if result.recurring_clusters:
            parts.append(
                f"Detected {len(result.recurring_clusters)} recurring symptom "
                f"cluster(s) that co-occur in current and historical visits."
            )

        if result.severity_changes:
            worsening = [s for s in result.severity_changes
                         if s["trend"] == "Worsening"]
            improving = [s for s in result.severity_changes
                         if s["trend"] == "Improving"]
            if worsening:
                names = [s["symptom"] for s in worsening]
                parts.append(
                    f"[WARNING] Severity worsening: {', '.join(names)} show(s) worsening "
                    f"severity compared to previous visits."
                )
            if improving:
                names = [s["symptom"] for s in improving]
                parts.append(
                    f"[IMPROVEMENT] Severity improvement noted for: {', '.join(names)}."
                )

        if result.new_symptoms:
            parts.append(
                f"New symptom(s) detected: {', '.join(result.new_symptoms)}. "
                f"These symptoms were not reported in prior consultations "
                f"and may warrant additional clinical attention."
            )

        if not parts:
            parts.append(
                "No significant symptom correlations found between "
                "current and historical consultations."
            )

        return " ".join(parts)

    def _compute_confidence(
        self,
        current_symptoms: List[Symptom],
        historical: List[Consultation],
        result: SymptomCorrelationResult
    ) -> float:
        """
        Compute a confidence score for the correlation results.
        Factors considered:
        - Number of historical consultations available
        - Proportion of current symptoms that matched historical ones
        - Number of data points (severity, duration) available
        """
        if not current_symptoms or not historical:
            return 0.0

        # Factor 1: Historical data richness (30%)
        history_factor = min(len(historical) / 10.0, 1.0) * 0.3

        # Factor 2: Match rate (50%)
        matched_count = sum(len(r["historical_matches"])
                            for r in result.recurring_symptoms)
        total_possible = len(current_symptoms) * len(historical)
        match_rate = min(matched_count / max(total_possible, 1), 1.0)
        match_factor = match_rate * 0.5

        # Factor 3: Data completeness (20%)
        has_severity = sum(1 for s in current_symptoms if s.severity)
        severity_factor = (has_severity / max(len(current_symptoms), 1)) * 0.2

        return round(min(history_factor + match_factor + severity_factor, 1.0), 2)