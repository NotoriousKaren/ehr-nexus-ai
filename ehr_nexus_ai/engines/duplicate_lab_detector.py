"""
EHR Nexus - Duplicate Laboratory Detection Engine
===================================================
Detects possible duplicate uploads of laboratory results by comparing:
- File hashes (MD5/SHA256) if available
- Test name, parameter name, value, and unit combinations
- Upload timestamps within a configurable time window
- Similarity scoring for near-duplicate detection
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import hashlib

from ..core.data_models import (
    PatientRecord, Consultation, LaboratoryResult
)


class DuplicateLabDetectionResult:
    """Encapsulates duplicate lab detection outputs."""
    def __init__(self):
        self.exact_duplicates: List[Dict[str, Any]] = []
        self.near_duplicates: List[Dict[str, Any]] = []
        self.duplicate_count: int = 0
        self.unique_results: int = 0
        self.summary: str = ""
        self.flagged_for_review: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "duplicate_count": self.duplicate_count,
            "unique_results": self.unique_results,
            "summary": self.summary,
            "flagged_for_review": self.flagged_for_review
        }


class DuplicateLabDetector:
    """
    Engine for detecting duplicate laboratory uploads.
    Uses multiple checks to identify both exact and near-duplicate results.
    """

    def __init__(
        self,
        time_window_hours: int = 24,
        value_similarity_threshold: float = 0.95,
        name_similarity_threshold: float = 0.90
    ):
        """
        Args:
            time_window_hours: Maximum hours between uploads to consider
                               as potential duplicates.
            value_similarity_threshold: Threshold (0-1) for matching values.
            name_similarity_threshold: Threshold for matching test names.
        """
        self.time_window_hours = time_window_hours
        self.value_similarity_threshold = value_similarity_threshold
        self.name_similarity_threshold = name_similarity_threshold

    def detect(
        self,
        record: PatientRecord,
        new_labs: List[LaboratoryResult],
        consultation_id: Optional[str] = None
    ) -> DuplicateLabDetectionResult:
        """
        Detect duplicates among newly uploaded lab results compared to
        all existing lab results in the patient's record.

        Args:
            record: Complete patient record.
            new_labs: List of newly uploaded laboratory results.
            consultation_id: Current consultation ID for context.

        Returns:
            DuplicateLabDetectionResult with all findings.
        """
        result = DuplicateLabDetectionResult()

        if not new_labs:
            result.summary = "No laboratory results to check for duplicates."
            return result

        # Collect all existing lab results
        existing_labs = record.get_all_lab_results()

        if not existing_labs:
            result.summary = (
                "No historical laboratory results to compare against."
            )
            result.unique_results = len(new_labs)
            return result

        # Check each new lab against all existing labs
        for new_lab in new_labs:
            best_match = self._find_best_match(new_lab, existing_labs)

            if best_match:
                match_type, match_info = best_match
                if match_type == "exact":
                    result.exact_duplicates.append(match_info)
                    result.duplicate_count += 1
                else:
                    result.near_duplicates.append(match_info)
                    result.duplicate_count += 1
            else:
                result.unique_results += 1

        # Flag suspicious patterns
        result.flagged_for_review = self._flag_suspicious_patterns(
            new_labs, existing_labs, result
        )

        # Generate summary
        result.summary = self._generate_summary(
            result, len(new_labs)
        )

        return result

    def _compute_file_hash(
        self, lab: LaboratoryResult
    ) -> Optional[str]:
        """
        Compute a content-based hash from lab result fields.
        This serves as a fingerprint when file_hash is not provided.
        """
        # Create a deterministic string from key fields
        content = (
            f"{lab.test_name}:{lab.parameter_name}:"
            f"{lab.value}:{lab.unit}:"
            f"{lab.reference_range_low}:{lab.reference_range_high}"
        ).encode("utf-8")
        return hashlib.md5(content).hexdigest()

    def _find_best_match(
        self,
        new_lab: LaboratoryResult,
        existing_labs: List[LaboratoryResult]
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Find the best matching existing lab for a new lab result.
        Returns (match_type, match_info) or None if no match found.
        """
        best_score = 0.0
        best_match_info = None

        for existing_lab in existing_labs:
            # Check 1: File hash exact match
            hash_match = (
                new_lab.file_hash and existing_lab.file_hash and
                new_lab.file_hash == existing_lab.file_hash
            )
            # Compute content hash if no file hash available
            content_hash_new = self._compute_file_hash(new_lab)
            content_hash_existing = self._compute_file_hash(existing_lab)
            content_hash_match = content_hash_new == content_hash_existing

            if hash_match or content_hash_match:
                return ("exact", {
                    "new_lab_id": new_lab.lab_result_id,
                    "duplicate_of_id": existing_lab.lab_result_id,
                    "match_type": "hash_match",
                    "test_name": new_lab.test_name,
                    "parameter": new_lab.parameter_name,
                    "value": new_lab.value,
                    "unit": new_lab.unit,
                    "confidence": 1.0,
                    "explanation": (
                        "Identical content hash detected. "
                        "This appears to be the same result uploaded again."
                    )
                })

            # Check 2: Same test + same parameter + same value + same unit
            if (
                new_lab.test_name == existing_lab.test_name and
                new_lab.parameter_name == existing_lab.parameter_name and
                abs(new_lab.value - existing_lab.value) < 0.001 and
                new_lab.unit == existing_lab.unit
            ):
                # Check time window
                if (new_lab.uploaded_at and existing_lab.uploaded_at and
                    abs((new_lab.uploaded_at -
                         existing_lab.uploaded_at).total_seconds())
                    <= self.time_window_hours * 3600):
                    return ("exact", {
                        "new_lab_id": new_lab.lab_result_id,
                        "duplicate_of_id": existing_lab.lab_result_id,
                        "match_type": "exact_value_match",
                        "test_name": new_lab.test_name,
                        "parameter": new_lab.parameter_name,
                        "value": new_lab.value,
                        "unit": new_lab.unit,
                        "existing_uploaded_at":
                            existing_lab.uploaded_at.isoformat()
                            if existing_lab.uploaded_at else None,
                        "new_uploaded_at":
                            new_lab.uploaded_at.isoformat()
                            if new_lab.uploaded_at else None,
                        "confidence": 0.98,
                        "explanation": (
                            "Exact same test, parameter, value, and unit "
                            "found in existing records within the "
                            f"{self.time_window_hours}-hour time window."
                        )
                    })

            # Check 3: Near-duplicate (fuzzy match)
            score = self._compute_similarity_score(new_lab, existing_lab)
            if score > best_score:
                best_score = score
                best_match_info = {
                    "new_lab_id": new_lab.lab_result_id,
                    "duplicate_of_id": existing_lab.lab_result_id,
                    "match_type": "near_duplicate",
                    "test_name": new_lab.test_name,
                    "parameter": new_lab.parameter_name,
                    "new_value": new_lab.value,
                    "existing_value": existing_lab.value,
                    "unit": new_lab.unit,
                    "similarity_score": round(score, 3),
                    "confidence": round(score, 3),
                    "explanation": (
                        f"Similar result found: "
                        f"test='{new_lab.test_name}', "
                        f"parameter='{new_lab.parameter_name}', "
                        f"new value={new_lab.value}, "
                        f"existing value={existing_lab.value}. "
                        f"Similarity score: {score:.1%}."
                    )
                }

        # Return best near-duplicate if above threshold and not exact
        if (best_score >= self.value_similarity_threshold and
            best_match_info):
            return ("near_duplicate", best_match_info)

        return None

    def _compute_similarity_score(
        self,
        lab_a: LaboratoryResult,
        lab_b: LaboratoryResult
    ) -> float:
        """
        Compute a similarity score (0-1) between two lab results.
        Combines multiple factors:
        - Test name similarity
        - Parameter name similarity
        - Value proximity
        - Unit match
        """
        scores = []
        weights = []

        # Test name similarity (weight: 0.25)
        name_score = SequenceMatcher(
            None,
            lab_a.test_name.lower(),
            lab_b.test_name.lower()
        ).ratio()
        scores.append(name_score)
        weights.append(0.25)

        # Parameter name similarity (weight: 0.25)
        param_score = SequenceMatcher(
            None,
            lab_a.parameter_name.lower(),
            lab_b.parameter_name.lower()
        ).ratio()
        scores.append(param_score)
        weights.append(0.25)

        # Unit match (weight: 0.15)
        unit_score = 1.0 if lab_a.unit == lab_b.unit else 0.0
        scores.append(unit_score)
        weights.append(0.15)

        # Value proximity (weight: 0.35)
        if lab_a.unit == lab_b.unit and lab_b.value != 0:
            ratio = min(lab_a.value, lab_b.value) / max(lab_a.value, lab_b.value)
            value_score = ratio
        else:
            value_score = 0.0
        scores.append(value_score)
        weights.append(0.35)

        # Weighted average
        total_score = sum(s * w for s, w in zip(scores, weights))
        return total_score

    def _flag_suspicious_patterns(
        self,
        new_labs: List[LaboratoryResult],
        existing_labs: List[LaboratoryResult],
        result: DuplicateLabDetectionResult
    ) -> List[Dict[str, Any]]:
        """
        Flag suspicious patterns that may indicate systematic duplication
        or data entry issues.
        """
        flags = []

        # Check if all labs from a new upload are duplicates
        if result.duplicate_count > 0:
            duplicate_rate = result.duplicate_count / len(new_labs)
            if duplicate_rate >= 0.8:
                flags.append({
                    "severity": "HIGH",
                    "pattern": "Bulk Duplicate Detection",
                    "description": (
                        f"{result.duplicate_count} out of {len(new_labs)} "
                        f"uploaded results ({duplicate_rate:.0%}) are "
                        f"duplicates. Possible bulk re-upload."
                    ),
                    "recommendation": (
                        "Verify if the entire upload was intentional "
                        "or a system error."
                    )
                })

        # Check for same test uploaded multiple times with slight variations
        test_counts: Dict[str, int] = {}
        for lab in new_labs:
            key = f"{lab.test_name}:{lab.parameter_name}"
            test_counts[key] = test_counts.get(key, 0) + 1

        for key, count in test_counts.items():
            if count > 1:
                flags.append({
                    "severity": "MEDIUM",
                    "pattern": "Multiple Similar Results",
                    "description": (
                        f"'{key}' uploaded {count} times in this batch. "
                        f"May indicate unintended duplication."
                    ),
                    "recommendation": (
                        "Review the upload batch for unintentional duplicates."
                    )
                })

        return flags

    def _generate_summary(
        self,
        result: DuplicateLabDetectionResult,
        total_new: int
    ) -> str:
        """Generate a human-readable summary."""
        parts = []

        if result.duplicate_count > 0:
            parts.append(
                f"Found {result.duplicate_count} potential duplicate(s) "
                f"out of {total_new} uploaded results "
                f"({result.duplicate_count/total_new*100:.0f}% duplicate rate)."
            )

            if result.exact_duplicates:
                parts.append(
                    f"Exact duplicates: {len(result.exact_duplicates)}."
                )
            if result.near_duplicates:
                parts.append(
                    f"Near-duplicates (similar results): "
                    f"{len(result.near_duplicates)}."
                )

            parts.append(
                f"Unique results retained: {result.unique_results}."
            )
        else:
            parts.append(
                f"All {total_new} uploaded results appear to be unique. "
                "No duplicates detected."
            )

        if result.flagged_for_review:
            high_flags = [f for f in result.flagged_for_review
                          if f["severity"] == "HIGH"]
            if high_flags:
                parts.append(
                    f"⚠ {len(high_flags)} high-severity flag(s) raised "
                    f"requiring attention."
                )

        return " ".join(parts)