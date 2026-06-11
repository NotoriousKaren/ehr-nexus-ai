"""
EHR Nexus - Explainable Correlation Engine
============================================
Produces explanations for how correlation conclusions are generated.
Every output includes:
- Which historical records contributed to the finding
- What specific data was matched (symptoms, diagnoses, labs, etc.)
- The matching methodology (exact match, fuzzy match, statistical)
- Confidence scores for each correlation
- Source references back to original consultations
"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import json

from ..core.data_models import (
    PatientRecord, Consultation, Symptom, ClinicalDiagnosis,
    LaboratoryResult, Medication, DoctorOrder
)


class ExplanationNode:
    """
    A single node in the explanation tree.
    Each node represents one correlation finding with full provenance.
    """
    def __init__(
        self,
        finding_type: str,         # e.g., "symptom_correlation", "lab_trend"
        description: str,          # Human-readable description
        confidence: float,         # 0.0 - 1.0
        source_consultations: List[str],  # IDs of contributing consultations
        evidence: List[Dict[str, Any]],   # Specific data points used
        methodology: str,          # How the conclusion was reached
        severity: str = "INFO"     # INFO, WARNING, CRITICAL
    ):
        self.finding_type = finding_type
        self.description = description
        self.confidence = confidence
        self.source_consultations = source_consultations
        self.evidence = evidence
        self.methodology = methodology
        self.severity = severity
        self.children: List["ExplanationNode"] = []
        self.timestamp = datetime.now()

    def add_child(self, child: "ExplanationNode") -> None:
        """Add a child explanation node for hierarchical detail."""
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_type": self.finding_type,
            "description": self.description,
            "confidence": self.confidence,
            "severity": self.severity,
            "source_consultations": self.source_consultations,
            "evidence": self.evidence,
            "methodology": self.methodology,
            "children": [c.to_dict() for c in self.children],
            "timestamp": self.timestamp.isoformat()
        }


class ExplainableCorrelationReport:
    """
    Complete explainable report that aggregates findings from all engines.
    Every conclusion is traceable back to its source data.
    """
    def __init__(self):
        self.patient_id: str = ""
        self.current_consultation_id: str = ""
        self.analysis_timestamp: str = ""
        self.nodes: List[ExplanationNode] = []
        self.overall_confidence: float = 0.0
        self.executive_summary: str = ""

    def add_node(self, node: ExplanationNode) -> None:
        self.nodes.append(node)

    @property
    def total_findings(self) -> int:
        return len(self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "current_consultation_id": self.current_consultation_id,
            "analysis_timestamp": self.analysis_timestamp,
            "overall_confidence": self.overall_confidence,
            "executive_summary": self.executive_summary,
            "findings": [n.to_dict() for n in self.nodes],
            "total_findings": len(self.nodes)
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ExplainableEngine:
    """
    Orchestrates the creation of explainable outputs by wrapping
    each engine's results with provenance data.
    """

    def __init__(self, record: PatientRecord):
        self.record = record

    def build_report(
        self,
        symptom_result: Any = None,
        diagnosis_result: Any = None,
        lab_result: Any = None,
        duplicate_result: Any = None,
        medication_result: Any = None,
        current_consultation: Optional[Consultation] = None
    ) -> ExplainableCorrelationReport:
        """
        Build a comprehensive explainable report from all engine results.

        Args:
            symptom_result: Output from SymptomCorrelationEngine
            diagnosis_result: Output from DiagnosisCorrelationEngine
            lab_result: Output from LabTrendAnalysisEngine
            duplicate_result: Output from DuplicateLabDetector
            medication_result: Output from MedicationReminderEngine
            current_consultation: Current consultation being analyzed

        Returns:
            ExplainableCorrelationReport with full provenance.
        """
        report = ExplainableCorrelationReport()

        if current_consultation is None:
            current_consultation = self.record.get_latest_consultation()

        if current_consultation:
            report.patient_id = self.record.patient.patient_id
            report.current_consultation_id = current_consultation.consultation_id
        else:
            report.patient_id = self.record.patient.patient_id
            report.current_consultation_id = "N/A"

        report.analysis_timestamp = datetime.now().isoformat()

        # Build explanation nodes from each engine's results
        if symptom_result:
            self._add_symptom_nodes(report, symptom_result)

        if diagnosis_result:
            self._add_diagnosis_nodes(report, diagnosis_result)

        if lab_result:
            self._add_lab_nodes(report, lab_result)

        if duplicate_result:
            self._add_duplicate_nodes(report, duplicate_result)

        if medication_result:
            self._add_medication_nodes(report, medication_result)

        # Compute overall confidence
        if report.nodes:
            report.overall_confidence = round(
                sum(n.confidence for n in report.nodes) / len(report.nodes),
                2
            )
        else:
            report.overall_confidence = 0.0

        # Generate executive summary
        report.executive_summary = self._generate_executive_summary(report)

        return report

    def _add_symptom_nodes(
        self,
        report: ExplainableCorrelationReport,
        result: Any
    ) -> None:
        """Add symptom correlation explanation nodes."""
        if not hasattr(result, 'recurring_symptoms'):
            return

        if result.recurring_symptoms:
            node = ExplanationNode(
                finding_type="symptom_correlation",
                description=(
                    f"Found {len(result.recurring_symptoms)} recurring "
                    f"symptom(s) when comparing current visit with "
                    f"historical consultations."
                ),
                confidence=result.confidence_score,
                source_consultations=list(set(
                    m["consultation_id"]
                    for r in result.recurring_symptoms
                    for m in r["historical_matches"]
                )),
                evidence=[
                    {
                        "symptom": r["current_symptom"],
                        "current_severity": r.get("current_severity", ""),
                        "match_count": r["match_count"],
                        "unique_consultations": r["unique_consultations"]
                    }
                    for r in result.recurring_symptoms[:10]
                ],
                methodology=(
                    "Fuzzy string matching with synonym resolution "
                    f"(threshold >= {0.7}) across all historical "
                    "consultations. Each current symptom is compared "
                    "against all historical symptom records."
                ),
                severity="WARNING" if len(result.recurring_symptoms) > 3 else "INFO"
            )

            # Add severity change children
            if hasattr(result, 'severity_changes') and result.severity_changes:
                severity_node = ExplanationNode(
                    finding_type="symptom_severity_trend",
                    description=(
                        f"{len(result.severity_changes)} symptom(s) show "
                        f"severity changes over time."
                    ),
                    confidence=0.8,
                    source_consultations=[],
                    evidence=[
                        {
                            "symptom": s["symptom"],
                            "trend": s["trend"],
                            "current_severity": s["current_severity"]
                        }
                        for s in result.severity_changes
                    ],
                    methodology=(
                        "Severity mapped to numeric scale (Mild=1, "
                        "Moderate=2, Severe=3). Trend determined by "
                        "comparing current severity to historical average."
                    )
                )
                node.add_child(severity_node)

            report.add_node(node)

        # New symptoms node
        if hasattr(result, 'new_symptoms') and result.new_symptoms:
            new_node = ExplanationNode(
                finding_type="new_symptoms_detected",
                description=(
                    f"{len(result.new_symptoms)} new symptom(s) detected "
                    f"that were not present in any historical consultation."
                ),
                confidence=0.95,
                source_consultations=[],
                evidence=[{"symptom": s} for s in result.new_symptoms],
                methodology=(
                    "Each current symptom is checked against all historical "
                    "symptoms using fuzzy matching. Symptom with no match "
                    "above threshold is classified as 'new'."
                ),
                severity="WARNING" if len(result.new_symptoms) > 2 else "INFO"
            )
            report.add_node(new_node)

    def _add_diagnosis_nodes(
        self,
        report: ExplainableCorrelationReport,
        result: Any
    ) -> None:
        """Add diagnosis correlation explanation nodes."""
        if not hasattr(result, 'recurring_diagnoses'):
            return

        if result.recurring_diagnoses:
            node = ExplanationNode(
                finding_type="diagnosis_correlation",
                description=(
                    f"Found {len(result.recurring_diagnoses)} recurring "
                    f"diagnosis(es). These diagnoses have been recorded "
                    f"in previous consultations."
                ),
                confidence=result.confidence_score,
                source_consultations=list(set(
                    m["consultation_id"]
                    for r in result.recurring_diagnoses
                    for m in r["historical_matches"]
                )),
                evidence=[
                    {
                        "diagnosis": r["current_diagnosis"],
                        "code": r.get("current_code", ""),
                        "category": r.get("category", ""),
                        "match_count": r["match_count"]
                    }
                    for r in result.recurring_diagnoses[:10]
                ],
                methodology=(
                    "Diagnoses matched using ICD-10 code comparison "
                    "(exact and category-level) and fuzzy name matching "
                    f"(threshold >= {0.75})."
                ),
                severity="INFO"
            )

            # Add progression child
            if hasattr(result, 'diagnosis_progression') and result.diagnosis_progression:
                prog_node = ExplanationNode(
                    finding_type="diagnosis_progression",
                    description=(
                        f"{len(result.diagnosis_progression)} diagnosis(es) "
                        f"show identifiable progression patterns."
                    ),
                    confidence=0.85,
                    source_consultations=[],
                    evidence=[
                        {
                            "diagnosis": p["current_diagnosis"],
                            "pattern": p["progression_pattern"],
                            "occurrences": p["total_occurrences"]
                        }
                        for p in result.diagnosis_progression
                    ],
                    methodology=(
                        "Progression pattern determined by analyzing "
                        "temporal sequence of diagnosis codes and names "
                        "across all consultations."
                    )
                )
                node.add_child(prog_node)

            report.add_node(node)

        if hasattr(result, 'new_diagnoses') and result.new_diagnoses:
            new_node = ExplanationNode(
                finding_type="new_diagnosis_detected",
                description=(
                    f"New diagnosis(es): {', '.join(result.new_diagnoses)}. "
                    f"These have not been previously recorded."
                ),
                confidence=0.95,
                source_consultations=[],
                evidence=[
                    {"diagnosis": d} for d in result.new_diagnoses
                ],
                methodology=(
                    "Each current diagnosis compared against all historical "
                    "diagnoses using ICD-10 and name matching. No match = new."
                ),
                severity="WARNING"
            )
            report.add_node(new_node)

    def _add_lab_nodes(
        self,
        report: ExplainableCorrelationReport,
        result: Any
    ) -> None:
        """Add laboratory analysis explanation nodes."""
        if not hasattr(result, 'trends_by_parameter'):
            return

        if result.trends_by_parameter:
            node = ExplanationNode(
                finding_type="lab_trend_analysis",
                description=(
                    f"Analyzed trends for {len(result.trends_by_parameter)} "
                    f"laboratory parameter(s). "
                    f"{len(result.significant_changes)} significant "
                    f"change(s) detected."
                ),
                confidence=result.confidence_score,
                source_consultations=list(set(
                    t["value_timeline"][0]["date"]
                    for t in result.trends_by_parameter
                    for _ in t["value_timeline"]
                )) if result.trends_by_parameter else [],
                evidence=[
                    {
                        "parameter": t["parameter"],
                        "direction": t["trend"]["direction"],
                        "mean": t["statistics"]["mean"],
                        "std_dev": t["statistics"]["standard_deviation"],
                        "measurements": t["number_of_measurements"]
                    }
                    for t in result.trends_by_parameter[:10]
                ],
                methodology=(
                    "Statistical trend analysis using mean, standard "
                    "deviation, slope estimation, and z-score calculation "
                    f"(threshold: |z| >= {2.0}). Values compared against "
                    "laboratory reference ranges."
                ),
                severity="WARNING" if result.significant_changes else "INFO"
            )

            # Add recurring abnormalities child
            if hasattr(result, 'recurring_abnormalities') and result.recurring_abnormalities:
                abnormal_node = ExplanationNode(
                    finding_type="recurring_abnormalities",
                    description=(
                        f"{len(result.recurring_abnormalities)} parameter(s) "
                        f"show recurring abnormal findings."
                    ),
                    confidence=0.9,
                    source_consultations=[],
                    evidence=[
                        {
                            "parameter": a["parameter"],
                            "abnormal_rate": a["abnormal_rate"],
                            "most_common_flag": a["most_common_flag"],
                            "currently_abnormal": a["currently_abnormal"]
                        }
                        for a in result.recurring_abnormalities[:10]
                    ],
                    methodology=(
                        "Parameter flagged as 'recurring abnormal' if "
                        "abnormal values appeared in 2+ consultations."
                    )
                )
                node.add_child(abnormal_node)

            report.add_node(node)

    def _add_duplicate_nodes(
        self,
        report: ExplainableCorrelationReport,
        result: Any
    ) -> None:
        """Add duplicate detection explanation nodes."""
        if not hasattr(result, 'duplicate_count'):
            return

        if result.duplicate_count > 0:
            node = ExplanationNode(
                finding_type="duplicate_lab_detection",
                description=(
                    f"Found {result.duplicate_count} potential duplicate "
                    f"laboratory result(s) out of the uploaded batch."
                ),
                confidence=min(
                    len(result.exact_duplicates) * 0.98 +
                    len(result.near_duplicates) * 0.85,
                    0.99
                ) if (result.exact_duplicates or result.near_duplicates) else 0.5,
                source_consultations=[],
                evidence=[
                    {
                        "test_name": d.get("test_name", ""),
                        "parameter": d.get("parameter", ""),
                        "match_type": d.get("match_type", ""),
                        "confidence": d.get("confidence", 0)
                    }
                    for d in (result.exact_duplicates +
                              result.near_duplicates)[:10]
                ],
                methodology=(
                    "Three-level duplicate detection: (1) File hash "
                    "comparison, (2) Exact value/same-parameter matching "
                    "within time window, (3) Weighted similarity scoring "
                    "for near-duplicates."
                ),
                severity="WARNING" if result.duplicate_count > 2 else "INFO"
            )
            report.add_node(node)

    def _add_medication_nodes(
        self,
        report: ExplainableCorrelationReport,
        result: Any
    ) -> None:
        """Add medication reminder explanation nodes."""
        if not hasattr(result, 'overdue_medications'):
            return

        if result.overdue_medications:
            node = ExplanationNode(
                finding_type="medication_overdue",
                description=(
                    f"{len(result.overdue_medications)} medication(s) "
                    f"are overdue and require attention."
                ),
                confidence=0.95,
                source_consultations=[],
                evidence=[
                    {
                        "medication": m["medication_name"],
                        "dosage": m.get("dosage", ""),
                        "severity": m.get("severity", "MEDIUM"),
                        "reasons": m.get("overdue_reasons", [])
                    }
                    for m in result.overdue_medications
                ],
                methodology=(
                    "Overdue detection based on: (1) explicit overdue flag, "
                    "(2) past end_date beyond grace period, "
                    "(3) past refill_date beyond grace period."
                ),
                severity="CRITICAL" if any(
                    m.get("severity") == "HIGH"
                    for m in result.overdue_medications
                ) else "WARNING"
            )
            report.add_node(node)

        if result.pending_doctor_orders:
            order_node = ExplanationNode(
                finding_type="pending_doctor_orders",
                description=(
                    f"{len(result.pending_doctor_orders)} doctor's "
                    f"order(s) currently pending."
                ),
                confidence=0.95,
                source_consultations=[],
                evidence=[
                    {
                        "order_type": o["order_type"],
                        "description": o["description"],
                        "days_pending": o.get("days_pending", 0),
                        "severity": o.get("severity", "LOW")
                    }
                    for o in result.pending_doctor_orders
                ],
                methodology=(
                    "Order status checked against pending threshold "
                    f"({self.medication_engine.order_pending_threshold_days} days "
                    f"for MEDIUM, 7+ for HIGH). Flagged orders separately noted."
                ),
                severity="WARNING"
            )
            report.add_node(order_node)

    @property
    def medication_engine(self):
        """Get reference to medication engine defaults."""
        from ..engines.medication_reminder_engine import MedicationReminderEngine
        return MedicationReminderEngine()

    def _generate_executive_summary(
        self,
        report: ExplainableCorrelationReport
    ) -> str:
        """Generate a concise executive summary from all findings."""
        total_findings = len(report.nodes)

        if total_findings == 0:
            return (
                "No significant correlations or findings were identified "
                "for this consultation. This may indicate insufficient "
                "historical data for meaningful analysis."
            )

        # Categorize findings by severity
        critical = [n for n in report.nodes if n.severity == "CRITICAL"]
        warnings = [n for n in report.nodes if n.severity == "WARNING"]
        infos = [n for n in report.nodes if n.severity == "INFO"]

        parts = []
        parts.append(
            f"Clinical Decision Support Summary: {total_findings} "
            f"finding(s) generated with overall confidence of "
            f"{report.overall_confidence:.0%}."
        )

        if critical:
            parts.append(
                f"[CRITICAL] {len(critical)} issue(s) requiring "
                f"immediate attention."
            )

        if warnings:
            parts.append(
                f"[WARNING] {len(warnings)} issue(s) flagged for "
                f"physician review."
            )

        if infos:
            parts.append(
                f"[INFO] {len(infos)} observation(s) for reference."
            )

        parts.append(
            "\nAll findings are correlations and recommendations "
            "for physician review. No findings constitute a "
            "medical diagnosis or treatment recommendation."
        )

        return " ".join(parts)