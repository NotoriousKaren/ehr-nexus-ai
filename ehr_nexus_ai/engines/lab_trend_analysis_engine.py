"""
EHR Nexus - Laboratory Trend Analysis Engine
==============================================
Analyzes laboratory results across multiple visits to identify:
- Trends and patterns in lab values over time
- Recurring abnormal findings
- Significant changes in lab values
- Longitudinal comparisons of current vs. historical results
"""

from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import statistics
import math

from ..core.data_models import (
    PatientRecord, Consultation, LaboratoryResult
)


class LabTrendAnalysisResult:
    """Encapsulates lab trend analysis outputs."""
    def __init__(self):
        self.trends_by_parameter: List[Dict[str, Any]] = []
        self.recurring_abnormalities: List[Dict[str, Any]] = []
        self.significant_changes: List[Dict[str, Any]] = []
        self.current_vs_historical_comparisons: List[Dict[str, Any]] = []
        self.summary: str = ""
        self.confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trends_by_parameter": self.trends_by_parameter,
            "recurring_abnormalities": self.recurring_abnormalities,
            "significant_changes": self.significant_changes,
            "current_vs_historical_comparisons":
                self.current_vs_historical_comparisons,
            "summary": self.summary,
            "confidence_score": self.confidence_score
        }


class LabTrendAnalysisEngine:
    """
    Rule-based engine for analyzing laboratory result trends.
    Uses statistical methods (mean, standard deviation, z-scores)
    to detect meaningful changes and patterns in lab values.
    """

    def __init__(
        self,
        significant_change_threshold: float = 2.0,  # z-score threshold
        min_data_points: int = 3,
        trend_window_months: int = 24  # 2-year lookback
    ):
        self.significant_change_threshold = significant_change_threshold
        self.min_data_points = min_data_points
        self.trend_window_months = trend_window_months

    def analyze(
        self,
        record: PatientRecord,
        current_consultation: Optional[Consultation] = None
    ) -> LabTrendAnalysisResult:
        """
        Main entry point: analyze laboratory trends.

        Args:
            record: Complete patient record.
            current_consultation: Current consultation. If None, uses latest.

        Returns:
            LabTrendAnalysisResult with all findings.
        """
        result = LabTrendAnalysisResult()

        if current_consultation is None:
            current_consultation = record.get_latest_consultation()

        if current_consultation is None:
            result.summary = "No consultation data available."
            return result

        historical = record.get_historical_consultations(exclude_latest=True)

        # Organize all labs by parameter name
        all_labs = self._collect_all_labs_by_parameter(
            current_consultation, historical
        )

        if not all_labs:
            result.summary = "No laboratory data available for analysis."
            return result

        current_labs = current_consultation.laboratory_results
        if not current_labs:
            result.summary = (
                "No laboratory results in the current consultation."
            )
            return result

        # 1. Analyze trends for each parameter
        result.trends_by_parameter = self._analyze_trends(all_labs)

        # 2. Detect recurring abnormalities
        result.recurring_abnormalities = self._find_recurring_abnormalities(
            all_labs, current_labs
        )

        # 3. Detect significant changes
        result.significant_changes = self._find_significant_changes(
            current_labs, all_labs
        )

        # 4. Compare current vs historical
        result.current_vs_historical_comparisons = (
            self._compare_current_with_historical(
                current_labs, all_labs
            )
        )

        # 5. Generate summary
        result.summary = self._generate_summary(result)

        # 6. Compute confidence
        result.confidence_score = self._compute_confidence(
            current_labs, all_labs, result
        )

        return result

    def _collect_all_labs_by_parameter(
        self,
        current: Consultation,
        historical: List[Consultation]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect all laboratory results across all consultations,
        organized by parameter name.
        """
        all_labs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Add historical labs
        for cons in historical:
            for lab in cons.laboratory_results:
                all_labs[lab.parameter_name].append({
                    "lab": lab,
                    "consultation_id": cons.consultation_id,
                    "consultation_date": cons.consultation_date,
                    "visit_type": cons.visit_type
                })

        # Add current labs
        for lab in current.laboratory_results:
            all_labs[lab.parameter_name].append({
                "lab": lab,
                "consultation_id": current.consultation_id,
                "consultation_date": current.consultation_date,
                "visit_type": current.visit_type,
                "is_current": True
            })

        return dict(all_labs)

    def _analyze_trends(
        self,
        all_labs: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        For each lab parameter, analyze the trend over time.
        Computes statistical measures and trend direction.
        """
        trends = []

        for param_name, records in all_labs.items():
            # Sort by date
            sorted_records = sorted(
                records, key=lambda r: r["consultation_date"]
            )

            if len(sorted_records) < self.min_data_points:
                continue

            values = [r["lab"].value for r in sorted_records]
            dates = [r["consultation_date"] for r in sorted_records]
            units = sorted_records[0]["lab"].unit
            ref_low = sorted_records[0]["lab"].reference_range_low
            ref_high = sorted_records[0]["lab"].reference_range_high

            # Basic statistics
            mean_val = statistics.mean(values)
            stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
            min_val = min(values)
            max_val = max(values)
            first_val = values[0]
            last_val = values[-1]

            # Determine trend direction using linear regression approximation
            # (simple slope calculation)
            n = len(values)
            if n >= 3 and stdev_val > 0:
                x_mean = (n - 1) / 2.0
                y_mean = mean_val
                numerator = sum(
                    (i - x_mean) * (val - y_mean)
                    for i, val in enumerate(values)
                )
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                slope = numerator / denominator if denominator != 0 else 0

                # Normalize slope relative to ref range width
                ref_range_width = max(ref_high - ref_low, 1.0)
                normalized_slope = slope / ref_range_width * n  # per-n-visit change

                if normalized_slope > 0.3:
                    direction = "Increasing"
                elif normalized_slope < -0.3:
                    direction = "Decreasing"
                else:
                    direction = "Stable"
            else:
                slope = 0.0
                direction = "Insufficient data"

            # Determine if values are trending toward/away from abnormal
            percent_abnormal = (
                sum(1 for v in values if v < ref_low or v > ref_high)
                / len(values) * 100
            )

            trends.append({
                "parameter": param_name,
                "unit": units,
                "number_of_measurements": n,
                "date_range": {
                    "first": dates[0].isoformat(),
                    "last": dates[-1].isoformat()
                },
                "statistics": {
                    "mean": round(mean_val, 2),
                    "standard_deviation": round(stdev_val, 2),
                    "minimum": round(min_val, 2),
                    "maximum": round(max_val, 2),
                    "first_value": round(first_val, 2),
                    "latest_value": round(last_val, 2),
                    "reference_range": f"{ref_low}-{ref_high}"
                },
                "trend": {
                    "direction": direction,
                    "slope": round(slope, 4),
                    "percent_change_from_first": round(
                        ((last_val - first_val) / max(abs(first_val), 0.01)) * 100,
                        2
                    )
                },
                "percent_abnormal": round(percent_abnormal, 1),
                "value_timeline": [
                    {
                        "date": r["consultation_date"].isoformat(),
                        "value": r["lab"].value,
                        "flag": r["lab"].flag,
                        "is_current": r.get("is_current", False)
                    }
                    for r in sorted_records
                ]
            })

        trends.sort(
            key=lambda t: abs(t["trend"]["percent_change_from_first"]),
            reverse=True
        )
        return trends

    def _find_recurring_abnormalities(
        self,
        all_labs: Dict[str, List[Dict[str, Any]]],
        current_labs: List[LaboratoryResult]
    ) -> List[Dict[str, Any]]:
        """
        Find lab parameters that have repeatedly shown abnormal values.
        """
        recurring = []

        for param_name, records in all_labs.items():
            abnormal_instances = [
                r for r in records
                if r["lab"].is_abnormal or r["lab"].flag not in ("Normal", "")
            ]

            if len(abnormal_instances) >= 2:
                # Check if current lab for this param is also abnormal
                current_abnormal = any(
                    lab.parameter_name == param_name and
                    (lab.is_abnormal or lab.flag not in ("Normal", ""))
                    for lab in current_labs
                )

                # Determine most common flag
                flag_counts: Dict[str, int] = defaultdict(int)
                for r in abnormal_instances:
                    if r["lab"].flag:
                        flag_counts[r["lab"].flag] += 1

                most_common_flag = max(flag_counts, key=flag_counts.get) \
                    if flag_counts else "Abnormal"

                recurring.append({
                    "parameter": param_name,
                    "abnormal_count": len(abnormal_instances),
                    "total_count": len(records),
                    "abnormal_rate": round(
                        len(abnormal_instances) / len(records) * 100, 1
                    ),
                    "most_common_flag": most_common_flag,
                    "currently_abnormal": current_abnormal,
                    "abnormal_timeline": [
                        {
                            "date": r["consultation_date"].isoformat(),
                            "value": r["lab"].value,
                            "flag": r["lab"].flag,
                            "reference_range": (
                                f"{r['lab'].reference_range_low}-"
                                f"{r['lab'].reference_range_high}"
                            )
                        }
                        for r in abnormal_instances
                    ]
                })

        recurring.sort(key=lambda x: x["abnormal_count"], reverse=True)
        return recurring

    def _find_significant_changes(
        self,
        current_labs: List[LaboratoryResult],
        all_labs: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Detect significant changes in current lab values compared to
        historical averages using z-score methodology.
        """
        significant_changes = []

        for lab in current_labs:
            param = lab.parameter_name
            if param not in all_labs:
                continue

            records = all_labs[param]
            historical_records = [
                r for r in records if not r.get("is_current", False)
            ]

            if len(historical_records) < 2:
                continue

            hist_values = [r["lab"].value for r in historical_records]
            hist_mean = statistics.mean(hist_values)
            hist_stdev = statistics.stdev(hist_values) if len(hist_values) > 1 else 0.0

            if hist_stdev == 0.0:
                continue

            # Calculate z-score
            z_score = (lab.value - hist_mean) / hist_stdev

            if abs(z_score) >= self.significant_change_threshold:
                direction = "Increased" if z_score > 0 else "Decreased"
                significant_changes.append({
                    "parameter": param,
                    "current_value": lab.value,
                    "unit": lab.unit,
                    "historical_mean": round(hist_mean, 2),
                    "historical_std_dev": round(hist_stdev, 2),
                    "z_score": round(z_score, 2),
                    "direction": direction,
                    "change_magnitude": round(abs(z_score), 2),
                    "interpretation": (
                        f"{direction} significantly from historical "
                        f"average ({round(hist_mean, 2)} ± "
                        f"{round(hist_stdev, 2)}) "
                        f"with z-score of {round(z_score, 2)}"
                    ),
                    "current_flag": lab.flag,
                    "num_historical_points": len(historical_records)
                })

        significant_changes.sort(
            key=lambda x: x["change_magnitude"], reverse=True
        )
        return significant_changes

    def _compare_current_with_historical(
        self,
        current_labs: List[LaboratoryResult],
        all_labs: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        For each current lab result, compare with historical results
        for the same parameter.
        """
        comparisons = []

        for lab in current_labs:
            param = lab.parameter_name
            if param not in all_labs:
                comparisons.append({
                    "parameter": param,
                    "current_value": lab.value,
                    "unit": lab.unit,
                    "flag": lab.flag,
                    "status": "No historical data available for comparison"
                })
                continue

            records = all_labs[param]
            historical_records = [
                r for r in records if not r.get("is_current", False)
            ]

            if not historical_records:
                comparisons.append({
                    "parameter": param,
                    "current_value": lab.value,
                    "unit": lab.unit,
                    "flag": lab.flag,
                    "status": "No historical data available for comparison"
                })
                continue

            hist_values = [r["lab"].value for r in historical_records]
            hist_mean = statistics.mean(hist_values) if hist_values else 0.0
            hist_min = min(hist_values)
            hist_max = max(hist_values)

            # Determine percentile of current value relative to historical
            below_count = sum(1 for v in hist_values if v < lab.value)
            percentile = (below_count / len(hist_values)) * 100

            comparisons.append({
                "parameter": param,
                "current_value": lab.value,
                "unit": lab.unit,
                "flag": lab.flag,
                "reference_range": (
                    f"{lab.reference_range_low}-{lab.reference_range_high}"
                ),
                "historical": {
                    "mean": round(hist_mean, 2),
                    "min": round(hist_min, 2),
                    "max": round(hist_max, 2),
                    "num_previous_measurements": len(historical_records),
                    "percentile_rank": round(percentile, 1)
                },
                "comparison": (
                    f"Current value ({lab.value}) is at the "
                    f"{percentile:.1f}th percentile of historical values "
                    f"(range: {hist_min:.1f} - {hist_max:.1f}, "
                    f"mean: {hist_mean:.1f})"
                ),
                "is_consistent_with_history":
                    hist_min <= lab.value <= hist_max
            })

        return comparisons

    def _generate_summary(self, result: LabTrendAnalysisResult) -> str:
        """Generate a human-readable summary."""
        parts = []

        if result.trends_by_parameter:
            increasing = [
                t for t in result.trends_by_parameter
                if t["trend"]["direction"] == "Increasing"
            ]
            decreasing = [
                t for t in result.trends_by_parameter
                if t["trend"]["direction"] == "Decreasing"
            ]
            if increasing:
                names = [t["parameter"] for t in increasing[:3]]
                parts.append(
                    f"[UP] {len(increasing)} parameter(s) show increasing "
                    f"trend: {', '.join(names)}."
                )
            if decreasing:
                names = [t["parameter"] for t in decreasing[:3]]
                parts.append(
                    f"[DOWN] {len(decreasing)} parameter(s) show decreasing "
                    f"trend: {', '.join(names)}."
                )

        if result.recurring_abnormalities:
            parts.append(
                f"Found {len(result.recurring_abnormalities)} parameter(s) "
                f"with recurring abnormal findings."
            )

        if result.significant_changes:
            parts.append(
                f"[WARNING] Detected {len(result.significant_changes)} significant "
                f"change(s) in laboratory values that deviate substantially "
                f"from historical patterns."
            )

        if not parts:
            parts.append(
                "No significant laboratory trends or abnormalities detected."
            )

        return " ".join(parts)

    def _compute_confidence(
        self,
        current_labs: List[LaboratoryResult],
        all_labs: Dict[str, List[Dict[str, Any]]],
        result: LabTrendAnalysisResult
    ) -> float:
        """Compute confidence score for lab analysis."""
        if not current_labs or not all_labs:
            return 0.0

        # How many current labs have historical data
        params_with_history = len([
            lab for lab in current_labs
            if lab.parameter_name in all_labs and
            len([r for r in all_labs[lab.parameter_name]
                 if not r.get("is_current")]) > 0
        ])
        history_coverage = params_with_history / max(len(current_labs), 1)

        # Average data points per parameter
        data_points = [
            len(records) for records in all_labs.values()
        ]
        avg_data_points = (
            sum(data_points) / max(len(data_points), 1)
        ) if data_points else 0

        depth_factor = min(avg_data_points / 10.0, 1.0)

        return round(min(history_coverage * 0.6 + depth_factor * 0.4, 1.0), 2)