"""
EHR Nexus - Configuration Module
==================================
Central configuration for all AI engine parameters.
All settings are configurable and provide sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SymptomEngineConfig:
    """Configuration for SymptomCorrelationEngine."""
    similarity_threshold: float = 0.70
    enabled: bool = True


@dataclass
class DiagnosisEngineConfig:
    """Configuration for DiagnosisCorrelationEngine."""
    similarity_threshold: float = 0.75
    enabled: bool = True


@dataclass
class LabTrendEngineConfig:
    """Configuration for LabTrendAnalysisEngine."""
    significant_change_threshold: float = 2.0  # z-score
    min_data_points: int = 3
    trend_window_months: int = 24
    enabled: bool = True


@dataclass
class DuplicateLabConfig:
    """Configuration for DuplicateLabDetector."""
    time_window_hours: int = 24
    value_similarity_threshold: float = 0.95
    name_similarity_threshold: float = 0.90
    enabled: bool = True


@dataclass
class MedicationReminderConfig:
    """Configuration for MedicationReminderEngine."""
    refill_reminder_days_before: int = 7
    overdue_grace_days: int = 2
    order_pending_threshold_days: int = 3
    enabled: bool = True


@dataclass
class EHRNexusAIConfig:
    """Master configuration for the entire AI module."""
    symptom: SymptomEngineConfig = field(default_factory=SymptomEngineConfig)
    diagnosis: DiagnosisEngineConfig = field(default_factory=DiagnosisEngineConfig)
    lab_trend: LabTrendEngineConfig = field(default_factory=LabTrendEngineConfig)
    duplicate_lab: DuplicateLabConfig = field(default_factory=DuplicateLabConfig)
    medication: MedicationReminderConfig = field(default_factory=MedicationReminderConfig)

    # System-wide settings
    log_level: str = "INFO"
    enable_all_engines: bool = True

    def to_dict(self) -> dict:
        return {
            "symptom": {
                "similarity_threshold": self.symptom.similarity_threshold,
                "enabled": self.symptom.enabled and self.enable_all_engines
            },
            "diagnosis": {
                "similarity_threshold": self.diagnosis.similarity_threshold,
                "enabled": self.diagnosis.enabled and self.enable_all_engines
            },
            "lab_trend": {
                "significant_change_threshold":
                    self.lab_trend.significant_change_threshold,
                "min_data_points": self.lab_trend.min_data_points,
                "trend_window_months": self.lab_trend.trend_window_months,
                "enabled": self.lab_trend.enabled and self.enable_all_engines
            },
            "duplicate_lab": {
                "time_window_hours": self.duplicate_lab.time_window_hours,
                "value_similarity_threshold":
                    self.duplicate_lab.value_similarity_threshold,
                "enabled": self.duplicate_lab.enabled and self.enable_all_engines
            },
            "medication": {
                "refill_reminder_days_before":
                    self.medication.refill_reminder_days_before,
                "overdue_grace_days": self.medication.overdue_grace_days,
                "enabled": self.medication.enabled and self.enable_all_engines
            },
            "system": {
                "log_level": self.log_level
            }
        }


# Default configuration singleton
DEFAULT_CONFIG = EHRNexusAIConfig()