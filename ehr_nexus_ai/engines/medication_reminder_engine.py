"""
EHR Nexus - Medication Reminder & Doctor's Order Monitor
===========================================================
Monitors medications and doctor's orders to:
- Flag overdue medications (past end/refill dates)
- Flag pending doctor's orders
- Detect medication refill needs
- Generate reminders for both patients and physicians
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, date, timedelta

from ..core.data_models import (
    PatientRecord, Consultation, Medication, DoctorOrder
)


class MedicationReminderResult:
    """Encapsulates medication and order monitoring outputs."""
    def __init__(self):
        self.overdue_medications: List[Dict[str, Any]] = []
        self.upcoming_refills: List[Dict[str, Any]] = []
        self.active_medications: List[Dict[str, Any]] = []
        self.pending_doctor_orders: List[Dict[str, Any]] = []
        self.flagged_orders: List[Dict[str, Any]] = []
        self.summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overdue_medications": self.overdue_medications,
            "upcoming_refills": self.upcoming_refills,
            "active_medications": self.active_medications,
            "pending_doctor_orders": self.pending_doctor_orders,
            "flagged_orders": self.flagged_orders,
            "summary": self.summary
        }


class MedicationReminderEngine:
    """
    Monitors medication adherence and doctor's order completion.
    Generates actionable reminders for healthcare providers.
    """

    def __init__(
        self,
        refill_reminder_days_before: int = 7,
        overdue_grace_days: int = 2,
        order_pending_threshold_days: int = 3
    ):
        """
        Args:
            refill_reminder_days_before: Days before refill to generate reminder.
            overdue_grace_days: Days past due before flagging as overdue.
            order_pending_threshold_days: Days after which a pending order is flagged.
        """
        self.refill_reminder_days_before = refill_reminder_days_before
        self.overdue_grace_days = overdue_grace_days
        self.order_pending_threshold_days = order_pending_threshold_days

    def analyze(
        self,
        record: PatientRecord,
        current_consultation: Optional[Consultation] = None
    ) -> MedicationReminderResult:
        """
        Main entry point: analyze medications and orders.

        Args:
            record: Complete patient record.
            current_consultation: Current consultation (optional).

        Returns:
            MedicationReminderResult with all findings.
        """
        result = MedicationReminderResult()
        today = date.today()
        now = datetime.now()

        # Collect all medications from all consultations
        all_medications = record.get_all_medications()

        # Collect all doctor's orders
        all_orders = []
        for cons in record.consultations:
            all_orders.extend(cons.doctor_orders)

        if not all_medications and not all_orders:
            result.summary = (
                "No medications or doctor's orders found in the patient's record."
            )
            return result

        # 1. Check medication statuses
        self._check_medications(all_medications, today, result)

        # 2. Check doctor's orders
        self._check_orders(all_orders, now, result)

        # 3. Generate summary
        result.summary = self._generate_summary(result)

        return result

    def _check_medications(
        self,
        medications: List[Medication],
        today: date,
        result: MedicationReminderResult
    ) -> None:
        """Check each medication for overdue, refill-needed, and active status."""
        for med in medications:
            # Active medications
            if med.status in ("Active", "active", ""):
                result.active_medications.append({
                    "medication_name": med.medication_name,
                    "dosage": med.dosage,
                    "frequency": med.frequency,
                    "route": med.route,
                    "start_date": med.start_date.isoformat()
                                   if med.start_date else None,
                    "end_date": med.end_date.isoformat()
                                 if med.end_date else None,
                    "notes": med.notes
                })

            # Overdue detection
            overdue = False
            overdue_reasons = []

            if med.is_overdue:
                overdue = True
                overdue_reasons.append("Marked as overdue in record.")

            if med.end_date and med.end_date < today:
                days_overdue = (today - med.end_date).days
                if days_overdue > self.overdue_grace_days:
                    overdue = True
                    overdue_reasons.append(
                        f"End date ({med.end_date}) was "
                        f"{days_overdue} day(s) ago."
                    )

            if med.next_refill_date and med.next_refill_date < today:
                days_since_refill = (today - med.next_refill_date).days
                if days_since_refill > self.overdue_grace_days:
                    overdue = True
                    overdue_reasons.append(
                        f"Refill was due on {med.next_refill_date} "
                        f"({days_since_refill} day(s) ago)."
                    )

            if overdue:
                result.overdue_medications.append({
                    "medication_id": med.medication_id,
                    "medication_name": med.medication_name,
                    "dosage": med.dosage,
                    "frequency": med.frequency,
                    "start_date": med.start_date.isoformat()
                                   if med.start_date else None,
                    "end_date": med.end_date.isoformat()
                                 if med.end_date else None,
                    "next_refill_date": med.next_refill_date.isoformat()
                                         if med.next_refill_date else None,
                    "overdue_reasons": overdue_reasons,
                    "notes": med.notes,
                    "severity": "HIGH" if len(overdue_reasons) >= 2 else "MEDIUM",
                    "recommendation": (
                        "Review medication status and consider "
                        "refill or discontinuation."
                    )
                })

            # Upcoming refill reminders
            if med.next_refill_date:
                days_until_refill = (med.next_refill_date - today).days
                if 0 <= days_until_refill <= self.refill_reminder_days_before:
                    result.upcoming_refills.append({
                        "medication_name": med.medication_name,
                        "dosage": med.dosage,
                        "next_refill_date":
                            med.next_refill_date.isoformat(),
                        "days_until_refill": days_until_refill,
                        "recommendation": (
                            f"Refill due in {days_until_refill} day(s). "
                            "Please arrange refill."
                        )
                    })

    def _check_orders(
        self,
        orders: List[DoctorOrder],
        now: datetime,
        result: MedicationReminderResult
    ) -> None:
        """Check doctor's orders for pending and flagged items."""
        for order in orders:
            # Pending orders
            if order.status in ("Pending", "pending", ""):
                order_info = {
                    "order_id": order.order_id,
                    "order_type": order.order_type,
                    "description": order.description,
                    "created_at": order.created_at.isoformat()
                                   if order.created_at else None,
                    "status": order.status,
                }

                # Calculate days pending
                if order.created_at:
                    days_pending = (now - order.created_at).days
                    order_info["days_pending"] = days_pending

                    if days_pending >= self.order_pending_threshold_days:
                        order_info["severity"] = "HIGH" if days_pending >= 7 else "MEDIUM"
                        order_info["recommendation"] = (
                            f"Order has been pending for {days_pending} day(s). "
                            "Review and take action."
                        )
                    else:
                        order_info["severity"] = "LOW"
                        order_info["recommendation"] = "Monitor."

                result.pending_doctor_orders.append(order_info)

            # Flagged orders
            if order.is_flagged:
                result.flagged_orders.append({
                    "order_id": order.order_id,
                    "order_type": order.order_type,
                    "description": order.description,
                    "created_at": order.created_at.isoformat()
                                   if order.created_at else None,
                    "status": order.status,
                    "flag_reason": "Marked as flagged in system.",
                    "recommendation": "Review flagged order immediately."
                })

    def _generate_summary(self, result: MedicationReminderResult) -> str:
        """Generate a human-readable summary."""
        parts = []

        if result.overdue_medications:
            names = [m["medication_name"]
                     for m in result.overdue_medications[:5]]
            parts.append(
                f"ALERT: {len(result.overdue_medications)} medication(s) overdue: "
                f"{', '.join(names)}. Immediate attention required."
            )

        if result.upcoming_refills:
            parts.append(
                f"REMINDER: {len(result.upcoming_refills)} medication(s) due for "
                f"refill within {self.refill_reminder_days_before} days."
            )

        if result.pending_doctor_orders:
            high_pending = [o for o in result.pending_doctor_orders
                            if o.get("severity") == "HIGH"]
            if high_pending:
                parts.append(
                    f"URGENT: {len(high_pending)} high-priority doctor's order(s) "
                    f"have been pending for an extended period."
                )
            else:
                parts.append(
                    f"PENDING: {len(result.pending_doctor_orders)} doctor's order(s) "
                    f"currently pending."
                )

        if result.flagged_orders:
            parts.append(
                f"FLAGGED: {len(result.flagged_orders)} doctor's order(s) flagged "
                f"for immediate review."
            )

        if result.active_medications and not result.overdue_medications:
            parts.append(
                f"OK: {len(result.active_medications)} active medication(s) "
                f"with no overdue items."
            )

        if not parts:
            parts.append(
                "No medication reminders or pending orders at this time."
            )

        return " ".join(parts)