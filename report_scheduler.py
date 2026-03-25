import json
from pathlib import Path
from datetime import datetime, timedelta
import schedule
import threading

SCHEDULES_FILE = Path("report_schedules.json")


def _load_schedules():
    """Load all scheduled reports."""
    if not SCHEDULES_FILE.exists():
        return {}
    try:
        with open(SCHEDULES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_schedules(schedules):
    """Save scheduled reports."""
    with open(SCHEDULES_FILE, "w", encoding="utf-8") as f:
        json.dump(schedules, f, indent=2, ensure_ascii=False)


def create_report_schedule(
    schedule_id,
    user_id,
    schedule_name,
    frequency,
    recipients,
    include_metrics=True,
    include_charts=True,
    include_patient_summary=False
):
    """
    Create a new report schedule.
    
    Args:
        schedule_id: Unique schedule identifier
        user_id: User who created the schedule
        schedule_name: Name of the schedule
        frequency: "Daily", "Weekly", "Monthly"
        recipients: List of email addresses
        include_metrics: Include performance metrics
        include_charts: Include visualization charts
        include_patient_summary: Include patient scan summary
    
    Returns:
        dict: Schedule details
    """
    schedules = _load_schedules()
    
    if schedule_id in schedules:
        raise ValueError(f"Schedule ID '{schedule_id}' already exists")
    
    schedule_data = {
        "id": schedule_id,
        "user_id": user_id,
        "name": schedule_name,
        "frequency": frequency,
        "recipients": recipients,
        "include_metrics": include_metrics,
        "include_charts": include_charts,
        "include_patient_summary": include_patient_summary,
        "created_at": datetime.now().isoformat(),
        "last_sent": None,
        "next_send": _calculate_next_send(frequency),
        "is_active": True,
        "send_count": 0
    }
    
    schedules[schedule_id] = schedule_data
    _save_schedules(schedules)
    
    return schedule_data


def get_schedule(schedule_id):
    """Get a specific schedule."""
    schedules = _load_schedules()
    return schedules.get(schedule_id)


def list_user_schedules(user_id):
    """List all schedules for a user."""
    schedules = _load_schedules()
    return [
        schedule for schedule in schedules.values()
        if schedule.get("user_id") == user_id
    ]


def update_schedule(schedule_id, **kwargs):
    """Update schedule settings."""
    schedules = _load_schedules()
    
    if schedule_id not in schedules:
        raise ValueError("Schedule not found")
    
    # Only allow updating specific fields
    allowed_fields = [
        "name", "frequency", "recipients", "include_metrics",
        "include_charts", "include_patient_summary", "is_active"
    ]
    
    for key, value in kwargs.items():
        if key in allowed_fields:
            schedules[schedule_id][key] = value
    
    # Recalculate next send if frequency changed
    if "frequency" in kwargs:
        schedules[schedule_id]["next_send"] = _calculate_next_send(kwargs["frequency"])
    
    _save_schedules(schedules)
    return schedules[schedule_id]


def delete_schedule(schedule_id):
    """Delete a schedule."""
    schedules = _load_schedules()
    if schedule_id in schedules:
        del schedules[schedule_id]
        _save_schedules(schedules)
        return True
    return False


def mark_schedule_sent(schedule_id):
    """Mark a schedule as sent and calculate next send time."""
    schedules = _load_schedules()
    
    if schedule_id not in schedules:
        raise ValueError("Schedule not found")
    
    schedule = schedules[schedule_id]
    schedule["last_sent"] = datetime.now().isoformat()
    schedule["send_count"] = schedule.get("send_count", 0) + 1
    schedule["next_send"] = _calculate_next_send(schedule["frequency"])
    
    _save_schedules(schedules)
    return schedule


def get_pending_schedules():
    """Get all schedules that are due to be sent."""
    schedules = _load_schedules()
    pending = []
    
    for schedule in schedules.values():
        if not schedule.get("is_active"):
            continue
        
        next_send = datetime.fromisoformat(schedule["next_send"])
        if datetime.now() >= next_send:
            pending.append(schedule)
    
    return pending


def _calculate_next_send(frequency):
    """Calculate next send time based on frequency."""
    now = datetime.now()
    
    if frequency == "Daily":
        # Send at 9 AM tomorrow
        next_send = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    elif frequency == "Weekly":
        # Send at 9 AM next Monday
        days_ahead = 0 - now.weekday()  # Monday is 0
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        next_send = (now + timedelta(days=days_ahead)).replace(hour=9, minute=0, second=0, microsecond=0)
    elif frequency == "Monthly":
        # Send at 9 AM on the first day of next month
        if now.month == 12:
            next_send = now.replace(year=now.year + 1, month=1, day=1, hour=9, minute=0, second=0, microsecond=0)
        else:
            next_send = now.replace(month=now.month + 1, day=1, hour=9, minute=0, second=0, microsecond=0)
    else:
        next_send = now + timedelta(days=1)
    
    return next_send.isoformat()


def get_schedule_statistics():
    """Get statistics about all schedules."""
    schedules = _load_schedules()
    
    stats = {
        "total_schedules": len(schedules),
        "active_schedules": sum(1 for s in schedules.values() if s.get("is_active")),
        "inactive_schedules": sum(1 for s in schedules.values() if not s.get("is_active")),
        "by_frequency": {
            "Daily": sum(1 for s in schedules.values() if s.get("frequency") == "Daily" and s.get("is_active")),
            "Weekly": sum(1 for s in schedules.values() if s.get("frequency") == "Weekly" and s.get("is_active")),
            "Monthly": sum(1 for s in schedules.values() if s.get("frequency") == "Monthly" and s.get("is_active"))
        },
        "total_reports_sent": sum(s.get("send_count", 0) for s in schedules.values())
    }
    
    return stats


class ReportScheduler:
    """Background scheduler for automated reports."""
    
    def __init__(self):
        self.scheduler_thread = None
        self.is_running = False
    
    def start(self):
        """Start the background scheduler."""
        if self.is_running:
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
    
    def stop(self):
        """Stop the background scheduler."""
        self.is_running = False
    
    def _run_scheduler(self):
        """Run scheduled tasks."""
        while self.is_running:
            pending = get_pending_schedules()
            
            for schedule in pending:
                # Here you would call the actual report generation and sending
                # This is a placeholder for integration with email_notifications.py
                print(f"Would send report for schedule: {schedule['id']}")
                mark_schedule_sent(schedule["id"])
            
            # Check every 60 seconds
            import time
            time.sleep(60)
