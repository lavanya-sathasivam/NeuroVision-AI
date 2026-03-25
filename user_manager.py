import json
import os
from pathlib import Path
from datetime import datetime
import hashlib

USERS_FILE = Path("users.json")
USER_SETTINGS_FOLDER = Path("user_settings")
USER_SETTINGS_FOLDER.mkdir(exist_ok=True)


def _load_json(path):
    """Load JSON file safely."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path, data):
    """Save JSON file safely."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    """Load all users from database."""
    return _load_json(USERS_FILE)


def save_users(users):
    """Save users to database."""
    _save_json(USERS_FILE, users)


def user_exists(user_id):
    """Check if user exists."""
    users = load_users()
    return user_id in users


def create_user(user_id, password, full_name, email, department, hospital="", phone=""):
    """Create a new user account."""
    if not user_id or not password or not full_name or not email:
        raise ValueError("User ID, password, full name, and email are required")
    
    users = load_users()
    
    if user_id in users:
        raise ValueError(f"User ID '{user_id}' already exists")
    
    users[user_id] = {
        "password_hash": hash_password(password),
        "full_name": full_name,
        "email": email,
        "department": department,
        "hospital": hospital,
        "phone": phone,
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "is_active": True
    }
    
    save_users(users)
    _initialize_user_settings(user_id)
    
    return users[user_id]


def authenticate_user(user_id, password):
    """Authenticate user with password. Returns True if valid."""
    users = load_users()
    
    if user_id not in users:
        return False
    
    user = users[user_id]
    if not user.get("is_active"):
        return False
    
    stored_hash = user.get("password_hash")
    return stored_hash == hash_password(password)


def update_last_login(user_id):
    """Update last login timestamp."""
    users = load_users()
    if user_id in users:
        users[user_id]["last_login"] = datetime.now().isoformat()
        save_users(users)


def get_user(user_id):
    """Get user profile."""
    users = load_users()
    if user_id in users:
        user = users[user_id].copy()
        user.pop("password_hash", None)  # Don't return password hash
        return user
    return None


def update_user_profile(user_id, **kwargs):
    """Update user profile information."""
    users = load_users()
    
    if user_id not in users:
        raise ValueError("User not found")
    
    # Don't allow updating password through this method
    safe_fields = ["full_name", "email", "department", "hospital", "phone"]
    
    for key, value in kwargs.items():
        if key in safe_fields:
            users[user_id][key] = value
    
    save_users(users)
    return users[user_id]


def change_password(user_id, old_password, new_password):
    """Change user password."""
    if not authenticate_user(user_id, old_password):
        raise ValueError("Current password is incorrect")
    
    users = load_users()
    users[user_id]["password_hash"] = hash_password(new_password)
    save_users(users)


def list_all_users():
    """Get list of all users (without passwords)."""
    users = load_users()
    result = []
    for user_id, user_data in users.items():
        user = user_data.copy()
        user["user_id"] = user_id
        user.pop("password_hash", None)
        result.append(user)
    return result


def deactivate_user(user_id):
    """Deactivate a user account."""
    users = load_users()
    if user_id in users:
        users[user_id]["is_active"] = False
        save_users(users)
        return True
    return False


def activate_user(user_id):
    """Activate a user account."""
    users = load_users()
    if user_id in users:
        users[user_id]["is_active"] = True
        save_users(users)
        return True
    return False


def _initialize_user_settings(user_id):
    """Initialize default settings for new user."""
    settings_file = USER_SETTINGS_FOLDER / f"{user_id}_settings.json"
    
    default_settings = {
        "display": {
            "dark_mode": False,
            "show_advanced_metrics": False,
            "chart_style": "Default",
            "animation_enabled": True,
            "sidebar_width": "Medium"
        },
        "model": {
            "confidence_threshold": 0.7,
            "alert_threshold": 0.85,
            "caution_threshold": 0.70,
            "batch_processing": False,
            "gpu_acceleration": True
        },
        "notifications": {
            "enabled": True,
            "notify_high_confidence": True,
            "notify_alerts": True,
            "notify_uncertain": True,
            "notify_analysis_complete": True,
            "sound_enabled": True
        },
        "privacy": {
            "auto_save": True,
            "anonymize_exports": True,
            "data_retention_days": "Never",
            "log_retention_days": "30 days",
            "hipaa_mode": False
        },
        "email": {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "email_address": "",
            "email_password": "",
            "notifications_via_email": False
        },
        "reports": {
            "auto_generate_reports": False,
            "report_frequency": "Weekly",
            "report_recipients": []
        }
    }
    
    _save_json(settings_file, default_settings)


def load_user_settings(user_id):
    """Load user-specific settings."""
    settings_file = USER_SETTINGS_FOLDER / f"{user_id}_settings.json"
    return _load_json(settings_file)


def save_user_settings(user_id, settings):
    """Save user-specific settings."""
    settings_file = USER_SETTINGS_FOLDER / f"{user_id}_settings.json"
    _save_json(settings_file, settings)


def update_user_settings(user_id, category, key, value):
    """Update a specific setting for a user."""
    settings = load_user_settings(user_id)
    
    if category not in settings:
        settings[category] = {}
    
    settings[category][key] = value
    save_user_settings(user_id, settings)
    
    return settings
