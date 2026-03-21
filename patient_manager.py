import json
import os
from pathlib import Path
from datetime import datetime

PATIENT_FILE = Path("patients.json")
IMG_FOLDER = Path("history_images")
IMG_FOLDER.mkdir(exist_ok=True)


def _load_json(path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_patients():
    return _load_json(PATIENT_FILE)


def save_patients(patients):
    _save_json(PATIENT_FILE, patients)


def add_patient(patient_id, name, age=None, gender=None):
    if not patient_id or not name:
        raise ValueError("Patient ID and name are required")

    patients = load_patients()
    if patient_id in patients:
        raise ValueError(f"Patient ID '{patient_id}' already exists")

    patients[patient_id] = {
        "name": name,
        "age": age,
        "gender": gender,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scans": []
    }
    save_patients(patients)
    return patients[patient_id]


def update_patient(patient_id, name=None, age=None, gender=None):
    patients = load_patients()
    if patient_id not in patients:
        raise ValueError(f"Patient ID '{patient_id}' not found")

    if name:
        patients[patient_id]["name"] = name
    if age is not None:
        patients[patient_id]["age"] = age
    if gender:
        patients[patient_id]["gender"] = gender

    save_patients(patients)
    return patients[patient_id]


def get_patient(patient_id):
    return load_patients().get(patient_id)


def list_patients():
    patients = load_patients()
    return [
        {"id": pid, **pdata}
        for pid, pdata in patients.items()
    ]


def search_patients(term=""):
    term = str(term).strip().lower()
    patients = load_patients()
    if not term:
        return list_patients()
    results = []
    for pid, pdata in patients.items():
        if term in pid.lower() or term in str(pdata.get("name", "")).lower():
            results.append({"id": pid, **pdata})
    return results


def get_patient_history(patient_id):
    patient = get_patient(patient_id)
    if not patient:
        return []
    return sorted(patient.get("scans", []), key=lambda x: x.get("timestamp", ""), reverse=True)


def add_scan_record(patient_id, scan_record):
    if not patient_id or not scan_record:
        raise ValueError("Patient ID and scan record are required")

    patients = load_patients()
    if patient_id not in patients:
        raise ValueError(f"Patient ID '{patient_id}' not found")

    patients[patient_id].setdefault("scans", []).append(scan_record)
    save_patients(patients)


def get_scan_record(patient_id, scan_id):
    patient = get_patient(patient_id)
    if not patient:
        return None
    scans = patient.get("scans", [])
    for scan in scans:
        if scan.get("id") == scan_id:
            return scan
    return None


def delete_scan_record(patient_id, scan_id):
    patients = load_patients()
    if patient_id not in patients:
        raise ValueError(f"Patient ID '{patient_id}' not found")

    scans = patients[patient_id].get("scans", [])
    patients[patient_id]["scans"] = [s for s in scans if s.get("id") != scan_id]
    save_patients(patients)
