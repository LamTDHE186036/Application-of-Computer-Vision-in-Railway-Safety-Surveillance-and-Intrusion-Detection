import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Toàn bộ file cấu hình YAML/JSON trong configs/
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
ROI_FILE = os.path.join(CONFIGS_DIR, "roi.json")
CLASSES_FILE = os.path.join(CONFIGS_DIR, "classes.json")
DETECTION_FILE = os.path.join(CONFIGS_DIR, "detection.json")
BYTETRACK_FILE = os.path.join(CONFIGS_DIR, "bytetrack.yaml")

# Thư mục dữ liệu runtime trong data/
DATA_DIR = os.path.join(BASE_DIR, "data")
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
INTRUSION_LOGS_DIR = os.path.join(DATA_DIR, "intrusion_logs")

os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(INTRUSION_LOGS_DIR, exist_ok=True)

DEFAULT_CLASSES = ["person", "car", "truck-bus", "motorcycle"]
DEFAULT_CONF = 0.5
DEFAULT_DWELL_TIME = 5.0
DEFAULT_TRAIN_MODE = "train_approaching"

def _load_detection_json():
    if os.path.exists(DETECTION_FILE):
        try:
            with open(DETECTION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_detection_json(data):
    current = _load_detection_json()
    current.update(data)
    with open(DETECTION_FILE, 'w', encoding='utf-8') as f:
        json.dump(current, f, indent=2)

def load_roi_from_file():
    if os.path.exists(ROI_FILE):
        try:
            with open(ROI_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("polygon", [])
        except Exception:
            return []
    return []

def save_roi_to_file(polygon):
    with open(ROI_FILE, 'w', encoding='utf-8') as f:
        json.dump({"polygon": polygon}, f, indent=2)

def load_conf_from_file():
    det_data = _load_detection_json()
    if "conf" in det_data:
        return float(det_data["conf"])
    return DEFAULT_CONF

def save_conf_to_file(conf):
    _save_detection_json({"conf": float(conf)})

def load_classes_from_file():
    if os.path.exists(CLASSES_FILE):
        try:
            with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("classes", DEFAULT_CLASSES)
        except Exception:
            return DEFAULT_CLASSES
    return DEFAULT_CLASSES

def save_classes_to_file(classes):
    with open(CLASSES_FILE, 'w', encoding='utf-8') as f:
        json.dump({"classes": classes}, f, indent=2)

def load_dwell_from_file():
    det_data = _load_detection_json()
    if "dwell_time" in det_data:
        return float(det_data["dwell_time"])
    return DEFAULT_DWELL_TIME

def save_dwell_to_file(dwell_time):
    _save_detection_json({"dwell_time": float(dwell_time)})

def load_train_mode_from_file():
    det_data = _load_detection_json()
    if "train_mode" in det_data:
        return det_data["train_mode"]
    return DEFAULT_TRAIN_MODE

def save_train_mode_to_file(mode):
    _save_detection_json({"train_mode": mode})

DEFAULT_DROIDCAM_URL = "http://192.168.1.15:4747/video"

def load_droidcam_url_from_file():
    det_data = _load_detection_json()
    if "droidcam_url" in det_data and det_data["droidcam_url"]:
        return str(det_data["droidcam_url"]).strip()
    return DEFAULT_DROIDCAM_URL

def save_droidcam_url_to_file(url):
    _save_detection_json({"droidcam_url": str(url).strip()})

# --- KHỞI TẠO BIẾN TOÀN CỤC (GLOBAL STATE) ---
roi_polygon = load_roi_from_file()
initial_conf = load_conf_from_file()
selected_classes = load_classes_from_file()
dwell_time = load_dwell_from_file()
train_mode = load_train_mode_from_file()
droidcam_url = load_droidcam_url_from_file()

latest_data = {
    "webcam": {"text": "", "alarm": False},
    "video": {"text": "", "alarm": False},
    "droidcam": {"text": "", "alarm": False}
}
stream_active = {"webcam": False, "video": False, "droidcam": False}
ai_active = {"webcam": False, "video": False, "droidcam": False} 
current_conf = {
    "webcam": initial_conf,
    "video": initial_conf,
    "droidcam": initial_conf
}

