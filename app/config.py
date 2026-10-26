import os
import json
import sys
import shutil

# Xác định thư mục ứng dụng động:
# - Khi đóng gói bằng PyInstaller (sys.frozen = True):
#   os.path.dirname(sys.executable) chính là thư mục chứa file .exe thực tế
# - Khi chạy trong môi trường dev (python main.py):
#   APP_DIR chính là thư mục gốc của dự án Web_flask
if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
    BUNDLE_DIR = getattr(sys, '_MEIPASS', APP_DIR)
else:
    APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    BUNDLE_DIR = APP_DIR

# BASE_DIR luôn trỏ vào APP_DIR (ngang hàng với .exe hoặc main.py)
BASE_DIR = APP_DIR

# 1. Thư mục dữ liệu runtime trong data/ (LUÔN nằm ngang hàng với file .exe)
DATA_DIR = os.path.join(BASE_DIR, "data")
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
INTRUSION_LOGS_DIR = os.path.join(DATA_DIR, "intrusion_logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(INTRUSION_LOGS_DIR, exist_ok=True)

# 2. Thư mục cấu hình configs/ (Nằm ngang hàng với file .exe để tiện chỉnh sửa)
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
os.makedirs(CONFIGS_DIR, exist_ok=True)

# Tự động sao chép các file cấu hình mặc định từ bundle nếu ngoài thư mục app chưa có
bundle_configs_dir = os.path.join(BUNDLE_DIR, "configs")
if os.path.exists(bundle_configs_dir) and os.path.abspath(bundle_configs_dir) != os.path.abspath(CONFIGS_DIR):
    for item in os.listdir(bundle_configs_dir):
        src = os.path.join(bundle_configs_dir, item)
        dst = os.path.join(CONFIGS_DIR, item)
        if not os.path.exists(dst) and os.path.isfile(src):
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

ROI_FILE = os.path.join(CONFIGS_DIR, "roi.json")
CLASSES_FILE = os.path.join(CONFIGS_DIR, "classes.json")
DETECTION_FILE = os.path.join(CONFIGS_DIR, "detection.json")
BYTETRACK_FILE = os.path.join(CONFIGS_DIR, "bytetrack.yaml")

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

DEFAULT_OBS_CAM = "auto"

def load_obs_cam_from_file():
    det_data = _load_detection_json()
    if "obs_cam" in det_data and det_data["obs_cam"] is not None:
        return str(det_data["obs_cam"]).strip()
    return DEFAULT_OBS_CAM

def save_obs_cam_to_file(cam):
    _save_detection_json({"obs_cam": str(cam).strip()})

# Giữ tương thích ngược tạm thời nếu có lời gọi cũ
def load_droidcam_url_from_file():
    return ""

def save_droidcam_url_to_file(url):
    pass

# --- KHỞI TẠO BIẾN TOÀN CỤC (GLOBAL STATE) ---
roi_polygon = load_roi_from_file()
initial_conf = load_conf_from_file()
selected_classes = load_classes_from_file()
dwell_time = load_dwell_from_file()
train_mode = load_train_mode_from_file()
obs_cam = load_obs_cam_from_file()

latest_data = {
    "obs": {"text": "", "alarm": False},
    "webcam": {"text": "", "alarm": False},
    "video": {"text": "", "alarm": False}
}
stream_active = {"obs": False, "webcam": False, "video": False}
ai_active = {"obs": False, "webcam": False, "video": False} 
current_conf = {
    "obs": initial_conf,
    "webcam": initial_conf,
    "video": initial_conf
}

