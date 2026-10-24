import os
import json

# Đường dẫn file lưu cấu hình độc lập
ROI_FILE = 'save_config/roi_config.json'
CONF_FILE = 'save_config/conf_config.json'

def load_roi_from_file():
    if os.path.exists(ROI_FILE):
        try:
            with open(ROI_FILE, 'r') as f:
                data = json.load(f)
                return data.get("polygon", [])
        except:
            return []
    return []

def save_roi_to_file(polygon):
    # Dùng chung cho cả việc lưu tọa độ mới và làm rỗng tọa độ
    with open(ROI_FILE, 'w') as f:
        json.dump({"polygon": polygon}, f)

def load_conf_from_file():
    if os.path.exists(CONF_FILE):
        try:
            with open(CONF_FILE, 'r') as f:
                data = json.load(f)
                return data.get("conf", 0.5)
        except:
            return 0.5
    return 0.5

def save_conf_to_file(conf):
    os.makedirs(os.path.dirname(CONF_FILE), exist_ok=True)
    with open(CONF_FILE, 'w') as f:
        json.dump({"conf": conf}, f)

CLASSES_FILE = 'save_config/classes_config.json'
DEFAULT_CLASSES = ["person", "car", "truck-bus", "motorcycle"]

def load_classes_from_file():
    if os.path.exists(CLASSES_FILE):
        try:
            with open(CLASSES_FILE, 'r') as f:
                data = json.load(f)
                return data.get("classes", DEFAULT_CLASSES)
        except:
            return DEFAULT_CLASSES
    return DEFAULT_CLASSES

def save_classes_to_file(classes):
    os.makedirs(os.path.dirname(CLASSES_FILE), exist_ok=True)
    with open(CLASSES_FILE, 'w') as f:
        json.dump({"classes": classes}, f)

DWELL_FILE = 'save_config/dwell_config.json'
DEFAULT_DWELL_TIME = 5.0

def load_dwell_from_file():
    if os.path.exists(DWELL_FILE):
        try:
            with open(DWELL_FILE, 'r') as f:
                data = json.load(f)
                return float(data.get("dwell_time", DEFAULT_DWELL_TIME))
        except:
            return DEFAULT_DWELL_TIME
    return DEFAULT_DWELL_TIME

def save_dwell_to_file(dwell_time):
    os.makedirs(os.path.dirname(DWELL_FILE), exist_ok=True)
    with open(DWELL_FILE, 'w') as f:
        json.dump({"dwell_time": float(dwell_time)}, f)

TRAIN_MODE_FILE = 'save_config/train_mode_config.json'
DEFAULT_TRAIN_MODE = 'train_approaching'  # 'train_approaching' hoặc 'no_train'

def load_train_mode_from_file():
    if os.path.exists(TRAIN_MODE_FILE):
        try:
            with open(TRAIN_MODE_FILE, 'r') as f:
                data = json.load(f)
                return data.get("train_mode", DEFAULT_TRAIN_MODE)
        except:
            return DEFAULT_TRAIN_MODE
    return DEFAULT_TRAIN_MODE

def save_train_mode_to_file(mode):
    os.makedirs(os.path.dirname(TRAIN_MODE_FILE), exist_ok=True)
    with open(TRAIN_MODE_FILE, 'w') as f:
        json.dump({"train_mode": mode}, f)

# --- KHỞI TẠO BIẾN TOÀN CỤC (GLOBAL STATE) ---
roi_polygon = load_roi_from_file()
initial_conf = load_conf_from_file()
selected_classes = load_classes_from_file()
dwell_time = load_dwell_from_file()
train_mode = load_train_mode_from_file()

latest_data = {"webcam": {"text": "", "alarm": False}, "video": {"text": "", "alarm": False}}
stream_active = {"webcam": False, "video": False}
ai_active = {"webcam": False, "video": False} 
current_conf = {"webcam": initial_conf, "video": initial_conf}