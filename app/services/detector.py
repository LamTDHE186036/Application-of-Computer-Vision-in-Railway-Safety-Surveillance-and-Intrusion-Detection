import cv2
import numpy as np
from ultralytics import YOLO
import time
import datetime
import os
import json
import re
from collections import deque
import openvino as ov


# Cho phép Ultralytics tự động nhận diện thư mục model_openvino theo kiến trúc dự án
try:
    from ultralytics.nn import autobackend
    _orig_model_type = autobackend.AutoBackend._model_type

    def _patched_model_type(p, dnn=False):
        fmt = _orig_model_type(p, dnn)
        if fmt is None and 'openvino' in str(p).lower():
            return 'openvino'
        return fmt

    autobackend.AutoBackend._model_type = staticmethod(_patched_model_type)
except Exception:
    pass

from app import config
from app.utils.video_utils import (
    convert_video_to_h264_async,
    format_time_str,
    format_file_time_str,
    draw_roi as util_draw_roi
)
from app.utils.camera_utils import find_obs_camera_index
from app.services.tracker import (
    reset_tracker as svc_reset_tracker,
    check_roi_intersection,
    update_dwell_and_evaluate_violation
)

# --- THIẾT BỊ SUY LUẬN TỐI ƯU (OPENVINO INTEL GPU / CPU) ---
def get_optimal_device():
    try:
        core = ov.Core()
        if "GPU" in core.available_devices:
            return "intel:gpu"
    except Exception:
        pass
    return "cpu"

def check_system_resources():
    """
    Kiểm tra tài nguyên phần cứng hệ thống để tối ưu hóa thiết bị suy luận (CUDA GPU / Intel iGPU / CPU) cho YOLO26s.
    """
    has_gpu = False
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            has_gpu = True
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    ram_gb = 0.0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            ram_gb = stat.ullTotalPhys / (1024 ** 3)
        except Exception:
            ram_gb = 8.0

    return {
        "has_gpu": has_gpu,
        "gpu_name": gpu_name,
        "ram_gb": round(ram_gb, 1),
        "chosen_model": "yolo26s"
    }

# --- NẠP MÔ HÌNH YOLO26s ---
def load_yolo_model():
    base_dirs = [config.BASE_DIR]
    if hasattr(config, 'BUNDLE_DIR') and config.BUNDLE_DIR != config.BASE_DIR:
        base_dirs.append(config.BUNDLE_DIR)

    res_info = check_system_resources()

    print("=" * 75)
    print(" HỆ THỐNG PHÂN TÍCH TÀI NGUYÊN & KHỞI ĐỘNG MÔ HÌNH YOLO26s")
    print(f"[*] Card đồ họa rời (NVIDIA GPU) : {res_info['gpu_name'] if res_info['has_gpu'] else 'Không có (Dùng Intel iGPU / CPU)'}")
    print(f"[*] Tổng dung lượng RAM hệ thống : {res_info['ram_gb']} GB")
    print("[*] Mô hình kích hoạt           : YOLO26s (Mô hình phát hiện vi phạm đường sắt)")
    print("=" * 75)

    model_candidates = []
    for b_dir in base_dirs:
        m_dir = os.path.join(b_dir, "models", "yolo26s")
        pt_dir = os.path.join(m_dir, "model_pytorch")
        ov_dir = os.path.join(m_dir, "model_openvino")

        pt_files = [
            os.path.join(pt_dir, "train-railway-yolo26s-all.pt"),
            os.path.join(pt_dir, "yolo26s.pt"),
        ]
        if os.path.exists(pt_dir):
            for f in os.listdir(pt_dir):
                if f.endswith(".pt"):
                    full_p = os.path.join(pt_dir, f)
                    if full_p not in pt_files:
                        pt_files.append(full_p)

        if res_info["has_gpu"]:
            model_candidates.extend(pt_files)
            model_candidates.append(ov_dir)
        else:
            model_candidates.append(ov_dir)
            model_candidates.extend(pt_files)

    selected_model_path = None
    for p in model_candidates:
        if os.path.exists(p):
            selected_model_path = p
            break

    if not selected_model_path:
        raise FileNotFoundError("Không tìm thấy mô hình trong thư mục models/yolo26s")

    if "openvino" in selected_model_path.lower():
        device = get_optimal_device()
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    model_instance = YOLO(selected_model_path)
    print(f"[*] Đã nạp thành công mô hình: {selected_model_path} trên thiết bị {device}")
    return model_instance, device, selected_model_path, res_info

model, DEVICE, active_model_path, system_res_info = load_yolo_model()

# --- THƯ MỤC LƯU TRỮ DỮ LIỆU RUNTIME (DATA/) ---
TXT_DIR = config.INTRUSION_LOGS_DIR
VID_DIR = config.RECORDINGS_DIR
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(VID_DIR, exist_ok=True)

# --- TỪ ĐIỂN DỊCH NHÃN YOLO SANG TIẾNG VIỆT CHUẨN (MODEL YOLO26s HIỆN TẠI) ---
CLASS_NAMES_VI = {
    "car": "Ô tô",
    "motorcycle": "Xe máy",
    "person": "Người",
    "truck-bus": "Xe tải/Xe buýt",
    "train": "Tàu hỏa"
} 

def translate_class(eng_name):
    return CLASS_NAMES_VI.get(eng_name.lower(), eng_name)

# --- THAM SỐ CẤU HÌNH TỐI ƯU HÓA HỆ THỐNG ---
PRE_ROLL_SECONDS = 3.0  # Lưu trước sự kiện vi phạm 3 giây (Hộp đen Ring Buffer)
TIMEOUT_SECONDS = 3.0   # Duy trì ghi thêm 3 giây sau sự kiện (Post-roll) và timeout đối tượng
MIN_CONFIRM_SECONDS = 0.5  # Lọc nhiễu: Phải ở trong ROI >= 0.5s mới xác nhận vi phạm (chống chớp 1-2 frame)
MIN_EVENT_DURATION = 1.0   # Lọc nhiễu: Tổng thời lượng vi phạm phải >= 1.0s mới ghi lịch sử và lưu video

# --- BIẾN TOÀN CỤC QUẢN LÝ TRẠNG THÁI ---
active_objects = {}   
completed_events = [] 
current_date_str = ""
recording_start_date_str = ""

is_recording = False
video_writer = None
recording_start_time_str = ""
temp_video_filename = ""
last_activity_time = 0.0
has_valid_event_in_recording = False
roi_dwell_tracker = {}  # {track_id: {"enter_time": float, "last_seen": float, "violating": bool}}
pre_roll_buffer = deque(maxlen=90) # Bộ đệm xoay vòng lưu trước sự kiện
pre_roll_meta = deque(maxlen=90)   # Bộ đệm xoay vòng lưu metadata trước sự kiện
current_recording_meta = []         # Danh sách metadata các khung hình của clip đang ghi

def save_recording_metadata(final_video_path, fps, meta_frames):
    """Lưu metadata khung hình (tọa độ bbox chuẩn hóa, track_id, trạng thái vi phạm) ra file .json đồng bộ"""
    if not final_video_path or not meta_frames:
        return
    try:
        json_path = os.path.splitext(final_video_path)[0] + ".json"
        payload = {
            "video": os.path.basename(final_video_path),
            "fps": round(fps, 2) if fps > 0 else 30.0,
            "total_frames": len(meta_frames),
            "frames": meta_frames
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"[*] Đã lưu metadata vi phạm: {os.path.basename(json_path)} ({len(meta_frames)} frames)")
    except Exception as e:
        print(f"[!] Lỗi khi lưu metadata video vi phạm: {e}")

# --- HÀM KHÔI PHỤC LỊCH SỬ HÔM NAY KHI MỞ PHẦN MỀM ---
def load_today_events():
    global completed_events, current_date_str
    now = datetime.datetime.now()
    current_date_str = now.strftime("%d-%m-%Y")
    completed_events.clear()
    
    filename = os.path.join(TXT_DIR, f"Lich_su_{current_date_str}.txt")
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(r'(?:(\d{2}-\d{2}-\d{4})\s+)?(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2}):\s*(.*?)\s*\(ID\s*(.*?)\)\s*(.*)', line)
                if match:
                    _, start, end, name, tid, desc = match.groups()
                    completed_events.insert(0, {"id": tid, "name": name, "start": start, "end": end, "desc": desc.strip()})

def check_day_transition():
    """Kiểm tra và tự động chuyển ngày mới (qua nửa đêm)"""
    global current_date_str
    now_date = datetime.datetime.now().strftime("%d-%m-%Y")
    if now_date != current_date_str:
        load_today_events()
        return True
    return False

load_today_events()

def update_txt_file(date_str, tid, start_time, end_time, obj_name, status, mode="train_approaching", dwell_sec=0):
    class_name_vn = translate_class(obj_name)
    filename = os.path.join(TXT_DIR, f"Lich_su_{date_str}.txt")
    
    target_prefix = f"{date_str} {start_time}: {class_name_vn} (ID {tid})"
    if mode == "no_train":
        target_str_active = f"{target_prefix} ở trong vùng nguy hiểm {dwell_sec} giây\n"
        target_str_done = f"{date_str} {start_time} - {end_time}: {class_name_vn} (ID {tid}) ở trong vùng nguy hiểm {dwell_sec} giây\n"
    else:
        target_str_active = f"{target_prefix} đang ở trong vùng nguy hiểm\n"
        target_str_done = f"{date_str} {start_time} - {end_time}: {class_name_vn} (ID {tid}) đã đi vào vùng nguy hiểm\n"

    lines = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
    target_str = target_str_active if status == "active" else target_str_done
    found = False
    for i in range(len(lines)):
        if lines[i].startswith(target_prefix):
            lines[i] = target_str
            found = True
            break
    if not found:
        lines.append(target_str)

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    except Exception:
        pass

def remove_active_txt_entry(date_str, tid, start_time, obj_name):
    """Xóa bỏ dòng ghi nhận tạm thời nếu đối tượng xuất hiện quá ngắn (dưới 1 giây)"""
    class_name_vn = translate_class(obj_name)
    filename = os.path.join(TXT_DIR, f"Lich_su_{date_str}.txt")
    if not os.path.exists(filename):
        return

    target_prefix = f"{date_str} {start_time}: {class_name_vn} (ID {tid})"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        new_lines = [l for l in lines if not l.startswith(target_prefix)]
        if len(new_lines) != len(lines):
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
    except Exception:
        pass

def generate_html():
    check_day_transition()

    html_text = ""
    for data in active_objects.values():
        name_vn = translate_class(data['name'])
        if data.get('mode') == 'no_train':
            dwell_s = max(int(data.get('last_seen', 0) - data.get('enter_time', 0)), int(getattr(config, 'dwell_time', 5)))
            html_text += f"<div class='event-item' style='background: #fff0f0; padding: 8px; border-radius: 6px; border-left: 3px solid var(--red);'><div class='time' style='color: var(--red);'>{data['start_time_str']}</div><div class='desc'><b>{name_vn}:</b> <span style='color: var(--red);'>ở trong vùng nguy hiểm {dwell_s} giây</span></div></div>"
        else:
            html_text += f"<div class='event-item' style='background: #fff0f0; padding: 8px; border-radius: 6px; border-left: 3px solid var(--red);'><div class='time' style='color: var(--red);'>{data['start_time_str']}</div><div class='desc'><b>{name_vn}</b> <span style='color: var(--red);'>đang ở trong vùng nguy hiểm</span></div></div>"
        
    for ev in completed_events:
        desc = ev.get('desc', 'đã đi vào vùng nguy hiểm')
        html_text += f"<div class='event-item' style='padding: 8px;'><div class='time'>{ev['start']} - {ev['end']}</div><div class='desc'><b>{ev['name']}</b> {desc}</div></div>"
    return html_text

def draw_roi(frame):
    return util_draw_roi(frame, config.roi_polygon)

def process_and_draw(frame, results, fps, stream_type="webcam", current_time=None, current_dt=None, base_video_datetime=None):
    global active_objects, completed_events, current_date_str
    global is_recording, video_writer, temp_video_filename, recording_start_time_str, last_activity_time
    global pre_roll_buffer, pre_roll_meta, current_recording_meta, has_valid_event_in_recording, recording_start_date_str

    if current_dt is None:
        current_dt = datetime.datetime.now()
    if current_time is None:
        current_time = time.time()

    date_str = current_dt.strftime("%d-%m-%Y")
    time_str = current_dt.strftime("%H:%M:%S")
    file_time_str = current_dt.strftime("%Hh%Mm%Ss")

    h, w = frame.shape[:2]
    has_danger = False
    
    has_roi = config.roi_polygon and len(config.roi_polygon) >= 3
    if has_roi:
        pts = np.array([[int(p['x'] * w), int(p['y'] * h)] for p in config.roi_polygon], np.int32)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [pts], 255)
    
    frame = draw_roi(frame)
    # Bản sao khung hình sạch (có ROI nhưng KHÔNG vẽ đè bbox/ID) dùng để ghi video vi phạm
    rec_frame = frame.copy()
    current_ids_in_roi = set()
    current_frame_objs = []
    
    if results and len(results) > 0 and results[0].boxes is not None:
        classes = results[0].names
        for box in results[0].boxes:
            cls_id = int(box.cls)
            obj_name = classes[cls_id]
            
            # Bỏ qua nếu người dùng không chọn nhận diện lớp này trong Cài đặt
            if hasattr(config, 'selected_classes') and config.selected_classes is not None:
                if obj_name not in config.selected_classes and obj_name != "train":
                    continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            obj_name_vn = translate_class(obj_name)
            
            track_id = str(int(box.id[0])) if box.id is not None else "?"

            # Nếu là Tàu hỏa: Tàu lưu thông hợp lệ trên ray, KHÔNG vẽ bbox, KHÔNG lưu sự kiện, KHÔNG cảnh báo vi phạm
            if obj_name == "train":
                if track_id in active_objects:
                    data = active_objects[track_id]
                    remove_active_txt_entry(date_str, track_id, data.get("start_time_str", ""), data.get("name", "train"))
                    del active_objects[track_id]
                if track_id in roi_dwell_tracker:
                    del roi_dwell_tracker[track_id]
                continue

            label = f"{obj_name_vn} - ID: {track_id}"

            intersect = False
            if has_roi:
                intersect = check_roi_intersection((x1, y1, x2, y2), roi_mask, pts, (h, w))
            
            is_violation = False
            current_train_mode = getattr(config, 'train_mode', 'train_approaching')

            if intersect:
                dwell_limit = float(getattr(config, 'dwell_time', 5.0))
                is_violation, dwell_duration, label_suffix = update_dwell_and_evaluate_violation(
                    track_id, current_time, current_train_mode, dwell_limit,
                    roi_dwell_tracker, MIN_CONFIRM_SECONDS
                )
                label = f"{obj_name_vn} - ID: {track_id} {label_suffix}"

                if is_violation:
                    color = (0, 0, 255)  # Màu đỏ cảnh báo vi phạm
                    thickness = 2
                    label_bg = (0, 0, 255)
                    text_color = [255, 255, 255]
                    has_danger = True
                    current_ids_in_roi.add(track_id)
                    
                    enter_t = roi_dwell_tracker[track_id]["enter_time"]
                    enter_str = format_time_str(enter_t, stream_type, base_video_datetime)

                    if track_id not in active_objects:
                        active_objects[track_id] = {
                            "name": obj_name,
                            "mode": current_train_mode,
                            "entry_date": date_str,
                            "start_time_str": enter_str,
                            "enter_time": enter_t,
                            "last_seen": current_time,
                            "stream_type": stream_type,
                            "base_dt": base_video_datetime
                        }
                        dwell_s = int(current_time - enter_t) if current_train_mode == 'no_train' else 0
                        update_txt_file(date_str, track_id, enter_str, None, obj_name, "active", mode=current_train_mode, dwell_sec=dwell_s)
                    else:
                        active_objects[track_id]["last_seen"] = current_time
                else:
                    color = (0, 215, 255)  # Màu vàng cam (đối tượng mới chạm ROI hoặc chưa quá thời gian)
                    thickness = 2
                    label_bg = (0, 215, 255)
                    text_color = [255, 255, 255]
            else:
                if track_id in roi_dwell_tracker:
                    if current_time - roi_dwell_tracker[track_id]["last_seen"] > 1.5:
                        del roi_dwell_tracker[track_id]

            # Ghi nhận metadata tọa độ chuẩn hóa (0.0 -> 1.0)
            norm_box = [
                round(max(0.0, min(1.0, x1 / w)), 4),
                round(max(0.0, min(1.0, y1 / h)), 4),
                round(max(0.0, min(1.0, x2 / w)), 4),
                round(max(0.0, min(1.0, y2 / h)), 4)
            ]
            current_frame_objs.append({
                "id": track_id,
                "name": obj_name,
                "name_vn": obj_name_vn,
                "box": norm_box,
                "violating": bool(is_violation),
                "dwell": round(dwell_duration, 1) if is_violation else 0.0
            })

            if not intersect:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, label_bg, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
            
    ids_to_remove = []
    for tid, data in active_objects.items():
        if tid not in current_ids_in_roi:
            if current_time - data["last_seen"] > TIMEOUT_SECONDS:
                stype = data.get("stream_type", stream_type)
                bdt = data.get("base_dt", base_video_datetime)
                end_time_str = format_time_str(data["last_seen"], stype, bdt)
                mode = data.get("mode", "train_approaching")
                name_vn = translate_class(data["name"])

                entry_d = data.get("entry_date", date_str)

                # KIỂM TRA ĐỘ DÀI SỰ KIỆN: Nếu xuất hiện quá ngắn (< MIN_EVENT_DURATION) thì HỦY BỎ hoàn toàn
                total_duration = max(0.0, data["last_seen"] - data.get("enter_time", data["last_seen"]))
                if total_duration < MIN_EVENT_DURATION:
                    remove_active_txt_entry(entry_d, tid, data["start_time_str"], data["name"])
                    ids_to_remove.append(tid)
                    continue

                has_valid_event_in_recording = True
                overnight_note = f" (vắt qua ngày {date_str})" if entry_d != date_str else ""

                if mode == "no_train":
                    total_dwell = max(1, int(total_duration))
                    desc = f"ở trong vùng nguy hiểm {total_dwell} giây{overnight_note}"
                    update_txt_file(entry_d, tid, data["start_time_str"], end_time_str, data["name"], "done", mode="no_train", dwell_sec=total_dwell)
                else:
                    desc = f"đã đi vào vùng nguy hiểm{overnight_note}"
                    update_txt_file(entry_d, tid, data["start_time_str"], end_time_str, data["name"], "done", mode="train_approaching")

                completed_events.insert(0, {"id": tid, "name": name_vn, "start": data["start_time_str"], "end": end_time_str, "desc": desc})
                ids_to_remove.append(tid)

    for tid in ids_to_remove:
        del active_objects[tid]

    # In ngày giờ theo timeline video / đồng hồ thực lên cả live frame và recording frame
    cv2.putText(frame, f"{date_str} {time_str}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(rec_frame, f"{date_str} {time_str}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    rec_fps = fps if (fps and fps > 0) else 30.0

    # XỬ LÝ LƯU VIDEO VI PHẠM (PRE-ROLL VÀ POST-ROLL)
    if len(active_objects) > 0:
        last_activity_time = current_time
        if not is_recording:
            is_recording = True
            recording_start_date_str = date_str
            has_valid_event_in_recording = False
            current_recording_meta = []
            
            # Đổ toàn bộ metadata từ pre_roll_meta vào current_recording_meta
            for p_idx, p_objs in enumerate(pre_roll_meta):
                current_recording_meta.append({
                    "t": round(p_idx / rec_fps, 3),
                    "objs": p_objs
                })
            pre_roll_meta.clear()

            pre_roll_len = len(pre_roll_buffer)
            pre_roll_sec = pre_roll_len / rec_fps
            rec_start_t = max(0.0, current_time - pre_roll_sec)
            recording_start_time_str = format_file_time_str(rec_start_t, stream_type, base_video_datetime)
            
            day_dir = os.path.join(VID_DIR, date_str)
            os.makedirs(day_dir, exist_ok=True)
            temp_video_filename = os.path.join(day_dir, f"{date_str}_{recording_start_time_str}_dang_ghi.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_video_filename, fourcc, rec_fps, (w, h))
            
            # Ghi toàn bộ khung hình trong pre-roll buffer vào đầu video
            while len(pre_roll_buffer) > 0:
                buffered_frame = pre_roll_buffer.popleft()
                video_writer.write(buffered_frame)
    else:
        # Nếu không còn đối tượng vi phạm, duy trì ghi tiếp cho đến khi hết TIMEOUT_SECONDS (Post-roll)
        if is_recording and (current_time - last_activity_time > TIMEOUT_SECONDS):
            is_recording = False
            try:
                if video_writer:
                    video_writer.release()
                    video_writer = None

                    # Nếu không có sự kiện vi phạm hợp lệ nào (chỉ là nhiễu nháy) -> Xóa video tạm
                    if not has_valid_event_in_recording:
                        if os.path.exists(temp_video_filename):
                            try:
                                os.remove(temp_video_filename)
                            except Exception:
                                pass
                        current_recording_meta.clear()
                    else:
                        recording_end_time_str = file_time_str
                        rec_date = recording_start_date_str if recording_start_date_str else date_str
                        day_dir = os.path.join(VID_DIR, rec_date)
                        os.makedirs(day_dir, exist_ok=True)
                        final_filename = os.path.join(day_dir, f"{rec_date}_{recording_start_time_str}_den_{recording_end_time_str}.mp4")
                        try:
                            os.rename(temp_video_filename, final_filename)
                            save_recording_metadata(final_filename, rec_fps, current_recording_meta)
                            convert_video_to_h264_async(final_filename)
                        except Exception:
                            pass
                        current_recording_meta.clear()
            except Exception as e_rec:
                print(f"[!] Canh bao khi dong goi video vi pham: {e_rec}")
                current_recording_meta.clear()

    # Ghi frame sạch hiện tại vào video nếu đang ghi, hoặc lưu vào pre_roll_buffer nếu chưa ghi
    if is_recording and video_writer:
        try:
            video_writer.write(rec_frame)
            current_recording_meta.append({
                "t": round(len(current_recording_meta) / rec_fps, 3),
                "objs": current_frame_objs
            })
        except Exception:
            pass
    else:
        pre_roll_buffer.append(rec_frame.copy())
        pre_roll_meta.append(current_frame_objs)

    return frame, generate_html(), has_danger

def cleanup_recording(stream_type="webcam", current_dt=None, base_dt=None):
    global is_recording, video_writer, temp_video_filename, recording_start_time_str
    global active_objects, completed_events, pre_roll_buffer, has_valid_event_in_recording, recording_start_date_str
    
    if current_dt is None:
        now = datetime.datetime.now()
    else:
        now = current_dt

    date_str = now.strftime("%d-%m-%Y")
    
    for tid, data in list(active_objects.items()):
        entry_d = data.get("entry_date", date_str)
        total_duration = max(0.0, data["last_seen"] - data.get("enter_time", data["last_seen"]))
        if total_duration < MIN_EVENT_DURATION:
            remove_active_txt_entry(entry_d, tid, data["start_time_str"], data["name"])
            continue

        has_valid_event_in_recording = True
        stype = data.get("stream_type", stream_type)
        bdt = data.get("base_dt", base_dt)
        end_time_str = format_time_str(data["last_seen"], stype, bdt)
        mode = data.get("mode", "train_approaching")
        name_vn = translate_class(data["name"])

        overnight_note = f" (vắt qua ngày {date_str})" if entry_d != date_str else ""

        if mode == "no_train":
            total_dwell = max(1, int(total_duration))
            desc = f"ở trong vùng nguy hiểm {total_dwell} giây{overnight_note}"
            update_txt_file(entry_d, tid, data["start_time_str"], end_time_str, data["name"], "done", mode="no_train", dwell_sec=total_dwell)
        else:
            desc = f"đã đi vào vùng nguy hiểm{overnight_note}"
            update_txt_file(entry_d, tid, data["start_time_str"], end_time_str, data["name"], "done", mode="train_approaching")

        completed_events.insert(0, {"id": tid, "name": name_vn, "start": data["start_time_str"], "end": end_time_str, "desc": desc})
    
    active_objects.clear() 
    roi_dwell_tracker.clear()
    pre_roll_buffer.clear()
    pre_roll_meta.clear()

    if is_recording and video_writer:
        video_writer.release()
        video_writer = None
        is_recording = False
        if not has_valid_event_in_recording:
            if os.path.exists(temp_video_filename):
                try:
                    os.remove(temp_video_filename)
                except Exception:
                    pass
            current_recording_meta.clear()
        else:
            now_str = now.strftime("%Hh%Mm%Ss")
            day_str = recording_start_date_str if recording_start_date_str else (current_date_str if current_date_str else date_str)
            day_dir = os.path.join(VID_DIR, day_str)
            os.makedirs(day_dir, exist_ok=True)
            final_filename = os.path.join(day_dir, f"{day_str}_{recording_start_time_str}_den_{now_str}.mp4")
            try:
                os.rename(temp_video_filename, final_filename)
                save_recording_metadata(final_filename, 30.0, current_recording_meta)
                convert_video_to_h264_async(final_filename)
            except Exception:
                pass
            current_recording_meta.clear()


def reset_tracker():
    svc_reset_tracker(model)

def generate_frames(stream_type):
    global pre_roll_buffer, pre_roll_meta, current_recording_meta
    if stream_type in ("obs", "webcam"):
        obs_idx = find_obs_camera_index(config.load_obs_cam_from_file())
        print(f"[*] Đang mở camera OBS Virtual Camera (Index {obs_idx})...")
        cap = cv2.VideoCapture(obs_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(obs_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(config.TEMP_VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or np.isnan(fps) or fps <= 0 or fps > 120:
        fps = 30.0
    delay = 1.0 / fps

    # Đặt kích thước buffer cho pre-roll (3 giây)
    max_buf = max(30, int(fps * PRE_ROLL_SECONDS))
    pre_roll_buffer = deque(maxlen=max_buf)
    pre_roll_meta = deque(maxlen=max_buf)
    current_recording_meta.clear()
    
    frame_idx = 0
    base_video_datetime = datetime.datetime.now()
    current_dt = base_video_datetime
    current_time = 0.0

    reset_tracker()

    tracker_cfg = config.BYTETRACK_FILE if os.path.exists(config.BYTETRACK_FILE) else os.path.join(getattr(config, 'BUNDLE_DIR', config.BASE_DIR), "configs", "bytetrack.yaml")
    if not os.path.exists(tracker_cfg):
        tracker_cfg = "bytetrack.yaml"

    camera_retry = 0
    try:
        while cap.isOpened() and config.stream_active[stream_type]:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                if stream_type == "video":
                    # Video kết thúc -> Tự động phát lặp lại vô tận từ đầu tệp
                    try:
                        cleanup_recording(stream_type, current_dt, base_video_datetime)
                    except Exception:
                        pass
                    reset_tracker()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        try:
                            cap.release()
                            cap = cv2.VideoCapture(config.TEMP_VIDEO_PATH)
                            ret, frame = cap.read()
                        except Exception:
                            pass

                    if not ret or frame is None:
                        time.sleep(0.05)
                        continue

                    frame_idx = 0
                    base_video_datetime = datetime.datetime.now()
                    current_dt = base_video_datetime
                    current_time = 0.0
                else:
                    camera_retry += 1
                    if camera_retry < 60 and config.stream_active[stream_type]:
                        time.sleep(0.05)
                        continue
                    break
            camera_retry = 0
            
            frame_idx += 1
            if stream_type == "video":
                video_time_sec = frame_idx / fps
                current_time = video_time_sec
                current_dt = base_video_datetime + datetime.timedelta(seconds=video_time_sec)
            else:
                current_time = time.time()
                current_dt = datetime.datetime.now()

            if config.ai_active[stream_type]:
                if not config.roi_polygon or len(config.roi_polygon) < 3:
                    config.ai_active[stream_type] = False
                    if is_recording:
                        try:
                            cleanup_recording(stream_type, current_dt, base_video_datetime)
                        except Exception:
                            pass
                    res_img = draw_roi(frame.copy())
                    config.latest_data[stream_type] = {"text": generate_html(), "alarm": False}
                else:
                    conf = config.current_conf[stream_type]
                    try:
                        results = model.track(
                            source=frame, conf=conf, imgsz=640,
                            device=DEVICE, verbose=False, persist=True,
                            tracker=tracker_cfg
                        )
                    except Exception:
                        try:
                            results = model.track(
                                source=frame, conf=conf, imgsz=640,
                                device=DEVICE, verbose=False, persist=True,
                                tracker="bytetrack.yaml"
                            )
                        except Exception as e2:
                            print(f"[!] AI track error: {e2}")
                            results = None

                    try:
                        res_img, html_text, has_danger = process_and_draw(
                            frame, results, fps, stream_type=stream_type,
                            current_time=current_time, current_dt=current_dt, base_video_datetime=base_video_datetime
                        )
                        config.latest_data[stream_type] = {"text": html_text, "alarm": has_danger}
                    except Exception as e_proc:
                        print(f"[!] Canh bao loi xu ly AI (da cach ly an toan): {e_proc}")
                        res_img = draw_roi(frame.copy())
                        cv2.putText(res_img, f"{current_dt.strftime('%d-%m-%Y %H:%M:%S')}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                if is_recording:
                    try:
                        cleanup_recording(stream_type, current_dt, base_video_datetime)
                    except Exception:
                        pass
                res_img = draw_roi(frame.copy())
                cv2.putText(res_img, f"{current_dt.strftime('%d-%m-%Y %H:%M:%S')}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                config.latest_data[stream_type] = {"text": generate_html(), "alarm": False} 

            ret_enc, buffer = cv2.imencode('.jpg', res_img)
            if ret_enc:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            elapsed = time.time() - start_time
            if elapsed < delay:
                time.sleep(delay - elapsed)

    finally:
        cap.release()
        cleanup_recording(stream_type, current_dt, base_video_datetime) 
        reset_tracker()
        config.latest_data[stream_type] = {"text": generate_html(), "alarm": False}
