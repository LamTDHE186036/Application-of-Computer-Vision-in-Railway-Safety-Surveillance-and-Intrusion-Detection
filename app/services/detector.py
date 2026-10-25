import cv2
import numpy as np
from ultralytics import YOLO
import time
import datetime
import os
import re
from collections import deque

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
from app.services.tracker import (
    reset_tracker as svc_reset_tracker,
    check_roi_intersection,
    update_dwell_and_evaluate_violation
)

# --- THIẾT BỊ SUY LUẬN TỐI ƯU (OPENVINO INTEL GPU / CPU) ---
def get_optimal_device():
    try:
        import openvino as ov
        core = ov.Core()
        if "GPU" in core.available_devices:
            return "intel:gpu"
    except Exception:
        pass
    return "cpu"

# --- NẠP MÔ HÌNH YOLO TỪ THƯ MỤC MODELS/ ---
def load_yolo_model():
    base_dirs = [config.BASE_DIR]
    if hasattr(config, 'BUNDLE_DIR') and config.BUNDLE_DIR != config.BASE_DIR:
        base_dirs.append(config.BUNDLE_DIR)

    model_candidates = []
    for b_dir in base_dirs:
        model_candidates.extend([
            os.path.join(b_dir, "models", "yolo26s", "model_openvino"),
            os.path.join(b_dir, "models", "yolo26s", "model.pt"),
            os.path.join(b_dir, "models", "yolo11s", "model_openvino"),
            os.path.join(b_dir, "models", "yolo11s", "model.pt"),
        ])

    selected_model_path = None
    for p in model_candidates:
        if os.path.exists(p):
            selected_model_path = p
            break

    if not selected_model_path:
        raise FileNotFoundError("Không tìm thấy mô hình huấn luyện trong thư mục models/yolo26s hoặc models/yolo11s")

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
    return model_instance, device, selected_model_path

model, DEVICE, MODEL_PATH = load_yolo_model()
OPENVINO_MODEL_PATH = MODEL_PATH
PT_MODEL_PATH = MODEL_PATH

# --- THƯ MỤC LƯU TRỮ DỮ LIỆU RUNTIME (DATA/) ---
TXT_DIR = config.INTRUSION_LOGS_DIR
VID_DIR = config.RECORDINGS_DIR
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(VID_DIR, exist_ok=True)

# --- TỪ ĐIỂN DỊCH NHÃN YOLO SANG TIẾNG VIỆT CHUẨN ---
CLASS_NAMES_VI = {
    "person": "Người",
    "motorcycle": "Xe máy",
    "bicycle": "Xe đạp",
    "car": "Ô tô",
    "truck": "Xe tải",
    "bus": "Xe buýt",
    "truck-bus": "Xe tải/Xe buýt",
    "train": "Tàu hỏa"
}

def translate_class(eng_name):
    return CLASS_NAMES_VI.get(eng_name.lower(), eng_name)

# --- THAM SỐ CẤU HÌNH TỐI ƯU HÓA HỆ THỐNG ---
FRAME_SKIP = 2          # Chạy AI mỗi 2 frame để giảm tải CPU/GPU và loại bỏ giật lag
PRE_ROLL_SECONDS = 3.0  # Lưu trước sự kiện vi phạm 3 giây (Hộp đen Ring Buffer)
TIMEOUT_SECONDS = 3.0   # Duy trì ghi thêm 3 giây sau sự kiện (Post-roll) và timeout đối tượng
MIN_CONFIRM_SECONDS = 0.5  # Lọc nhiễu: Phải ở trong ROI >= 0.5s mới xác nhận vi phạm (chống chớp 1-2 frame)
MIN_EVENT_DURATION = 1.0   # Lọc nhiễu: Tổng thời lượng vi phạm phải >= 1.0s mới ghi lịch sử và lưu video

# --- BIẾN TOÀN CỤC QUẢN LÝ TRẠNG THÁI ---
active_objects = {}   
completed_events = [] 
current_date_str = ""

is_recording = False
video_writer = None
recording_start_time_str = ""
temp_video_filename = ""
last_activity_time = 0.0
has_valid_event_in_recording = False
roi_dwell_tracker = {}  # {track_id: {"enter_time": float, "last_seen": float, "violating": bool}}
pre_roll_buffer = deque(maxlen=90) # Bộ đệm xoay vòng lưu trước sự kiện

# --- HÀM KHÔI PHỤC LỊCH SỬ HÔM NAY KHI MỞ PHẦN MỀM ---
def load_today_events():
    global completed_events, current_date_str
    now = datetime.datetime.now()
    current_date_str = now.strftime("%d-%m-%Y")
    
    filename = os.path.join(TXT_DIR, f"Lich_su_{current_date_str}.txt")
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(r'(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2}): (.*?) \(ID (.*?)\) (.*)', line)
                if match:
                    start, end, name, tid, desc = match.groups()
                    completed_events.insert(0, {"id": tid, "name": name, "start": start, "end": end, "desc": desc.strip()})
                elif "đã đi vào" in line:
                    match_old = re.search(r'(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2}): (.*?) \(ID (.*?)\)', line)
                    if match_old:
                        start, end, name, tid = match_old.groups()
                        completed_events.insert(0, {"id": tid, "name": name, "start": start, "end": end, "desc": "đã đi vào vùng nguy hiểm"})

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
            
    if status == "active":
        found = False
        for i in range(len(lines)):
            if lines[i].startswith(target_prefix):
                lines[i] = target_str_active
                found = True
                break
        if not found:
            lines.append(target_str_active)
    elif status == "done":
        found = False
        for i in range(len(lines)):
            if lines[i].startswith(target_prefix):
                lines[i] = target_str_done
                found = True
                break
        if not found:
            lines.append(target_str_done)

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
    global completed_events, current_date_str
    
    now_date = datetime.datetime.now().strftime("%d-%m-%Y")
    if now_date != current_date_str:
        completed_events.clear()
        current_date_str = now_date

    html_text = ""
    for tid, data in active_objects.items():
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
    global pre_roll_buffer, has_valid_event_in_recording

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
    current_ids_in_roi = set()
    
    if results and len(results) > 0 and results[0].boxes is not None:
        classes = results[0].names
        for box in results[0].boxes:
            cls_id = int(box.cls)
            obj_name = classes[cls_id]
            
            # Bỏ qua nếu người dùng không chọn nhận diện lớp này trong Cài đặt
            if hasattr(config, 'selected_classes') and config.selected_classes is not None:
                is_selected = False
                if obj_name in config.selected_classes:
                    is_selected = True
                elif obj_name in ["truck", "bus"] and "truck-bus" in config.selected_classes:
                    is_selected = True
                elif obj_name == "bicycle" and "motorcycle" in config.selected_classes:
                    is_selected = True
                elif obj_name == "train":
                    is_selected = True

                if not is_selected:
                    continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf)
            obj_name_vn = translate_class(obj_name)
            
            track_id = str(int(box.id[0])) if box.id is not None else "?"
            label = f"{obj_name_vn} - ID: {track_id}"

            intersect = False
            if has_roi:
                intersect = check_roi_intersection((x1, y1, x2, y2), roi_mask, pts, (h, w))

            # Nếu là Tàu hỏa: Tàu lưu thông hợp lệ trên ray, KHÔNG vẽ bbox, KHÔNG lưu sự kiện, KHÔNG cảnh báo vi phạm
            if obj_name == "train":
                if track_id in active_objects:
                    data = active_objects[track_id]
                    remove_active_txt_entry(date_str, track_id, data.get("start_time_str", ""), data.get("name", "train"))
                    del active_objects[track_id]
                if track_id in roi_dwell_tracker:
                    del roi_dwell_tracker[track_id]
                continue
            
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

                # KIỂM TRA ĐỘ DÀI SỰ KIỆN: Nếu xuất hiện quá ngắn (< MIN_EVENT_DURATION) thì HỦY BỎ hoàn toàn
                total_duration = max(0.0, data["last_seen"] - data.get("enter_time", data["last_seen"]))
                if total_duration < MIN_EVENT_DURATION:
                    remove_active_txt_entry(date_str, tid, data["start_time_str"], data["name"])
                    ids_to_remove.append(tid)
                    continue

                has_valid_event_in_recording = True

                if mode == "no_train":
                    total_dwell = max(1, int(total_duration))
                    desc = f"ở trong vùng nguy hiểm {total_dwell} giây"
                    update_txt_file(date_str, tid, data["start_time_str"], end_time_str, data["name"], "done", mode="no_train", dwell_sec=total_dwell)
                else:
                    desc = "đã đi vào vùng nguy hiểm"
                    update_txt_file(date_str, tid, data["start_time_str"], end_time_str, data["name"], "done", mode="train_approaching")

                completed_events.insert(0, {"id": tid, "name": name_vn, "start": data["start_time_str"], "end": end_time_str, "desc": desc})
                ids_to_remove.append(tid)

    for tid in ids_to_remove:
        del active_objects[tid]

    # In ngày giờ theo timeline video / đồng hồ thực
    cv2.putText(frame, f"{date_str} {time_str}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # XỬ LÝ LƯU VIDEO VI PHẠM (PRE-ROLL VÀ POST-ROLL)
    if len(active_objects) > 0:
        last_activity_time = current_time
        if not is_recording:
            is_recording = True
            has_valid_event_in_recording = False
            pre_roll_len = len(pre_roll_buffer)
            pre_roll_sec = pre_roll_len / fps if fps > 0 else 0
            rec_start_t = max(0.0, current_time - pre_roll_sec)
            recording_start_time_str = format_file_time_str(rec_start_t, stream_type, base_video_datetime)
            
            day_dir = os.path.join(VID_DIR, date_str)
            os.makedirs(day_dir, exist_ok=True)
            temp_video_filename = os.path.join(day_dir, f"{date_str}_{recording_start_time_str}_dang_ghi.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_video_filename, fourcc, fps, (w, h))
            
            # Ghi toàn bộ khung hình trong pre-roll buffer vào đầu video
            while len(pre_roll_buffer) > 0:
                buffered_frame = pre_roll_buffer.popleft()
                video_writer.write(buffered_frame)
    else:
        # Nếu không còn đối tượng vi phạm, duy trì ghi tiếp cho đến khi hết TIMEOUT_SECONDS (Post-roll)
        if is_recording and (current_time - last_activity_time > TIMEOUT_SECONDS):
            is_recording = False
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
                else:
                    recording_end_time_str = file_time_str
                    day_dir = os.path.join(VID_DIR, date_str)
                    os.makedirs(day_dir, exist_ok=True)
                    final_filename = os.path.join(day_dir, f"{date_str}_{recording_start_time_str}_den_{recording_end_time_str}.mp4")
                    try:
                        os.rename(temp_video_filename, final_filename)
                        convert_video_to_h264_async(final_filename)
                    except Exception:
                        pass

    # Ghi frame hiện tại vào video nếu đang ghi, hoặc lưu vào pre_roll_buffer nếu chưa ghi
    if is_recording and video_writer:
        video_writer.write(frame)
    else:
        pre_roll_buffer.append(frame.copy())
            
    return frame, generate_html(), has_danger 

def cleanup_recording(stream_type="webcam", current_dt=None, base_dt=None):
    global is_recording, video_writer, temp_video_filename, recording_start_time_str
    global active_objects, completed_events, pre_roll_buffer, has_valid_event_in_recording
    
    if current_dt is None:
        now = datetime.datetime.now()
    else:
        now = current_dt

    date_str = now.strftime("%d-%m-%Y")
    
    for tid, data in list(active_objects.items()):
        total_duration = max(0.0, data["last_seen"] - data.get("enter_time", data["last_seen"]))
        if total_duration < MIN_EVENT_DURATION:
            remove_active_txt_entry(date_str, tid, data["start_time_str"], data["name"])
            continue

        has_valid_event_in_recording = True
        stype = data.get("stream_type", stream_type)
        bdt = data.get("base_dt", base_dt)
        end_time_str = format_time_str(data["last_seen"], stype, bdt)
        mode = data.get("mode", "train_approaching")
        name_vn = translate_class(data["name"])

        if mode == "no_train":
            total_dwell = max(1, int(total_duration))
            desc = f"ở trong vùng nguy hiểm {total_dwell} giây"
            update_txt_file(date_str, tid, data["start_time_str"], end_time_str, data["name"], "done", mode="no_train", dwell_sec=total_dwell)
        else:
            desc = "đã đi vào vùng nguy hiểm"
            update_txt_file(date_str, tid, data["start_time_str"], end_time_str, data["name"], "done", mode="train_approaching")

        completed_events.insert(0, {"id": tid, "name": name_vn, "start": data["start_time_str"], "end": end_time_str, "desc": desc})
    
    active_objects.clear() 
    roi_dwell_tracker.clear()
    pre_roll_buffer.clear()

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
        else:
            now_str = now.strftime("%Hh%Mm%Ss")
            day_str = current_date_str if current_date_str else date_str
            day_dir = os.path.join(VID_DIR, day_str)
            os.makedirs(day_dir, exist_ok=True)
            final_filename = os.path.join(day_dir, f"{day_str}_{recording_start_time_str}_den_{now_str}.mp4")
            try:
                os.rename(temp_video_filename, final_filename)
                convert_video_to_h264_async(final_filename)
            except Exception:
                pass

def reset_tracker():
    svc_reset_tracker(model)

def generate_frames(stream_type):
    global pre_roll_buffer
    if stream_type == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            try:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            except Exception:
                pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    elif stream_type == "droidcam":
        raw_url = config.load_droidcam_url_from_file()
        url = raw_url.strip()
        if not (url.startswith("http://") or url.startswith("https://") or url.startswith("rtsp://")):
            url = f"http://{url}"
        if not any(url.endswith(x) for x in ["/video", "/mjpegfeed"]):
            url = f"{url.rstrip('/')}/video"
        print(f"[*] Đang kết nối camera điện thoại (DroidCam): {url}")
        cap = cv2.VideoCapture(url)
    else:
        temp_path = os.path.join(config.BASE_DIR, "temp_video.mp4")
        cap = cv2.VideoCapture(temp_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or np.isnan(fps) or fps <= 0 or fps > 120:
        fps = 30.0
    delay = 1.0 / fps

    # Đặt kích thước buffer cho pre-roll (3 giây)
    pre_roll_buffer = deque(maxlen=max(30, int(fps * PRE_ROLL_SECONDS)))
    
    frame_idx = 0
    base_video_datetime = datetime.datetime.now()
    current_dt = base_video_datetime
    current_time = 0.0

    reset_tracker()

    tracker_cfg = config.BYTETRACK_FILE if os.path.exists(config.BYTETRACK_FILE) else os.path.join(getattr(config, 'BUNDLE_DIR', config.BASE_DIR), "configs", "bytetrack.yaml")
    if not os.path.exists(tracker_cfg):
        tracker_cfg = "bytetrack.yaml"

    try:
        while cap.isOpened() and config.stream_active[stream_type]:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                if stream_type == "video": 
                    cleanup_recording(stream_type, current_dt, base_video_datetime)
                    reset_tracker()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    base_video_datetime = datetime.datetime.now()
                    continue
                break
            
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
                        cleanup_recording(stream_type, current_dt, base_video_datetime)
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
                    except Exception as e:
                        try:
                            results = model.track(
                                source=frame, conf=conf, imgsz=640,
                                device=DEVICE, verbose=False, persist=True,
                                tracker="bytetrack.yaml"
                            )
                        except Exception as e2:
                            print(f"[!] AI track error: {e2}")
                            results = None

                    res_img, html_text, has_danger = process_and_draw(
                        frame, results, fps, stream_type=stream_type,
                        current_time=current_time, current_dt=current_dt, base_video_datetime=base_video_datetime
                    )
                    config.latest_data[stream_type] = {"text": html_text, "alarm": has_danger}
            else:
                if is_recording:
                    cleanup_recording(stream_type, current_dt, base_video_datetime)
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
