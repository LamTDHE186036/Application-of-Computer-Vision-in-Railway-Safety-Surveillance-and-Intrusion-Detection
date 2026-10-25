import os
import cv2
import numpy as np
import datetime
import subprocess
import threading

def convert_video_to_h264_async(video_path):
    """
    Tự động chuyển đổi video đã ghi sang chuẩn H.264 (+faststart) để tương thích 100% với trình duyệt web
    """
    def _run():
        if not os.path.exists(video_path):
            return
        tmp_converted = video_path + ".h264.mp4"
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-movflags', '+faststart',
            tmp_converted
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.exists(tmp_converted) and os.path.getsize(tmp_converted) > 1000:
                os.replace(tmp_converted, video_path)
            elif os.path.exists(tmp_converted):
                os.remove(tmp_converted)
        except Exception:
            if os.path.exists(tmp_converted):
                try:
                    os.remove(tmp_converted)
                except Exception:
                    pass

    threading.Thread(target=_run, daemon=True).start()

def format_time_str(t, stream_type="webcam", base_dt=None):
    """
    Định dạng thời gian theo hệ quy chiếu video hoặc đồng hồ thực (HH:MM:SS)
    """
    if stream_type == "video" and base_dt is not None:
        return (base_dt + datetime.timedelta(seconds=float(t))).strftime("%H:%M:%S")
    try:
        return datetime.datetime.fromtimestamp(float(t)).strftime("%H:%M:%S")
    except Exception:
        return datetime.datetime.now().strftime("%H:%M:%S")

def format_file_time_str(t, stream_type="webcam", base_dt=None):
    """
    Định dạng thời gian để đặt tên file an toàn (HHhMMmSSs)
    """
    if stream_type == "video" and base_dt is not None:
        return (base_dt + datetime.timedelta(seconds=float(t))).strftime("%Hh%Mm%Ss")
    try:
        return datetime.datetime.fromtimestamp(float(t)).strftime("%Hh%Mm%Ss")
    except Exception:
        return datetime.datetime.now().strftime("%Hh%Mm%Ss")

def draw_roi(frame, roi_polygon):
    """
    Vẽ vùng nhận diện ROI (vàng mờ và viền vàng cam) lên khung hình
    """
    if roi_polygon and len(roi_polygon) >= 3:
        h, w = frame.shape[:2]
        pts = np.array([[int(p['x'] * w), int(p['y'] * h)] for p in roi_polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 200, 255), 2)
    return frame
