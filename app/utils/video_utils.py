import os
import cv2
import numpy as np
import datetime
import subprocess
import threading
import queue
import time

import sys
import shutil

_cached_ffmpeg_cmd = None

_video_queue = queue.Queue()
_worker_thread = None
_worker_lock = threading.Lock()
_is_converting = False
_stop_worker_signal = False

def get_ffmpeg_path():
    """
    Tự động dò tìm đường dẫn ffmpeg khả dụng theo thứ tự ưu tiên:
    1. Trong thư mục ứng dụng (ngang hàng với .exe hoặc main.py)
    2. Trong thư mục bundle PyInstaller (_internal hoặc _MEIPASS)
    3. Thư mục bin/
    4. Từ thư viện imageio_ffmpeg
    5. Từ biến môi trường PATH hệ thống
    """
    global _cached_ffmpeg_cmd
    if _cached_ffmpeg_cmd and os.path.isfile(_cached_ffmpeg_cmd):
        return _cached_ffmpeg_cmd

    search_dirs = []
    try:
        from app import config
        if hasattr(config, 'BASE_DIR') and config.BASE_DIR:
            search_dirs.append(config.BASE_DIR)
        if hasattr(config, 'BUNDLE_DIR') and config.BUNDLE_DIR:
            search_dirs.append(config.BUNDLE_DIR)
    except Exception:
        pass

    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        search_dirs.extend([exe_dir, getattr(sys, '_MEIPASS', exe_dir)])
    else:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        search_dirs.append(root_dir)

    for b_dir in search_dirs:
        for fname in ("ffmpeg.exe", "ffmpeg"):
            candidate = os.path.join(b_dir, fname)
            if os.path.isfile(candidate):
                _cached_ffmpeg_cmd = candidate
                return candidate
            candidate_bin = os.path.join(b_dir, "bin", fname)
            if os.path.isfile(candidate_bin):
                _cached_ffmpeg_cmd = candidate_bin
                return candidate_bin

    # Dò từ imageio_ffmpeg nếu có
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.isfile(p):
            _cached_ffmpeg_cmd = p
            return p
    except Exception:
        pass

    # Dò từ PATH hệ thống
    sys_path = shutil.which("ffmpeg")
    if sys_path:
        _cached_ffmpeg_cmd = sys_path
        return sys_path

    return None

def _process_single_video(video_path):
    if not os.path.exists(video_path):
        return

    ffmpeg_bin = get_ffmpeg_path()
    if not ffmpeg_bin:
        print(f"[!] Canh bao: Khong tim thay ffmpeg.exe! Video '{os.path.basename(video_path)}' giu nguyen dinh dang goc.")
        return

    tmp_converted = video_path + ".h264.mp4"
    cmd = [
        ffmpeg_bin, '-y', '-i', video_path,
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-movflags', '+faststart',
        tmp_converted
    ]

    creation_flags = 0
    if sys.platform.startswith('win'):
        creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags
        )
        if res.returncode == 0 and os.path.exists(tmp_converted) and os.path.getsize(tmp_converted) > 1000:
            os.replace(tmp_converted, video_path)
            print(f"[*] Da chuyen doi thanh cong video sang chuan web H.264: {os.path.basename(video_path)}")
        elif os.path.exists(tmp_converted):
            os.remove(tmp_converted)
    except Exception as e:
        print(f"[!] Loi khi thuc hien chuyen doi video bang ffmpeg: {e}")
        if os.path.exists(tmp_converted):
            try:
                os.remove(tmp_converted)
            except Exception:
                pass

def _video_worker_loop():
    global _is_converting
    while True:
        try:
            video_path = _video_queue.get(timeout=1.0)
        except queue.Empty:
            if _stop_worker_signal:
                break
            continue

        if video_path is None:
            _video_queue.task_done()
            break

        _is_converting = True
        try:
            _process_single_video(video_path)
        finally:
            _is_converting = False
            _video_queue.task_done()

def _ensure_worker_started():
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_video_worker_loop, daemon=True, name="FFmpegWorker")
            _worker_thread.start()

def convert_video_to_h264_async(video_path):
    """
    Đưa video vào hàng đợi (Queue) để Worker chuyển đổi tuần tự sang H.264 (+faststart),
    đảm bảo chỉ chạy tối đa 1 tiến trình ffmpeg tại 1 thời điểm, tránh tràn CPU/RAM.
    """
    if not video_path:
        return
    _ensure_worker_started()
    _video_queue.put(video_path)
    pending = _video_queue.qsize()
    if pending > 1:
        print(f"[*] Da dua video '{os.path.basename(video_path)}' vao hang doi chuyen doi H.264 (Dang cho: {pending} video).")

def wait_for_video_converter(timeout=30):
    """
    Đợi Worker giải quyết nốt các video còn tồn trong hàng đợi trước khi tắt ứng dụng.
    """
    global _stop_worker_signal
    total_pending = _video_queue.unfinished_tasks
    if total_pending <= 0 and not _is_converting:
        return

    print("\n" + "=" * 65)
    print("  [*] Dang hoan tat luu cac video cuoi cung, vui long cho vai giay...")
    print(f"      (Con lai: {total_pending} video can chuyen doi sang H.264)")
    print("=" * 65)

    start_t = time.time()
    while _video_queue.unfinished_tasks > 0 or _is_converting:
        if time.time() - start_t > timeout:
            print("[!] Canh bao: Het thoi gian cho xu ly video, ung dung se dong.")
            break
        time.sleep(0.3)

    _stop_worker_signal = True
    print("[*] Da hoan tat toan bo video trong hang doi.")

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
