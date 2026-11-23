import os
import re
import cv2
import json
import time
import datetime
from flask import Blueprint, request, jsonify, Response, send_from_directory
from app import config

history_bp = Blueprint('history', __name__)

TXT_DIR = config.INTRUSION_LOGS_DIR
VID_DIR = config.RECORDINGS_DIR

@history_bp.route('/api/history/events', methods=['GET'])
def get_history_events():
    date_param = request.args.get('date', '').strip()
    if not date_param:
        date_param = datetime.datetime.now().strftime("%d-%m-%Y")
    
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_param):
        parts = date_param.split('-')
        date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
    else:
        date_str = date_param

    def parse_t(s):
        p = list(map(int, s.replace('h', ':').replace('m', ':').replace('s', '').split(':')))
        return p[0] * 3600 + p[1] * 60 + p[2]

    vid_dir = os.path.join(VID_DIR, date_str)
    videos = []
    video_track_ids = {}
    video_has_meta = {}
    if os.path.exists(vid_dir):
        for f in sorted(os.listdir(vid_dir)):
            if f.endswith('.mp4') and not f.endswith('.h264.mp4') and not f.endswith('_dang_ghi.mp4'):
                m = re.search(r'(\d{2}h\d{2}m\d{2}s)_den_(\d{2}h\d{2}m\d{2}s)', f)
                if m:
                    # Nạp danh sách các Track ID có trong tệp JSON đi kèm video để phục vụ khớp chính xác 100%
                    v_meta_fn = os.path.splitext(f)[0] + ".json"
                    v_meta_path = os.path.join(vid_dir, v_meta_fn)
                    track_ids = set()
                    has_v_meta = os.path.exists(v_meta_path)
                    if has_v_meta:
                        try:
                            with open(v_meta_path, 'r', encoding='utf-8') as jf:
                                jd = json.load(jf)
                                for fr in jd.get('frames', []):
                                    for ob in fr.get('objs', []):
                                        if 'id' in ob and ob['id'] is not None:
                                            track_ids.add(str(ob['id']))
                        except Exception:
                            pass

                    video_track_ids[f] = track_ids
                    video_has_meta[f] = has_v_meta
                    videos.append({
                        'filename': f,
                        'start_sec': parse_t(m.group(1)),
                        'end_sec': parse_t(m.group(2))
                    })

    def normalize_obj_name(name):
        if not name:
            return name
        n = name.strip()
        if n in ["Xe máy", "motorcycle"]:
            return "Xe máy"
        if n in ["Ô tô", "car"]:
            return "Ô tô"
        if n in ["Xe tải/Xe buýt", "truck-bus"]:
            return "Xe tải/Xe buýt"
        if n in ["Người", "person"]:
            return "Người"
        return n

    txt_path = os.path.join(TXT_DIR, f"Lich_su_{date_str}.txt")
    events = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                m = re.search(r'(?:(\d{2}-\d{2}-\d{4})\s+)?(\d{2}:\d{2}:\d{2})(?:\s*-\s*(\d{2}:\d{2}:\d{2}))?:\s*(.*?)\s*\(ID\s*(.*?)\)\s*(.*)', line)
                if m:
                    _, start_t, end_t, obj_name, tid, desc = m.groups()
                    if not end_t:
                        end_t = start_t
                    p = list(map(int, start_t.split(':')))
                    ev_sec = p[0] * 3600 + p[1] * 60 + p[2]
                    target_tid_str = str(tid).strip() if tid else None

                    # THUẬT TOÁN GHÉP NỐI ĐA TẦNG (MULTI-TIER MATCHING) - KHẮC PHỤC TRIỆT ĐỂ LỖI GÁN LỆCH CLIP:
                    best_v = None

                    # Tầng 1 (Ưu tiên tuyệt đối): Video bao trùm thời gian VÀ có chứa đúng Track ID trong tệp metadata JSON
                    for v in videos:
                        if v['start_sec'] - 2 <= ev_sec <= v['end_sec'] + 2:
                            if target_tid_str and target_tid_str in video_track_ids.get(v['filename'], set()):
                                best_v = v
                                break

                    # Tầng 2: Mốc sự kiện nằm THỰC SỰ BÊN TRONG khoảng thời gian của video (start_sec <= ev_sec <= end_sec)
                    # Loại bỏ hoàn toàn việc nới lỏng +4s ở đuôi khiến sự kiện bị gán nhầm sang clip trước đó!
                    if not best_v:
                        for v in videos:
                            if v['start_sec'] <= ev_sec <= v['end_sec']:
                                best_v = v
                                break

                    # Tầng 3: Nếu vẫn chưa có, tìm video lân cận trong bán kính 10 giây có chứa đúng Track ID
                    if not best_v and target_tid_str:
                        for v in videos:
                            if min(abs(ev_sec - v['start_sec']), abs(ev_sec - v['end_sec'])) <= 10:
                                if target_tid_str in video_track_ids.get(v['filename'], set()):
                                    best_v = v
                                    break

                    # Tầng 4: Tìm video gần nhất theo khoảng cách thời gian (dung sai tối đa 60 giây)
                    if not best_v:
                        min_diff = 999999
                        for v in videos:
                            diff = min(abs(ev_sec - v['start_sec']), abs(ev_sec - v['end_sec']))
                            if diff < min_diff and diff <= 60:
                                min_diff = diff
                                best_v = v

                    fn = best_v['filename'] if best_v else None
                    v_url = f"/history_video/{date_str}/{fn}" if fn else None
                    v_dur = max(0, best_v['end_sec'] - best_v['start_sec']) if best_v else 0

                    offset_sec = max(0, ev_sec - best_v['start_sec']) if best_v else 0
                    has_meta = bool(fn and video_has_meta.get(fn, False))
                    meta_fn = os.path.splitext(fn)[0] + ".json" if has_meta else None
                    meta_url = f"/history_metadata/{date_str}/{meta_fn}" if has_meta else None

                    events.append({
                        'date': date_str,
                        'start_time': start_t,
                        'end_time': end_t,
                        'object_name': normalize_obj_name(obj_name),
                        'track_id': tid,
                        'desc': desc.strip(),
                        'video_file': fn,
                        'video_url': v_url,
                        'video_duration': v_dur,
                        'offset_sec': offset_sec,
                        'has_metadata': has_meta,
                        'metadata_url': meta_url
                    })

    events_reversed = list(reversed(events))

    return jsonify({
        "status": "success",
        "date": date_str,
        "events": events_reversed,
        "videos": videos
    })

@history_bp.route('/history_metadata/<date_str>/<filename>')
def serve_history_metadata(date_str, filename):
    dir_path = os.path.abspath(os.path.join(VID_DIR, date_str))
    if not filename.endswith('.json') or not os.path.exists(os.path.join(dir_path, filename)):
        return jsonify({"error": "Metadata không tồn tại"}), 404

    return send_from_directory(dir_path, filename, mimetype='application/json')

@history_bp.route('/history_video/<date_str>/<filename>')
def serve_history_video(date_str, filename):
    dir_path = os.path.abspath(os.path.join(VID_DIR, date_str))
    actual_path = os.path.join(dir_path, filename)
    tmp_converted = actual_path + ".h264.mp4"

    # Nếu có tệp nén H.264 đang chờ, thử thay thế ngay lập tức
    if os.path.exists(tmp_converted):
        try:
            os.replace(tmp_converted, actual_path)
        except Exception:
            # Nếu actual_path vẫn đang bị tiến trình khác giữ khóa, phục vụ thẳng tệp H.264 đã nén hoàn chỉnh
            return send_from_directory(dir_path, filename + ".h264.mp4", mimetype='video/mp4', conditional=True)

    if not os.path.exists(actual_path):
        return "Video không tồn tại", 404

    resp = send_from_directory(dir_path, filename, mimetype='video/mp4', conditional=True)
    resp.headers['Accept-Ranges'] = 'bytes'
    return resp

@history_bp.route('/history_video_stream')
def stream_history_video():
    date_str = request.args.get('date', '').strip()
    filename = request.args.get('file', '').strip()
    try:
        start_sec = float(request.args.get('start_sec', '0'))
    except (ValueError, TypeError):
        start_sec = 0.0

    filepath = os.path.join(VID_DIR, date_str, filename)
    if not os.path.exists(filepath):
        return "Video không tồn tại", 404

    def stream_gen():
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        if start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
        delay = 1.0 / max(5, min(fps, 60))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(delay)
        cap.release()

    return Response(stream_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
