import os
import re
import cv2
import time
import datetime
from flask import Blueprint, request, jsonify, Response, send_from_directory
from app import config

history_bp = Blueprint('history', __name__)

TXT_DIR = config.INTRUSION_LOGS_DIR
VID_DIR = config.RECORDINGS_DIR

@history_bp.route('/api/history/dates', methods=['GET'])
def get_history_dates():
    dates = set()
    
    if os.path.exists(TXT_DIR):
        for f in os.listdir(TXT_DIR):
            m = re.match(r'Lich_su_(\d{2}-\d{2}-\d{4})\.txt', f)
            if m:
                dates.add(m.group(1))

    if os.path.exists(VID_DIR):
        for d in os.listdir(VID_DIR):
            if os.path.isdir(os.path.join(VID_DIR, d)) and re.match(r'^\d{2}-\d{2}-\d{4}$', d):
                dates.add(d)

    def parse_d(d_str):
        try:
            return datetime.datetime.strptime(d_str, "%d-%m-%Y")
        except Exception:
            return datetime.datetime.min

    sorted_dates = sorted(list(dates), key=parse_d, reverse=True)
    today_str = datetime.datetime.now().strftime("%d-%m-%Y")
    if today_str not in sorted_dates:
        sorted_dates.insert(0, today_str)

    return jsonify({"dates": sorted_dates, "today": today_str})

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

    vid_dir = os.path.join(VID_DIR, date_str)
    videos = []
    if os.path.exists(vid_dir):
        for f in sorted(os.listdir(vid_dir)):
            if f.endswith('.mp4'):
                m = re.search(r'(\d{2}h\d{2}m\d{2}s)_den_(\d{2}h\d{2}m\d{2}s)', f)
                if m:
                    def parse_t(s):
                        p = list(map(int, s.replace('h', ':').replace('m', ':').replace('s', '').split(':')))
                        return p[0] * 3600 + p[1] * 60 + p[2]
                    videos.append({
                        'filename': f,
                        'start_sec': parse_t(m.group(1)),
                        'end_sec': parse_t(m.group(2)),
                        'start_time': m.group(1).replace('h', ':').replace('m', ':').replace('s', ''),
                        'end_time': m.group(2).replace('h', ':').replace('m', ':').replace('s', '')
                    })

    txt_path = os.path.join(TXT_DIR, f"Lich_su_{date_str}.txt")
    events = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            for idx, raw_line in enumerate(f):
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

                    best_v = None
                    min_diff = 999999
                    for v in videos:
                        if v['start_sec'] - 4 <= ev_sec <= v['end_sec'] + 4:
                            best_v = v
                            break
                        diff = min(abs(ev_sec - v['start_sec']), abs(ev_sec - v['end_sec']))
                        if diff < min_diff and diff <= 60:
                            min_diff = diff
                            best_v = v

                    fn = best_v['filename'] if best_v else None
                    v_url = f"/history_video/{date_str}/{fn}" if fn else None
                    stream_url = f"/history_video_stream?date={date_str}&file={fn}" if fn else None
                    v_dur = max(0, best_v['end_sec'] - best_v['start_sec']) if best_v else 0
                    events.append({
                        'index': idx + 1,
                        'date': date_str,
                        'start_time': start_t,
                        'end_time': end_t,
                        'object_name': obj_name,
                        'track_id': tid,
                        'desc': desc.strip(),
                        'video_file': fn,
                        'video_url': v_url,
                        'video_stream_url': stream_url,
                        'video_duration': v_dur
                    })

    events_reversed = list(reversed(events))

    return jsonify({
        "status": "success",
        "date": date_str,
        "total": len(events),
        "events": events_reversed,
        "videos": videos
    })

@history_bp.route('/history_video/<date_str>/<filename>')
def serve_history_video(date_str, filename):
    dir_path = os.path.abspath(os.path.join(VID_DIR, date_str))
    if not os.path.exists(os.path.join(dir_path, filename)):
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
