from core import ai_detector
from flask import request, jsonify, Response, render_template, send_from_directory
import cv2
import base64
from core import configs as config
import os
import re
import datetime
import time
import core.ai_detector     
from core.ai_detector import generate_frames
import atexit  # Thêm thư viện canh gác khi tắt app

# --- HÀM TỰ ĐỘNG DỌN RÁC KHI TẮT PHẦN MỀM ---
def cleanup_on_shutdown():
    if os.path.exists("temp_video.mp4"):
        try:
            os.remove("temp_video.mp4")
        except:
            pass

# Đăng ký hàm dọn dẹp chạy tự động khi tắt hệ thống
atexit.register(cleanup_on_shutdown)

def register_routes(app):
    @app.route('/upload_video', methods=['POST'])
    def upload_video():
        file = request.files['video']
        file.save("temp_video.mp4")
        
        cap = cv2.VideoCapture("temp_video.mp4")
        ret, frame = cap.read()
        cap.release()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({"status": "OK", "frame": f"data:image/jpeg;base64,{img_b64}"})
        return jsonify({"status": "OK", "frame": ""})

    @app.route('/delete_video', methods=['POST'])
    def delete_video():
        try:
            if os.path.exists("temp_video.mp4"):
                os.remove("temp_video.mp4")
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})

    @app.route('/get_config', methods=['GET'])
    def get_config():
        return jsonify({
            "polygon": config.load_roi_from_file(), 
            "conf": config.load_conf_from_file(),
            "classes": config.load_classes_from_file(),
            "dwell_time": config.load_dwell_from_file(),
            "train_mode": config.load_train_mode_from_file()
        })

    @app.route('/set_train_mode', methods=['POST'])
    def set_train_mode():
        data = request.json or {}
        mode = data.get('train_mode', 'train_approaching')
        if mode in ['train_approaching', 'no_train']:
            config.train_mode = mode
            config.save_train_mode_to_file(mode)
        return jsonify({"status": "success", "train_mode": config.train_mode})

    @app.route('/set_roi', methods=['POST'])
    def set_roi():
        data = request.json
        if data and data.get('enabled'):
            config.roi_polygon = data['polygon']
            config.save_roi_to_file(config.roi_polygon)
        else:
            config.roi_polygon = []
            config.save_roi_to_file([]) 
            
        return jsonify({"status": "success"})

    @app.route('/set_conf', methods=['POST'])
    def set_conf():
        data = request.json
        if data and 'conf' in data:
            val = float(data['conf'])
            config.current_conf["webcam"] = val
            config.current_conf["video"] = val
            config.save_conf_to_file(val)
        return jsonify({"status": "success"})

    @app.route('/set_settings', methods=['POST'])
    def set_settings():
        data = request.json or {}
        if 'conf' in data:
            val = float(data['conf'])
            config.current_conf["webcam"] = val
            config.current_conf["video"] = val
            config.save_conf_to_file(val)
        if 'classes' in data and isinstance(data['classes'], list):
            config.selected_classes = data['classes']
            config.save_classes_to_file(data['classes'])
        if 'dwell_time' in data:
            val = max(1.0, float(data['dwell_time']))
            config.dwell_time = val
            config.save_dwell_to_file(val)
        return jsonify({
            "status": "success",
            "conf": config.load_conf_from_file(),
            "classes": config.load_classes_from_file(),
            "dwell_time": config.load_dwell_from_file(),
            "train_mode": config.load_train_mode_from_file()
        })

    @app.route('/start_stream/<stream_type>')
    def start_stream(stream_type):
        config.stream_active[stream_type] = True
        config.ai_active[stream_type] = request.args.get('ai', '1') == '1'
        config.current_conf[stream_type] = float(request.args.get('conf', config.load_conf_from_file()))
        return "Started"

    @app.route('/set_ai/<stream_type>')
    def set_ai(stream_type):
        config.ai_active[stream_type] = request.args.get('ai', '1') == '1'
        return "OK"

    @app.route('/stop_stream/<stream_type>')
    def stop_stream(stream_type):
        config.stream_active[stream_type] = False
        return "Stopped"

    @app.route('/video_feed/<stream_type>')
    def video_feed(stream_type):
        return Response(generate_frames(stream_type), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/get_data/<stream_type>')
    def get_data(stream_type):
        return jsonify(config.latest_data.get(stream_type, {"text": "", "alarm": False}))

    @app.route('/get_events', methods=['GET'])
    def get_events():
        return jsonify({"text": ai_detector.generate_html()})

    # ===== API LỊCH SỬ VI PHẠM & VIDEO =====
    @app.route('/api/history/dates', methods=['GET'])
    def get_history_dates():
        dates = set()
        if os.path.exists("lichsu_xampham"):
            for f in os.listdir("lichsu_xampham"):
                m = re.match(r'Lich_su_(\d{2}-\d{2}-\d{4})\.txt', f)
                if m:
                    dates.add(m.group(1))
        if os.path.exists("lichsu_video"):
            for d in os.listdir("lichsu_video"):
                if os.path.isdir(os.path.join("lichsu_video", d)) and re.match(r'^\d{2}-\d{2}-\d{4}$', d):
                    dates.add(d)

        def parse_d(d_str):
            try:
                return datetime.datetime.strptime(d_str, "%d-%m-%Y")
            except:
                return datetime.datetime.min

        sorted_dates = sorted(list(dates), key=parse_d, reverse=True)
        today_str = datetime.datetime.now().strftime("%d-%m-%Y")
        if today_str not in sorted_dates:
            sorted_dates.insert(0, today_str)

        return jsonify({"dates": sorted_dates, "today": today_str})

    @app.route('/api/history/events', methods=['GET'])
    def get_history_events():
        date_param = request.args.get('date', '').strip()
        if not date_param:
            date_param = datetime.datetime.now().strftime("%d-%m-%Y")
        
        # Chuẩn hóa về định dạng DD-MM-YYYY
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_param):
            parts = date_param.split('-')
            date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
        else:
            date_str = date_param

        vid_dir = os.path.join("lichsu_video", date_str)
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

        txt_path = os.path.join("lichsu_xampham", f"Lich_su_{date_str}.txt")
        events = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                for idx, raw_line in enumerate(f):
                    line = raw_line.strip()
                    if not line:
                        continue
                    m = re.search(r'(?:(\d{2}-\d{2}-\d{4})\s+)?(\d{2}:\d{2}:\d{2})(?:\s*-\s*(\d{2}:\d{2}:\d{2}))?:\s*(.*?)\s*\(ID\s*(.*?)\)\s*(.*)', line)
                    if m:
                        d_prefix, start_t, end_t, obj_name, tid, desc = m.groups()
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

        # Đảo ngược để sự kiện mới nhất lên trên
        events_reversed = list(reversed(events))

        return jsonify({
            "status": "success",
            "date": date_str,
            "total": len(events),
            "events": events_reversed,
            "videos": videos
        })

    @app.route('/history_video/<date_str>/<filename>')
    def serve_history_video(date_str, filename):
        dir_path = os.path.abspath(os.path.join("lichsu_video", date_str))
        if not os.path.exists(os.path.join(dir_path, filename)):
            return "Video không tồn tại", 404
        resp = send_from_directory(dir_path, filename, mimetype='video/mp4', conditional=True)
        resp.headers['Accept-Ranges'] = 'bytes'
        return resp

    @app.route('/history_video_stream')
    def stream_history_video():
        date_str = request.args.get('date', '').strip()
        filename = request.args.get('file', '').strip()
        try:
            start_sec = float(request.args.get('start_sec', '0'))
        except (ValueError, TypeError):
            start_sec = 0.0
        filepath = os.path.join("lichsu_video", date_str, filename)
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

    @app.route('/')
    def index():
        return render_template('index.html')