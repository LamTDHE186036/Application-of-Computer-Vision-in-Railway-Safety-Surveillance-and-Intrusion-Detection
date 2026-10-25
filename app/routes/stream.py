import os
import cv2
import base64
from flask import Blueprint, request, jsonify, Response, render_template

from app import config
from app.services import detector

stream_bp = Blueprint('stream', __name__)

@stream_bp.route('/')
def index():
    return render_template('index.html')

@stream_bp.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['video']
    temp_path = os.path.join(config.BASE_DIR, "temp_video.mp4")
    file.save(temp_path)
    
    cap = cv2.VideoCapture(temp_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"status": "OK", "frame": f"data:image/jpeg;base64,{img_b64}"})
    return jsonify({"status": "OK", "frame": ""})

@stream_bp.route('/delete_video', methods=['POST'])
def delete_video():
    try:
        temp_path = os.path.join(config.BASE_DIR, "temp_video.mp4")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@stream_bp.route('/start_stream/<stream_type>')
def start_stream(stream_type):
    config.stream_active[stream_type] = True
    config.ai_active[stream_type] = request.args.get('ai', '1') == '1'
    config.current_conf[stream_type] = float(request.args.get('conf', config.load_conf_from_file()))
    return "Started"

@stream_bp.route('/set_ai/<stream_type>')
def set_ai(stream_type):
    config.ai_active[stream_type] = request.args.get('ai', '1') == '1'
    return "OK"

@stream_bp.route('/stop_stream/<stream_type>')
def stop_stream(stream_type):
    config.stream_active[stream_type] = False
    return "Stopped"

@stream_bp.route('/video_feed/<stream_type>')
def video_feed(stream_type):
    return Response(detector.generate_frames(stream_type), mimetype='multipart/x-mixed-replace; boundary=frame')

@stream_bp.route('/get_data/<stream_type>')
def get_data(stream_type):
    return jsonify(config.latest_data.get(stream_type, {"text": "", "alarm": False}))

@stream_bp.route('/get_events', methods=['GET'])
def get_events():
    return jsonify({"text": detector.generate_html()})
