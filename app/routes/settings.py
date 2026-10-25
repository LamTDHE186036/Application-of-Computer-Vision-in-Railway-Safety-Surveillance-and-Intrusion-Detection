from flask import Blueprint, request, jsonify
from app import config

settings_bp = Blueprint('settings', __name__)

@settings_bp.route('/get_config', methods=['GET'])
def get_config():
    return jsonify({
        "polygon": config.load_roi_from_file(), 
        "conf": config.load_conf_from_file(),
        "classes": config.load_classes_from_file(),
        "dwell_time": config.load_dwell_from_file(),
        "train_mode": config.load_train_mode_from_file(),
        "droidcam_url": config.load_droidcam_url_from_file()
    })

@settings_bp.route('/set_droidcam_url', methods=['POST'])
def set_droidcam_url():
    data = request.json or {}
    url = data.get('url', '').strip()
    if url:
        config.droidcam_url = url
        config.save_droidcam_url_to_file(url)
    return jsonify({"status": "success", "droidcam_url": config.load_droidcam_url_from_file()})

@settings_bp.route('/set_train_mode', methods=['POST'])
def set_train_mode():
    data = request.json or {}
    mode = data.get('train_mode', 'train_approaching')
    if mode in ['train_approaching', 'no_train']:
        config.train_mode = mode
        config.save_train_mode_to_file(mode)
    return jsonify({"status": "success", "train_mode": config.train_mode})

@settings_bp.route('/set_roi', methods=['POST'])
def set_roi():
    data = request.json
    if data and data.get('enabled'):
        config.roi_polygon = data['polygon']
        config.save_roi_to_file(config.roi_polygon)
    else:
        config.roi_polygon = []
        config.save_roi_to_file([]) 
        
    return jsonify({"status": "success"})

@settings_bp.route('/set_conf', methods=['POST'])
def set_conf():
    data = request.json
    if data and 'conf' in data:
        val = float(data['conf'])
        config.current_conf["webcam"] = val
        config.current_conf["video"] = val
        config.current_conf["droidcam"] = val
        config.save_conf_to_file(val)
    return jsonify({"status": "success"})

@settings_bp.route('/set_settings', methods=['POST'])
def set_settings():
    data = request.json or {}
    if 'conf' in data:
        val = float(data['conf'])
        config.current_conf["webcam"] = val
        config.current_conf["video"] = val
        config.current_conf["droidcam"] = val
        config.save_conf_to_file(val)
    if 'classes' in data and isinstance(data['classes'], list):
        config.selected_classes = data['classes']
        config.save_classes_to_file(data['classes'])
    if 'dwell_time' in data:
        val = max(1.0, float(data['dwell_time']))
        config.dwell_time = val
        config.save_dwell_to_file(val)
    if 'droidcam_url' in data:
        url = str(data['droidcam_url']).strip()
        if url:
            config.droidcam_url = url
            config.save_droidcam_url_to_file(url)
    return jsonify({
        "status": "success",
        "conf": config.load_conf_from_file(),
        "classes": config.load_classes_from_file(),
        "dwell_time": config.load_dwell_from_file(),
        "train_mode": config.load_train_mode_from_file(),
        "droidcam_url": config.load_droidcam_url_from_file()
    })
