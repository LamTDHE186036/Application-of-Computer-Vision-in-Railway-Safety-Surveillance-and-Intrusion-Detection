from flask import Blueprint, request, jsonify
from app import config
from app.utils.camera_utils import get_system_cameras, find_obs_camera_index

settings_bp = Blueprint('settings', __name__)

@settings_bp.route('/get_config', methods=['GET'])
def get_config():
    return jsonify({
        "polygon": config.load_roi_from_file(), 
        "conf": config.load_conf_from_file(),
        "classes": config.load_classes_from_file(),
        "dwell_time": config.load_dwell_from_file(),
        "train_mode": config.load_train_mode_from_file(),
        "obs_cam": config.load_obs_cam_from_file()
    })

@settings_bp.route('/get_cameras', methods=['GET'])
def get_cameras():
    cameras = get_system_cameras()
    configured = config.load_obs_cam_from_file()
    active_index = find_obs_camera_index(configured, cameras=cameras)
    return jsonify({
        "status": "success",
        "cameras": cameras,
        "configured": configured,
        "active_index": active_index
    })

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

@settings_bp.route('/set_settings', methods=['POST'])
def set_settings():
    data = request.json or {}
    if 'conf' in data:
        val = float(data['conf'])
        config.current_conf["obs"] = val
        config.current_conf["video"] = val
        config.save_conf_to_file(val)
    if 'classes' in data and isinstance(data['classes'], list):
        config.selected_classes = data['classes']
        config.save_classes_to_file(data['classes'])
    if 'dwell_time' in data:
        val = max(1.0, float(data['dwell_time']))
        config.dwell_time = val
        config.save_dwell_to_file(val)
    if 'obs_cam' in data:
        cam = str(data['obs_cam']).strip()
        config.obs_cam = cam
        config.save_obs_cam_to_file(cam)
    return jsonify({
        "status": "success",
        "conf": config.current_conf.get("obs", config.initial_conf),
        "classes": config.selected_classes,
        "dwell_time": config.dwell_time,
        "train_mode": config.train_mode,
        "obs_cam": config.obs_cam
    })
