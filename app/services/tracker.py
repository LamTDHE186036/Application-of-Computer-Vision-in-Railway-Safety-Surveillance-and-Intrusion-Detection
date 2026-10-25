import cv2
import numpy as np

def reset_tracker(model):
    """
    Khôi phục trạng thái bộ theo dõi ByteTrack khi đổi stream hoặc lặp lại video
    """
    try:
        if hasattr(model, 'predictor') and hasattr(model.predictor, 'trackers'):
            for t in model.predictor.trackers:
                if hasattr(t, 'reset'):
                    t.reset()
    except Exception:
        pass

def check_roi_intersection(box_xyxy, roi_mask, pts, frame_shape):
    """
    Kiểm tra xem bounding box có giao cắt với vùng ROI hay không.
    Dùng kết hợp cả mặt nạ (mask overlap) và kiểm tra tâm đáy vật thể (pointPolygonTest).
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box_xyxy
    y1_c, y2_c = max(0, y1), min(h, y2)
    x1_c, x2_c = max(0, x1), min(w, x2)
    
    if y1_c < y2_c and x1_c < x2_c:
        box_mask = roi_mask[y1_c:y2_c, x1_c:x2_c]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        in_polygon = cv2.pointPolygonTest(pts, (cx, cy), False) >= 0
        return cv2.countNonZero(box_mask) > 0 or in_polygon
    return False

def update_dwell_and_evaluate_violation(
    track_id, current_time, current_train_mode, dwell_limit,
    roi_dwell_tracker, min_confirm_seconds=0.5
):
    """
    Cập nhật thời gian lưu trú trong ROI và đánh giá trạng thái vi phạm:
    - Chế độ 1 (train_approaching): Vi phạm ngay khi đối tượng ở trong ROI >= min_confirm_seconds (chống nhiễu nháy)
    - Chế độ 2 (no_train): Chỉ vi phạm khi ở trong ROI quá dwell_limit giây
    """
    if track_id not in roi_dwell_tracker:
        roi_dwell_tracker[track_id] = {
            "enter_time": current_time,
            "last_seen": current_time,
            "violating": False
        }
    else:
        roi_dwell_tracker[track_id]["last_seen"] = current_time

    dwell_duration = current_time - roi_dwell_tracker[track_id]["enter_time"]

    if current_train_mode == 'train_approaching':
        is_violation = dwell_duration >= min_confirm_seconds
        roi_dwell_tracker[track_id]["violating"] = is_violation
        label_suffix = f"({dwell_duration:.1f}s)"
    else:
        if dwell_duration >= dwell_limit:
            is_violation = True
            roi_dwell_tracker[track_id]["violating"] = True
            label_suffix = f"(DUNG {dwell_duration:.0f}s)"
        else:
            is_violation = False
            label_suffix = f"({dwell_duration:.1f}s/{dwell_limit:.0f}s)"

    return is_violation, dwell_duration, label_suffix
